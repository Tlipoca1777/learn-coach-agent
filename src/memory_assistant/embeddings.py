r"""
向量化层（Embeddings）
================================================================

职责：把文本变成向量。这是长期记忆的入口 —— 没有向量就没有语义检索。

--------------------------------------------------------------------
⚠️ 这一层踩过的坑（都是实测出来的，不是猜的）
--------------------------------------------------------------------

【坑 1】DeepSeek 没有 embedding 接口
    DeepSeek 只提供对话模型，**没有任何向量化模型**。
    （参考 https://github.com/deepseek-ai/DeepSeek-V3/issues/806）
    所以「用 DeepSeek 的 key」这件事，只能解决聊天，解决不了检索。
    向量必须另找来源。

【坑 2】Chroma 自带的默认向量模型在中文上基本不可用
    Chroma 默认用 `all-MiniLM-L6-v2`，那是个**英文模型**。
    我实测了同一组中文查询（4 条候选句，预算内谁排第一）：

    查询                              英文 MiniLM          中文 BGE-small-zh
    --------------------------------  -------------------  ------------------
    我平时喝什么咖啡？                 咖啡句排第 3 ❌      咖啡句排第 1 ✅
                                      (0.64，输给"猫"和    (0.67，第 2 名只有
                                       "花生过敏"的 0.68)   0.33，区分度很大)
    我家猫咪叫什么名字？               猫句排第 2 ❌        猫句排第 1 ✅
                                      (0.56，输给 0.61)    (0.66)
    我吃花生会怎么样？                 第 1 ✅ (0.89)       第 1 ✅ (0.64)
                                      字面重合才行
    无关查询最高分                     0.59（几乎无区分度）  0.39（明显偏低）

    规律很清楚：**英文模型只在「字面有重合」时有效**
    （"花生"两个字都在），一旦要做语义改写（"喝什么咖啡" → "喜欢喝手冲咖啡"）
    就完全失效。
    这就是为什么必须换中文模型 —— 这个结论是跑出来的，不是想出来的。

【坑 3】HuggingFace 在国内连不上，必须用镜像
    实测：
        huggingface.co   12 秒超时，不可达
        hf-mirror.com    330 毫秒，可达  ✅
        modelscope.cn    273 毫秒，可达  ✅

    所以必须设置 HF_ENDPOINT。而且有个**顺序陷阱**：
    环境变量必须在 `import fastembed` **之前**设置，
    因为 huggingface_hub 在 import 时就会把 endpoint 读成模块级常量，
    之后再改 os.environ 就没用了。

    这个模块用「延迟 import + 先配环境」的方式解决了它 ——
    使用者不需要改任何系统环境变量。

【坑 4】Windows 上 fastembed 的缓存默认放在系统临时目录
    临时目录会被清理，模型动辄一百多 MB，重下一次很烦。
    我们强制把它指向项目内的 `data/models/`。

--------------------------------------------------------------------
选型结论
--------------------------------------------------------------------
    默认模型：BAAI/bge-small-zh-v1.5
        512 维，约 90MB，中文优化，纯 ONNX（**不需要装 PyTorch**）
        实测：语义改写场景 4/4 命中，区分度清晰

    为什么不选别的：
        · all-MiniLM-L6-v2（Chroma 默认）—— 实测中文效果差，见坑 2
        · sentence-transformers 系 —— 要装 PyTorch，2GB+，对新手太重
        · 调用别家的 embedding API —— 要多一个 key，多一份钱，
          而且个人助理这种低并发场景没必要
        · bge-large-zh —— 效果更好但 1.3GB，个人项目不值当

    ⚠️ 一个实测细节：BGE 官方建议给 query 加指令前缀
       （"为这个句子生成表示以用于检索相关文章："），
       但在 bge-**small**-zh-v1.5 上实测**反而略微变差**：
           咖啡查询：不加前缀 0.6694  >  加前缀 0.6283
       所以我们默认**不加**。query_instruction 参数留着，
       换大模型时你可以自己再测一遍。
       —— 不要迷信文档里的建议，要自己测。

--------------------------------------------------------------------
怎么用
--------------------------------------------------------------------
    from memory_assistant.config import Config
    from memory_assistant.embeddings import create_embeddings

    config = Config.from_env()
    embedder = create_embeddings(config)          # 真实模型（首次会下载）
    embedder = create_embeddings(config, fake=True)  # 假模型（测试用，秒级）

    vectors = embedder.embed_documents(["我喜欢手冲咖啡", "我养了只猫"])
    query_vector = embedder.embed_query("我喝什么咖啡？")
    print(embedder.dimension)
"""

import os
import sys
import zlib
from pathlib import Path
from typing import Protocol

# 默认模型：中文优化、体积小、纯 ONNX
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 国内可用的 HuggingFace 镜像。
# 如果你的网络能直连 huggingface.co，把它设为 None 即可。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 环境变量名（抽成常量，避免拼错字符串）
ENV_HF_ENDPOINT = "HF_ENDPOINT"
ENV_FASTEMBED_CACHE = "FASTEMBED_CACHE_PATH"
ENV_DISABLE_SYMLINK_WARNING = "HF_HUB_DISABLE_SYMLINKS_WARNING"


# ====================================================================
# 一、接口
# ====================================================================
class EmbeddingProvider(Protocol):
    """
    任何能把文本变成向量的对象都要满足这个接口。

    为什么要抽象成接口？
        因为"用哪个向量模型"是会变的：
        现在用本地 BGE，将来可能换成 API，测试时用假向量。
        上层代码只依赖这个接口，就不需要跟着改。
    """

    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批文档变成向量（用于入库）。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """把一条查询变成向量（用于检索）。"""
        ...


# ====================================================================
# 二、环境准备 —— 必须在 import fastembed 之前调用
# ====================================================================
def prepare_hf_environment(
    hf_endpoint: str | None = DEFAULT_HF_ENDPOINT,
    cache_dir: Path | str | None = None,
) -> dict[str, str]:
    """
    配置 HuggingFace 相关的环境变量。

    ⚠️ 必须在 `import fastembed` 之前调用，原因见模块文档【坑 3】。

    参数 hf_endpoint：
        镜像地址。传 None 表示用官方站（需要能直连）。
        用 setdefault 语义：如果你已经在系统里显式设过，我们不会覆盖你。

    返回：实际设置的环境变量字典（方便自检脚本打印和测试断言）。
    """
    applied: dict[str, str] = {}

    if hf_endpoint:
        # setdefault：不覆盖用户已有的设置。
        # 这样"我想用官方站"的显式配置不会被我们强行改成镜像。
        if ENV_HF_ENDPOINT not in os.environ:
            os.environ[ENV_HF_ENDPOINT] = hf_endpoint
        applied[ENV_HF_ENDPOINT] = os.environ[ENV_HF_ENDPOINT]

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ[ENV_FASTEMBED_CACHE] = str(cache_path)
        applied[ENV_FASTEMBED_CACHE] = str(cache_path)

    # Windows 不支持符号链接，huggingface_hub 会啰嗦地警告一大段，关掉它
    os.environ[ENV_DISABLE_SYMLINK_WARNING] = "1"
    applied[ENV_DISABLE_SYMLINK_WARNING] = "1"

    # ---------- 顺序陷阱的自动修正 ----------
    # huggingface_hub 在 import 时会把一批环境变量读成**模块级常量**，例如：
    #     constants.py:69   ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    #     constants.py:282  HF_HUB_DISABLE_SYMLINKS_WARNING = _is_true(os.getenv(...))
    #
    # 所以如果它已经被 import 过，我们再改环境变量就完全没用了。
    #
    # ⭐ 这是个通用规律，值得记住：
    #     **任何在模块顶层用 os.getenv() 读的配置，都会在 import 那一刻被冻结。**
    #     你要么保证在 import 之前设好环境变量，要么事后去改它的常量。
    #     这类 bug 的表现是"我明明设了环境变量却不生效"，极难排查。
    #
    # 我们两手都做：
    #   1. 先在 os.environ 里设好 —— 应对"还没被 import"的正常情况
    #   2. 再检查是否已经被 import，是就把常量改掉 —— 应对"已经被 import"的最坏情况
    # 这样无论导入顺序如何都能正常工作，调用方完全不用操心。
    if "huggingface_hub.constants" in sys.modules:
        import huggingface_hub.constants as constants

        wanted = os.environ.get(ENV_HF_ENDPOINT, "").rstrip("/")
        current = str(constants.ENDPOINT).rstrip("/")

        if wanted and wanted != current:
            constants.ENDPOINT = wanted
            # 这个模板是 import 时按旧 ENDPOINT 拼好的，必须一起改，
            # 否则文件下载地址还是指向老的域名。
            constants.HUGGINGFACE_CO_URL_TEMPLATE = (
                wanted + "/{repo_id}/resolve/{revision}/{filename}"
            )
            applied["ENDPOINT_patched"] = wanted

        # 同样的道理：这个开关也是 import 时固化的
        if not constants.HF_HUB_DISABLE_SYMLINKS_WARNING:
            constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
            applied["symlink_warning_patched"] = "1"

    return applied


# ====================================================================
# 三、真实向量模型（本地 ONNX）
# ====================================================================
class LocalEmbeddings:
    """
    基于 fastembed 的本地向量模型。

    特点：
        · 纯 ONNX 推理，**不需要 PyTorch**（省 2GB 依赖）
        · 模型在本地跑，不联网、不花钱、数据不出本机
        · 首次使用会下载模型（约 90MB），之后走本地缓存

    为什么不用异步？
        ONNX 推理是 CPU 密集型的同步操作，套 async 没有意义
        （它不会把 CPU 让出去）。真要提高吞吐得用多进程，
        个人助理这种低并发场景完全不需要。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        cache_dir: Path | str | None = None,
        hf_endpoint: str | None = DEFAULT_HF_ENDPOINT,
        query_instruction: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction

        # ⚠️ 顺序很关键：先配环境，再 import fastembed
        self.environment = prepare_hf_environment(hf_endpoint, cache_dir)

        from fastembed import TextEmbedding  # 延迟 import，见坑 3

        kwargs: dict = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if threads is not None:
            kwargs["threads"] = threads

        self._model = TextEmbedding(**kwargs)
        self._dimension: int | None = None

    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        """
        向量维度。

        为什么要延迟获取？因为查维度需要先把模型加载起来，
        而加载模型要花几秒。只有真正需要时才付这个成本。

        ⚠️ 这里有个很容易写错的地方（我自己就写错了，被测试抓到）：
            embed_documents() 返回的是 list[list[float]]，
            所以 len(结果) 是**文本条数**，不是维度！
            要取 len(结果[0]) 才是维度。
        """
        if self._dimension is None:
            vectors = self.embed_documents(["维度探测"])
            self._dimension = len(vectors[0])
        return self._dimension

    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        把一批文档变成向量。

        注意返回的是 list[list[float]]，而不是 numpy 数组。
        为什么？因为：
            · numpy 数组不能直接 JSON 序列化（存数据库、传给前端会挂）
            · 上层代码不该被迫依赖 numpy
        代价是多了一次类型转换，在这种数据量下可以忽略。
        """
        if not texts:
            return []
        # fastembed 返回的是 numpy 数组的生成器，转成普通 list
        return [[float(value) for value in vector] for vector in self._model.embed(texts)]

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        """
        把查询变成向量。

        query_instruction 默认是 None，即不加前缀 —— 这是**实测**的结论，
        见模块文档末尾。换成更大的 BGE 模型时建议重新测一遍。
        """
        if self.query_instruction:
            text = self.query_instruction + text
        return self.embed_documents([text])[0]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<LocalEmbeddings {self.model_name}>"


# ====================================================================
# 四、假向量模型（测试和离线开发用）
# ====================================================================
class FakeEmbeddings:
    """
    确定性的假向量模型 —— 这是你写测试时的主力工具。

    它不下载模型、不联网、毫秒级返回，而且**结果完全可复现**。

    原理：字符哈希词袋模型
        把每个字符用 crc32 映射到向量的某一维，统计出现次数，然后归一化。
        于是"共享字符越多"的两段文本，向量越接近。

    这当然不是一个好的语义模型（"猫咪"和"橘猫"共享 0 个字符，
    相似度是 0），但对测试**足够了**：
        · 相同的文本 → 相同的向量（确定性）
        · 有字面重合的文本 → 相似度更高（能验证排序逻辑）
        · 完全不相关的文本 → 相似度接近 0（能验证阈值逻辑）

    ⚠️ 一个必须知道的坑：Python 内置的 hash() 对字符串是**每个进程随机**的
       （为了防哈希碰撞攻击）。如果用它做向量，同一段文本在两次运行里
       会得到不同向量，测试就会时好时坏。
       所以这里用 zlib.crc32 —— 它稳定、跨进程一致。
    """

    def __init__(self, model_name: str = "fake-hash-embeddings", dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension 必须大于 0，收到的是 {dimension}")
        self.model_name = model_name
        self.dimension = dimension
        self.call_count = 0

    # ------------------------------------------------------------------
    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        for char in text:
            # crc32 是稳定的（同一输入永远同一输出，跨进程也一样）
            index = zlib.crc32(char.encode("utf-8")) % self.dimension
            vector[index] += 1.0

        # L2 归一化：让向量长度为 1，这样点积就等于余弦相似度
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            # 空文本：返回一个固定的零向量，而不是崩溃
            return vector
        return [value / norm for value in vector]

    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._vector(text) for text in texts]

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        return self._vector(text)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<FakeEmbeddings dim={self.dimension}>"


# ====================================================================
# 五、工厂函数
# ====================================================================
def create_embeddings(
    config=None,
    *,
    fake: bool = False,
    model_name: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """
    根据配置创建向量模型。

    参数 fake：
        True 时返回假模型，用于测试和离线开发。
        这和 `create_llm(config, fake=True)` 是同一个套路 ——
        保持一致的接口，上层代码切换起来毫无成本。
    """
    if fake:
        return FakeEmbeddings(**kwargs)

    if model_name is not None:
        return LocalEmbeddings(model_name, **kwargs)

    if config is None:
        raise ValueError("不传 fake=True 时，必须提供 config 或 model_name")

    return LocalEmbeddings(
        config.embedding_model,
        cache_dir=config.embedding_cache_dir,
        hf_endpoint=config.hf_endpoint,
        **kwargs,
    )
