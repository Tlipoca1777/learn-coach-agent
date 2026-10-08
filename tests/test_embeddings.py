r"""
向量化层验收测试
================================================================

这些测试覆盖已经实现好的 `embeddings.py`，和你的作业无关。

运行：
    # 快速测试（秒级，用假向量模型）
    .\.venv\Scripts\python.exe -m pytest tests/test_embeddings.py -v

    # 加上真实模型测试（首次会下载约 90MB）
    $env:RUN_SLOW_TESTS="1"
    .\.venv\Scripts\python.exe -m pytest tests/test_embeddings.py -v

--------------------------------------------------------------------
这里演示两个值得学的测试技巧
--------------------------------------------------------------------
【1】用假实现测逻辑，用真实现测效果
    `FakeEmbeddings` 让你在毫秒内测完"向量维度对不对、归一化没有、
    相似度排序合不合理"这些**逻辑**问题。
    而"中文语义检索到底准不准"这种**效果**问题，只能拿真模型测，
    所以那几条被标记成慢测试，默认跳过。

    这个划分很重要：如果你所有测试都要加载真模型，
    跑一次测试要几分钟，你就不会想跑测试了，最后测试形同虚设。

【2】为"踩过的坑"写回归测试
    注意下面那条 `test_chinese_semantics_work`。
    它测的正是我实测发现的那个严重问题：
    Chroma 默认的英文模型会让"我平时喝什么咖啡？"
    匹配到"我养了一只橘猫"（因为它不认识中文语义）。

    这条测试的价值是：**如果以后有人把向量模型换回英文的，
    测试会立刻失败**，而不是等到上线后发现检索全是错的。
    给已知 bug 写一条测试，是防止它复发的最有效手段。
"""

import math
import os

import pytest

from memory_assistant.embeddings import (
    DEFAULT_HF_ENDPOINT,
    ENV_DISABLE_SYMLINK_WARNING,
    ENV_FASTEMBED_CACHE,
    ENV_HF_ENDPOINT,
    FakeEmbeddings,
    create_embeddings,
    prepare_hf_environment,
)

# 需要真实模型（下载 90MB）的测试，默认跳过
slow = pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="真实向量模型测试较慢（首次要下载约 90MB）。设置 RUN_SLOW_TESTS=1 才跑。",
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@pytest.fixture
def clean_env():
    """
    保护 os.environ 不被测试污染。

    为什么必须要有？因为 prepare_hf_environment() 会直接改 os.environ。
    如果不保存和恢复，这些改动会**泄漏到后面的测试**，
    造成"单独跑通过、一起跑就失败"这种最难排查的问题。
    """
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def restore_hf_constants():
    """
    保护 huggingface_hub 的模块常量不被测试污染。

    因为 prepare_hf_environment() 会去改 `constants.ENDPOINT`。
    如果不恢复，后面所有真实下载都会走一个不存在的域名 ——
    而失败信息是"下载超时"，你会完全想不到是测试改坏了全局状态。

    ⚠️ 这类"测试污染了全局可变状态"的问题，是新手最难 debug 的一类 bug。
       解决办法就是像这样：**谁改谁负责恢复**。
    """
    import huggingface_hub.constants as constants

    saved = (
        constants.ENDPOINT,
        constants.HUGGINGFACE_CO_URL_TEMPLATE,
        constants.HF_HUB_DISABLE_SYMLINKS_WARNING,
    )
    yield
    (
        constants.ENDPOINT,
        constants.HUGGINGFACE_CO_URL_TEMPLATE,
        constants.HF_HUB_DISABLE_SYMLINKS_WARNING,
    ) = saved


# ====================================================================
# 组 1：FakeEmbeddings 的基本行为
# ====================================================================


def test_fake_dimension():
    embedder = FakeEmbeddings(dimension=64)
    assert embedder.dimension == 64

    vectors = embedder.embed_documents(["你好", "世界"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 64


def test_fake_rejects_bad_dimension():
    with pytest.raises(ValueError):
        FakeEmbeddings(dimension=0)
    with pytest.raises(ValueError):
        FakeEmbeddings(dimension=-1)


def test_fake_is_deterministic():
    """
    同样的文本必须永远得到同样的向量。

    ⚠️ 这里有个真实的坑：如果实现里用了 Python 内置的 hash()，
       同一段文本在**不同的进程**里会得到不同的向量
       （Python 对字符串的 hash 默认加了随机盐）。
       那样测试就会时好时坏。所以实现用的是 zlib.crc32。
    """
    a = FakeEmbeddings()
    b = FakeEmbeddings()

    va = a.embed_documents(["我叫小明"])[0]
    vb = b.embed_documents(["我叫小明"])[0]

    assert va == vb


def test_fake_empty_input_returns_empty_list():
    embedder = FakeEmbeddings()
    assert embedder.embed_documents([]) == []


def test_fake_handles_empty_string_without_crashing():
    """空字符串应该返回零向量，而不是除以零崩掉。"""
    embedder = FakeEmbeddings(dimension=16)
    vector = embedder.embed_query("")
    assert len(vector) == 16
    assert all(value == 0.0 for value in vector)


def test_fake_vectors_are_l2_normalized():
    """
    向量长度应该是 1。

    为什么重要？因为归一化之后，余弦相似度就等于点积，
    计算更快，而且不同长度文本之间比较才公平。
    """
    embedder = FakeEmbeddings(dimension=64)
    vector = embedder.embed_documents(["我喜欢喝手冲咖啡，不加糖"])[0]
    norm = math.sqrt(sum(value * value for value in vector))
    assert abs(norm - 1.0) < 1e-9


def test_fake_similar_texts_score_higher():
    """有字面重合的文本，相似度应该更高 —— 这是排序逻辑的基础。"""
    embedder = FakeEmbeddings()
    base = embedder.embed_query("我喜欢喝手冲咖啡")

    similar = embedder.embed_documents(["我喜欢喝手冲咖啡，不加糖"])[0]
    unrelated = embedder.embed_documents(["相对论描述了时空的几何结构"])[0]

    assert cosine(base, similar) > cosine(base, unrelated)


def test_fake_identical_text_scores_one():
    embedder = FakeEmbeddings()
    a = embedder.embed_query("完全一样的文本")
    b = embedder.embed_documents(["完全一样的文本"])[0]
    assert abs(cosine(a, b) - 1.0) < 1e-9


def test_fake_counts_calls():
    """计数功能让测试可以验证"有没有多余调用"。"""
    embedder = FakeEmbeddings()
    assert embedder.call_count == 0

    embedder.embed_documents(["a"])
    embedder.embed_query("b")
    assert embedder.call_count == 2


def test_fake_repr():
    assert "FakeEmbeddings" in repr(FakeEmbeddings(dimension=32))


# ====================================================================
# 组 2：工厂函数
# ====================================================================


def test_create_embeddings_fake_mode():
    embedder = create_embeddings(fake=True, dimension=32)
    assert isinstance(embedder, FakeEmbeddings)
    assert embedder.dimension == 32


def test_create_embeddings_requires_config_or_model_name():
    with pytest.raises(ValueError, match="必须提供 config 或 model_name"):
        create_embeddings()


# ====================================================================
# 组 3：环境变量准备（顺序陷阱那一块）
# ====================================================================


def test_prepare_env_sets_endpoint_and_cache(clean_env, tmp_path):
    os.environ.pop(ENV_HF_ENDPOINT, None)
    cache = tmp_path / "models"

    applied = prepare_hf_environment(DEFAULT_HF_ENDPOINT, cache)

    assert os.environ[ENV_HF_ENDPOINT] == DEFAULT_HF_ENDPOINT
    assert os.environ[ENV_FASTEMBED_CACHE] == str(cache)
    assert os.environ[ENV_DISABLE_SYMLINK_WARNING] == "1"
    assert applied[ENV_HF_ENDPOINT] == DEFAULT_HF_ENDPOINT


def test_prepare_env_creates_cache_directory(clean_env, tmp_path):
    cache = tmp_path / "还不存在的" / "目录"
    assert not cache.exists()

    prepare_hf_environment(DEFAULT_HF_ENDPOINT, cache)

    assert cache.is_dir()


def test_prepare_env_does_not_override_existing_endpoint(clean_env, tmp_path):
    """
    如果用户已经显式设置了 HF_ENDPOINT，我们不能覆盖他。

    场景：有人就是能直连 huggingface.co，或者用公司内网镜像。
    强行改成我们的默认值会破坏他的配置。
    """
    os.environ[ENV_HF_ENDPOINT] = "https://my-company-mirror.example.com"

    prepare_hf_environment(DEFAULT_HF_ENDPOINT, tmp_path)

    assert os.environ[ENV_HF_ENDPOINT] == "https://my-company-mirror.example.com"


def test_prepare_env_with_none_endpoint_skips_it(clean_env, tmp_path):
    """传 None 表示"用官方站"，那就不该设置这个环境变量。"""
    os.environ.pop(ENV_HF_ENDPOINT, None)

    applied = prepare_hf_environment(None, tmp_path)

    assert ENV_HF_ENDPOINT not in os.environ
    assert ENV_HF_ENDPOINT not in applied
    # 但缓存目录和警告开关还是要设置
    assert os.environ[ENV_FASTEMBED_CACHE] == str(tmp_path)


def test_prepare_env_without_cache_dir(clean_env):
    os.environ.pop(ENV_HF_ENDPOINT, None)

    applied = prepare_hf_environment("https://example.com", None)

    assert ENV_FASTEMBED_CACHE not in os.environ
    assert ENV_FASTEMBED_CACHE not in applied


def test_prepare_env_patches_endpoint_if_hub_already_imported(
    clean_env, restore_hf_constants, tmp_path
):
    """
    ⭐ 顺序陷阱的**自动修正**。

    背景：`huggingface_hub` 在 import 时就把 HF_ENDPOINT 固化成了模块常量，
    之后再改环境变量是无效的。所以如果有人（或某个库）提前 import 了它，
    我们的镜像设置就会静默失效 —— 表现是"下载一直超时"，很难排查。

    我们的做法不是"警告一下就算了"，而是**直接把常量改掉**。
    能这么做是因为它内部几乎所有地方都是调用时才读 `constants.ENDPOINT`，
    而不是 import 时绑定的局部名字。

    这条测试验证修正真的生效。
    """
    import huggingface_hub.constants as constants

    os.environ.pop(ENV_HF_ENDPOINT, None)
    target = "https://definitely-not-the-real-endpoint.example"

    applied = prepare_hf_environment(target, tmp_path)

    assert constants.ENDPOINT == target
    assert applied.get("ENDPOINT_patched") == target
    # 下载地址模板也要跟着改，否则文件还是从老域名拿
    assert constants.HUGGINGFACE_CO_URL_TEMPLATE.startswith(target + "/")


def test_prepare_env_does_not_patch_when_endpoint_already_correct(
    clean_env, restore_hf_constants, tmp_path
):
    """如果常量已经是对的，就不该多此一举地"修正"。"""
    import huggingface_hub.constants as constants

    prepare_hf_environment(DEFAULT_HF_ENDPOINT, tmp_path)
    # 先让它变成目标值
    assert constants.ENDPOINT == DEFAULT_HF_ENDPOINT

    # 再调一次，不该产生 patched 标记
    applied = prepare_hf_environment(DEFAULT_HF_ENDPOINT, tmp_path)
    assert "ENDPOINT_patched" not in applied


def test_prepare_env_patches_symlink_warning_flag(clean_env, restore_hf_constants, tmp_path):
    """
    这是个更隐蔽的同类问题。

    `HF_HUB_DISABLE_SYMLINKS_WARNING` 和 `ENDPOINT` 一样，
    也是在 huggingface_hub 的模块顶层用 os.getenv 读的，所以同样会被冻结。

    症状：Windows 上每次下载模型都会打印一大段关于"不支持符号链接"的警告。
    虽然不影响功能，但会让日志很吵，而且新手会以为出错了。

    （这个坑是我自己实测时发现的：明明设了环境变量，警告还是照样打印。
      能用上"环境变量在 import 时被冻结"这条规律，才定位到原因。）
    """
    import huggingface_hub.constants as constants

    constants.HF_HUB_DISABLE_SYMLINKS_WARNING = False

    applied = prepare_hf_environment(DEFAULT_HF_ENDPOINT, tmp_path)

    assert constants.HF_HUB_DISABLE_SYMLINKS_WARNING is True
    assert applied.get("symlink_warning_patched") == "1"


# ====================================================================
# 组 4：真实向量模型（默认跳过，用 RUN_SLOW_TESTS=1 开启）
# ====================================================================


@slow
def test_local_embeddings_dimension():
    from memory_assistant.config import Config

    embedder = create_embeddings(Config.from_env(require_key=False))
    assert embedder.dimension == 512
    assert "bge-small-zh" in embedder.model_name


@slow
def test_local_embeddings_is_deterministic():
    from memory_assistant.config import Config

    embedder = create_embeddings(Config.from_env(require_key=False))
    a = embedder.embed_query("我叫小明")
    b = embedder.embed_query("我叫小明")
    assert a == b


@slow
def test_chinese_semantics_work():
    """
    ⭐ 这是为"我实测踩过的坑"写的回归测试。

    背景：Chroma 默认的英文向量模型（all-MiniLM-L6-v2）在中文上基本失效。
    实测数据（同一组候选项）：

        查询"我平时喝什么咖啡？"
            英文模型：咖啡句排第 3（0.64），输给"我养了一只橘猫"(0.68)
            中文模型：咖啡句排第 1（0.67），第 2 名只有 0.33

    这条测试确保：**换了中文模型之后，语义改写能正确检索**。
    如果以后有人把模型换回英文的，或者换成别的差模型，这条会立刻失败。

    这比"检查维度是 512"有意义得多 —— 维度对不代表效果对。
    """
    from memory_assistant.config import Config

    embedder = create_embeddings(Config.from_env(require_key=False))

    corpus = [
        "我喜欢喝手冲咖啡，不加糖",
        "我养了一只橘猫叫豆豆",
        "我对花生过敏，点菜要注意",
        "相对论描述了时空的几何结构",
    ]
    vectors = embedder.embed_documents(corpus)

    def top_hit(query: str) -> str:
        query_vector = embedder.embed_query(query)
        scored = sorted(
            ((cosine(query_vector, v), text) for v, text in zip(vectors, corpus)),
            reverse=True,
        )
        return scored[0][1]

    # 这三条都是"语义改写"—— 和候选句几乎没有字面重合。
    # 英文模型全部会挂，中文模型应该全中。
    assert top_hit("我平时喝什么咖啡？") == "我喜欢喝手冲咖啡，不加糖"
    assert top_hit("我家猫咪叫什么名字？") == "我养了一只橘猫叫豆豆"
    assert top_hit("我吃花生会怎么样？") == "我对花生过敏，点菜要注意"


@slow
def test_relevant_scores_beat_irrelevant():
    """
    相关查询的相似度要明显高于无关查询。

    这条测的是"区分度"。如果模型对所有文本都给出 0.6 左右的相似度，
    那它就等于没用 —— 检索结果会变成随机抽签。
    """
    from memory_assistant.config import Config

    embedder = create_embeddings(Config.from_env(require_key=False))

    corpus = [
        "我喜欢喝手冲咖啡，不加糖",
        "我养了一只橘猫叫豆豆",
        "我对花生过敏，点菜要注意",
        "相对论描述了时空的几何结构",
    ]
    vectors = embedder.embed_documents(corpus)

    def best_score(query: str) -> float:
        query_vector = embedder.embed_query(query)
        return max(cosine(query_vector, v) for v in vectors)

    relevant = min(
        best_score("我平时喝什么咖啡？"),
        best_score("我家猫咪叫什么名字？"),
        best_score("我吃花生会怎么样？"),
    )
    irrelevant = best_score("请解释一下深度学习中的注意力机制")

    assert relevant > irrelevant, (
        f"相关查询最高分 {relevant:.4f} 应该高于无关查询 {irrelevant:.4f}。\n"
        "如果这条挂了，说明向量模型没有区分度，检索会失去意义。"
    )
