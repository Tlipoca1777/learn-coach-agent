r"""
配置模块
================================================================

职责：把所有「可以调整的东西」集中到一个地方。

为什么要这么做？
    新手最常见的写法是把 API Key、模型名、各种数字散落在几十个文件里。
    等到要改一个参数时，你得全局搜索，还容易漏。
    正确的做法是：**所有配置只在这里读一次，其他地方都从这里拿。**

这个文件已经帮你写好了（属于"管道代码"，不是学习重点），
你可以直接 import 使用：

    from memory_assistant.config import Config

    config = Config.from_env()
    print(config.model)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------
# 项目根目录
# --------------------------------------------------------------------
# __file__ 是当前文件的路径：D:\Agent开发\src\memory_assistant\config.py
# .resolve() 转成绝对路径
# .parents 是"所有上级目录"的列表：
#     parents[0] = D:\Agent开发\src\memory_assistant
#     parents[1] = D:\Agent开发\src
#     parents[2] = D:\Agent开发          ← 我们要的
#
# 这么写的好处：不管你在哪个目录下运行程序，都能正确定位到项目根目录。
# 如果你写死 "D:\Agent开发"，项目一换电脑就废了。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """
    全部配置项。

    关于 @dataclass：
        它是一个装饰器，自动帮你生成 __init__ / __repr__ / __eq__。
        没有它你得手写一大堆 self.xxx = xxx，很容易写错。
        这是我们第一次用"类 + 装饰器"，可以先照着用，后面学类的时候再回头看。

    关于 frozen=True：
        表示"创建之后就不能改了"，也就是不可变对象。
        好处是配置一旦加载就不会被中途意外篡改，是个好习惯。
    """

    # ---------- 模型相关 ----------
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int

    # ---------- 记忆相关 ----------
    # 模型一次能处理的 token 上限（上下文窗口）。
    # 注意：这个值随模型版本变化，请以官网文档为准。
    context_window: int

    # 我们给自己留的安全余量：prompt 最多只能占到 context_window 的这个比例。
    # 为什么不占满？因为：
    #   1. 要给模型的回答留出空间（max_tokens）
    #   2. token 估算本身有误差，必须留 buffer
    prompt_budget_ratio: float

    # ---------- 向量化相关 ----------
    # ⚠️ DeepSeek 没有 embedding 接口，所以向量必须用别的模型。
    #    默认用本地的中文 BGE 小模型（约 90MB，纯 ONNX，不需要 PyTorch）。
    #    选型理由和实测数据见 embeddings.py 的模块文档。
    embedding_model: str

    # HuggingFace 端点。国内必须走镜像，否则下载会超时。
    # 设为 None 表示用官方站（需要能直连 huggingface.co）。
    hf_endpoint: str | None

    # ---------- 路径相关 ----------
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    # ------------------------------------------------------------------
    # 属性（property）：像访问变量一样访问它，实际是算出来的
    # ------------------------------------------------------------------
    @property
    def prompt_token_budget(self) -> int:
        """
        单次请求中，prompt（输入）最多能用的 token 数。

        这是整个记忆系统的**硬约束**：
        所有记忆层加起来都不能超过这个数，超了就必须裁剪。
        """
        return int(self.context_window * self.prompt_budget_ratio)

    @property
    def embedding_cache_dir(self) -> Path:
        """
        向量模型的缓存目录。

        为什么强制放在项目内，而不是用 fastembed 的默认位置？
            因为它的默认位置是系统临时目录（%TEMP%），
            而临时目录是会被清理的。模型有一百多 MB，重下一次很烦。
        """
        return self.data_dir / "models"

    # ------------------------------------------------------------------
    # 构造函数
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, require_key: bool = True) -> "Config":
        """
        从 .env 文件 + 系统环境变量里读取配置。

        参数 require_key：
            默认 True，表示"没有 API Key 就直接报错"。
            写测试的时候传 False，因为测试不需要真的调模型。

        关于 @classmethod：
            它表示"这个方法属于类本身，不属于某个对象"。
            所以调用时写 Config.from_env()，而不是先创建对象再调用。
            你会在大量 Python 库里看到这个模式，是很常见的写法。
        """
        # 读取项目根目录下的 .env 文件。
        # override=False 的意思是：如果系统环境变量里已经有了，就不要被 .env 覆盖。
        # 这样 CI/CD 或临时用命令行设置的环境变量优先级更高，符合预期。
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

        if require_key and not api_key:
            raise RuntimeError(
                "\n"
                "=" * 64 + "\n"
                "[配置错误] 没有找到 DEEPSEEK_API_KEY\n"
                "=" * 64 + "\n"
                "请按下面两步操作：\n"
                "  1. 在项目根目录执行：Copy-Item .env.example .env\n"
                "  2. 打开 .env，把 DEEPSEEK_API_KEY= 后面填上你的真实 Key\n"
                "     申请地址：https://platform.deepseek.com/api_keys\n"
            )

        # HF_ENDPOINT 允许显式设为空字符串，表示"用官方站，不要镜像"。
        # 所以要区分"没设置"和"设置成空"：
        #   os.getenv 返回 None  → 没设置 → 用默认镜像
        #   os.getenv 返回 ""    → 显式清空 → 用官方站（None）
        raw_endpoint = os.getenv("HF_ENDPOINT")
        if raw_endpoint is None:
            hf_endpoint: str | None = "https://hf-mirror.com"
        else:
            hf_endpoint = raw_endpoint.strip() or None

        config = cls(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            # DeepSeek 已将旧的 `deepseek-chat` 兼容名路由到 flash。
            # 使用当前公开模型名，避免依赖服务端的隐式别名。
            model=os.getenv("LLM_MODEL", "deepseek-flash").strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            context_window=int(os.getenv("LLM_CONTEXT_WINDOW", "64000")),
            prompt_budget_ratio=float(os.getenv("LLM_PROMPT_BUDGET_RATIO", "0.5")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip(),
            hf_endpoint=hf_endpoint,
            data_dir=Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))),
        )
        config.validate()
        return config

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        检查配置是否合理，不合理就立刻报错。

        为什么要"立刻"报错？
            如果不检查，一个 temperature=5 的错误配置可能要等到程序跑了一半
            才莫名其妙地失败，排查起来非常痛苦。
            这叫"快速失败"（fail fast），是工程上的好习惯。
        """
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature 必须在 0.0~2.0 之间，当前是 {self.temperature}。"
                "建议 Agent 类应用用 0.2~0.5，需要创意时再用 0.8 以上。"
            )

        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须大于 0，当前是 {self.max_tokens}")

        if self.max_tokens >= self.context_window:
            raise ValueError(
                f"max_tokens({self.max_tokens}) 不能大于等于 "
                f"context_window({self.context_window})，否则模型没有空间放输入。"
            )

        if not 0.0 < self.prompt_budget_ratio <= 1.0:
            raise ValueError(
                f"prompt_budget_ratio 必须在 (0, 1] 之间，当前是 {self.prompt_budget_ratio}"
            )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def ensure_data_dir(self) -> Path:
        """确保数据目录存在，返回它。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def describe(self) -> str:
        """返回一段人类可读的配置摘要，打印出来方便排查问题。"""
        # ⚠️ 注意：这里绝对不能打印 api_key！
        #     日志里泄露密钥是真实项目中非常常见且严重的事故。
        #     所以下面只显示前 6 位和长度。
        masked = f"{self.api_key[:6]}...（共 {len(self.api_key)} 位）" if self.api_key else "（未设置）"
        return (
            f"模型        : {self.model}\n"
            f"接口地址    : {self.base_url}\n"
            f"API Key     : {masked}\n"
            f"temperature : {self.temperature}\n"
            f"max_tokens  : {self.max_tokens}\n"
            f"上下文窗口  : {self.context_window}\n"
            f"prompt 预算 : {self.prompt_token_budget} tokens"
            f"（占窗口 {self.prompt_budget_ratio:.0%}）\n"
            f"向量模型    : {self.embedding_model}\n"
            f"HF 端点     : {self.hf_endpoint or '官方站（需要能直连）'}\n"
            f"数据目录    : {self.data_dir}"
        )


# 说明：这个文件刻意不写 if __name__ == "__main__" 的自检代码。
# 因为如果用 `python -m memory_assistant.config` 运行它，
# 包被 import 时已经加载过一次 config 模块，再当脚本执行会导致模块被加载两次，
# Python 会给出 RuntimeWarning，容易让初学者困惑。
#
# 想检查配置请运行专门的自检脚本：
#     .\.venv\Scripts\python.exe scripts\check_env.py
