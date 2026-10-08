r"""
大模型客户端封装
================================================================

职责：把"怎么调用大模型"这件事封装起来，让上层代码不用关心 HTTP 细节。

为什么要封装？
    假设你的项目里有 20 个地方要调模型。如果每处都写一遍
    client.chat.completions.create(...)，那么：
      - 模型名换了，你要改 20 个地方
      - 想加个重试，你要改 20 个地方
      - 想知道一共花了多少钱，你根本统计不了

    封装成一个类之后，这些事都只在一个地方处理。

这个文件也已经帮你写好了（属于"管道代码"）。
但里面有两个东西**你必须看懂**，因为后面天天用：

    1. estimate_tokens()      —— token 估算，是记忆系统的基础
    2. FakeLLM                —— 假模型，让你不花钱、不联网也能写代码

关于 FakeLLM 为什么重要：
    你是个初学者，调一次真模型要等几秒、还要花钱。
    如果每写一行代码都要真实调用一次，你的学习效率会非常低。
    用假模型，你可以：
      - 一秒跑完 100 次"对话"
      - 反复测试记忆逻辑对不对
      - 不用联网，在火车上也能写代码
    这是专业开发者的基本工作方式，叫"测试替身"（test double）。
"""

import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from openai import OpenAI

from memory_assistant.config import Config


# ====================================================================
# 一、Token 估算 —— 整个记忆系统的度量衡
# ====================================================================
# 为什么不能直接用 len(text)？
#   因为英文和中文的"密度"完全不同：
#     "hello world"     12 个字符，约 2 个 token
#     "你好世界"         4 个字符，约 4 个 token
#   英文大约 4 个字符 = 1 个 token，中文大约 1 个字符 = 1 个 token。
#
# 为什么不用精确的 tokenizer（比如 tiktoken）？
#   因为精确分词需要下载词表、绑定特定模型，而且每次都要跑一遍，
#   在一秒钟要估算上千条记忆的场景下太慢。
#   **工程上，"误差 5% 但快 100 倍"通常比"绝对精确但很慢"更好用。**
#   这个取舍面试时可以直接讲。
#
# ⚠️ 重要：估算一定会有误差，所以你的预算永远要留安全余量，
#         不能算出来刚好 64000 就真发 64000 过去。


def estimate_tokens(text: str) -> int:
    """
    粗略估算一段文本的 token 数。

    规则：
        中日韩字符（CJK）  → 1 个字符 ≈ 1 token
        其他字符（英文等）  → 4 个字符 ≈ 1 token

    这个规则是经验值，对 DeepSeek / GPT 系列都大致适用。
    """
    if not text:
        return 0

    cjk_count = 0
    other_count = 0

    for char in text:
        # 用 Unicode 码点范围判断是不是中日韩字符。
        # ord(char) 把字符转成它的编码数字。
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF      # 中日韩统一表意文字（汉字）
            or 0x3040 <= code <= 0x30FF   # 日文平假名 / 片假名
            or 0xAC00 <= code <= 0xD7AF   # 韩文
            or 0x3000 <= code <= 0x303F   # 中文标点（。，、等）
            or 0xFF00 <= code <= 0xFFEF   # 全角字符
        ):
            cjk_count += 1
        else:
            other_count += 1

    # (other_count + 3) // 4 是"向上取整的除法"，
    # 因为 1 个英文字符也算 1 个 token，不能算成 0。
    return cjk_count + (other_count + 3) // 4


# 每条消息除了正文，还有 role、分隔符等额外开销。
# 这是 OpenAI / DeepSeek API 的固定格式开销，经验值约 4 个 token。
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_message_tokens(message: dict) -> int:
    """估算单条消息（字典）占用的 token 数。"""
    return estimate_tokens(message.get("content", "")) + MESSAGE_OVERHEAD_TOKENS


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算整个消息列表占用的 token 数。"""
    return sum(estimate_message_tokens(m) for m in messages)


# ====================================================================
# 二、用量统计
# ====================================================================
# 养成从第一天就统计成本的意识。
# 面试时如果你能说"这个项目每次请求平均 X token、单次成本 Y 元"，
# 比只会说"我调用了 API"强太多。


@dataclass
class Usage:
    """一次或累计的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        """把另一次调用的用量累加进来。"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.requests += other.requests

    def __str__(self) -> str:
        return (
            f"{self.requests} 次请求 / 输入 {self.prompt_tokens} / "
            f"输出 {self.completion_tokens} / 合计 {self.total_tokens} tokens"
        )


# ====================================================================
# 三、模型接口（Protocol）
# ====================================================================
# Protocol 是"接口声明"：它规定"一个能当模型用的对象必须有哪些方法"。
#
# 好处：只要你的类实现了 chat()，它就能当模型用。
#       于是"真模型"和"假模型"可以随意替换，上层代码完全不用改。
#
# 这在面试里叫"面向接口编程"或"依赖倒置"，是能加分的表述。


class ChatModel(Protocol):
    """任何能聊天的对象都要满足这个接口。"""

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发一次请求，返回完整回答文本。"""
        ...

    def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """发一次请求，一块一块地吐出回答文本。"""
        ...


# ====================================================================
# 四、真模型客户端
# ====================================================================


class DeepSeekClient:
    """
    DeepSeek 客户端。

    DeepSeek 的接口和 OpenAI 完全兼容，所以我们直接复用官方的 openai 库，
    只要把 base_url 指向 DeepSeek 就行。
    """

    def __init__(self, config: Config, client: OpenAI | None = None) -> None:
        """
        参数 client：
            允许从外面传一个已经建好的 OpenAI 客户端进来（默认自己建）。
            为什么留这个口子？因为写单元测试时可以传一个假的客户端进来，
            这样测试就不会真的发网络请求。这个技巧叫"依赖注入"。
        """
        self.config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            # 超时设置很重要！没有超时的话，网络一卡你的程序就永久挂住。
            timeout=60.0,
            # 让库在遇到 429（限流）/ 5xx 时自动重试
            max_retries=3,
        )

        # 累计用量
        self.total_usage = Usage()
        # 最近一次调用的用量
        self.last_usage = Usage()

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发一次普通请求（会等模型全部写完才返回）。"""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )

        # 记录用量
        if response.usage:
            self.last_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                requests=1,
            )
            self.total_usage.add(self.last_usage)

        # 注意：content 有可能是 None（比如模型只返回了工具调用），所以要兜底
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        发一次流式请求（写一个字就吐一个字）。

        关于 yield：
            这个函数里有 yield，所以它是一个"生成器函数"。
            调用它**不会立刻执行**，而是返回一个可以逐个取值的对象。
            每次别人取一个值，代码就跑到下一个 yield 然后暂停。
            这就是"边收边显示"的实现原理。
        """
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=True,
            # 让接口在最后一个块里带上用量信息，否则流式模式下拿不到 token 数
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # 坑 1：最后一个 chunk 的 choices 是空列表，只带用量信息
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self.last_usage = Usage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        requests=1,
                    )
                    self.total_usage.add(self.last_usage)
                continue

            # 坑 2：有些 chunk 的 content 是 None（只携带角色信息）
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    # ------------------------------------------------------------------
    def close(self) -> None:
        """关闭底层连接。用完最好调一下，或者用 try/finally。"""
        self._client.close()


# ====================================================================
# 五、假模型 —— 你未来两周最重要的工具
# ====================================================================


class FakeLLM:
    """
    假的大模型，用于离线开发和测试。

    它长得和 DeepSeekClient 一模一样（都实现了 chat / stream_chat），
    所以代码里可以无缝替换。

    两种用法：

        用法 1：排队式（按顺序返回预设答案）
            fake = FakeLLM(["回答一", "回答二"])
            fake.chat(msgs)   # -> "回答一"
            fake.chat(msgs)   # -> "回答二"
            fake.chat(msgs)   # -> 队列空了，返回默认值

        用法 2：函数式（根据输入动态生成答案）
            fake = FakeLLM(reply_fn=lambda msgs: f"你说了 {len(msgs)} 句话")

    事后可以检查它收到了什么：
            fake.calls          # 每次调用传进来的 messages 列表
            fake.call_count     # 被调用了几次
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        default_response: str = "（这是假模型的回答）",
        reply_fn=None,
        chunk_size: int = 3,
        delay: float = 0.0,
    ) -> None:
        # 下划线开头的属性是"私有"约定，表示"外面别直接改"
        self._responses = list(responses or [])
        self._default_response = default_response
        self._reply_fn = reply_fn
        self._chunk_size = chunk_size
        self._delay = delay

        self.calls: list[list[dict]] = []
        self.total_usage = Usage()
        self.last_usage = Usage()

    # ------------------------------------------------------------------
    @property
    def call_count(self) -> int:
        """被调用了几次。"""
        return len(self.calls)

    # ------------------------------------------------------------------
    def _next_response(self, messages: list[dict]) -> str:
        """决定这次该返回什么。"""
        # 先记下这次收到的输入，方便测试断言
        self.calls.append(messages)

        if self._reply_fn is not None:
            return self._reply_fn(messages)

        if self._responses:
            # pop(0) 取出第一个（队列，先进先出）
            return self._responses.pop(0)

        return self._default_response

    # ------------------------------------------------------------------
    def _record_usage(self, messages: list[dict], answer: str) -> None:
        """按照真实模型的算法，记录一份"假的"用量，方便测试成本统计逻辑。"""
        self.last_usage = Usage(
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(answer),
            requests=1,
        )
        self.total_usage.add(self.last_usage)

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # ⚠️ 注意这里：即使是假模型，我们也要把 messages **复制**一份再存。
        # 否则调用方后面修改了 messages，我们记录的历史也会跟着变，
        # 测试就会莫名其妙地失败。这个坑非常隐蔽，一定要养成习惯。
        snapshot = [dict(m) for m in messages]

        answer = self._next_response(snapshot)
        self._record_usage(snapshot, answer)
        return answer

    # ------------------------------------------------------------------
    def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        snapshot = [dict(m) for m in messages]
        answer = self._next_response(snapshot)

        # 模拟真实流式：把回答切成小块一块块吐出去
        for i in range(0, len(answer), self._chunk_size):
            if self._delay:
                time.sleep(self._delay)
            yield answer[i : i + self._chunk_size]

        self._record_usage(snapshot, answer)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """假模型不需要关闭，但为了接口一致还是提供一个空方法。"""
        return None


# ====================================================================
# 六、工厂函数
# ====================================================================


def create_llm(config: Config, *, fake: bool = False, **fake_kwargs) -> ChatModel:
    """
    根据配置创建模型客户端。

    参数 fake：
        True 时返回假模型，用于测试和离线开发。

    为什么要用"工厂函数"而不是直接 new 一个类？
        因为将来你可能要支持多家模型（DeepSeek / 通义 / 本地 Ollama）。
        有了工厂函数，上层代码只写 create_llm(config)，
        换模型时只需要改这一个函数，其他代码一行都不用动。
    """
    if fake:
        return FakeLLM(**fake_kwargs)
    return DeepSeekClient(config)
