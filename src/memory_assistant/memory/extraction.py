r"""
事实抽取器 —— 从对话里抽出结构化的事实
================================================================

⭐⭐ 这是你的第四个作业。规格写在下面，代码留给你实现。
    验收标准在 tests/test_extraction.py。

--------------------------------------------------------------------
这一层解决什么问题？
--------------------------------------------------------------------
上一周你做出了 `LongTermMemory`：它能存事实、能检索、能衰减。
但**事实从哪来**？现在库里是空的。

            对话  ──[?]──▶  结构化事实  ──▶  LongTermMemory
                          ↑
                    你这一周要做的就是这一步

它要做的事：
    输入：一段对话（几条消息）
    输出：[ExtractedFact(subject="user", predicate="住在", object="杭州", ...), ...]

--------------------------------------------------------------------
为什么这是本项目最"工程"的一层？
--------------------------------------------------------------------
因为这里要处理一个很现实的矛盾：

    **大模型很擅长理解自然语言，但它输出的 JSON 非常不可靠。**

你让它"返回 JSON 数组"，它可能给你：

    1. 用 markdown 代码块包起来：  ```json\n[{...}]\n```
    2. 前面加一段解释：           好的，我抽取到以下事实：\n[{...}]
    3. 后面加一段总结：           [{...}]\n以上共 3 条。
    4. 返回对象而不是数组：        {"facts": [{...}]}
    5. 只有一条时忘了套数组：      {"subject": "user", ...}
    6. 字段缺失：                 {"predicate": "住在"}     ← 没有 object
    7. confidence 给了百分数：     95  而不是  0.95
    8. 干脆不是 JSON：            抱歉，我无法从这段对话中抽取事实。

**你的代码必须能扛住上面全部 8 种情况，而且不能崩。**
这就是"和 LLM 打交道"的真实工作量 —— 也是为什么第 8 周用 LangGraph 时，
你会发现那些结构化输出的抽象突然变得很好理解：
因为你自己手写过一遍，知道它们到底在解决什么问题。

--------------------------------------------------------------------
接口
--------------------------------------------------------------------

    from memory_assistant.llm import FakeLLM
    from memory_assistant.memory.extraction import FactExtractor

    llm = FakeLLM(['[{"subject":"user","predicate":"住在","object":"杭州","confidence":0.95}]'])
    extractor = FactExtractor(llm)

    facts = extractor.extract([
        {"role": "user", "content": "我叫小明，在杭州做后端开发"},
    ])

    for fact in facts:
        print(fact.subject, fact.predicate, fact.object, fact.confidence)

--------------------------------------------------------------------
详细规格
--------------------------------------------------------------------

【S1】数据模型 ExtractedFact（用 pydantic）
    class ExtractedFact(BaseModel):
        subject: str = "user"
        predicate: str
        object: str
        confidence: float = 1.0
        evidence: str = ""

    字段含义：
        subject     事实关于谁。通常是 "user"，也支持 "老婆" / "豆豆（猫）"
        predicate   什么关系。如 "住在" / "喜欢" / "职业是" / "对……过敏"
        object      什么值。如 "杭州" / "手冲咖啡" / "花生"
        confidence  0~1，抽取的把握
        evidence    原文里支撑这条事实的那句话（调试和审计用）

    另外要有一个 `text` 属性，返回 "subject predicate object"
    （和 LongTermMemory 的文本渲染规则保持一致，这样能直接喂给它）

【S2】构造函数
    __init__(self, llm, *, min_confidence=0.6, max_facts=10)

    - llm：任何有 `.chat(messages) -> str` 的对象。测试传 FakeLLM。
    - min_confidence：低于这个置信度的事实会被丢弃，默认 0.6。
      必须满足 0 <= min_confidence <= 1，否则抛 ValueError。
    - max_facts：一次最多返回几条，默认 10。必须 > 0，否则抛 ValueError。

    还要初始化两个状态：
        self.last_error: Exception | None   —— 最近一次失败的原因
        self.last_skipped: int              —— 最近一次解析中被跳过的条目数

【S3】build_prompt(self, messages, known_facts=None) -> list[dict]
    构造发给模型的 prompt。**单独拆成方法是为了能脱离模型测试它。**

    参数：
        messages     待抽取的对话，格式 [{"role": ..., "content": ...}]
        known_facts  可选。已知事实的文本列表（如 ["user 住在 杭州"]），
                     用来告诉模型"这些别重复抽了"。

    ⚠️ 请照抄下面的措辞（测试会检查关键词，理由和上周一样：prompt 是接口契约）

        你是信息抽取器。从下面的对话里抽取关于用户的**长期有效**的事实。

        只抽取长期稳定的信息：
          - 姓名、职业、所在城市
          - 偏好、厌恶、习惯
          - 长期拥有的东西（宠物、设备等）
          - 过敏、禁忌这类需要长期记住的信息

        不要抽取：
          - 一次性的临时安排（"我明天要开会"）
          - 寒暄、情绪、对助手的提问
          - 助手的回答内容（只看用户说了什么）

        输出格式：严格的 JSON 数组，不要输出任何其他文字、不要用代码块包裹。
        每个元素包含字段：subject / predicate / object / confidence / evidence
        confidence 是 0 到 1 之间的小数，表示你的把握程度。
        evidence 是原文中支撑这条事实的那句话。
        没有可抽取的内容时，输出 []。

        [如果 known_facts 非空，插入这一段]
        已知事实（不要重复抽取）：
        - user 住在 杭州
        - user 喜欢 手冲咖啡

        [最后]
        需要抽取的对话：
        user: 我叫小明，在杭州做后端开发
        assistant: 记住了

    对话部分按 "{role}: {content}" 逐行拼起来。

【S4】parse_response(self, raw: str) -> list[ExtractedFact]
    ⭐ **这是整个作业的核心。** 它只做一件事：把模型返回的原始文本变成事实列表。
    它不调用模型，所以可以脱离模型独立测试 —— 请把它写成一个尽量纯粹的函数。

    健壮性要求（对应开头列的 8 种坏情况）：

    (a) 先去掉 markdown 代码块围栏（```json ... ``` 或 ``` ... ```）
    (b) 如果整段文本不是合法 JSON，就**从中截取第一个 JSON 数组或对象**：
        找到第一个 "[" 或 "{" 和它对应的最后一个 "]" 或 "}"。
        —— 这样情况 2 和 3（前后有解释文字）就能处理
    (c) 解析结果如果是一个 dict：
          · 如果它有一个列表类型的字段（任一字段都行，比如 "facts"），取那个列表
          · 否则把它当成单条事实，包成一个单元素列表
        —— 处理情况 4 和 5
    (d) 逐条转成 ExtractedFact：
          · predicate 或 object 缺失/为空白 → **跳过这一条**，并让 last_skipped 加一
            （注意：不要因为一条坏数据就丢掉整批，这叫"部分成功"）
          · subject 缺失 → 用默认值 "user"
          · confidence 缺失或不是数字 → 用默认值 1.0
          · confidence 在 (1, 100] 之间 → 当成百分数，除以 100
            （模型经常给 95 而不是 0.95，这个坑很常见）
          · confidence 仍然越界 → clamp 到 [0, 1]
    (e) 完全无法解析时（空字符串、纯文本、JSON 语法错误且截取不到）：
          · 返回空列表
          · 把异常记进 self.last_error
          · **不要抛异常** —— 抽取失败不该让整个对话流程崩掉

    成功解析时（即使结果为空数组）要把 last_error 重置为 None。
    每次调用都要重置 last_skipped。

【S5】extract(self, messages, *, known_facts=None) -> list[ExtractedFact]
    完整流程：
        1. 对话为空 → 直接返回 []（**不要调模型**，省一次钱）
        2. prompt = self.build_prompt(messages, known_facts)
        3. raw = self.llm.chat(prompt)   —— 失败时记 last_error 并返回 []
        4. facts = self.parse_response(raw)
        5. 丢掉 confidence < min_confidence 的
        6. **批内去重**：subject/predicate/object 三者（去掉首尾空白、转小写）
           完全相同的算重复，只保留 confidence 最高的那条
        7. 按 confidence 从高到低排序，取前 max_facts 条
        8. 返回

    ⚠️ 三层去重是**分工**的，别搞混：
        · 批内去重（这里）   —— 模型在同一次回答里说了两遍同样的话
        · 长期记忆的去重      —— 上周做的，跨会话、基于向量相似度
        · 冲突消解（第 10 周） —— 用户改主意了（"我戒咖啡了"）
      这一层只做最简单的那一种（字符串完全相等），因为它不需要向量，
      也不该依赖 LongTermMemory（保持模块独立、可单独测试）。

【S6】__repr__
    类似 <FactExtractor 最少置信度=0.6 最多=10条>

--------------------------------------------------------------------
为什么这一层不碰数据库、不碰向量库？
--------------------------------------------------------------------
和 SummaryMemory 一样的理由：**职责分离**。
    FactExtractor  —— 只管"从文本里抽事实"
    LongTermMemory  —— 只管"事实怎么存和检索"
上层（对话引擎）负责把两者接起来：

    facts = extractor.extract(messages)
    for fact in facts:
        memory.add(fact.predicate, fact.object,
                   subject=fact.subject,
                   confidence=fact.confidence,
                   source_message_id=last_message_id)

好处：这一层的 30 多个测试**一行数据库代码都不需要**，跑起来毫秒级。

--------------------------------------------------------------------
怎么开始
--------------------------------------------------------------------
    1. 先读 tests/test_extraction.py，重点看「组 3：parse_response 健壮性」
    2. 跑测试看全红：
       .\.venv\Scripts\python.exe -m pytest tests/test_extraction.py -v
    3. 按这个顺序实现：
       ExtractedFact → 构造函数 → __repr__
       → build_prompt（不碰模型，最好上手）
       → parse_response（核心，占一半时间）
       → extract
    4. 全绿后挑战进阶任务

💡 写 parse_response 的建议：**不要试图用一个复杂的正则搞定一切**。
   分步骤来：
       先剥代码块围栏 → 再尝试 json.loads → 失败了就截取括号范围再试 → ...
   每一步失败都退回上一步的结果，而不是直接放弃。
   这种"逐级降级"的写法在解析不可靠输入时非常常见。
"""

from pydantic import BaseModel, Field


# ====================================================================
# 数据模型
# ====================================================================
class ExtractedFact(BaseModel):
    """
    一条被抽取出来的事实。

    关于 pydantic：
        它是"数据校验"库。用 BaseModel 定义一个类，
        它在创建对象时就会自动校验类型。
        模型返回的脏数据（比如 confidence 是字符串 "很高"）会被挡在外面，
        而不是流进你的数据库。

        ⚠️ 注意它和 Python 的 @dataclass 的区别：
           dataclass 只帮你生成 __init__，**不做任何校验**；
           pydantic 会真的检查类型和范围。

    关于 Field：
        用来给字段加默认值和说明。default 表示"可以不传"。
    """

    subject: str = Field(default="user", description="事实关于谁")
    predicate: str = Field(description="什么关系，如「住在」")
    object: str = Field(description="什么值，如「杭州」")
    confidence: float = Field(default=1.0, description="抽取把握，0~1")
    evidence: str = Field(default="", description="支撑这条事实的原文句子")

    @property
    def text(self) -> str:
        """渲染成 "subject predicate object"，可以直接喂给 LongTermMemory。"""
        # TODO【S1】请实现
        raise NotImplementedError("【S1】请实现 text 属性")

    def __str__(self) -> str:
        return f"{self.text}（把握 {self.confidence:.2f}）"


# ====================================================================
# 抽取器
# ====================================================================
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MAX_FACTS = 10


class FactExtractor:
    """从对话里抽取结构化事实。规格见模块文档字符串。"""

    def __init__(
        self,
        llm,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_facts: int = DEFAULT_MAX_FACTS,
    ) -> None:
        # ---- 给你一个开头，剩下的自己写 ----

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence 必须在 [0, 1] 之间，收到的是 {min_confidence}"
            )
        if max_facts <= 0:
            raise ValueError(f"max_facts 必须大于 0，收到的是 {max_facts}")

        self.llm = llm
        self.min_confidence = min_confidence
        self.max_facts = max_facts

        # TODO【S2】把这两个状态初始化好
        self.last_error: Exception | None = None
        self.last_skipped: int = 0

    # ==================================================================
    # 你的任务从这里开始
    # ==================================================================

    def build_prompt(self, messages: list[dict], known_facts=None) -> list[dict]:
        """构造 prompt。规格见【S3】。"""
        raise NotImplementedError("【S3】请实现 build_prompt()")

    def parse_response(self, raw: str) -> list[ExtractedFact]:
        """把模型返回的原始文本解析成事实列表。规格见【S4】。这是核心。"""
        raise NotImplementedError("【S4】请实现 parse_response()")

    def extract(self, messages: list[dict], *, known_facts=None) -> list[ExtractedFact]:
        """完整抽取流程。规格见【S5】。"""
        raise NotImplementedError("【S5】请实现 extract()")

    def __repr__(self) -> str:
        """规格见【S6】。"""
        raise NotImplementedError("【S6】请实现 __repr__()")


# ====================================================================
# 进阶任务
# ====================================================================
#
# 【进阶 1】用 JSON 模式提高成功率
#     DeepSeek / OpenAI 都支持 response_format={"type": "json_object"}，
#     能显著减少"返回一段解释性文字"的情况。
#     但有个坑：开启 JSON 模式时，prompt 里**必须出现 "JSON" 这个词**，
#     否则接口会报错。去查一下你用的模型的文档确认这个要求。
#     改造点：给 FactExtractor 加一个 use_json_mode 开关，
#     并在 build_prompt 里确保措辞满足要求。
#     想清楚：开了 JSON 模式之后，parse_response 的健壮性还要不要保留？
#     （提示：要。因为模式只是"提高成功率"，不是"保证"。）
#
# 【进阶 2】让模型给出证据并校验
#     evidence 字段现在只是存着好看。可以进一步：
#     校验 evidence 是否真的出现在原始对话里（子串匹配）。
#     如果不在，说明模型在编造 —— 这条事实的置信度应该打折或者直接丢弃。
#     这叫"可溯源性校验"，是降低幻觉的实用手段。
#
# 【进阶 3】处理否定与条件
#     现在的三元组表达能力有限，下面这些抽出来会失真：
#         "我不喜欢甜的"           → 是 dislikes 还是 likes 的否定？
#         "如果下雨我就不去爬山"     → 条件事实
#         "以前喜欢，现在不喜欢了"   → 时间变化
#     试着给 ExtractedFact 加一个 polarity（肯定/否定）字段，
#     看看能不能至少把第一种情况表达清楚。
#     —— 想清楚这个问题的边界在哪，比硬做出来更有价值。
#
# 【进阶 4】批量抽取的成本优化
#     现在每轮对话都要调一次模型抽取。可以改成：
#     攒够 N 轮对话（或者累计 M 个 token）再一起抽。
#     好处是省调用次数，坏处是记忆有延迟。
#     实现之后量化两种模式的延迟和成本差异，把数据记下来 ——
#     这就是"热路径 vs 后台写入"那个取舍的实证。
