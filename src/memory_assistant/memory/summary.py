r"""
中期记忆 —— 递归增量摘要
================================================================

⭐⭐ 这是你的第二个正式作业。规格写在下面，代码留给你实现。
    验收标准在 tests/test_summary.py。

--------------------------------------------------------------------
先想清楚问题
--------------------------------------------------------------------
短期记忆有自己的预算上限，装不下的消息会被挤出去。
如果直接丢掉，这些信息就永久消失了：

    第 1 轮："我叫小明，在杭州做后端开发"
    ...（聊了 50 轮，第 1 轮早就被挤出窗口了）...
    第 51 轮："我是做什么工作的？"
    助理："抱歉，我不知道。"

中期记忆的职责就是：**把被挤出去的消息压缩成一段摘要，保住要点。**
摘要只占几百 token，但能承载几千 token 的信息。

--------------------------------------------------------------------
为什么是「递归增量」，而不是「每次都重压一遍」？
--------------------------------------------------------------------
方案 A（每次都重压）：把「全部历史」重新丢给模型压缩一次。
    问题 1：历史越长越贵。第 50 轮时你要把 50 轮的对话重新压一遍，
            成本随轮数线性增长，而摘要本身却在缩小 —— 纯浪费。
    问题 2：每次都是"从原始对话压缩"，早期信息经过多次重压会被反复稀释，
            越压越模糊。

方案 B（递归增量，本项目采用）：
    新摘要 = LLM(旧摘要 + 这次新挤出的消息)
    好处 1：每次只处理一小段新消息，**成本恒定**。
    好处 2：旧摘要被反复"强化"，重要信息不容易丢。

方案 B 的代价：误差会累积（摘要的摘要的摘要……）。
    缓解手段：在 prompt 里明确要求结构化输出、明确要求保留哪几类信息。
    验证手段：第 11 周的评测会专门检查"第 1 轮说过的信息在第 50 轮能否召回"。

--------------------------------------------------------------------
你要实现的接口
--------------------------------------------------------------------

    llm = FakeLLM(["小明在杭州做后端开发，偏好手冲咖啡不加糖。"])
    memory = SummaryMemory(llm, trigger_tokens=100, max_summary_tokens=200)

    # 窗口挤出来的消息交给它
    memory.add_evicted([
        {"id": 1, "role": "user", "content": "我叫小明，在杭州做后端开发"},
        {"id": 2, "role": "assistant", "content": "记住了。"},
    ])

    if memory.should_compress():          # 攒够了才压，不是每轮都调模型
        memory.maybe_compress()

    memory.get_summary()                  # "小明在杭州做后端开发，偏好手冲咖啡不加糖。"
    memory.covered_until                  # 2  ← 摘要已覆盖到 id=2 的消息

--------------------------------------------------------------------
详细规格（逐条对照实现）
--------------------------------------------------------------------

【S1】构造函数
    __init__(self, llm, *, max_summary_tokens: int = 800,
             trigger_tokens: int = 400,
             token_counter: Callable[[str], int] | None = None)

    - llm：任何有 `.chat(messages) -> str` 方法的对象。
      默认用 llm.py 里的 DeepSeekClient；**测试时传 FakeLLM**
      （FakeLLM 就是为了这个才存在的 —— 让你不花钱、不联网也能测）。
    - max_summary_tokens：**摘要本身**的长度上限，默认 800。
    - trigger_tokens：待压缩消息攒到多少 token 才触发压缩，默认 400。
    - token_counter：文本 token 计数器，默认用 estimate_tokens。
      和短期记忆一样，允许注入是为了让测试不依赖估算算法的细节。

    ⚠️ 必须校验 max_summary_tokens > 0、trigger_tokens > 0，否则抛 ValueError。

【S2】add_evicted(self, messages: list[dict]) -> int
    把被短期记忆挤出来的消息加入待压缩队列，返回实际接收的条数。

    - 忽略 role == "system" 的消息（系统提示词是永久的，不该被摘要）
    - 忽略 content 为**空白**的消息（注意：`"   "` 这种只有空格也算空白，
      要用 `content.strip()` 判断，不能只写 `if content`）
    - 消息如果带 "id" 字段，要记住最大的那个 id，用于 covered_until

【S3】should_compress(self) -> bool
    待压缩消息的 token 数 >= trigger_tokens 时返回 True。

    ⭐ **为什么要这个阈值？这是本模块最重要的设计点。**
       如果不设阈值，每挤出一条消息你就调一次模型：
         · 成本爆炸（每轮多一次 LLM 调用）
         · 摘要质量差（一次只压两条寒暄，模型看不出重点）
       攒够一批再压，成本可控，而且模型能看到完整的一段对话，
       更容易抽出「什么才是重点」。

【S4】compress(self) -> str | None
    立刻执行一次压缩（不判断阈值）。待压缩队列为空时返回 None。

    流程：
        1. 如果队列为空 → 返回 None
        2. prompt = self.build_compression_prompt()
        3. 调用 self.llm.chat(prompt) 拿到新摘要
        4. **校验结果**（见 S6），不合法就当失败处理
        5. 成功：self.summary = 新摘要；清空队列；返回新摘要
        6. 失败：队列和旧摘要都**原样保留**，把异常记进 self.last_error，返回 None

【S5】build_compression_prompt(self) -> list[dict]
    构造发给模型的 messages 列表。**把它单独拆成方法是有意的设计**：
    这样测试可以直接检查 prompt 的内容，而不需要真的调用模型。

    必须包含的内容：
        · 要求把对话压缩成不超过 max_summary_tokens 字
        · 「必须保留」清单：人物、地点、时间；用户偏好与厌恶；已做出的决定；
          未完成的待办
        · 「可以丢弃」清单：寒暄客套、重复表述、无关的临时内容
        · 待压缩的对话原文
    如果已有旧摘要，还必须包含：
        · 明确告诉模型「请在此基础上增量更新，不要丢失已有摘要里的信息」
        · 旧摘要的原文
    （`docs/记忆架构设计.md` 第 3.2 节有一个模板初稿可以参考）

    ⚠️ **请照抄下面这段措辞。** 为什么要指定得这么死？
       因为测试要检查 prompt 内容，如果让你自由发挥措辞，
       测试就只能靠猜你用了哪个词，那样很不讲道理。
       这本身也是真实工程里的做法：**prompt 是一种接口契约**，
       一旦定下来就应该稳定，不能今天写"偏好"明天写"喜好"。

    必须包含的措辞（首次压缩）：

        你是对话摘要器。把下面的对话压缩成不超过 {max_summary_tokens} 字的摘要。

        必须保留：
          - 涉及的人物、地点、时间
          - 用户的偏好、厌恶、习惯
          - 已经做出的决定和承诺
          - 未完成的待办

        可以丢弃：
          - 寒暄、客套
          - 重复表述
          - 与用户长期信息无关的临时内容

        需要压缩的新对话：
        {把每条消息按 "角色: 内容" 逐行拼起来}

    有旧摘要时，在「需要压缩的新对话」之前**额外插入**这一段：

        已有摘要（请在此基础上做增量更新，不要丢失其中已有的事实）：
        {旧摘要}

    （也就是说：首次压缩时 prompt 里**不该出现**「已有摘要」和「增量更新」这两个词）

【S6】结果校验 —— 这一条决定了系统的健壮性
    模型是会犯错的：会返回空字符串、会返回一大段废话、会返回 "好的" 就完事。

    校验规则：
        (a) 结果是空白字符串 → 视为失败（**绝对不能覆盖掉旧摘要！**）
            这是最危险的情况：模型抽风返回空，你把旧摘要清了，
            等于把用户的历史记忆全删了。
        (b) 结果超过 max_summary_tokens → 截断到该长度，并在末尾加 "…"
            （这是安全网，正常情况不该触发，因为 prompt 里已经要求了长度）

    为什么「失败时保留旧摘要和队列」？
        因为摘要压缩是"尽力而为"的后台任务，不是关键路径。
        失败了大不了下次再压一遍，但**绝不能丢数据**。
        这叫"优雅降级"。面试里能讲出这一点很加分。

【S7】maybe_compress(self) -> str | None
    should_compress() 为 True 时才调 compress()，否则直接返回 None。

【S8】其它
    - get_summary(self) -> str | None：返回当前摘要，没有就返回 None
    - covered_until 属性：已覆盖到的最大消息 id，没有就返回 0
    - pending_tokens 属性：待压缩队列的 token 数
    - pending_count 属性：待压缩队列的消息条数
    - clear(self) -> None：清空摘要、队列、covered_until、last_error
      注意这里和短期记忆不同 —— 摘要**不保留**，因为 clear 的语义是
      "彻底忘掉这段对话"（被遗忘权）
    - last_error 属性：最近一次压缩**失败**的原因（Exception 或 None）。
      压缩成功后必须重置为 None。（"最近一次尝试的结果"这个语义更清晰）
    - __repr__：类似
      <SummaryMemory 摘要 120 token / 待压缩 3 条 45 token / 阈值 400>

【S9】restore(self, summary: str | None, covered_until: int = 0) -> None
    ⭐ 从持久化存储里**恢复状态**。程序重启后必须调它，否则会出大问题。

    它要做的：
        · self.summary = summary
        · self._covered_until = covered_until
        · 不动待压缩队列，也不动 last_error

    ⚠️ 为什么这个方法是必需的？想清楚这个场景：

        第 1 天：聊了 50 轮，摘要已经覆盖到 id=50，存进了数据库。
        第 2 天：程序重启。如果不调 restore：
                  · self.summary 是 None
                  · 新的压缩会算成 "新摘要 = LLM(空 + 新消息)"
                  · **结果：前 50 轮的记忆全部丢失，用户的历史被清空**

        这是记忆系统里最危险的一类 bug —— 不报错，只是"用户发现助手失忆了"。

    参数 summary 传 None 时，等价于"只恢复覆盖位置，没有摘要"。
    如果摘要为**空白**（`strip()` 之后为空，比如 `"   "`），也当成 None 处理
    （空摘要没有意义，而且会让 prompt 里出现一段空内容）。

    （注意：这个方法和 clear() 的语义完全不同。
      clear 是"忘掉一切"，restore 是"想起来"。别搞混。）

--------------------------------------------------------------------
怎么开始
--------------------------------------------------------------------
    1. 先读 tests/test_summary.py —— 那是验收标准
    2. 跑测试，看它们全部失败：
       .\.venv\Scripts\python.exe -m pytest tests/test_summary.py -v
    3. 按这个顺序实现（从易到难）：
       add_evicted → should_compress → build_compression_prompt
       → get_summary / 属性 / __repr__ → compress → maybe_compress
    4. 全部变绿后，再挑战文件末尾的进阶任务

⚠️ 这个作业比短期记忆难，因为要通过 FakeLLM 验证"和模型的交互"。
   卡住是正常的。卡超过 30 分钟就来找我，把报错和你的代码一起发过来。

--------------------------------------------------------------------
一个重要的设计说明：为什么这里不碰数据库？
--------------------------------------------------------------------
你可能会想："摘要不是要存进 SQLite 的 summaries 表吗？为什么这里没有 SQL？"

因为**职责分离**：
    SummaryMemory  —— 只负责"怎么把对话压成摘要"这个逻辑
    SummaryRepository —— 只负责"摘要怎么持久化"

好处是 SummaryMemory 可以完全脱离数据库做测试（下面 18 个测试没有一行 SQL），
而且将来换存储实现不用动这里的代码。
把这两件事揉在一起是新手最常见的架构错误 —— 会导致"想测逻辑就必须先建库"。

实际使用时由 cli.py 把两者接起来：
    summary_memory.add_evicted(evicted)
    if summary_memory.maybe_compress():
        store.summaries.add(session_id, summary_memory.get_summary(),
                            covered_until=summary_memory.covered_until)
"""

from typing import Callable

from memory_assistant.llm import estimate_tokens

DEFAULT_MAX_SUMMARY_TOKENS = 800
DEFAULT_TRIGGER_TOKENS = 400


class SummaryMemory:
    """中期记忆：递归增量摘要。规格见模块文档字符串。"""

    def __init__(
        self,
        llm,
        *,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
        trigger_tokens: int = DEFAULT_TRIGGER_TOKENS,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        # ---- 给你一个开头，剩下的自己写 ----

        if max_summary_tokens <= 0:
            raise ValueError(f"max_summary_tokens 必须大于 0，收到的是 {max_summary_tokens}")
        if trigger_tokens <= 0:
            raise ValueError(f"trigger_tokens 必须大于 0，收到的是 {trigger_tokens}")

        self.llm = llm
        self.max_summary_tokens = max_summary_tokens
        self.trigger_tokens = trigger_tokens
        self._token_counter = token_counter or estimate_tokens

        # 当前摘要（没有就是 None）
        self.summary: str | None = None
        # 待压缩的消息队列
        self._pending: list[dict] = []
        # 摘要已覆盖到的最大消息 id
        self._covered_until: int = 0
        # 最近一次失败的原因
        self.last_error: Exception | None = None

    # ==================================================================
    # 你的任务从这里开始。把每个 raise NotImplementedError 换成真正的实现。
    # ==================================================================

    def add_evicted(self, messages: list[dict]) -> int:
        """把被挤出的消息加入待压缩队列。规格见【S2】。"""
        accepted = 0
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system" or not isinstance(content, str) or not content.strip():
                continue

            # 保存副本，避免调用方之后修改消息时改变待压缩内容。
            self._pending.append(dict(message))
            accepted += 1
            message_id = message.get("id")
            if message_id is not None:
                self._covered_until = max(self._covered_until, int(message_id))
        return accepted

    def should_compress(self) -> bool:
        """待压缩消息攒够阈值了吗？规格见【S3】。"""
        return self.pending_tokens >= self.trigger_tokens

    def build_compression_prompt(self) -> list[dict]:
        """构造发给模型的 prompt。规格见【S5】。"""
        instructions = (
            f"你是对话摘要器。把下面的对话压缩成不超过 {self.max_summary_tokens} token 的摘要。\n\n"
            "必须保留：\n"
            "  - 涉及的人物、地点、时间\n"
            "  - 用户的偏好、厌恶、习惯\n"
            "  - 已经做出的决定和承诺\n"
            "  - 未完成的待办\n\n"
            "可以丢弃：\n"
            "  - 寒暄、客套\n"
            "  - 重复表述\n"
            "  - 与用户长期信息无关的临时内容"
        )

        sections: list[str] = []
        if self.summary:
            sections.append(
                "已有摘要（请在此基础上做增量更新，不要丢失其中已有的事实）：\n"
                f"{self.summary}"
            )

        conversation = "\n".join(
            f"{message['role']}: {message['content']}" for message in self._pending
        )
        sections.append(f"需要压缩的新对话：\n{conversation}")

        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def compress(self) -> str | None:
        """立刻压缩一次。规格见【S4】【S6】。"""
        if not self._pending:
            return None

        try:
            raw_summary = self.llm.chat(self.build_compression_prompt())
            if not isinstance(raw_summary, str) or not raw_summary.strip():
                raise ValueError("模型返回了空摘要")

            new_summary = raw_summary.strip()
            if self._token_counter(new_summary) > self.max_summary_tokens:
                # 为省略号预留空间，找出仍能满足 token 上限的最长前缀。
                low, high = 0, len(new_summary)
                while low < high:
                    middle = (low + high + 1) // 2
                    if self._token_counter(new_summary[:middle] + "…") <= self.max_summary_tokens:
                        low = middle
                    else:
                        high = middle - 1
                new_summary = new_summary[:low].rstrip() + "…"

            self.summary = new_summary
            self._pending.clear()
            self.last_error = None
            return new_summary
        except Exception as error:
            # 摘要属于尽力而为的维护任务；失败时保留旧状态供后续重试。
            self.last_error = error
            return None

    def maybe_compress(self) -> str | None:
        """攒够阈值才压缩。规格见【S7】。"""
        if not self.should_compress():
            return None
        return self.compress()

    def get_summary(self) -> str | None:
        """返回当前摘要。规格见【S8】。"""
        return self.summary

    @property
    def covered_until(self) -> int:
        """摘要已覆盖到的最大消息 id。规格见【S8】。"""
        return self._covered_until

    @property
    def pending_tokens(self) -> int:
        """待压缩队列的 token 数。规格见【S8】。"""
        return sum(self._token_counter(message["content"]) for message in self._pending)

    @property
    def pending_count(self) -> int:
        """待压缩队列的消息条数。规格见【S8】。"""
        return len(self._pending)

    def clear(self) -> None:
        """清空摘要和队列。规格见【S8】。"""
        self.summary = None
        self._pending.clear()
        self._covered_until = 0
        self.last_error = None

    def restore(self, summary: str | None, covered_until: int = 0) -> None:
        """从持久化存储恢复状态。规格见【S9】。"""
        self.summary = summary.strip() or None if summary is not None else None
        self._covered_until = covered_until

    def __repr__(self) -> str:
        """调试用的字符串表示。规格见【S8】。"""
        summary_tokens = self._token_counter(self.summary) if self.summary else 0
        return (
            f"<SummaryMemory 摘要 {summary_tokens} token"
            f" / 待压缩 {self.pending_count} 条 {self.pending_tokens} token"
            f" / 阈值 {self.trigger_tokens}>"
        )


# ====================================================================
# 进阶任务（做完上面的再看）
# ====================================================================
#
# 【进阶 1】超长摘要的智能重压
#     现在超过 max_summary_tokens 就硬截断，可能截到句子中间。
#     更好的做法：再调用一次模型，明确要求「太长了，压到 N 字以内」，
#     只有第二次还超长才硬截断。
#     想想这带来的成本：每次超长都要多一次 LLM 调用。值得吗？
#
# 【进阶 2】摘要质量的自检
#     用另一个 LLM 调用（或者同一模型的另一次调用）来检查：
#     "新摘要是否包含了旧摘要里的所有关键信息？"
#     如果发现信息丢失，就重试一次。
#     这叫 "LLM as a judge"，是当前很流行的评估手段。
#     代价是成本翻倍 —— 想清楚什么场景下值得。
#
# 【进阶 3】分层摘要
#     摘要攒得太多之后，摘要本身也会超预算。
#     这时候可以做"摘要的摘要"：把最早的几版摘要再压成一段"远期摘要"。
#     这就是分层记忆的雏形。（这个可以留到第 10 周再做）
#
# 【进阶 4】把 prompt 模板抽成配置
#     现在 prompt 是硬编码在 build_compression_prompt 里的。
#     如果以后想支持不同领域（比如代码助手、医疗助手），
#     "必须保留什么"的清单应该不一样。
#     试着把模板抽成构造函数的参数，并给一个默认值。
