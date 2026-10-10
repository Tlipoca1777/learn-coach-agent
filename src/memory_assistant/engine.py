r"""
对话引擎 —— 把四层记忆串成一个能跑的应用
================================================================

职责：定义**一轮对话的完整流程**，把各个模块按正确顺序调用起来。

                                   ┌──────────────────────────┐
    用户输入 ─────────────────────▶│  1. 落库 + 进短期记忆      │
                                   └────────────┬─────────────┘
                                                ▼
                                   ┌──────────────────────────┐
                                   │  2. 挤出的消息 → 中期摘要  │
                                   └────────────┬─────────────┘
                                                ▼
        ┌───────────────────────────────────────────────────────┐
        │  3. 组装 prompt（按固定顺序）                          │
        │     系统提示 → 长期记忆 → 中期摘要 → 近期对话窗口        │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                          ┌──────────────────┐
                          │  4. 调大模型      │
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  5. 回答落库      │
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  6. 抽事实 → 长期库│
                          └──────────────────┘

--------------------------------------------------------------------
这个文件为什么由我提供，而不是你写？
--------------------------------------------------------------------
因为你已经写了四个模块（短期/中期/长期/抽取），
它们各自都有 40~60 个测试把关。集成层是**管道代码**——
它的价值在于"顺序正确"，而不在于算法。

但更重要的是：**这一层验证了我的四份规格能不能拼在一起。**
我在写它的时候发现了一个真实的接口缺口：
`SummaryMemory` 原本没有恢复状态的方法，程序重启后摘要会丢，
新摘要会只覆盖新消息、把之前的一切抹掉。于是我补了 `restore()`。

如果等你把四个模块都实现完才发现这个问题，代价会大得多。

--------------------------------------------------------------------
两个关键设计点（面试可以讲）
--------------------------------------------------------------------

【设计点 1】依赖注入让引擎可以脱离记忆模块测试
    构造函数接受 `short_term` / `summary` / `long_term` / `extractor` 参数。
    不传就按配置创建真的，传了就用你给的。

    于是 `tests/test_engine.py` 可以注入几个轻量替身，
    在没有 chromadb、没有向量模型、没有实现任何记忆模块的情况下，
    验证"调用顺序、prompt 组装、落库时机"这些**编排逻辑**。

    如果不做这个设计，引擎就只能等四个模块全部实现完才能测 ——
    到那时出问题，你根本分不清是引擎的错还是某个模块的错。

【设计点 2】摘要的"喂入"基于数据库 id，而不是内存状态
    引擎自己维护 `_covered_until`，它来自 `summaries` 表的记录。
    只有 id 大于它的消息才会被喂给摘要层。

    为什么不用摘要层自己的 `covered_until`？
        因为重启后它是 0 —— 内存状态不可信，**持久化的 id 才可信**。
        这是所有"有状态系统重启"场景的通用原则：
        恢复状态时，以持久化存储为准，而不是以对象属性为准。

--------------------------------------------------------------------
怎么用
--------------------------------------------------------------------
    from memory_assistant.config import Config
    from memory_assistant.engine import ConversationEngine
    from memory_assistant.tools import create_default_tools

    config = Config.from_env()
    engine = ConversationEngine(config)
    engine.start()

    result = engine.respond("我叫小明，在杭州做后端开发")
    print(result.reply)
    print(result.budget)          # 各部分 token 占用
    # 默认还提供 calculator / get_current_time / record_answer / get_weak_topics 工具

    或者流式：
    engine.respond("你好", on_token=lambda piece: print(piece, end="", flush=True))

    命令行方式：
        .\.venv\Scripts\python.exe -m memory_assistant.cli
"""

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from memory_assistant.config import Config
from memory_assistant.llm import estimate_tokens, estimate_messages_tokens

# ====================================================================
# 默认的系统提示词
# ====================================================================
DEFAULT_SYSTEM_PROMPT = (
    "你是一个有记忆的私人助理。你记得用户之前告诉过你的信息。\n"
    "回答要简洁、具体、有用。如果用户询问个人信息，先检查当前上下文；"
    "若没有明确答案，按需调用 recall_memory 检索相关过去事实；"
    "检索仍无结果时，诚实地说你不记得，不要编造。\n"
    "如果用户正在练习知识点，请先清楚说明题目和评分依据；判定用户答案后，"
    "必须调用 record_answer 记录 topic_name、question、user_answer、verdict、score 和反馈。"
    "用户询问薄弱知识点或学习进度时，调用 get_weak_topics 查询后再回答；"
    "不得声称已记录或查到数据，除非对应工具已成功返回。"
)

# prompt 里两段附加内容的小标题
LONG_TERM_HEADER = "【关于用户的已知信息（来自你的长期记忆）】\n"
SUMMARY_HEADER = "【更早对话的摘要】\n"

DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_LONG_TERM_TOKEN_BUDGET = 2000


# ====================================================================
# 一轮对话的结果
# ====================================================================
@dataclass
class TurnResult:
    """
    一轮对话的完整结果。

    为什么要把这么多东西都返回？
        因为调试一个记忆系统时，你最需要知道的就是
        "这一轮到底往 prompt 里放了什么"。
        CLI 的 /prompt 命令就是把它打印出来。
    """

    reply: str = ""
    prompt: list[dict] = field(default_factory=list)
    retrieved_facts: list[str] = field(default_factory=list)
    summary_used: str | None = None
    new_summary: str | None = None
    extracted_facts: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ====================================================================
# 引擎
# ====================================================================
class ConversationEngine:
    """把四层记忆串成一轮完整对话。"""

    def __init__(
        self,
        config: Config,
        *,
        session_id: str | None = None,
        title: str | None = None,
        user_id: str = "default",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
        long_term_token_budget: int = DEFAULT_LONG_TERM_TOKEN_BUDGET,
        # ---------- 下面这些留了口子，测试时可以注入替身 ----------
        llm=None,
        store=None,
        short_term=None,
        summary=None,
        long_term=None,
        extractor=None,
        tools=None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.config = config
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.title = title
        self.user_id = user_id
        self.system_prompt = system_prompt
        self.retrieve_top_k = retrieve_top_k
        self.long_term_token_budget = long_term_token_budget
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.max_tool_rounds = max_tool_rounds

        self.last_error: Exception | None = None

        # 会话内全部消息（含 id），从数据库恢复，每轮追加。
        # 它是"谁是窗口内、谁被挤出去了"的判断依据。
        self._history: list[dict] = []
        # 摘要已经**压缩进内容**的位置（来自数据库，不以内存状态为准）
        self._covered_until: int = 0
        # 已经**交给摘要层排队**的位置。
        #
        # ⚠️ 为什么要和 _covered_until 分开？这是我自己写这一层时踩到的 bug：
        #    消息"喂进去"和"压缩完成"之间有延迟 —— 要攒够阈值才会真正压缩。
        #    如果只看 _covered_until（它只在压缩成功后才前进），
        #    同一批消息就会在每一轮被重复喂一次，
        #    摘要的待压缩队列里堆满重复内容，越压越乱，还白花 token。
        #
        #    教训：**"提交了"和"完成了"是两个不同的状态，别用一个变量表示。**
        self._fed_until: int = 0
        self._started = False

        # ---------- 依赖注入 ----------
        self.llm = llm if llm is not None else self._build_llm()
        self.store = store if store is not None else self._build_store()
        self.short_term = (
            short_term if short_term is not None else self._build_short_term()
        )
        self.summary = summary if summary is not None else self._build_summary()
        self.long_term = long_term if long_term is not None else self._build_long_term()
        self.extractor = (
            extractor if extractor is not None else self._build_extractor()
        )
        if tools is None:
            from memory_assistant.tools import (
                create_default_tools,
                create_learning_tools,
                create_memory_tools,
            )

            self.tools = create_default_tools()
            for tool in create_memory_tools(self.long_term, user_id=self.user_id):
                self.tools.register(tool)
            learning_repository = getattr(self.store, "learning", None)
            if learning_repository is not None:
                for tool in create_learning_tools(
                    learning_repository, user_id=self.user_id
                ):
                    self.tools.register(tool)
        else:
            self.tools = tools

    # ------------------------------------------------------------------
    # 按配置创建各个部件
    # ------------------------------------------------------------------
    def _build_llm(self):
        from memory_assistant.llm import create_llm

        return create_llm(self.config)

    def _build_store(self):
        from memory_assistant.storage import Database, Store

        self.config.ensure_data_dir()
        database = Database(self.config.data_dir / "assistant.db")
        database.initialize()
        return Store(database)

    def _build_short_term(self):
        """
        注意：这里**不传** system_prompt。

        为什么？因为系统提示词由引擎统一管理。
        如果两边都管，prompt 里就会出现两条 system，而且短期记忆的
        token 预算还会把系统提示词算进去 —— 职责重叠容易出错。
        """
        from memory_assistant.memory import ShortTermMemory

        max_tokens = int(self.config.prompt_token_budget * 0.4)
        return ShortTermMemory(max_tokens=max_tokens)

    def _build_summary(self):
        """
        ⚠️ 摘要的阈值必须**从配置推导**，不能全用默认值。

        这是我做集成验证时发现的真实问题：
            SummaryMemory 的默认阈值是 trigger_tokens=400（要攒够 400 token 才压缩）。
            如果用户的 context_window 配得比较小（比如测试用的 400），
            那么 prompt 总预算只有 200，短期窗口 80 —— 攒到 400 是不可能的。
            结果就是**摘要永远不会触发**，被挤出的消息全部堆在队列里，
            用户会发现"聊了很久，助手还是把早期的事忘了"。

        现在按 docs/记忆架构设计.md 第 5 节的预算比例推导：
            prompt 预算的 20% 给摘要（存起来的内容）
            prompt 预算的 10% 作为触发阈值
        """
        from memory_assistant.memory import SummaryMemory

        budget = self.config.prompt_token_budget
        return SummaryMemory(
            self.llm,
            max_summary_tokens=max(50, int(budget * 0.2)),
            trigger_tokens=max(20, int(budget * 0.1)),
        )

    def _build_long_term(self):
        from memory_assistant.embeddings import create_embeddings
        from memory_assistant.memory import LongTermMemory

        embedder = create_embeddings(self.config)
        return LongTermMemory(embedder, persist_dir=self.config.data_dir / "chroma")

    def _build_extractor(self):
        from memory_assistant.memory import FactExtractor

        return FactExtractor(self.llm)

    # ==================================================================
    # 启动 / 恢复
    # ==================================================================
    def start(self) -> str:
        """
        开始一个会话：不存在就创建，存在就**从数据库恢复全部状态**。

        返回 session_id。

        ⭐ 恢复这一步是"关掉程序再打开还记得你"的全部秘密。
           要做三件事：
             1. 把历史消息重新灌进短期记忆（窗口会自动裁到预算内）
             2. 把最新摘要 restore 回摘要层
             3. 记住摘要覆盖到哪个 id
        """
        self.store.sessions.get_or_create(self.session_id, self.title)

        rows = self.store.messages.list_by_session(self.session_id)
        self._history = [
            {"id": row["id"], "role": row["role"], "content": row["content"]}
            for row in rows
            if row["role"] != "system"
        ]
        for message in self._history:
            self.short_term.add(message["role"], message["content"])

        latest = self.store.summaries.latest(self.session_id)
        if latest is not None:
            self.summary.restore(latest["content"], int(latest["covered_until"]))
            self._covered_until = int(latest["covered_until"])
        else:
            self._covered_until = 0
        # 已经压缩过的不该再排队
        self._fed_until = self._covered_until

        self._started = True
        return self.session_id

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    # ==================================================================
    # 一轮对话
    # ==================================================================
    def respond(
        self,
        text: str,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """
        处理用户的一句话，返回这一轮的结果。

        参数 on_token：
            给了就用流式输出，每收到一小段文字就回调一次。
            没给就等模型全部写完再返回。
        """
        self._ensure_started()
        text = (text or "").strip()
        if not text:
            return TurnResult(error=ValueError("输入为空"))

        result = TurnResult()

        # ---------- 1. 落库 + 进短期记忆 ----------
        message_id = self.store.messages.append(
            self.session_id, "user", text, token_count=estimate_tokens(text)
        )
        self._history.append({"id": message_id, "role": "user", "content": text})
        self.short_term.add("user", text)

        # ---------- 2. 被挤出窗口的消息 → 中期摘要 ----------
        result.new_summary = self._feed_evicted_to_summary()

        # ---------- 3. 组装 prompt ----------
        window = [m for m in self.short_term.get_window() if m["role"] != "system"]
        result.summary_used = self.summary.get_summary()

        facts = self.long_term.search(
            text, top_k=self.retrieve_top_k, user_id=self.user_id
        )
        fact_lines = self._format_facts(facts)
        result.retrieved_facts = fact_lines

        result.prompt = self._build_prompt(result.summary_used, fact_lines, window)
        result.budget = self._budget_report(result.prompt, window, fact_lines)

        # ---------- 4. 调模型 ----------
        reply = ""
        try:
            if on_token is not None and not hasattr(self.llm, "chat_with_tools"):
                pieces: list[str] = []
                for piece in self.llm.stream_chat(result.prompt):
                    pieces.append(piece)
                    on_token(piece)
                reply = "".join(pieces)
            elif hasattr(self.llm, "chat_with_tools"):
                reply = self._chat_with_tools(result.prompt)
                if on_token is not None:
                    on_token(reply)
            else:
                reply = self.llm.chat(result.prompt)
        except Exception as error:
            # 不让异常冒出去：用户的消息已经存好了，下一轮还能继续。
            # 引擎把错误放进返回值，由调用方决定怎么展示。
            self.last_error = error
            result.error = error
            if not reply:
                return result

        reply = (reply or "").strip()
        if result.error is not None and reply:
            self.last_error = result.error
            return result
        result.reply = reply

        # ---------- 5. 回答落库 ----------
        if reply:
            assistant_id = self.store.messages.append(
                self.session_id, "assistant", reply, token_count=estimate_tokens(reply)
            )
            self._history.append(
                {"id": assistant_id, "role": "assistant", "content": reply}
            )
            self.short_term.add("assistant", reply)

        # ---------- 6. 抽事实 → 长期记忆 ----------
        result.extracted_facts = self._extract_and_store(text, reply, message_id)

        return result

    def _chat_with_tools(self, prompt: list[dict]) -> str:
        """Run bounded assistant/tool turns and return the final assistant text."""
        messages = [dict(message) for message in prompt]
        schemas = self.tools.schemas()
        for _ in range(self.max_tool_rounds):
            response = self.llm.chat_with_tools(messages, schemas)
            assistant_message = {
                "role": "assistant",
                "content": response.get("content") or "",
            }
            calls = response.get("tool_calls") or []
            if not calls:
                return assistant_message["content"]
            assistant_message["tool_calls"] = calls
            messages.append(assistant_message)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                try:
                    output = self.tools.execute(name, function.get("arguments", "{}"))
                except (ValueError, TypeError, ArithmeticError) as error:
                    output = f"工具执行失败：{error}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": output,
                })
        raise RuntimeError(f"工具调用超过上限（{self.max_tool_rounds} 轮）")

    # ------------------------------------------------------------------
    # 步骤 2：把挤出去的消息交给摘要层
    # ------------------------------------------------------------------
    def _feed_evicted_to_summary(self) -> str | None:
        """
        找出"已经不在窗口里、且还没被摘要覆盖"的消息，喂给摘要层。

        怎么知道谁被挤出去了？
            短期记忆的窗口是「最近的连续一段」，所以被挤出去的
            就是 `_history` 的前面那一截。用长度差就能算出来。

        为什么要同时看 _covered_until 和 _fed_until？见构造函数里的注释。
        """
        window_size = len(self.short_term.get_window())
        kept = max(0, window_size - (1 if self.short_term.system_prompt else 0))
        evicted = self._history[: len(self._history) - kept] if kept < len(self._history) else []

        fresh = [
            message
            for message in evicted
            if message["id"] > self._covered_until and message["id"] > self._fed_until
        ]
        if fresh:
            self.summary.add_evicted(
                [
                    {"id": m["id"], "role": m["role"], "content": m["content"]}
                    for m in fresh
                ]
            )
            self._fed_until = max(m["id"] for m in fresh)

        new_summary = self.summary.maybe_compress()
        if new_summary:
            covered = self.summary.covered_until or self._covered_until
            self.store.summaries.add(
                self.session_id,
                new_summary,
                covered_until=covered,
                token_count=estimate_tokens(new_summary),
            )
            self._covered_until = covered
        return new_summary

    # ------------------------------------------------------------------
    # 步骤 3：组装 prompt
    # ------------------------------------------------------------------
    def _format_facts(self, facts: list[dict]) -> list[str]:
        """把检索到的事实渲染成短句，并限制总 token 不超过预算。"""
        lines: list[str] = []
        used = 0
        for fact in facts:
            line = f"- {fact['text']}"
            cost = estimate_tokens(line)
            if used + cost > self.long_term_token_budget:
                break
            lines.append(line)
            used += cost
        return lines

    def _build_prompt(
        self, summary_text: str | None, fact_lines: list[str], window: list[dict]
    ) -> list[dict]:
        """
        prompt 的固定顺序：

            系统提示 → 长期记忆 → 中期摘要 → 近期对话

        为什么是这个顺序？
            从"最稳定"到"最临时"：身份设定最稳定，长期事实次之，
            摘要再次之，最近几轮对话最临时。
            模型读起来也是从"我是谁、用户是谁"到"刚刚说了什么"，符合直觉。

        ⚠️ 所有 system 消息都放在最前面。虽然 OpenAI 兼容的接口
           允许 system 出现在中间，但放在最前面是最保险的写法。
        """
        prompt: list[dict] = [{"role": "system", "content": self.system_prompt}]

        if fact_lines:
            prompt.append(
                {"role": "system", "content": LONG_TERM_HEADER + "\n".join(fact_lines)}
            )
        if summary_text:
            prompt.append(
                {"role": "system", "content": SUMMARY_HEADER + summary_text}
            )

        prompt.extend({"role": m["role"], "content": m["content"]} for m in window)
        return prompt

    def _budget_report(
        self, prompt: list[dict], window: list[dict], fact_lines: list[str]
    ) -> dict:
        """
        统计 prompt 各部分的 token 占用。

        这张表就是 `/status` 里显示的东西，也是第 10 周
        「token 预算表」的雏形 —— 到那时候我们会把它做成硬约束，
        现在是"先测量，后优化"。
        """
        system_tokens = estimate_tokens(self.system_prompt)
        fact_tokens = (
            estimate_messages_tokens(
                [{"role": "system", "content": LONG_TERM_HEADER + "\n".join(fact_lines)}]
            )
            if fact_lines
            else 0
        )
        summary_text = self.summary.get_summary()
        summary_tokens = (
            estimate_messages_tokens(
                [{"role": "system", "content": SUMMARY_HEADER + summary_text}]
            )
            if summary_text
            else 0
        )
        window_tokens = estimate_messages_tokens(window)
        total = estimate_messages_tokens(prompt)
        budget = self.config.prompt_token_budget

        return {
            "system": system_tokens,
            "long_term": fact_tokens,
            "summary": summary_tokens,
            "recent": window_tokens,
            "total": total,
            "budget": budget,
            "usage": round(total / budget * 100, 1) if budget else 0.0,
        }

    # ------------------------------------------------------------------
    # 步骤 6：抽事实并写进长期记忆
    # ------------------------------------------------------------------
    def _extract_and_store(self, text: str, reply: str, source_message_id: int) -> list:
        """
        从**这一轮**的对话里抽事实。

        为什么只抽这一轮、而不是把整段历史重抽一遍？
            因为整段重抽既贵又慢，而且大部分事实早就抽过了。
            用 known_facts 把已有事实告诉模型，它就不会重复抽。

        为什么用 search 拿 known_facts，而不是"列出全部事实"？
            因为用户这句话相关的事实才有重复风险。
            这是个简化 —— 严格来说应该列出该用户最近的全部事实。
        """
        if self.extractor is None:
            return []

        conversation = [{"role": "user", "content": text}]
        if reply:
            conversation.append({"role": "assistant", "content": reply})

        try:
            known = [
                fact["text"]
                for fact in self.long_term.search(
                    text, top_k=10, user_id=self.user_id, record_hits=False
                )
            ]
            facts = self.extractor.extract(conversation, known_facts=known or None)
        except Exception as error:
            # 抽取失败不影响这一轮对话 —— 大不了这轮没有新记忆
            self.last_error = error
            return []

        for fact in facts:
            try:
                self.long_term.add(
                    fact.predicate,
                    fact.object,
                    subject=fact.subject,
                    user_id=self.user_id,
                    confidence=fact.confidence,
                    source_message_id=source_message_id,
                )
            except Exception as error:
                self.last_error = error

        return facts

    # ==================================================================
    # 状态与维护
    # ==================================================================
    def status(self) -> dict:
        """返回当前会话的状态，CLI 的 /status 用它。"""
        self._ensure_started()
        summary_text = self.summary.get_summary()
        window = [m for m in self.short_term.get_window() if m["role"] != "system"]
        return {
            "session_id": self.session_id,
            "messages": self.store.messages.count(self.session_id),
            "history": len(self._history),
            # ⚠️ 注意区分这两个数：
            #   stored    —— 短期记忆里**存着**多少条（会一直增长）
            #   window    —— 这一轮**真正发给模型**的有多少条（受预算裁剪）
            # 我第一版把这两个搞混了，显示成"窗口 12 条"，
            # 实际发给模型的只有 4 条 —— 会让人完全误判预算有没有生效。
            "stored_messages": len(self.short_term),
            "window_messages": len(window),
            "window_tokens": self.short_term.token_count,
            "summary_tokens": estimate_tokens(summary_text) if summary_text else 0,
            "covered_until": self._covered_until,
            "pending_evicted": self.summary.pending_count,
            "facts": self.long_term.count(user_id=self.user_id),
            "last_error": repr(self.last_error) if self.last_error else None,
        }

    def forget_all(self) -> None:
        """
        彻底忘掉这个会话的一切：消息、摘要、事实。

        这是"被遗忘权"的入口。注意要**四个地方都清**，
        只清一个地方的话，用户会发现"明明让它忘了，它还记得"。
        """
        self._ensure_started()
        self.store.delete_session(self.session_id)
        self.long_term.delete_user(self.user_id)
        self.short_term.clear()
        self.summary.clear()
        self._history.clear()
        self._covered_until = 0
        self._fed_until = 0
        self.last_error = None

    def close(self) -> None:
        """释放资源（关闭数据库连接）。"""
        database = getattr(self.store, "db", None)
        if database is not None:
            try:
                database.close()
            except Exception:
                pass

    def __enter__(self) -> "ConversationEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<ConversationEngine 会话={self.session_id}"
            f" 历史={len(self._history)}条"
            f" 是否已启动={'是' if self._started else '否'}>"
        )
