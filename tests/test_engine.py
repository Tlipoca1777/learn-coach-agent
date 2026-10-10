r"""
ConversationEngine（对话引擎）的测试
================================================================

这些测试**现在就能通过**，因为引擎用的是依赖注入：
所有记忆模块都被替换成了轻量替身，不碰 chromadb、不加载向量模型、
也不依赖那四份还没实现的作业。

运行：
    .\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v

--------------------------------------------------------------------
这个测试文件教你的三件事
--------------------------------------------------------------------
【1】依赖注入怎么让"集成层"变得可测
    看 `make_engine()`：它把五个部件全部换成替身，
    于是"调用顺序、prompt 组装、落库时机"这些**编排逻辑**
    可以在毫秒内验证完。

    如果不做依赖注入，引擎就只能等四个记忆模块都实现完才能测 ——
    到那时如果你发现 prompt 顺序错了，你根本分不清是引擎的问题
    还是某个记忆模块的问题。**测试的隔离性直接决定调试成本。**

【2】怎么测"顺序"
    编排逻辑的核心就是顺序。看 `test_prompt_order_*` 那一组：
    system → 长期记忆 → 摘要 → 近期对话，
    这个顺序写错了功能不会崩，只是效果变差 ——
    所以必须用测试把它钉住。

【3】替身不用很"真"，只要"够用"
    注意 `StubLongTerm.search()` —— 它根本不计算相似度，
    只是"返回前几条"。这够用吗？够。
    因为这里要验证的是"引擎有没有调用检索、有没有把结果放进 prompt"，
    而不是"检索准不准"（那是 test_long_term.py 的职责）。

    **每个测试只验证一件事。** 想让一个测试验证所有事，
    最后它会什么都验证不好。
"""

import pytest

from memory_assistant.config import Config
from memory_assistant.engine import (
    DEFAULT_SYSTEM_PROMPT,
    LONG_TERM_HEADER,
    SUMMARY_HEADER,
    ConversationEngine,
)
from memory_assistant.llm import FakeLLM, estimate_messages_tokens
from memory_assistant.storage import Database, Store


# ====================================================================
# 替身（test double）
# ====================================================================
class StubShortTerm:
    """极简短期记忆：只保留最后 N 条。"""

    system_prompt = None

    def __init__(self, max_messages: int = 4):
        self.max_messages = max_messages
        self._messages: list[dict] = []
        self.clear_count = 0

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_window(self) -> list[dict]:
        return [dict(m) for m in self._messages[-self.max_messages :]]

    def clear(self) -> None:
        self._messages.clear()
        self.clear_count += 1

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def token_count(self) -> int:
        return estimate_messages_tokens(self.get_window())


class StubSummary:
    """极简中期摘要：攒够 N 条就"压缩"成一句话。"""

    def __init__(self, compress_at: int = 2):
        self.summary: str | None = None
        self.covered = 0
        self.pending: list[dict] = []
        self.compress_at = compress_at
        self.compressions = 0
        self.fed_batches: list[list[dict]] = []

    def add_evicted(self, messages: list[dict]) -> int:
        self.pending.extend(messages)
        self.fed_batches.append(list(messages))
        return len(messages)

    def maybe_compress(self) -> str | None:
        if len(self.pending) < self.compress_at:
            return None
        self.compressions += 1
        self.summary = f"摘要v{self.compressions}"
        self.covered = max(m["id"] for m in self.pending)
        self.pending.clear()
        return self.summary

    def get_summary(self) -> str | None:
        return self.summary

    @property
    def covered_until(self) -> int:
        return self.covered

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def restore(self, summary, covered_until: int = 0) -> None:
        self.summary = (summary or "").strip() or None
        self.covered = int(covered_until)

    def clear(self) -> None:
        self.summary = None
        self.covered = 0
        self.pending.clear()


class StubLongTerm:
    """极简长期记忆：用字典存事实，检索就返回前几条。"""

    def __init__(self, seed: list[str] | None = None):
        self.facts: dict[str, dict] = {}
        self.next_id = 1
        self.searches: list[dict] = []
        for text in seed or []:
            subject, predicate, obj = text.split(" ", 2)
            self.add(predicate, obj, subject=subject)

    def add(
        self,
        predicate,
        object,
        *,
        subject="user",
        user_id="default",
        confidence=1.0,
        source_message_id=None,
        now=None,
    ) -> dict:
        fact_id = f"f{self.next_id}"
        self.next_id += 1
        self.facts[fact_id] = {
            "id": fact_id,
            "text": f"{subject} {predicate} {object}",
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "user_id": user_id,
            "confidence": confidence,
            "source_message_id": source_message_id,
            "hit_count": 0,
        }
        return {"id": fact_id, "action": "created", "similarity": 0.0}

    def search(
        self,
        query,
        *,
        top_k=5,
        user_id="default",
        include_invalid=False,
        record_hits=True,
        now=None,
    ):
        self.searches.append({
            "query": query, "record_hits": record_hits, "user_id": user_id
        })
        return [
            dict(f)
            for f in self.facts.values()
            if f["user_id"] == user_id
        ][:top_k]

    def count(self, *, user_id=None, include_invalid=False) -> int:
        return len(self.facts)

    def delete(self, fact_id: str) -> bool:
        return self.facts.pop(fact_id, None) is not None

    def delete_user(self, user_id: str) -> int:
        removed = len(self.facts)
        self.facts.clear()
        return removed


class FakeFact:
    """冒充 ExtractedFact（真正的那个还没实现 text 属性）。"""

    def __init__(self, predicate, object, subject="user", confidence=1.0):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = confidence

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


class StubExtractor:
    def __init__(self, facts=None):
        self.facts = list(facts or [])
        self.calls: list[dict] = []

    def extract(self, messages, *, known_facts=None):
        self.calls.append({"messages": list(messages), "known_facts": known_facts})
        return list(self.facts)


# ====================================================================
# fixture
# ====================================================================
@pytest.fixture
def store():
    """用真实存储层 + 内存数据库 —— 这部分是已实现的，不用替身。"""
    database = Database(":memory:")
    database.initialize()
    yield Store(database)
    database.close()


def make_engine(store, **kwargs) -> ConversationEngine:
    config = Config.from_env(require_key=False)
    parts = {
        "llm": FakeLLM(default_response="好的，我记住了。"),
        "store": store,
        "short_term": StubShortTerm(),
        "summary": StubSummary(),
        "long_term": StubLongTerm(),
        "extractor": StubExtractor(),
    }
    parts.update(kwargs)
    return ConversationEngine(config, session_id="test-session", **parts)


def system_contents(prompt: list[dict]) -> list[str]:
    return [m["content"] for m in prompt if m["role"] == "system"]


# ====================================================================
# 组 1：启动与恢复
# ====================================================================


def test_start_creates_session(store):
    engine = make_engine(store)
    session_id = engine.start()

    assert session_id == "test-session"
    assert store.sessions.get("test-session") is not None


def test_respond_auto_starts(store):
    """没显式调 start() 也应该能直接用。"""
    engine = make_engine(store)
    result = engine.respond("你好")

    assert result.ok
    assert store.sessions.get("test-session") is not None


def test_start_loads_history_from_database(store):
    """⭐ 启动时要把历史消息灌回短期记忆 —— 这是"记得你"的第一步。"""
    store.sessions.create(session_id="test-session")
    store.messages.append("test-session", "user", "我叫小明")
    store.messages.append("test-session", "assistant", "记住了")

    engine = make_engine(store)
    engine.start()

    window = engine.short_term.get_window()
    assert [m["content"] for m in window] == ["我叫小明", "记住了"]
    assert len(engine._history) == 2


def test_start_restores_summary(store):
    """⭐ 启动时要把最新摘要 restore 回去 —— 否则之前的历史会被抹掉。"""
    store.sessions.create(session_id="test-session")
    store.summaries.add("test-session", "用户叫小明，在杭州。", covered_until=42)

    summary = StubSummary()
    engine = make_engine(store, summary=summary)
    engine.start()

    assert summary.get_summary() == "用户叫小明，在杭州。"
    assert engine._covered_until == 42


def test_start_without_summary_sets_zero(store):
    engine = make_engine(store)
    engine.start()

    assert engine._covered_until == 0
    assert engine.summary.get_summary() is None


def test_start_skips_system_messages(store):
    """历史里的 system 消息不该被塞进窗口（引擎自己管系统提示）。"""
    store.sessions.create(session_id="test-session")
    store.messages.append("test-session", "system", "旧的系统提示")
    store.messages.append("test-session", "user", "你好")

    engine = make_engine(store)
    engine.start()

    assert [m["content"] for m in engine.short_term.get_window()] == ["你好"]


# ====================================================================
# 组 2：respond 的基本行为
# ====================================================================


def test_respond_persists_both_messages(store):
    engine = make_engine(store)
    engine.respond("我叫小明")

    rows = store.messages.list_by_session("test-session")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "我叫小明"
    assert rows[1]["content"] == "好的，我记住了。"


def test_respond_returns_reply(store):
    engine = make_engine(store)
    result = engine.respond("你好")

    assert result.reply == "好的，我记住了。"
    assert result.ok


def test_respond_adds_to_short_term(store):
    engine = make_engine(store, short_term=StubShortTerm(max_messages=10))
    engine.respond("我叫小明")

    contents = [m["content"] for m in engine.short_term.get_window()]
    assert contents == ["我叫小明", "好的，我记住了。"]


def test_respond_records_token_count(store):
    engine = make_engine(store)
    engine.respond("我叫小明")

    rows = store.messages.list_by_session("test-session")
    assert all(r["token_count"] is not None for r in rows)


def test_respond_rejects_empty_input(store):
    engine = make_engine(store)
    result = engine.respond("   ")

    assert not result.ok
    assert result.reply == ""
    assert store.messages.count("test-session") == 0, "空输入不该落库"


# ====================================================================
# 组 3：prompt 组装（顺序是关键）
# ====================================================================


def test_prompt_starts_with_system_prompt(store):
    engine = make_engine(store)
    result = engine.respond("你好")

    assert result.prompt[0]["role"] == "system"
    assert result.prompt[0]["content"] == engine.system_prompt


def test_prompt_ends_with_recent_window(store):
    engine = make_engine(store)
    result = engine.respond("我叫小明")

    tail = [m for m in result.prompt if m["role"] != "system"]
    assert tail[-1] == {"role": "user", "content": "我叫小明"}


def test_prompt_includes_long_term_facts(store):
    engine = make_engine(store, long_term=StubLongTerm(["user 住在 杭州"]))
    result = engine.respond("我住哪？")

    systems = system_contents(result.prompt)
    assert any(LONG_TERM_HEADER in text for text in systems)
    assert any("user 住在 杭州" in text for text in systems)
    assert result.retrieved_facts == ["- user 住在 杭州"]


def test_prompt_omits_long_term_section_when_no_facts(store):
    engine = make_engine(store, long_term=StubLongTerm())
    result = engine.respond("你好")

    assert all(LONG_TERM_HEADER not in text for text in system_contents(result.prompt))


def test_prompt_includes_summary(store):
    store.sessions.create(session_id="test-session")
    store.summaries.add("test-session", "用户叫小明。", covered_until=5)
    engine = make_engine(store)
    engine.start()

    result = engine.respond("你好")

    assert result.summary_used == "用户叫小明。"
    assert any(SUMMARY_HEADER in text for text in system_contents(result.prompt))


def test_prompt_order_is_system_facts_summary_recent(store):
    """
    ⭐ 钉住顺序：系统提示 → 长期记忆 → 中期摘要 → 近期对话。

    这个顺序写错了功能不会崩，只是效果变差 —— 所以必须用测试固定住。
    """
    store.sessions.create(session_id="test-session")
    store.summaries.add("test-session", "更早的摘要", covered_until=3)

    engine = make_engine(store, long_term=StubLongTerm(["user 住在 杭州"]))
    engine.start()
    result = engine.respond("你好")

    systems = system_contents(result.prompt)
    assert len(systems) == 3, f"期望三条 system 消息，实际 {len(systems)} 条"
    assert systems[0] == engine.system_prompt
    assert LONG_TERM_HEADER in systems[1]
    assert SUMMARY_HEADER in systems[2]

    # 所有 system 消息都必须在对话之前
    roles = [m["role"] for m in result.prompt]
    last_system = max(i for i, r in enumerate(roles) if r == "system")
    first_other = min(i for i, r in enumerate(roles) if r != "system")
    assert last_system < first_other


def test_budget_report_has_all_parts(store):
    engine = make_engine(store)
    result = engine.respond("你好")

    for key in ["system", "long_term", "summary", "recent", "total", "budget", "usage"]:
        assert key in result.budget, f"budget 报告缺少 {key}"


def test_long_term_budget_limits_injected_facts(store):
    """事实注入受 token 预算约束，不能无限往里塞。"""
    many = StubLongTerm([f"user 事实{i} 值{i}" for i in range(50)])
    engine = make_engine(store, long_term=many, long_term_token_budget=20)
    result = engine.respond("你好")

    assert 0 < len(result.retrieved_facts) < 50


# ====================================================================
# 组 4：把挤出的消息喂给摘要层
# ====================================================================


def test_evicted_messages_are_fed_to_summary(store):
    """⭐ 短期窗口装不下的消息，应该交给中期摘要，而不是直接丢掉。"""
    summary = StubSummary(compress_at=99)
    engine = make_engine(store, short_term=StubShortTerm(max_messages=2), summary=summary)

    engine.respond("第一句")   # 历史只有 1 条，窗口放得下 → 没挤出
    engine.respond("第二句")   # 历史 3 条，窗口 2 条 → 挤出最早那条
    engine.respond("第三句")   # 挤出更多

    fed = [m["content"] for batch in summary.fed_batches for m in batch]
    assert "第一句" in fed
    assert "好的，我记住了。" in fed, "助手的回答也应该被保留，而不是只有用户的话"


def test_evicted_messages_are_fed_only_once(store):
    """
    ⭐ 回归测试：同一批消息不能被重复喂入。

    消息"喂进摘要层"和"被压缩进摘要"之间有延迟 —— 要攒够阈值才会真正压缩。
    如果只用"已压缩位置"来判断该不该喂，同一批消息会在每一轮被重复喂一次，
    待压缩队列里堆满重复内容，越压越乱，还白花 token。

    （这个 bug 是我写集成层时发现的：模块各自都对，拼起来就错了。
      这就是"必须做集成验证"的最好例子。）
    """
    summary = StubSummary(compress_at=99)      # 阈值极高，永远不会真正压缩
    engine = make_engine(store, short_term=StubShortTerm(max_messages=2), summary=summary)

    for _ in range(6):
        engine.respond("说点什么")

    fed_ids = [m["id"] for batch in summary.fed_batches for m in batch]
    assert len(fed_ids) == len(set(fed_ids)), f"有消息被重复喂入：{fed_ids}"
    assert summary.pending_count == len(fed_ids)


def test_already_covered_messages_are_not_refed(store):
    """
    ⭐ 已经被摘要覆盖过的消息不能重复喂。

    否则每轮都会把同一批消息反复压缩，越压越乱。
    """
    summary = StubSummary(compress_at=1)
    engine = make_engine(store, short_term=StubShortTerm(max_messages=2), summary=summary)

    engine.respond("第一句")
    engine.respond("第二句")   # 挤出最早那条 → 立刻触发压缩 → 覆盖位置前进

    covered = engine._covered_until
    assert covered > 0

    summary.fed_batches.clear()
    engine.respond("第三句")

    fed_ids = [m["id"] for batch in summary.fed_batches for m in batch]
    assert fed_ids, "这一轮应该还有新挤出的消息要喂"
    for message_id in fed_ids:
        assert message_id > covered


def test_new_summary_is_persisted(store):
    summary = StubSummary(compress_at=1)
    engine = make_engine(store, short_term=StubShortTerm(max_messages=2), summary=summary)

    engine.respond("第一句")
    result = engine.respond("第二句")

    assert result.new_summary is not None
    saved = store.summaries.latest("test-session")
    assert saved is not None
    assert saved["content"] == result.new_summary
    assert saved["covered_until"] == engine._covered_until


def test_no_summary_when_below_threshold(store):
    summary = StubSummary(compress_at=99)
    engine = make_engine(store, short_term=StubShortTerm(max_messages=2), summary=summary)

    result = engine.respond("第一句")

    assert result.new_summary is None
    assert store.summaries.latest("test-session") is None


# ====================================================================
# 组 5：事实抽取与写入
# ====================================================================


def test_extracted_facts_are_stored(store):
    fact = FakeFact("住在", "杭州")
    extractor = StubExtractor([fact])
    engine = make_engine(store, extractor=extractor)

    result = engine.respond("我叫小明，在杭州")

    assert len(result.extracted_facts) == 1
    assert engine.long_term.count() == 1
    saved = next(iter(engine.long_term.facts.values()))
    assert saved["predicate"] == "住在"
    assert saved["object"] == "杭州"
    assert saved["source_message_id"] == 1, "事实要能溯源到是哪条消息"


def test_extractor_receives_current_turn_only(store):
    extractor = StubExtractor()
    engine = make_engine(store, extractor=extractor)

    engine.respond("第一句")
    engine.respond("第二句")

    assert len(extractor.calls) == 2
    # 每次只传这一轮的一问一答
    assert [m["role"] for m in extractor.calls[-1]["messages"]] == ["user", "assistant"]
    assert extractor.calls[-1]["messages"][0]["content"] == "第二句"


def test_known_facts_passed_to_extractor(store):
    """把已有事实告诉模型，避免重复抽取。"""
    extractor = StubExtractor()
    engine = make_engine(
        store, long_term=StubLongTerm(["user 住在 杭州"]), extractor=extractor
    )

    engine.respond("你好")

    assert extractor.calls[-1]["known_facts"] == ["user 住在 杭州"]


def test_known_facts_lookup_does_not_record_hits(store):
    """
    ⭐ 引擎为了拿 known_facts 而做的检索，不能记录命中次数。

    否则每轮对话都会给一批事实的 hit_count 加一，
    时间衰减和命中加权全被污染 —— 而且看不出来。
    """
    long_term = StubLongTerm(["user 住在 杭州"])
    engine = make_engine(store, long_term=long_term)

    engine.respond("你好")

    lookup_calls = [s for s in long_term.searches if not s["record_hits"]]
    assert len(lookup_calls) == 1, (
        "known_facts 的检索必须传 record_hits=False，而且只调一次"
    )


def test_extractor_failure_does_not_break_turn(store):
    class BrokenExtractor:
        def extract(self, messages, *, known_facts=None):
            raise RuntimeError("抽取器炸了")

    engine = make_engine(store, extractor=BrokenExtractor())
    result = engine.respond("你好")

    # 抽取失败不影响这一轮对话
    assert result.ok
    assert result.reply == "好的，我记住了。"
    assert engine.last_error is not None


# ====================================================================
# 组 6：流式输出
# ====================================================================


def test_on_token_is_called_incrementally(store):
    llm = FakeLLM(default_response="一二三四五六", chunk_size=2)
    engine = make_engine(store, llm=llm)

    pieces: list[str] = []
    result = engine.respond("你好", on_token=pieces.append)

    assert pieces, "应该收到多个片段"
    assert "".join(pieces) == "一二三四五六"
    assert result.reply == "一二三四五六"


def test_streamed_reply_is_persisted(store):
    llm = FakeLLM(default_response="流式回答", chunk_size=2)
    engine = make_engine(store, llm=llm)

    engine.respond("你好", on_token=lambda piece: None)

    rows = store.messages.list_by_session("test-session")
    assert rows[-1]["content"] == "流式回答"


def test_tool_call_is_executed_and_result_returned_to_model(store):
    llm = FakeLLM(
        tool_responses=[
            {
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"123*456"}'},
                }]
            },
            {"content": "123×456=56088。"},
        ]
    )
    engine = make_engine(store, llm=llm)

    result = engine.respond("123×456 等于多少？")

    assert result.ok
    assert result.reply == "123×456=56088。"
    assert llm.call_count == 2
    assert llm.calls[1][-1] == {
        "role": "tool", "tool_call_id": "call-1", "content": "56088"
    }


def test_tool_loop_has_a_configured_limit(store):
    repeated_call = {
        "tool_calls": [{
            "id": "loop",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }]
    }
    llm = FakeLLM(tool_responses=[repeated_call, repeated_call])
    engine = make_engine(store, llm=llm, max_tool_rounds=2)

    result = engine.respond("现在几点？")

    assert not result.ok
    assert isinstance(result.error, RuntimeError)
    assert "超过上限" in str(result.error)
    assert llm.call_count == 2


def test_recall_memory_tool_returns_fact_to_model(store):
    long_term = StubLongTerm()
    long_term.add(
        "喜欢", "深色主题", subject="user", user_id="alice", confidence=0.95
    )
    llm = FakeLLM(
        tool_responses=[
            {"tool_calls": [{
                "id": "recall-1",
                "type": "function",
                "function": {
                    "name": "recall_memory",
                    "arguments": '{"query":"我的主题偏好"}',
                },
            }]},
            {"content": "你之前说过喜欢深色主题。"},
        ]
    )
    engine = make_engine(store, llm=llm, long_term=long_term, user_id="alice")

    result = engine.respond("我之前说过喜欢什么主题？")

    assert result.ok
    assert "深色主题" in result.reply
    assert llm.call_count == 2
    assert llm.calls[1][-1]["role"] == "tool"
    assert llm.calls[1][-1]["tool_call_id"] == "recall-1"
    assert "深色主题" in llm.calls[1][-1]["content"]
    assert {
        "query": "我的主题偏好", "record_hits": True, "user_id": "alice"
    } in long_term.searches
    assert "recall_memory" in DEFAULT_SYSTEM_PROMPT


def test_repeated_recall_calls_stop_at_configured_limit(store):
    long_term = StubLongTerm()
    long_term.add("喜欢", "深色主题", user_id="alice")
    repeated_call = {
        "tool_calls": [{
            "id": "recall-loop",
            "type": "function",
            "function": {
                "name": "recall_memory",
                "arguments": '{"query":"主题偏好"}',
            },
        }]
    }
    llm = FakeLLM(tool_responses=[repeated_call, repeated_call])
    engine = make_engine(
        store, llm=llm, long_term=long_term, user_id="alice", max_tool_rounds=2
    )

    result = engine.respond("我之前说过喜欢什么主题？")

    assert not result.ok
    assert isinstance(result.error, RuntimeError)
    assert "超过上限" in str(result.error)
    assert llm.call_count == 2
    recall_searches = [
        search for search in long_term.searches
        if search["query"] == "主题偏好" and search["record_hits"]
    ]
    assert recall_searches == [
        {"query": "主题偏好", "record_hits": True, "user_id": "alice"},
        {"query": "主题偏好", "record_hits": True, "user_id": "alice"},
    ]


def test_current_time_tool_result_reaches_final_answer(store):
    llm = FakeLLM(
        tool_responses=[
            {
                "tool_calls": [{
                    "id": "time-1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }]
            },
            {"content": "现在是 2026-10-10 12:34。"},
        ]
    )
    engine = make_engine(store, llm=llm)

    result = engine.respond("现在几点？")

    assert result.ok
    assert result.reply == "现在是 2026-10-10 12:34。"
    assert llm.calls[1][-1]["role"] == "tool"
    assert "T" in llm.calls[1][-1]["content"]
    assert "+" in llm.calls[1][-1]["content"]


def test_learning_tools_are_available_in_engine_and_scoped_to_user(store):
    llm = FakeLLM(
        tool_responses=[
            {"tool_calls": [{
                "id": "record-1",
                "type": "function",
                "function": {
                    "name": "record_answer",
                    "arguments": (
                        '{"topic_name":"python.generator",'
                        '"question":"yield 是什么？",'
                        '"user_answer":"不知道",'
                        '"verdict":"wrong","score":0,'
                        '"feedback":"回忆它如何暂停函数。"}'
                    ),
                },
            }]},
            {"tool_calls": [{
                "id": "weak-1",
                "type": "function",
                "function": {"name": "get_weak_topics", "arguments": "{}"},
            }]},
            {"content": "我记录了你的回答，生成器与 yield 是目前的薄弱点。"},
        ]
    )
    engine = make_engine(store, llm=llm, user_id="alice")

    result = engine.respond("我答错了生成器题，请记录并告诉我薄弱点。")

    assert result.ok
    assert "薄弱点" in result.reply
    assert {schema["function"]["name"] for schema in engine.tools.schemas()} == {
        "calculator", "get_current_time", "recall_memory",
        "record_answer", "get_weak_topics"
    }
    assert "python.generator" in llm.calls[2][-1]["content"]
    assert store.learning.get_weak_topics(user_id="alice")[0]["topic_name"] == "python.generator"
    assert store.learning.get_weak_topics(user_id="bob") == []


# ====================================================================
# 组 7：错误处理
# ====================================================================


def test_llm_failure_returns_result_not_exception(store):
    class BrokenLLM:
        def chat(self, messages, **kwargs):
            raise RuntimeError("模拟网络故障")

        def stream_chat(self, messages, **kwargs):
            raise RuntimeError("模拟网络故障")

    engine = make_engine(store, llm=BrokenLLM())
    result = engine.respond("你好")

    assert not result.ok
    assert isinstance(result.error, RuntimeError)
    assert result.reply == ""


def test_user_message_survives_llm_failure(store):
    """⭐ 模型挂了，用户说的话也必须留下来 —— 否则他会白打一遍。"""

    class BrokenLLM:
        def chat(self, messages, **kwargs):
            raise RuntimeError("模拟网络故障")

    engine = make_engine(store, llm=BrokenLLM())
    engine.respond("这句很重要，别丢")

    rows = store.messages.list_by_session("test-session")
    assert [r["content"] for r in rows] == ["这句很重要，别丢"]


# ====================================================================
# 组 8：重启与遗忘
# ====================================================================


def test_restart_restores_full_state(store):
    """
    ⭐ 模拟"关掉程序再打开"：新建一个引擎，共用同一个数据库。

    期望：历史回来了、摘要回来了、窗口也回来了。
    """
    first = make_engine(store, short_term=StubShortTerm(max_messages=2))
    first.respond("我叫小明")
    first.respond("我在杭州")

    # ---- 重启：全新的引擎和全新的记忆对象 ----
    second = make_engine(store, short_term=StubShortTerm(max_messages=10))
    second.start()

    assert len(second._history) == 4
    assert [m["content"] for m in second.short_term.get_window()][0] == "我叫小明"


def test_forget_all_clears_everything(store):
    engine = make_engine(store, extractor=StubExtractor([FakeFact("住在", "杭州")]))
    engine.respond("我叫小明")
    assert store.messages.count("test-session") == 2
    assert engine.long_term.count() == 1

    engine.forget_all()

    assert store.messages.count("test-session") == 0
    assert store.summaries.latest("test-session") is None
    assert engine.long_term.count() == 0
    assert engine.short_term.get_window() == []
    assert engine._covered_until == 0


def test_status_reports_everything(store):
    engine = make_engine(store, extractor=StubExtractor([FakeFact("住在", "杭州")]))
    engine.respond("我叫小明")

    status = engine.status()

    assert status["session_id"] == "test-session"
    assert status["messages"] == 2
    assert status["history"] == 2
    assert status["facts"] == 1


# ====================================================================
# 组 9：其它
# ====================================================================


def test_repr(store):
    engine = make_engine(store)
    assert "ConversationEngine" in repr(engine)
    assert "test-session" in repr(engine)


def test_context_manager_closes(store):
    with make_engine(store) as engine:
        engine.respond("你好")
    assert "ConversationEngine" in repr(engine)
