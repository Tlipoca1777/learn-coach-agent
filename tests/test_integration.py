"""End-to-end checks for the real memory modules and the offline CLI."""

import json

import pytest

from memory_assistant.cli import main
from memory_assistant.config import Config
from memory_assistant.embeddings import FakeEmbeddings
from memory_assistant.engine import ConversationEngine
from memory_assistant.llm import FakeLLM
from memory_assistant.memory import FactExtractor, LongTermMemory, ShortTermMemory, SummaryMemory
from memory_assistant.storage import Database, Store


def make_fake_llm(*, tool_responses=None):
    def reply(messages):
        prompt = "\n".join(message["content"] for message in messages)
        if "信息抽取器" in prompt:
            return json.dumps(
                [
                    {
                        "subject": "user",
                        "predicate": "住在",
                        "object": "杭州",
                        "confidence": 0.98,
                        "evidence": "我住在杭州",
                    }
                ],
                ensure_ascii=False,
            )
        if "摘要器" in prompt:
            return "较早对话已压缩，用户住在杭州。"
        return "收到，我会记住。"

    return FakeLLM(reply_fn=reply, tool_responses=tool_responses)


def make_engine(
    tmp_path,
    session_id="integration-session",
    *,
    llm=None,
    max_tool_rounds=5,
):
    config = Config.from_env(require_key=False)
    database = Database(tmp_path / "assistant.db")
    database.initialize()
    llm = llm or make_fake_llm()
    store = Store(database)
    short_term = ShortTermMemory(
        max_tokens=8,
        token_counter=lambda messages: sum(len(message["content"]) for message in messages),
    )
    summary = SummaryMemory(
        llm,
        max_summary_tokens=200,
        trigger_tokens=1,
        token_counter=len,
    )
    long_term = LongTermMemory(
        FakeEmbeddings(), persist_dir=tmp_path / "chroma"
    )
    extractor = FactExtractor(llm)
    return ConversationEngine(
        config,
        session_id=session_id,
        llm=llm,
        store=store,
        short_term=short_term,
        summary=summary,
        long_term=long_term,
        extractor=extractor,
        max_tool_rounds=max_tool_rounds,
    )


def test_real_memory_modules_work_together_and_restore_after_restart(tmp_path):
    engine = make_engine(tmp_path)
    engine.start()

    first = engine.respond("我住在杭州")
    assert first.ok
    assert len(first.extracted_facts) == 1
    assert engine.long_term.count(user_id="default") == 1

    second = engine.respond("请告诉我住在哪里")
    assert second.ok
    assert any("user 住在 杭州" in line for line in second.retrieved_facts)
    assert any(
        "user 住在 杭州" in message["content"]
        for message in second.prompt
        if message["role"] == "system"
    )
    assert second.new_summary == "较早对话已压缩，用户住在杭州。"
    covered_until = engine._covered_until
    engine.close()

    # A fresh engine and fresh memory objects must recover persisted state.
    resumed = make_engine(tmp_path)
    resumed.start()
    assert resumed.summary.get_summary() == "较早对话已压缩，用户住在杭州。"
    assert resumed._covered_until == covered_until
    assert resumed.long_term.count(user_id="default") == 1

    third = resumed.respond("你记得我住哪里吗？")
    assert third.ok
    assert any("user 住在 杭州" in line for line in third.retrieved_facts)
    assert any("较早对话已压缩" in message["content"] for message in third.prompt)
    resumed.close()


def test_fake_cli_runs_chinese_conversation_and_reports_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    inputs = iter(["我住在杭州", "/status", "/memory", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    exit_code = main(["--fake", "--no-stream", "--session", "cli-integration"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "长期事实      : 1 条" in output
    assert "user 住在 杭州" in output
    assert "AI >" in output


def test_graph_engine_tool_memory_and_checkpoint_resume_end_to_end(tmp_path):
    sqlite_checkpoint = pytest.importorskip("langgraph.checkpoint.sqlite")
    from memory_assistant.graph import build_graph

    llm = make_fake_llm(
        tool_responses=[
            {
                "tool_calls": [
                    {
                        "id": "calculator-1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"123*456"}',
                        },
                    }
                ]
            },
            {"content": "计算结果是 56088。"},
        ]
    )
    engine = make_engine(tmp_path, llm=llm)
    config = {"configurable": {"thread_id": "d13-session"}}
    checkpoint_path = tmp_path / "graph-checkpoints.sqlite"

    with sqlite_checkpoint.SqliteSaver.from_conn_string(
        str(checkpoint_path)
    ) as saver:
        graph = build_graph(engine=engine, checkpointer=saver)
        first = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "我住在杭州，请计算 123*456"}
                ]
            },
            config=config,
        )

    assert first["reply"] == "计算结果是 56088。"
    assert first["error"] is None
    tool_followup = next(
        call
        for call in llm.calls
        if any(message.get("role") == "tool" for message in call)
    )
    assert any(
        message.get("role") == "tool"
        and message.get("content") == "56088"
        and message.get("tool_call_id") == "calculator-1"
        for message in tool_followup
    )
    first_rows = engine.store.messages.list_by_session(engine.session_id)
    assert [(row["role"], row["content"]) for row in first_rows] == [
        ("user", "我住在杭州，请计算 123*456"),
        ("assistant", "计算结果是 56088。"),
    ]
    assert engine.long_term.count(user_id="default") == 1
    assert any(
        "user 住在 杭州" in fact["text"]
        for fact in engine.long_term.search("我住在杭州", user_id="default")
    )
    engine.close()

    resumed = make_engine(tmp_path)
    with sqlite_checkpoint.SqliteSaver.from_conn_string(
        str(checkpoint_path)
    ) as saver:
        graph = build_graph(engine=resumed, checkpointer=saver)
        second = graph.invoke(
            {"messages": [{"role": "user", "content": "我住在哪里？"}]},
            config=config,
        )

    assert second["error"] is None
    assert len(second["messages"]) == 4
    assert [message["role"] for message in second["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert any("user 住在 杭州" in fact for fact in second["retrieved_facts"])
    assert resumed.summary.get_summary() == "较早对话已压缩，用户住在杭州。"
    assert any(
        "user 住在 杭州" in message["content"]
        for call in resumed.llm.calls
        for message in call
        if message["role"] == "system"
    )
    final_rows = resumed.store.messages.list_by_session(resumed.session_id)
    assert [(row["role"], row["content"]) for row in final_rows] == [
        ("user", "我住在杭州，请计算 123*456"),
        ("assistant", "计算结果是 56088。"),
        ("user", "我住在哪里？"),
        ("assistant", "收到，我会记住。"),
    ]
    assert resumed.long_term.count(user_id="default") == 1
    resumed.close()


def test_graph_can_continue_after_engine_tool_limit_failure(tmp_path):
    sqlite_checkpoint = pytest.importorskip("langgraph.checkpoint.sqlite")
    from memory_assistant.graph import build_graph

    llm = make_fake_llm(
        tool_responses=[
            {
                "tool_calls": [
                    {
                        "id": "calculator-1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ]
            }
        ]
    )
    engine = make_engine(tmp_path, llm=llm, max_tool_rounds=1)
    config = {"configurable": {"thread_id": "d13-failure-session"}}
    checkpoint_path = tmp_path / "failure-checkpoints.sqlite"

    with sqlite_checkpoint.SqliteSaver.from_conn_string(
        str(checkpoint_path)
    ) as saver:
        graph = build_graph(engine=engine, checkpointer=saver)
        failed = graph.invoke(
            {"messages": [{"role": "user", "content": "请计算 2+2"}]},
            config=config,
        )
        assert "工具调用超过上限" in failed["error"]
        assert failed["reply"] == ""

        continued = graph.invoke(
            {"messages": [{"role": "user", "content": "继续聊天"}]},
            config=config,
        )

    assert continued["error"] is None
    assert continued["reply"] == "收到，我会记住。"
    assert [message["content"] for message in continued["messages"]] == [
        "请计算 2+2",
        "继续聊天",
        "收到，我会记住。",
    ]
    rows = engine.store.messages.list_by_session(engine.session_id)
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "请计算 2+2"),
        ("user", "继续聊天"),
        ("assistant", "收到，我会记住。"),
    ]
    engine.close()
