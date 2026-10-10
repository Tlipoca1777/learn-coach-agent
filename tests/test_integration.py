"""End-to-end checks for the real memory modules and the offline CLI."""

import json

from memory_assistant.cli import main
from memory_assistant.config import Config
from memory_assistant.embeddings import FakeEmbeddings
from memory_assistant.engine import ConversationEngine
from memory_assistant.llm import FakeLLM
from memory_assistant.memory import FactExtractor, LongTermMemory, ShortTermMemory, SummaryMemory
from memory_assistant.storage import Database, Store


def make_fake_llm():
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

    return FakeLLM(reply_fn=reply)


def make_engine(tmp_path, session_id="integration-session"):
    config = Config.from_env(require_key=False)
    database = Database(tmp_path / "assistant.db")
    database.initialize()
    llm = make_fake_llm()
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
