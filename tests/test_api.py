"""Focused offline checks for the D19 API and D20 learning UI contracts."""

import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from memory_assistant.api import create_app
from memory_assistant.config import Config
from memory_assistant.embeddings import FakeEmbeddings
from memory_assistant.engine import ConversationEngine
from memory_assistant.llm import FakeLLM
from memory_assistant.memory import LongTermMemory, ShortTermMemory, SummaryMemory
from memory_assistant.storage import Database, Store


class EmptyExtractor:
    def extract(self, _messages, *, known_facts=None):
        return []


class BrokenLLM:
    def chat_with_tools(self, _messages, _tools):
        raise RuntimeError("private provider detail")


def _events(body: str) -> list[tuple[str, dict]]:
    parsed = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event, json.loads(data)))
    return parsed


@pytest.fixture
def api_fixture(tmp_path):
    db_path = tmp_path / "assistant.sqlite"
    chroma_path = tmp_path / "chroma"
    created_engines = []

    def engine_factory(user_id: str, session_id: str):
        database = Database(db_path)
        database.initialize()
        store = Store(database)
        def offline_reply(messages):
            system = messages[0]["content"] if messages else ""
            if "必须只返回 JSON" in system:
                return json.dumps({
                    "verdict": "wrong",
                    "score": 0,
                    "feedback": "想想这个概念解决的问题。",
                    "explanation": "完整说明这个概念及其用途。",
                }, ensure_ascii=False)
            if "生成一道关于指定知识点" in system:
                return "API 离线测试题"
            return "第一行\n第二行"

        llm = FakeLLM(reply_fn=offline_reply)
        config = Config(
            api_key="",
            base_url="https://example.invalid",
            model="offline-test",
            temperature=0,
            max_tokens=128,
            context_window=2048,
            prompt_budget_ratio=0.5,
            embedding_model="fake",
            hf_endpoint=None,
            data_dir=tmp_path,
        )
        engine = ConversationEngine(
            config,
            session_id=session_id,
            user_id=user_id,
            llm=llm,
            store=store,
            short_term=ShortTermMemory(max_tokens=1000),
            summary=SummaryMemory(llm),
            long_term=LongTermMemory(FakeEmbeddings(), persist_dir=chroma_path),
            extractor=EmptyExtractor(),
        )
        created_engines.append(engine)
        return engine

    app = create_app(engine_factory=engine_factory)

    # Seed user-scoped memory without touching .env or a real model.
    database = Database(db_path)
    database.initialize()
    store = Store(database)
    store.profiles.update("alice", {"喜欢": "深色主题"})
    database.close()
    seed_memory = LongTermMemory(FakeEmbeddings(), persist_dir=chroma_path)
    seed_memory.add(
        "喜欢", "深色主题", subject="user", user_id="alice", confidence=0.95
    )

    return TestClient(app), created_engines, db_path


def test_chat_streams_sse_and_returns_persistent_session_id(api_fixture):
    client, engines, db_path = api_fixture

    response = client.post(
        "/chat",
        json={"message": "你好", "user_id": "alice"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [event for event, _ in events] == ["meta", "token", "done"]
    session_id = events[0][1]["session_id"]
    assert events[0][1]["user_id"] == "alice"
    assert re.fullmatch(r"[0-9a-f]{12}", session_id)
    assert events[1][1]["text"] == "第一行\n第二行"
    assert events[2][1]["session_id"] == session_id

    resumed = client.post(
        "/chat",
        json={"message": "继续", "user_id": "alice", "session_id": session_id},
    )
    assert _events(resumed.text)[0][1]["session_id"] == session_id

    with pytest.raises(sqlite3.ProgrammingError):
        engines[0].store.db.conn.execute("SELECT 1")

    database = Database(db_path)
    database.initialize()
    rows = Store(database).messages.list_by_session(session_id)
    database.close()
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "你好"),
        ("assistant", "第一行\n第二行"),
        ("user", "继续"),
        ("assistant", "第一行\n第二行"),
    ]


def test_memory_is_user_scoped_and_rejects_foreign_session(api_fixture):
    client, _engines, _db_path = api_fixture
    client.post(
        "/chat",
        json={"message": "测试会话", "user_id": "alice", "session_id": "alice-session"},
    )

    alice = client.get("/memory", params={"user_id": "alice", "session_id": "alice-session"})
    assert alice.status_code == 200
    payload = alice.json()
    assert payload["profile"] == {"喜欢": "深色主题"}
    assert payload["session"]["session_id"] == "alice-session"
    assert payload["session"]["recent_messages"]
    assert any("深色主题" in fact["text"] for fact in payload["related_facts"])

    bob = client.get("/memory", params={"user_id": "bob", "session_id": "alice-session"})
    assert bob.status_code == 404
    bob_memory = client.get("/memory", params={"user_id": "bob"})
    assert bob_memory.status_code == 200
    assert bob_memory.json()["profile"] == {}
    assert bob_memory.json()["related_facts"] == []


def test_chat_hides_cross_user_session_and_validates_blank_message(api_fixture):
    client, _engines, _db_path = api_fixture
    client.post(
        "/chat",
        json={"message": "Alice 内容", "user_id": "alice", "session_id": "shared-session"},
    )

    response = client.post(
        "/chat",
        json={"message": "Bob 内容", "user_id": "bob", "session_id": "shared-session"},
    )
    assert response.status_code == 200
    assert _events(response.text) == [
        ("error", {"code": "session_unavailable", "message": "会话不存在或无权访问。"})
    ]
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_model_failure_is_an_error_event_without_provider_details(tmp_path):
    db_path = tmp_path / "failed.sqlite"
    created_engines = []

    def engine_factory(user_id: str, session_id: str):
        database = Database(db_path)
        database.initialize()
        llm = BrokenLLM()
        config = Config(
            api_key="",
            base_url="https://example.invalid",
            model="offline-test",
            temperature=0,
            max_tokens=128,
            context_window=2048,
            prompt_budget_ratio=0.5,
            embedding_model="fake",
            hf_endpoint=None,
            data_dir=tmp_path,
        )
        engine = ConversationEngine(
            config,
            session_id=session_id,
            user_id=user_id,
            llm=llm,
            store=Store(database),
            short_term=ShortTermMemory(max_tokens=1000),
            summary=SummaryMemory(FakeLLM()),
            long_term=LongTermMemory(FakeEmbeddings()),
            extractor=EmptyExtractor(),
        )
        created_engines.append(engine)
        return engine

    client = TestClient(create_app(engine_factory=engine_factory))
    response = client.post("/chat", json={"message": "请失败"})
    events = _events(response.text)
    assert [event for event, _ in events] == ["meta", "error"]
    assert "private provider detail" not in response.text
    with pytest.raises(sqlite3.ProgrammingError):
        created_engines[0].store.db.conn.execute("SELECT 1")

    database = Database(db_path)
    database.initialize()
    rows = Store(database).messages.list_by_session(events[0][1]["session_id"])
    database.close()
    assert [(row["role"], row["content"]) for row in rows] == [("user", "请失败")]


def test_learning_dashboard_returns_only_users_practiced_topics(api_fixture):
    client, _engines, db_path = api_fixture
    database = Database(db_path)
    database.initialize()
    store = Store(database)
    store.learning.record_answer(
        topic_name="python.generator", question="生成器是什么？", user_answer="不知道",
        verdict="wrong", score=0, user_id="alice",
    )
    store.learning.record_answer(
        topic_name="python.types", question="类型是什么？", user_answer="值的类别",
        verdict="correct", score=1, user_id="bob",
    )
    database.close()

    alice = client.get("/learning", params={"user_id": "alice"})
    assert alice.status_code == 200
    payload = alice.json()
    assert payload["counts"] == {"practiced_topics": 1, "weak_topics": 1, "due_reviews": 0}
    assert payload["topics"][0]["topic_name"] == "python.generator"
    assert payload["topics"][0]["mastery"] == 0.15
    assert payload["weak_topics"][0]["topic_name"] == "python.generator"

    bob = client.get("/learning", params={"user_id": "bob"}).json()
    assert [topic["topic_name"] for topic in bob["topics"]] == ["python.types"]
    assert bob["weak_topics"] == []


def test_quiz_api_preserves_retry_state_and_persists_both_attempts(api_fixture):
    client, _engines, db_path = api_fixture
    started = client.post("/quiz/start", json={
        "user_id": "alice", "session_id": "web-session", "topic_name": "python.generator",
    })
    assert started.status_code == 200
    start_payload = started.json()
    assert start_payload["result"]["status"] == "question"
    assert start_payload["result"]["question"] == "API 离线测试题"

    first = client.post("/quiz/answer", json={
        "user_id": "alice", "session_id": "web-session", "answer": "不知道",
    })
    assert first.status_code == 200
    retry = first.json()["result"]
    assert retry["status"] == "retry"
    assert retry["attempt_number"] == 1
    assert retry["question"] == start_payload["result"]["question"]
    assert "完整说明" not in retry["feedback"]

    second = client.post("/quiz/answer", json={
        "user_id": "alice", "session_id": "web-session", "answer": "还是不知道",
    })
    assert second.status_code == 200
    outcome = second.json()["result"]
    assert outcome["status"] == "explained"
    assert outcome["attempt_number"] == 2
    assert "完整说明" in outcome["explanation"]

    database = Database(db_path)
    database.initialize()
    rows = database.conn.execute(
        "SELECT question, user_answer, verdict FROM attempts WHERE user_id='alice' "
        "ORDER BY id"
    ).fetchall()
    database.close()
    assert [(row["question"], row["user_answer"], row["verdict"]) for row in rows] == [
        ("API 离线测试题", "不知道", "wrong"),
        ("API 离线测试题", "还是不知道", "wrong"),
    ]
    assert client.post("/quiz/answer", json={
        "user_id": "alice", "session_id": "web-session", "answer": "迟到的答案",
    }).status_code == 409


def test_quiz_api_state_is_user_scoped_and_requires_an_active_question(api_fixture):
    client, _engines, _db_path = api_fixture
    missing = client.post("/quiz/answer", json={
        "user_id": "bob", "session_id": "shared", "answer": "答案",
    })
    assert missing.status_code == 409

    started = client.post("/quiz/start", json={
        "user_id": "alice", "session_id": "shared", "topic_name": "python.generator",
    })
    assert started.status_code == 200
    foreign = client.post("/quiz/answer", json={
        "user_id": "bob", "session_id": "shared", "answer": "答案",
    })
    assert foreign.status_code == 409
    assert client.post("/quiz/answer", json={
        "user_id": "alice", "session_id": "shared", "answer": "   ",
    }).status_code == 422

    conflict = client.post("/quiz/start", json={
        "user_id": "bob", "session_id": "shared", "topic_name": "python.types",
    })
    assert conflict.status_code == 404
    assert conflict.json()["detail"] == "会话不存在或无权访问"

