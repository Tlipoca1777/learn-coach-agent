import json

import pytest

from memory_assistant.storage import Database, SCHEMA_VERSION, Store
from memory_assistant.tools import ToolRegistry, create_learning_tools


@pytest.fixture
def store():
    db = Database(":memory:")
    db.initialize()
    yield Store(db)
    db.close()


def test_record_answer_persists_attempt_and_updates_mastery(store):
    first = store.learning.record_answer(
        topic_name="python.generator",
        question="yield 有什么作用？",
        user_answer="我不太清楚",
        verdict="wrong",
        score=0,
        feedback="先回忆生成器如何暂停执行。",
        user_id="alice",
    )
    second = store.learning.record_answer(
        topic_name="python.generator",
        question="yield 有什么作用？",
        user_answer="暂停函数并产出一个值",
        verdict="correct",
        score=1,
        user_id="alice",
    )

    assert first == {
        "topic_name": "python.generator",
        "mastery": 0.15,
        "attempt_count": 1,
        "status": "weak",
    }
    assert second["mastery"] == 0.405
    assert second["attempt_count"] == 2
    rows = store.db.conn.execute(
        "SELECT verdict, score FROM attempts WHERE user_id = ? ORDER BY id", ("alice",)
    ).fetchall()
    assert [(row["verdict"], row["score"]) for row in rows] == [
        ("wrong", 0.0), ("correct", 1.0)
    ]


def test_partial_answer_uses_smaller_mastery_gain_and_crosses_weak_threshold(store):
    result = store.learning.record_answer(
        topic_name="python.scopes",
        question="什么是闭包？",
        user_answer="函数保存了外部变量",
        verdict="partial",
        score=0.5,
        feedback="还需说明函数和变量的关系。",
        user_id="alice",
    )

    assert result["mastery"] == 0.405
    assert result["status"] == "learning"
    assert store.learning.get_weak_topics(user_id="alice") == []


def test_get_weak_topics_returns_low_mastery_first_and_is_user_scoped(store):
    store.learning.record_answer(
        topic_name="python.decorators", question="q1", user_answer="a1",
        verdict="wrong", score=0, user_id="alice"
    )
    store.learning.record_answer(
        topic_name="python.generator", question="q2", user_answer="a2",
        verdict="wrong", score=0, feedback="再练一次", user_id="alice"
    )
    store.learning.record_answer(
        topic_name="python.generator", question="q3", user_answer="a3",
        verdict="wrong", score=0, feedback="再练一次", user_id="alice"
    )

    topics = store.learning.get_weak_topics(user_id="alice")

    assert [topic["topic_name"] for topic in topics] == [
        "python.generator", "python.decorators"
    ]
    assert topics[0]["status"] == "weak"
    assert topics[0]["latest_feedback"] == "再练一次"
    assert all(topic["mastery"] < 0.4 for topic in topics)
    assert store.learning.get_weak_topics(user_id="bob") == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"topic_name": " "},
        {"verdict": "excellent"},
        {"score": -0.1},
        {"score": 1.1},
        {"score": float("nan")},
    ],
)
def test_record_answer_rejects_invalid_inputs(store, overrides):
    values = {
        "topic_name": "python.generator",
        "question": "question",
        "user_answer": "answer",
        "verdict": "correct",
        "score": 1,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        store.learning.record_answer(**values)
    assert store.db.count("attempts") == 0


def test_learning_tools_call_repository_with_bound_user(store):
    registry = ToolRegistry(create_learning_tools(store.learning, user_id="alice"))

    recorded = json.loads(registry.execute(
        "record_answer",
        json.dumps({
            "topic_name": "python.generator",
            "question": "yield 是什么？",
            "user_answer": "不知道",
            "verdict": "wrong",
            "score": 0,
            "feedback": "提示：它可以暂停函数。",
        }),
    ))
    topics = json.loads(registry.execute("get_weak_topics", "{}"))
    names = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in registry.schemas()
    }

    assert recorded["status"] == "weak"
    assert topics[0]["topic_name"] == "python.generator"
    assert "user_id" not in names["record_answer"]["properties"]
    assert "user_id" not in names["get_weak_topics"]["properties"]
    assert store.learning.get_weak_topics(user_id="default") == []


def test_schema_v1_migrates_to_current_version_without_losing_rows(store):
    session_id = store.sessions.create(title="kept")
    store.db.conn.execute("DROP TABLE topic_mastery")
    store.db.conn.execute("DROP TABLE attempts")
    store.db.conn.execute("DROP TABLE topics")
    store.db.conn.execute(
        "UPDATE meta SET value='1' WHERE key='schema_version'"
    )
    store.db.conn.commit()

    store.db.initialize()

    version = store.db.conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert SCHEMA_VERSION == 2
    assert int(version) == SCHEMA_VERSION
    assert store.sessions.get(session_id)["title"] == "kept"
    assert {"topics", "attempts", "topic_mastery"}.issubset(
        set(store.db.table_names())
    )
