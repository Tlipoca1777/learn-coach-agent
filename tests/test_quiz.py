import json

import pytest

from memory_assistant.cli import main
from memory_assistant.config import Config
from memory_assistant.engine import ConversationEngine
from memory_assistant.llm import FakeLLM
from memory_assistant.quiz import AdaptiveQuiz
from memory_assistant.storage import Database, Store


@pytest.fixture
def store():
    db = Database(":memory:")
    db.initialize()
    value = Store(db)
    yield value
    db.close()


def make_quiz(store, llm, *, user_id="alice", session_id="session-1"):
    return AdaptiveQuiz(
        store.learning,
        llm,
        user_id=user_id,
        session_id=session_id,
        record_answer=lambda **answer: store.learning.record_answer(
            **answer, user_id=user_id
        ),
    )


def test_quiz_prefers_due_topic_over_weak_topic(store):
    clock = ["2026-01-01T00:00:00+00:00"]
    store.learning._clock = lambda: clock[0]
    store.learning.record_answer(
        topic_name="python.weak", question="q", user_answer="a",
        verdict="wrong", score=0, user_id="alice"
    )
    store.db.conn.execute(
        "UPDATE topic_mastery SET due_at = '2026-01-10T00:00:00+00:00' "
        "WHERE user_id = 'alice'"
    )
    store.db.conn.commit()
    store.learning.record_answer(
        topic_name="python.due", question="q", user_answer="a",
        verdict="correct", score=1, user_id="alice"
    )
    clock[0] = "2026-01-02T00:00:00+00:00"

    quiz = make_quiz(store, FakeLLM(["解释题", "{}"])),
    quiz = quiz[0]
    result = quiz.start()

    assert result.topic_name == "python.due"
    assert "到期" in result.reason


def test_first_wrong_answer_only_gives_hint_and_second_wrong_explains(store):
    llm = FakeLLM([
        "什么是生成器？",
        json.dumps({
            "verdict": "wrong", "score": 0,
            "feedback": "想想它如何暂停执行。",
            "explanation": "完整解释：生成器会逐步产出值。",
        }, ensure_ascii=False),
        json.dumps({
            "verdict": "wrong", "score": 0,
            "feedback": "仍需回忆暂停机制。",
            "explanation": "生成器通过 yield 暂停函数并逐个产出值。",
        }, ensure_ascii=False),
    ])
    quiz = make_quiz(store, llm)
    question = quiz.start("python.generator")

    first = quiz.answer("不知道")
    assert first.status == "retry"
    assert "完整解释" not in first.message
    assert "再试一次" in first.message
    assert store.db.count("attempts") == 1

    second = quiz.answer("还是不知道")
    assert second.status == "explained"
    assert "yield" in second.explanation
    assert store.db.count("attempts") == 2
    assert quiz.active is None


def test_correct_answer_is_saved_and_quiz_state_is_user_scoped(store):
    llm = FakeLLM([
        "题目", json.dumps({"verdict": "correct", "score": 1, "feedback": "很好"}),
    ])
    quiz = make_quiz(store, llm, user_id="alice", session_id="alice-session")
    quiz.start("python.types")
    result = quiz.answer("正确答案")

    assert result.status == "correct"
    assert store.db.conn.execute(
        "SELECT COUNT(*) AS count FROM attempts WHERE user_id = 'alice'"
    ).fetchone()["count"] == 1
    assert store.db.conn.execute(
        "SELECT COUNT(*) AS count FROM attempts WHERE user_id = 'bob'"
    ).fetchone()["count"] == 0

    with pytest.raises(ValueError, match="没有正在进行"):
        quiz.answer("再次回答")


def test_fake_cli_quiz_command_uses_next_inputs_as_answers(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    inputs = iter(["/quiz python.generator", "不知道", "还是不知道", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    assert main(["--fake", "--no-stream", "--session", "quiz-cli"]) == 0
    output = capsys.readouterr().out
    assert "python.generator" in output
    assert "再试一次" in output
    assert "完整解释" in output


def test_engine_rebuilds_quiz_on_user_switch_and_forget(store):
    engine = ConversationEngine(
        Config.from_env(require_key=False),
        session_id="quiz-session",
        user_id="alice",
        store=store,
        llm=FakeLLM(["alice题", "{}", "bob题", "{}"]),
        long_term=type("Memory", (), {
            "search": lambda self, *a, **k: [],
            "count": lambda self, **k: 0,
            "delete_user": lambda self, user_id: 0,
        })(),
        extractor=type("Extractor", (), {"extract": lambda self, *a, **k: []})(),
    )
    engine.start()
    first = engine.start_quiz("alice.topic")
    assert engine.quiz.active.user_id == "alice"
    engine.switch_user("bob")
    assert engine.quiz.active is None
    assert engine.quiz.user_id == "bob"
    engine.start_quiz("bob.topic")
    engine.forget_all()
    assert engine.quiz is not None
    assert engine.quiz.active is None
