"""Run the reproducible D18 selection-policy evaluation.

This is deliberately a small policy-level experiment.  It seeds 20 answer
attempts, freezes the resulting mastery snapshot, then compares 20 random
draws with 20 calls to the public ``AdaptiveQuiz.start`` entry point.  No
question bank, network call, or real LLM is involved.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

from memory_assistant.quiz import AdaptiveQuiz
from memory_assistant.storage import Database, Store


SEED = 20261013
USER_ID = "d18-eval"
NOW = "2026-01-10T00:00:00+00:00"
FUTURE_DUE = "2026-01-20T00:00:00+00:00"
OVERDUE_DUE = "2026-01-08T00:00:00+00:00"
DRAW_COUNT = 20


@dataclass(frozen=True)
class TopicFixture:
    name: str
    kind: str
    verdict: str
    repetitions: int


TOPIC_FIXTURES = (
    TopicFixture("python.generator", "weak", "wrong", 1),
    TopicFixture("python.decorators", "weak", "wrong", 1),
    TopicFixture("memory.retrieval", "weak", "wrong", 1),
    TopicFixture("llm.temperature", "weak", "wrong", 1),
    TopicFixture("python.scopes", "learning", "partial", 1),
    TopicFixture("memory.summaries", "learning", "partial", 1),
    TopicFixture("fastapi.dependencies", "learning", "partial", 1),
    TopicFixture("project.evaluation", "learning", "partial", 1),
    TopicFixture("python.types", "mastered", "correct", 4),
    TopicFixture("memory.layers", "mastered", "correct", 4),
    TopicFixture("tools.registry", "mastered", "correct", 4),
)


class StaticQuestionModel:
    """Deterministic stand-in used only for AdaptiveQuiz question generation."""

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> str:
        topic = messages[-1]["content"].split("（", 1)[0].replace("知识点：", "")
        return f"请解释「{topic}」的核心概念。"


def _seed_history(store: Store) -> None:
    store.learning._clock = lambda: NOW
    for fixture in TOPIC_FIXTURES:
        for index in range(fixture.repetitions):
            store.learning.record_answer(
                topic_name=fixture.name,
                question=f"D18 模拟题 {fixture.name} #{index + 1}",
                user_answer=f"D18 模拟回答 {fixture.name} #{index + 1}",
                verdict=fixture.verdict,
                score={"correct": 1.0, "partial": 0.5, "wrong": 0.0}[fixture.verdict],
                user_id=USER_ID,
            )

    # The fixture has one overdue weak topic so the evaluation also exercises
    # AdaptiveQuiz's due-review-first branch.  All other topics are kept out
    # of the due list; mastery values remain those produced by record_answer.
    store.db.conn.execute(
        "UPDATE topic_mastery SET due_at = ? "
        "WHERE user_id = ?",
        (FUTURE_DUE, USER_ID),
    )
    topic_id = store.db.conn.execute(
        "SELECT id FROM topics WHERE name = ?", ("python.generator",)
    ).fetchone()["id"]
    store.db.conn.execute(
        "UPDATE topic_mastery SET due_at = ? WHERE topic_id = ? AND user_id = ?",
        (OVERDUE_DUE, topic_id, USER_ID),
    )
    store.db.conn.commit()


def _mastery_snapshot(store: Store) -> dict[str, float]:
    rows = store.db.conn.execute(
        "SELECT t.name, m.mastery FROM topic_mastery m "
        "JOIN topics t ON t.id = m.topic_id WHERE m.user_id = ? ORDER BY t.name",
        (USER_ID,),
    ).fetchall()
    return {row["name"]: round(float(row["mastery"]), 4) for row in rows}


def _draw_metrics(draws: list[str], snapshot: dict[str, float]) -> dict[str, Any]:
    weak = sum(snapshot[name] < 0.4 for name in draws)
    mastered = sum(snapshot[name] >= 0.8 for name in draws)
    total = len(draws)
    return {
        "draws": total,
        "weak_draws": weak,
        "mastered_draws": mastered,
        "weak_coverage": round(weak / total, 4) if total else 0.0,
        "invalid_mastered_rate": round(mastered / total, 4) if total else 0.0,
        "selected_topics": {name: draws.count(name) for name in sorted(set(draws))},
    }


def run_evaluation() -> dict[str, Any]:
    db = Database(":memory:")
    db.initialize()
    try:
        store = Store(db)
        _seed_history(store)
        snapshot = _mastery_snapshot(store)
        candidates = sorted(snapshot)

        baseline_rng = random.Random(SEED)
        baseline_draws = [baseline_rng.choice(candidates) for _ in range(DRAW_COUNT)]

        quiz = AdaptiveQuiz(
            store.learning,
            StaticQuestionModel(),
            user_id=USER_ID,
            session_id="d18-selection-eval",
            record_answer=lambda **_: None,
        )
        adaptive_draws = [quiz.start().topic_name for _ in range(DRAW_COUNT)]

        return {
            "seed": SEED,
            "now": NOW,
            "history_attempts": sum(item.repetitions for item in TOPIC_FIXTURES),
            "draws_per_strategy": DRAW_COUNT,
            "candidate_topics": candidates,
            "mastery_snapshot": snapshot,
            "baseline_random": _draw_metrics(baseline_draws, snapshot),
            "adaptive_quiz": _draw_metrics(adaptive_draws, snapshot),
            "assumptions": [
                "The 20 seeded attempts are synthetic and only exercise repository mastery updates.",
                "Both strategies draw from the same 11 topic names and frozen mastery snapshot.",
                "AdaptiveQuiz.start is called without answer(), so selection state is intentionally static.",
            ],
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    result = run_evaluation()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("D18 自适应选题离线评测")
    print(f"history={result['history_attempts']} seed={result['seed']} draws={result['draws_per_strategy']}")
    for name in ("baseline_random", "adaptive_quiz"):
        metrics = result[name]
        print(
            f"{name}: weak_coverage={metrics['weak_coverage']:.4f} "
            f"invalid_mastered_rate={metrics['invalid_mastered_rate']:.4f} "
            f"weak_draws={metrics['weak_draws']}/{metrics['draws']} "
            f"mastered_draws={metrics['mastered_draws']}/{metrics['draws']}"
        )
        print(f"  selected_topics={metrics['selected_topics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
