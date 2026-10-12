"""Small adaptive quiz orchestration used by D17.

The project deliberately does not have a quiz bank yet.  A quiz therefore keeps
only the active question in memory and asks the injected chat model to generate
and judge it.  Durable learning state still goes through ``record_answer``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class QuizSession:
    """The one question currently being answered in a user/session pair."""

    user_id: str
    session_id: str
    topic_name: str
    display_name: str
    question: str
    reason: str
    attempts: int = 0


@dataclass
class QuizResult:
    """A quiz question or answer outcome suitable for the CLI and tests."""

    status: str
    topic_name: str
    display_name: str
    question: str
    reason: str = ""
    verdict: str | None = None
    score: float | None = None
    feedback: str = ""
    explanation: str = ""
    attempt_number: int = 0

    @property
    def completed(self) -> bool:
        return self.status in {"correct", "explained"}

    @property
    def message(self) -> str:
        if self.status == "question":
            return f"{self.reason}\n题目（{self.display_name}）：{self.question}\n请直接回答。"
        if self.status == "retry":
            return (
                f"判定：{self.verdict or '需要补充'}（第 {self.attempt_number} 次）\n"
                f"提示：{self.feedback}\n请再试一次。"
            )
        if self.status == "correct":
            return (
                f"判定：正确（第 {self.attempt_number} 次）。{self.feedback or '答得很好！'}"
            )
        return (
            f"判定：{self.verdict or '仍不正确'}（第 {self.attempt_number} 次）。\n"
            f"完整解释：{self.explanation}"
        )


class AdaptiveQuiz:
    """Select, generate and Socratically grade one question at a time."""

    def __init__(
        self,
        learning_repository,
        llm,
        *,
        user_id: str,
        session_id: str,
        record_answer: Callable[..., Any],
    ) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self.learning = learning_repository
        self.llm = llm
        self.user_id = user_id.strip()
        self.session_id = session_id.strip()
        self._record_answer = record_answer
        self.active: QuizSession | None = None

    def start(self, topic_name: str | None = None) -> QuizResult:
        """Start a new question, replacing any unfinished question."""
        selected, reason = self._select_topic(topic_name)
        topic = selected["topic_name"]
        display_name = selected.get("display_name") or topic
        question = self._generate_question(topic, display_name)
        self.active = QuizSession(
            user_id=self.user_id,
            session_id=self.session_id,
            topic_name=topic,
            display_name=display_name,
            question=question,
            reason=reason,
        )
        return QuizResult(
            status="question",
            topic_name=topic,
            display_name=display_name,
            question=question,
            reason=reason,
        )

    def answer(self, user_answer: str) -> QuizResult:
        """Judge the active answer and persist every attempt."""
        state = self.active
        if state is None:
            raise ValueError("没有正在进行的题目，请先使用 /quiz")
        if not isinstance(user_answer, str) or not user_answer.strip():
            raise ValueError("答案不能为空")

        evaluation = self._judge(state.question, user_answer.strip(), state.attempts + 1)
        verdict = evaluation["verdict"]
        score = evaluation["score"]
        attempt_number = state.attempts + 1

        if verdict == "correct":
            feedback = evaluation["feedback"] or "回答抓住了核心。"
            self._save(state, user_answer.strip(), verdict, score, feedback)
            state.attempts = attempt_number
            result = QuizResult(
                status="correct",
                topic_name=state.topic_name,
                display_name=state.display_name,
                question=state.question,
                verdict=verdict,
                score=score,
                feedback=feedback,
                attempt_number=attempt_number,
            )
            self.active = None
            return result

        if attempt_number == 1:
            hint = evaluation["feedback"] or self._default_hint(state.topic_name)
            self._save(state, user_answer.strip(), verdict, score, hint)
            state.attempts = attempt_number
            return QuizResult(
                status="retry",
                topic_name=state.topic_name,
                display_name=state.display_name,
                question=state.question,
                verdict=verdict,
                score=score,
                feedback=hint,
                attempt_number=attempt_number,
            )

        # A second failed attempt is the first point where a full explanation
        # is allowed.  Do not reuse a short first-step hint as the explanation.
        explanation = evaluation["explanation"] or self._default_explanation(state.topic_name)
        self._save(state, user_answer.strip(), verdict, score, explanation)
        state.attempts = attempt_number
        result = QuizResult(
            status="explained",
            topic_name=state.topic_name,
            display_name=state.display_name,
            question=state.question,
            verdict=verdict,
            score=score,
            explanation=explanation,
            attempt_number=attempt_number,
        )
        self.active = None
        return result

    def _select_topic(self, requested: str | None) -> tuple[dict[str, Any], str]:
        if requested is not None:
            if not isinstance(requested, str):
                raise ValueError("知识点必须是字符串")
            topic = requested.strip()
            if not topic:
                raise ValueError("知识点不能为空")
            return (
                {"topic_name": topic, "display_name": topic},
                f"按你的指定练习「{topic}」。",
            )

        due = self.learning.get_due_reviews(limit=50, user_id=self.user_id)
        weak = self.learning.get_weak_topics(limit=50, user_id=self.user_id)
        due_names = {item["topic_name"] for item in due}
        candidates: list[tuple[tuple[int, float, float, str], dict[str, Any]]] = []
        for item in due:
            mastery = float(item.get("mastery", 0.3))
            overdue = max(0.0, float(item.get("overdue_days", 0.0)))
            priority = (1.0 - mastery) * 0.6 + min(overdue / 7.0, 1.0) * 0.4
            candidates.append(((1, priority, -mastery, item["topic_name"]), item))
        for item in weak:
            if item["topic_name"] in due_names:
                continue
            mastery = float(item.get("mastery", 0.3))
            priority = (1.0 - mastery) * 0.6
            candidates.append(((0, priority, -mastery, item["topic_name"]), item))

        if candidates:
            _, selected = max(candidates, key=lambda item: item[0])
            if selected["topic_name"] in due_names:
                reason = (
                    f"优先安排「{selected.get('display_name', selected['topic_name'])}」："
                    "它已经到期复习。"
                )
            else:
                reason = (
                    f"优先安排「{selected.get('display_name', selected['topic_name'])}」："
                    f"当前掌握度 {float(selected.get('mastery', 0.0)):.3f}，属于薄弱知识点。"
                )
            return selected, reason

        return (
            {"topic_name": "general", "display_name": "综合基础"},
            "暂时没有已记录的到期或薄弱知识点，先做一道综合基础题。",
        )

    def _generate_question(self, topic: str, display_name: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是学习教练。请为用户生成一道关于指定知识点的短答题。"
                    "只输出题目本身，不要输出答案、提示或 JSON。"
                ),
            },
            {"role": "user", "content": f"知识点：{display_name}（{topic}）"},
        ]
        question = self.llm.chat(prompt)
        question = (question or "").strip()
        return question or f"请用自己的话解释「{display_name}」的核心概念，并举一个例子。"

    def _judge(self, question: str, answer: str, attempt_number: int) -> dict[str, Any]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是严格但友善的学习教练。判定用户答案，必须只返回 JSON，"
                    '格式为 {"verdict":"correct|partial|wrong","score":0到1之间的数字,'
                    '"feedback":"简短反馈或小提示","explanation":"完整解释"}。'
                    "第一次答错时 feedback 只能是小提示，不能泄露完整答案；"
                    "第二次答错时 explanation 才填写完整解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"第 {attempt_number} 次作答\n题目：{question}\n用户答案：{answer}"
                ),
            },
        ]
        raw = self.llm.chat(prompt)
        parsed = _parse_json_object(raw)
        verdict = str(parsed.get("verdict", "wrong")).strip().lower()
        if verdict not in {"correct", "partial", "wrong"}:
            verdict = "wrong"
        try:
            score = float(parsed.get("score", 1.0 if verdict == "correct" else 0.0))
        except (TypeError, ValueError):
            score = 1.0 if verdict == "correct" else 0.0
        score = min(1.0, max(0.0, score))
        return {
            "verdict": verdict,
            "score": score,
            "feedback": _text(parsed.get("feedback")),
            "explanation": _text(parsed.get("explanation")),
        }

    def _save(self, state: QuizSession, answer: str, verdict: str, score: float, feedback: str) -> None:
        self._record_answer(
            topic_name=state.topic_name,
            question=state.question,
            user_answer=answer,
            verdict=verdict,
            score=score,
            feedback=feedback,
        )

    @staticmethod
    def _default_hint(topic_name: str) -> str:
        return f"先说出「{topic_name}」解决什么问题，再回忆它的关键步骤。"

    @staticmethod
    def _default_explanation(topic_name: str) -> str:
        return f"完整解释需要说明「{topic_name}」的定义、作用，以及一个具体例子。"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


__all__ = ["AdaptiveQuiz", "QuizResult", "QuizSession"]
