"""FastAPI surface for chat, memory inspection and adaptive learning UI.

The application creates one engine per request. The synchronous chat engine,
including its thread-affine SQLite connection, is owned by a dedicated worker
thread while an SSE response is being produced.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from memory_assistant.config import Config
from memory_assistant.engine import ConversationEngine
from memory_assistant.quiz import QuizSession

EngineFactory = Callable[[str, str], ConversationEngine]
_END = object()


class ChatRequest(BaseModel):
    """One chat turn and its logical user/session scope."""

    message: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(default="default", min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @field_validator("user_id", "session_id")
    @classmethod
    def strip_scope(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("scope values must not be blank")
        return value


class QuizStartRequest(BaseModel):
    """Start a D17 adaptive question in one stable user/session scope."""

    user_id: str = Field(default="default", min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    topic_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("user_id", "session_id", "topic_name")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("scope and topic values must not be blank")
        return value


class QuizAnswerRequest(BaseModel):
    """Submit an answer for the active question in a stable scope."""

    user_id: str = Field(default="default", min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=20_000)

    @field_validator("user_id", "session_id", "answer")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id、session_id 和 answer 不能只包含空白")
        return value


def _default_engine_factory(user_id: str, session_id: str) -> ConversationEngine:
    # Environment and .env are read only when a request needs an engine, never
    # merely by importing this module. Production requests use the configured
    # model and embeddings through the existing engine builders.
    config = Config.from_env(require_key=True)
    return ConversationEngine(config, session_id=session_id, user_id=user_id)


def _sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _fact_for_response(fact: dict[str, Any]) -> dict[str, Any]:
    """Return a small, JSON-safe projection of one relevant fact hit."""
    fields = ("id", "text", "subject", "predicate", "object", "confidence", "score")
    return {field: fact[field] for field in fields if field in fact}


def create_app(*, engine_factory: EngineFactory | None = None) -> FastAPI:
    """Build the API app, with an injectable factory for offline tests."""
    make_engine = engine_factory or _default_engine_factory
    app = FastAPI(title="有记忆的私人助理 Agent", version="0.1.0")
    # D17 keeps its active QuizSession on ConversationEngine.quiz. The API
    # creates an engine per request, so retain only that small session snapshot
    # between quiz calls. This registry is intentionally process-local.
    app.state.quiz_sessions = {}
    app.state.quiz_locks = {}
    app.state.quiz_locks_guard = threading.Lock()

    def lock_for_quiz_scope(scope: tuple[str, str]) -> threading.Lock:
        with app.state.quiz_locks_guard:
            return app.state.quiz_locks.setdefault(scope, threading.Lock())

    def quiz_result_payload(result) -> dict[str, Any]:
        return asdict(result)

    @app.post("/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        session_id = request.session_id or uuid.uuid4().hex[:12]
        events: queue.Queue[str | object] = queue.Queue()

        def run_turn() -> None:
            engine = None
            try:
                engine = make_engine(request.user_id, session_id)
                engine.start()
                events.put(_sse("meta", {
                    "user_id": request.user_id,
                    "session_id": session_id,
                }))

                emitted_token = False

                def on_token(text: str) -> None:
                    nonlocal emitted_token
                    if text:
                        emitted_token = True
                        events.put(_sse("token", {"text": text}))

                result = engine.respond(request.message, on_token=on_token)
                if result.error is not None:
                    events.put(_sse("error", {
                        "code": "chat_failed",
                        "message": "对话处理失败，请稍后重试。",
                    }))
                else:
                    # Preserve the answer for compatible injected engines that
                    # return a TurnResult without invoking the callback.
                    if result.reply and not emitted_token:
                        events.put(_sse("token", {"text": result.reply}))
                    events.put(_sse("done", {"session_id": session_id}))
            except PermissionError:
                # Conceal whether a conflicting session id exists for another
                # user. The user_id value is a partition key, not authentication.
                events.put(_sse("error", {
                    "code": "session_unavailable",
                    "message": "会话不存在或无权访问。",
                }))
            except Exception:
                # Do not send exception text, provider details, or tracebacks
                # over the public stream.
                events.put(_sse("error", {
                    "code": "chat_failed",
                    "message": "对话服务暂时不可用，请检查服务配置后重试。",
                }))
            finally:
                try:
                    if engine is not None:
                        engine.close()
                finally:
                    events.put(_END)

        def stream() -> Iterator[str]:
            # SQLite connections enforce thread affinity. Run every engine
            # operation in this one worker; the response iterator only moves
            # serialized SSE strings between the queue and the HTTP client.
            worker = threading.Thread(target=run_turn, name="memory-assistant-chat", daemon=True)
            worker.start()
            try:
                while True:
                    item = events.get()
                    if item is _END:
                        break
                    yield item  # type: ignore[misc]
            finally:
                worker.join(timeout=1.0)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/memory")
    def memory(
        user_id: str = Query(default="default", min_length=1, max_length=128),
        session_id: str | None = Query(default=None, min_length=1, max_length=128),
        query: str = Query(default="用户", min_length=1, max_length=500),
    ) -> dict[str, Any]:
        user_id = user_id.strip()
        query = query.strip()
        if not user_id or not query or (session_id is not None and not session_id.strip()):
            raise HTTPException(status_code=422, detail="user_id、query 和 session_id 不能只包含空白")

        resolved_session_id = session_id.strip() if session_id is not None else None
        engine = None
        try:
            engine = make_engine(user_id, resolved_session_id or uuid.uuid4().hex[:12])

            if resolved_session_id is not None:
                row = engine.store.sessions.get(resolved_session_id)
                if row is None or row.get("user_id", "default") != user_id:
                    raise HTTPException(status_code=404, detail="会话不存在")
            else:
                recent = engine.store.sessions.list_recent(1, user_id=user_id)
                if recent:
                    resolved_session_id = recent[0]["id"]
                    engine.session_id = resolved_session_id

            session_snapshot = None
            if resolved_session_id is not None:
                engine.start()
                session_snapshot = {
                    "session_id": resolved_session_id,
                    "summary": engine.summary.get_summary(),
                    "recent_messages": engine.short_term.get_window(),
                }

            facts = engine.long_term.search(
                query,
                top_k=engine.retrieve_top_k,
                user_id=user_id,
                record_hits=False,
            )
            profile = engine.store.profiles.get(user_id) if hasattr(engine.store, "profiles") else {}
            return {
                "user_id": user_id,
                "profile": profile,
                "related_facts": [_fact_for_response(fact) for fact in facts],
                "session": session_snapshot,
            }
        finally:
            if engine is not None:
                engine.close()

    @app.get("/learning")
    def learning(
        user_id: str = Query(default="default", min_length=1, max_length=128),
    ) -> dict[str, Any]:
        """Return the user's practiced topics and existing weak/due subsets."""
        user_id = user_id.strip()
        if not user_id:
            raise HTTPException(status_code=422, detail="user_id 不能只包含空白")
        engine = None
        try:
            engine = make_engine(user_id, uuid.uuid4().hex[:12])
            learning_repository = getattr(engine.store, "learning", None)
            if learning_repository is None:
                raise HTTPException(status_code=503, detail="学习功能不可用")
            topics = learning_repository.get_topic_progress(user_id=user_id)
            weak = learning_repository.get_weak_topics(limit=50, user_id=user_id)
            due = learning_repository.get_due_reviews(limit=50, user_id=user_id)
            return {
                "user_id": user_id,
                "topics": topics,
                "weak_topics": weak,
                "due_reviews": due,
                "counts": {
                    "practiced_topics": len(topics),
                    "weak_topics": len(weak),
                    "due_reviews": len(due),
                },
            }
        finally:
            if engine is not None:
                engine.close()

    @app.post("/quiz/start")
    def quiz_start(request: QuizStartRequest) -> dict[str, Any]:
        session_id = request.session_id or uuid.uuid4().hex[:12]
        scope = (request.user_id, session_id)
        scope_lock = lock_for_quiz_scope(scope)
        with scope_lock:
            engine = None
            try:
                engine = make_engine(request.user_id, session_id)
                result = engine.start_quiz(request.topic_name)
                active = getattr(getattr(engine, "quiz", None), "active", None)
                if active is None:
                    raise HTTPException(status_code=503, detail="答题功能未能启动")
                app.state.quiz_sessions[scope] = QuizSession(
                    user_id=active.user_id,
                    session_id=active.session_id,
                    topic_name=active.topic_name,
                    display_name=active.display_name,
                    question=active.question,
                    reason=active.reason,
                    attempts=active.attempts,
                )
                return {
                    "user_id": request.user_id,
                    "session_id": session_id,
                    "result": quiz_result_payload(result),
                }
            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=503, detail="无法开始答题，请检查学习服务配置") from error
            except PermissionError as error:
                raise HTTPException(status_code=404, detail="会话不存在或无权访问") from error
            except Exception as error:
                raise HTTPException(status_code=503, detail="答题服务暂时不可用，请检查服务配置") from error
            finally:
                if engine is not None:
                    engine.close()

    @app.post("/quiz/answer")
    def quiz_answer(request: QuizAnswerRequest) -> dict[str, Any]:
        scope = (request.user_id, request.session_id)
        scope_lock = lock_for_quiz_scope(scope)
        with scope_lock:
            active = app.state.quiz_sessions.get(scope)
            if active is None:
                raise HTTPException(status_code=409, detail="当前会话没有正在进行的题目")
            engine = None
            try:
                engine = make_engine(request.user_id, request.session_id)
                engine.start()
                quiz = getattr(engine, "quiz", None)
                if quiz is None:
                    raise HTTPException(status_code=503, detail="答题功能不可用")
                # Rehydrate the D17 state object, then let its existing grader
                # and record_answer tool handle the answer unchanged.
                quiz.active = QuizSession(
                    user_id=active.user_id,
                    session_id=active.session_id,
                    topic_name=active.topic_name,
                    display_name=active.display_name,
                    question=active.question,
                    reason=active.reason,
                    attempts=active.attempts,
                )
                result = engine.answer_quiz(request.answer)
                if result.status == "retry":
                    app.state.quiz_sessions[scope] = QuizSession(
                        user_id=quiz.active.user_id,
                        session_id=quiz.active.session_id,
                        topic_name=quiz.active.topic_name,
                        display_name=quiz.active.display_name,
                        question=quiz.active.question,
                        reason=quiz.active.reason,
                        attempts=quiz.active.attempts,
                    )
                else:
                    app.state.quiz_sessions.pop(scope, None)
                return {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "result": quiz_result_payload(result),
                }
            except ValueError as error:
                raise HTTPException(status_code=409, detail="无法提交答案，请重新开始答题") from error
            except Exception as error:
                raise HTTPException(status_code=503, detail="判题服务暂时不可用，请稍后重试") from error
            finally:
                if engine is not None:
                    engine.close()

    return app


app = create_app()

