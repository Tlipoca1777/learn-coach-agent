"""LangGraph orchestration for one conversation turn.

D11 deliberately keeps the graph small.  ``ConversationEngine`` still owns
the memory pipeline and the bounded tool loop implemented in D1-D10; this
module only makes the high-level turn lifecycle explicit as graph nodes.

The graph is dependency-injected so it can be exercised without an API call::

    graph = build_graph(responder=lambda text: "答复: " + text)
    result = graph.invoke({"messages": [{"role": "user", "content": "你好"}]})

Checkpointing is intentionally left to D12.  The compiled graph returned by
``build_graph`` supports ``get_graph().draw_mermaid()`` for inspection.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError(
        "D11 的图编排需要安装 langgraph；请运行 "
        "python -m pip install 'langgraph>=1.2.0,<2'"
    ) from exc


class AgentState(TypedDict, total=False):
    """State passed between the D11 nodes.

    ``messages`` uses an append reducer: callers provide the existing
    conversation and the response node returns only the new assistant message.
    LangGraph then keeps both sides of the turn in the final state.
    """

    messages: Annotated[list[dict[str, Any]], operator.add]
    user_id: str
    user_input: str
    reply: str
    retrieved_facts: list[str]
    tool_rounds: int
    error: str | None
    route: str


Responder = Callable[[str], Any]


def _latest_user_message(messages: list[dict[str, Any]] | None) -> str:
    """Return the latest non-empty user message or raise a useful error."""

    for message in reversed(messages or []):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("图状态中至少需要一条非空的 user 消息")


def _result_to_reply(result: Any) -> tuple[str, str | None, dict[str, Any]]:
    """Normalize an engine result, string responder, or mapping.

    The production engine returns :class:`TurnResult`, while tests and small
    integrations often use a plain string.  Keeping this adapter here avoids
    making LangGraph nodes know about either concrete implementation.
    """

    if isinstance(result, str):
        return result.strip(), None, {}

    if isinstance(result, Mapping):
        reply = result.get("reply", "")
        error = result.get("error")
        metadata = {
            key: result[key]
            for key in ("retrieved_facts", "tool_rounds")
            if key in result
        }
    else:
        reply = getattr(result, "reply", "")
        error = getattr(result, "error", None)
        metadata = {}
        for key in ("retrieved_facts", "tool_rounds"):
            if hasattr(result, key):
                metadata[key] = getattr(result, key)

    error_text = None if error is None else str(error)
    return str(reply or "").strip(), error_text, metadata


def _resolve_responder(
    engine: Any | None,
    responder: Responder | None,
) -> Responder:
    if responder is not None and engine is not None:
        raise ValueError("engine 与 responder 只能传一个")
    if responder is not None:
        return responder
    if engine is not None:
        respond = getattr(engine, "respond", None)
        if not callable(respond):
            raise TypeError("engine 必须提供可调用的 respond(text) 方法")
        return respond
    raise ValueError("请传入 engine 或 responder，图节点才知道如何生成答复")


def build_graph(
    engine: Any | None = None,
    *,
    responder: Responder | None = None,
):
    """Build and compile the D11 one-turn conversation graph.

    The topology is ``START -> prepare -> respond``.  A conditional edge then
    routes successful turns to ``finish`` and failed turns to ``fail``; both
    nodes terminate at ``END``.  ``engine`` is normally a
    :class:`ConversationEngine`, while ``responder`` is convenient for tests
    and other adapters.
    """

    reply_fn = _resolve_responder(engine, responder)

    def prepare(state: AgentState) -> dict[str, Any]:
        supplied_input = state.get("user_input")
        user_input = (
            supplied_input.strip()
            if isinstance(supplied_input, str) and supplied_input.strip()
            else _latest_user_message(state.get("messages"))
        )
        return {"user_input": user_input, "route": "respond", "error": None}

    def respond(state: AgentState) -> dict[str, Any]:
        try:
            raw_result = reply_fn(state["user_input"])
            reply, result_error, metadata = _result_to_reply(raw_result)
            if result_error:
                return {
                    "reply": reply,
                    "error": result_error,
                    "route": "fail",
                    **metadata,
                }
            return {
                "reply": reply,
                "error": None,
                "route": "finish",
                "messages": ([{"role": "assistant", "content": reply}] if reply else []),
                **metadata,
            }
        except Exception as exc:  # graph boundary turns failures into state
            return {"reply": "", "error": str(exc), "route": "fail"}

    def route(state: AgentState) -> str:
        return "fail" if state.get("error") or state.get("route") == "fail" else "finish"

    def finish(state: AgentState) -> dict[str, Any]:
        return {"route": "finish"}

    def fail(state: AgentState) -> dict[str, Any]:
        return {"reply": "", "route": "fail"}

    workflow = StateGraph(AgentState)
    workflow.add_node("prepare", prepare)
    workflow.add_node("respond", respond)
    workflow.add_node("finish", finish)
    workflow.add_node("fail", fail)
    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "respond")
    workflow.add_conditional_edges(
        "respond",
        route,
        {"finish": "finish", "fail": "fail"},
    )
    workflow.add_edge("finish", END)
    workflow.add_edge("fail", END)
    return workflow.compile()


def build_conversation_graph(
    engine: Any | None = None,
    *,
    responder: Responder | None = None,
):
    """Descriptive alias for :func:`build_graph`."""

    return build_graph(engine, responder=responder)


create_graph = build_conversation_graph


__all__ = ["AgentState", "build_graph", "build_conversation_graph", "create_graph"]
