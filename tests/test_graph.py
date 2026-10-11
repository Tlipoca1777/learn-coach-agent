"""D11: small LangGraph orchestration contract tests.

These tests deliberately use a tiny engine double.  D11 is about the graph
topology and state hand-off; persistence/checkpointing belongs to D12 and the
real LLM belongs to the integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("langgraph")


class StubEngine:
    """The only engine contract a D11 graph needs: one text turn in/out."""

    def __init__(self, reply: str = "图工作流已运行") -> None:
        self.reply = reply
        self.calls: list[str] = []

    def respond(self, text: str, **_: Any) -> SimpleNamespace:
        self.calls.append(text)
        return SimpleNamespace(reply=self.reply, error=None, ok=True)


def _factory():
    """Resolve the public factory while allowing the descriptive D11 alias."""
    graph_module = pytest.importorskip("memory_assistant.graph")
    for name in ("build_graph", "build_conversation_graph", "create_graph"):
        factory = getattr(graph_module, name, None)
        if factory is not None:
            return factory
    pytest.fail(
        "memory_assistant.graph must export build_graph (or the alias "
        "build_conversation_graph/create_graph)"
    )


def _build(engine: StubEngine):
    factory = _factory()
    # D11 accepts either an engine object or a responder callable.  Prefer the
    # explicit engine keyword and fall back to the positional compatibility
    # form while the public API settles.
    try:
        return factory(engine=engine)
    except TypeError as keyword_error:
        try:
            return factory(engine)
        except TypeError:
            raise keyword_error


def _invoke(graph, state: dict[str, Any]) -> dict[str, Any]:
    result = graph.invoke(state)
    assert isinstance(result, dict)
    return result


def test_graph_compiles_and_exports_mermaid():
    graph = _build(StubEngine())
    drawable = graph.get_graph()
    mermaid = drawable.draw_mermaid()

    assert isinstance(mermaid, str)
    assert mermaid.strip()
    # These labels are stable across LangGraph releases and ensure both ends
    # of the compiled StateGraph are present in the exported diagram.
    assert "__start__" in mermaid
    assert "__end__" in mermaid
    assert len(drawable.nodes) >= 3
    assert len(drawable.edges) >= 3


def test_graph_runs_one_turn_and_preserves_state():
    engine = StubEngine("你好，图工作流已经回答。")
    graph = _build(engine)
    output = _invoke(
        graph,
        {
            "messages": [{"role": "user", "content": "你好"}],
            "user_id": "alice",
        },
    )

    assert engine.calls == ["你好"]
    assert output.get("reply") == "你好，图工作流已经回答。"
    messages = output.get("messages")
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert any(
        message.get("role") == "assistant"
        and message.get("content") == "你好，图工作流已经回答。"
        for message in messages
        if isinstance(message, dict)
    )
    assert output.get("user_id") == "alice"


def test_graph_accepts_a_responder_callable():
    graph_module = pytest.importorskip("memory_assistant.graph")
    factory = getattr(graph_module, "build_conversation_graph", None)
    if factory is None:
        pytest.skip("responder factory is optional for this graph implementation")

    graph = factory(responder=lambda text: "callable: " + text)
    output = _invoke(
        graph,
        {"messages": [{"role": "user", "content": "测试"}]},
    )
    assert output.get("reply") == "callable: 测试"


def test_graph_has_conditional_success_and_error_branches():
    graph = _build(StubEngine())
    drawable = graph.get_graph()
    names = set(drawable.nodes)
    lower_names = {str(name).lower() for name in names}

    # The D11 graph intentionally stays small: prepare -> respond -> finish
    # on success and either prepare/respond -> fail on an error.  This keeps
    # the conditional routing visible before D12 adds checkpointing.
    for expected in ("prepare", "respond", "finish", "fail"):
        assert any(expected in name for name in lower_names), (
            f"missing {expected!r} node in graph: {sorted(lower_names)}"
        )
    edges = {(str(edge.source), str(edge.target)) for edge in drawable.edges}
    assert any("prepare" in source and "respond" in target for source, target in edges)
    assert any("respond" in source and "finish" in target for source, target in edges)
    assert any("respond" in source and "fail" in target for source, target in edges)


class FailingEngine(StubEngine):
    def respond(self, text: str, **_: Any) -> SimpleNamespace:
        self.calls.append(text)
        raise RuntimeError("fake responder failure")


def test_graph_routes_responder_failure_to_fail_node():
    graph = _build(FailingEngine())
    output = _invoke(
        graph,
        {"messages": [{"role": "user", "content": "触发错误"}], "user_id": "alice"},
    )
    assert output.get("error")
    assert output.get("reply", "") == ""


def test_graph_rejects_missing_user_message():
    graph = _build(StubEngine())
    try:
        output = graph.invoke({"messages": [], "user_id": "alice"})
    except (ValueError, KeyError, TypeError):
        return
    assert output.get("error"), "an empty message list must take the error path"

