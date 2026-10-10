import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_assistant.tools import (
    Tool,
    ToolRegistry,
    calculate,
    create_default_tools,
    create_memory_tools,
    get_current_time,
)
from memory_assistant.config import Config
from memory_assistant.llm import DeepSeekClient


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("123*456", 56088), ("(2 + 3) * 4", 20), ("7 / 2", 3.5), ("-2**2", -4)],
)
def test_calculator(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", ["", "x * 2", "__import__('os')", "2 ** 1000"])
def test_calculator_rejects_unsafe_or_invalid_expression(expression):
    with pytest.raises((ValueError, ArithmeticError)):
        calculate(expression)


def test_current_time_has_timezone_offset():
    fixed = datetime(2026, 10, 10, 12, 34, 56, tzinfo=timezone.utc)
    assert get_current_time(now=lambda: fixed) == "2026-10-10T12:34:56+00:00"


def test_registry_exposes_schema_and_executes_tool():
    registry = create_default_tools()
    assert {schema["function"]["name"] for schema in registry.schemas()} == {
        "calculator",
        "get_current_time",
    }
    assert registry.execute("calculator", '{"expression":"123*456"}') == "56088"


def test_registry_rejects_duplicate_and_unknown_tool():
    tool = Tool("one", "test", {"type": "object"}, lambda: "ok")
    registry = ToolRegistry([tool])
    with pytest.raises(ValueError):
        registry.register(tool)
    with pytest.raises(ValueError):
        registry.execute("missing", "{}")


def test_recall_memory_tool_uses_bound_user_and_returns_relevant_fields():
    class Memory:
        calls = []

        def search(self, query, *, top_k, user_id, record_hits):
            self.calls.append({
                "query": query,
                "top_k": top_k,
                "user_id": user_id,
                "record_hits": record_hits,
            })
            return [{
                "id": "fact-1",
                "text": "user 喜欢深色主题",
                "subject": "user",
                "predicate": "喜欢",
                "object": "深色主题",
                "confidence": 0.95,
                "score": 0.82,
                "hit_count": 3,
            }]

    memory = Memory()
    tool = create_memory_tools(memory, user_id="alice")[0]
    registry = ToolRegistry([tool])
    schema = registry.schemas()[0]["function"]

    result = json.loads(registry.execute(
        "recall_memory", '{"query":"我的主题偏好","top_k":2}'
    ))

    assert memory.calls == [{
        "query": "我的主题偏好",
        "top_k": 2,
        "user_id": "alice",
        "record_hits": True,
    }]
    assert result == [{
        "id": "fact-1",
        "text": "user 喜欢深色主题",
        "subject": "user",
        "predicate": "喜欢",
        "object": "深色主题",
        "confidence": 0.95,
        "score": 0.82,
    }]
    assert "user_id" not in schema["parameters"]["properties"]
    with pytest.raises(TypeError):
        registry.execute(
            "recall_memory", '{"query":"偏好","user_id":"bob"}'
        )


@pytest.mark.parametrize("arguments", [
    '{"query":" "}',
    '{"query":"偏好","top_k":0}',
    '{"query":"偏好","top_k":11}',
    '{"query":"偏好","top_k":true}',
])
def test_recall_memory_tool_rejects_invalid_queries_and_limits(arguments):
    class Memory:
        def search(self, *args, **kwargs):
            pytest.fail("invalid input must be rejected before searching")

    registry = ToolRegistry(create_memory_tools(Memory()))
    with pytest.raises(ValueError):
        registry.execute("recall_memory", arguments)


def test_deepseek_client_sends_tool_schema_and_normalizes_tool_call():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="calculator", arguments='{"expression":"2+2"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
    response = SimpleNamespace(
        model="test-model",
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )
    requests = []
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: requests.append(kwargs) or response
            )
        )
    )
    client = DeepSeekClient(Config.from_env(require_key=False), client=fake_client)
    schemas = create_default_tools().schemas()

    result = client.chat_with_tools([{"role": "user", "content": "2+2"}], schemas)

    assert requests[0]["tools"] == schemas
    assert requests[0]["tool_choice"] == "auto"
    assert result["tool_calls"][0]["function"] == {
        "name": "calculator", "arguments": '{"expression":"2+2"}'
    }
    assert client.last_response_model == "test-model"
    assert client.last_usage.total_tokens == 13
