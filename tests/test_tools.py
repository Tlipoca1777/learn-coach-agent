from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_assistant.tools import (
    Tool,
    ToolRegistry,
    calculate,
    create_default_tools,
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
