"""Small, dependency-free tool framework and the D8 built-in tools."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """A callable tool and its OpenAI-compatible JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def to_api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Register tools by name, expose schemas and execute validated arguments."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"tool name is empty or already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_api_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        try:
            values = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("tool arguments must be a JSON object") from error
        if not isinstance(values, dict):
            raise ValueError("tool arguments must be a JSON object")
        result = tool.function(**values)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def calculate(expression: str) -> int | float:
    """Evaluate basic arithmetic without eval, names, calls, or other Python code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        root = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError) as error:
        raise ValueError("invalid arithmetic expression") from error

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent is too large")
            value = _BIN_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            value = _UNARY_OPS[type(node.op)](visit(node.operand))
        else:
            raise ValueError("only numeric arithmetic is allowed")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("result is outside the supported range")
        return value

    result = visit(root)
    return int(result) if result.is_integer() else result


def get_current_time(*, now: Callable[[], datetime] | None = None) -> str:
    """Return local wall-clock time with its UTC offset for unambiguous context."""
    current = (now or (lambda: datetime.now().astimezone()))()
    if current.tzinfo is None:
        raise ValueError("current time must include a timezone")
    return current.isoformat(timespec="seconds")


def create_default_tools(*, clock: Callable[[], datetime] | None = None) -> ToolRegistry:
    """Build the D8 calculator and current-time tools."""
    return ToolRegistry(
        [
            Tool(
                name="calculator",
                description="Calculate a basic arithmetic expression accurately.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Arithmetic expression, e.g. 123*456"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
                function=calculate,
            ),
            Tool(
                name="get_current_time",
                description="Get the current local date and time with timezone offset.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                function=lambda: get_current_time(now=clock),
            ),
        ]
    )


__all__ = ["Tool", "ToolRegistry", "calculate", "get_current_time", "create_default_tools"]
