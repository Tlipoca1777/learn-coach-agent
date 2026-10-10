"""Small, dependency-free tool framework and the built-in tools."""

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


def create_learning_tools(repository, *, user_id: str = "default") -> list[Tool]:
    """Build D9 study-coach tools bound to one user's learning repository."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    return [
        Tool(
            name="record_answer",
            description=(
                "Record a judged quiz answer and update mastery for the current user. "
                "Call this after evaluating the user's answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string", "description": "Stable topic key, e.g. python.generator"},
                    "question": {"type": "string"},
                    "user_answer": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "feedback": {"type": "string", "description": "Brief reason or next-step hint"},
                },
                "required": ["topic_name", "question", "user_answer", "verdict", "score"],
                "additionalProperties": False,
            },
            function=lambda **answer: repository.record_answer(
                **answer, user_id=user_id
            ),
        ),
        Tool(
            name="get_weak_topics",
            description=(
                "List this user's lowest-mastery topics first, including mastery and "
                "the latest feedback, to choose what to practice next."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5}
                },
                "additionalProperties": False,
            },
            function=lambda limit=5: repository.get_weak_topics(
                limit=limit, user_id=user_id
            ),
        ),
    ]


def create_memory_tools(memory, *, user_id: str = "default") -> list[Tool]:
    """Build D10 long-term-memory lookup tools bound to one user.

    The model supplies only a query. The application owns the memory object
    and binds ``user_id`` in the closure, so a tool call cannot read another
    user's facts by putting a different user id in JSON arguments.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    def recall_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer between 1 and 10")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10")
        facts = memory.search(
            query.strip(), top_k=top_k, user_id=user_id, record_hits=True
        )
        # Keep the model-facing result focused on useful, stable fields.
        return [
            {
                "id": fact["id"],
                "text": fact["text"],
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "object": fact["object"],
                "confidence": fact["confidence"],
                "score": fact.get("score"),
            }
            for fact in facts
        ]

    return [
        Tool(
            name="recall_memory",
            description=(
                "Search the user's long-term memory for facts relevant to a query. "
                "Use this when the user asks what they said or when a specific "
                "past fact is needed; an empty list means no matching fact was found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What past fact to look for",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            function=recall_memory,
        )
    ]


__all__ = [
    "Tool", "ToolRegistry", "calculate", "get_current_time",
    "create_default_tools", "create_learning_tools", "create_memory_tools",
]
