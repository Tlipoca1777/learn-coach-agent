"""Export the implemented LangGraph topology as Mermaid text.

This is intentionally offline: the graph receives a tiny responder adapter and
does not construct an API client, database, or vector model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_assistant.graph import build_graph


def export(path: Path) -> str:
    graph = build_graph(responder=lambda text: text)
    mermaid = graph.get_graph().draw_mermaid()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mermaid.rstrip() + "\n", encoding="utf-8")
    return mermaid


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 LangGraph Mermaid 拓扑")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/diagrams/conversation_graph.mmd"),
        help="输出 .mmd 文件路径",
    )
    args = parser.parse_args()
    export(args.output)
    print(f"已导出 Mermaid 图：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
