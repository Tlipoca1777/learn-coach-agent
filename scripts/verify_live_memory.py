"""Run D7's live-model, cross-session memory smoke test.

This script makes real API requests and may incur usage charges. It stores its
test data under ``data/d7-live-check`` so it does not touch normal sessions.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_assistant.config import Config  # noqa: E402
from memory_assistant.engine import ConversationEngine  # noqa: E402


@dataclass
class CheckResult:
    ok: bool
    detail: str


def _new_engine(
    config: Config,
    session_id: str,
    user_id: str,
    *,
    fake_embeddings: bool = False,
    embedding_cache_dir: Path | None = None,
) -> ConversationEngine:
    overrides = {}
    chroma_dir = config.data_dir / (
        "chroma-fake" if fake_embeddings else "chroma-live"
    )
    if fake_embeddings:
        from memory_assistant.embeddings import FakeEmbeddings
        from memory_assistant.memory import LongTermMemory

        overrides["long_term"] = LongTermMemory(
            FakeEmbeddings(), persist_dir=chroma_dir
        )
    elif embedding_cache_dir is not None:
        from memory_assistant.embeddings import LocalEmbeddings
        from memory_assistant.memory import LongTermMemory

        overrides["long_term"] = LongTermMemory(
            LocalEmbeddings(
                config.embedding_model,
                cache_dir=embedding_cache_dir,
                hf_endpoint=config.hf_endpoint,
            ),
            persist_dir=chroma_dir,
        )
    engine = ConversationEngine(
        config,
        session_id=session_id,
        title="D7 live memory verification",
        user_id=user_id,
        **overrides,
    )
    engine.start()
    return engine


def run_live_check(
    turns: int,
    name: str,
    *,
    fake_embeddings: bool = False,
    embedding_cache_dir: Path | None = None,
) -> CheckResult:
    if turns < 2:
        return CheckResult(False, "至少需要 2 轮，才能验证跨轮记忆。")

    config = replace(
        Config.from_env(require_key=True),
        data_dir=PROJECT_ROOT / "data" / "d7-live-check",
    )
    config.ensure_data_dir()

    # Each run gets a distinct session and user, while keeping data persistent
    # across the engine restart performed below.
    from uuid import uuid4

    run_id = uuid4().hex[:10]
    session_id = f"d7-{run_id}"
    user_id = f"d7-user-{run_id}"
    print(f"模型：{config.model}")
    print(f"测试轮数：{turns}")
    print("数据目录：data/d7-live-check")
    print(f"向量模型：{'FakeEmbeddings（只验证持久化链路）' if fake_embeddings else config.embedding_model}")
    print("开始真实 API 对话……")

    engine = _new_engine(
        config,
        session_id,
        user_id,
        fake_embeddings=fake_embeddings,
        embedding_cache_dir=embedding_cache_dir,
    )
    try:
        first = engine.respond(f"请记住：我叫{name}。请简短确认。")
        if not first.ok:
            return CheckResult(False, f"第 1 轮模型请求失败：{first.error!r}")
        print(f"第 1/{turns} 轮完成；抽取事实 {len(first.extracted_facts)} 条。")

        # Use varied ordinary turns to push the identifying statement out of
        # the short-term window and exercise summary creation as well.
        topics = [
            "请用一句话解释什么是递归。",
            "给我一个学习 Python 的小建议。",
            "说明列表和元组的一个区别。",
            "什么是函数的参数？简单解释。",
            "举例说明哈希表的用途。",
            "解释一下 API 是什么。",
            "给我一个专注学习 25 分钟的方法。",
            "简单说说版本控制的作用。",
            "什么叫单元测试？",
            "请解释缓存为什么能提高性能。",
            "用简单例子说明异常处理。",
            "介绍一个整理代码的小习惯。",
            "什么是数据库索引？",
            "简单解释同步和异步的区别。",
            "给我一个阅读技术文档的建议。",
            "什么是依赖注入？",
            "用一句话解释向量检索。",
            "写代码前如何把任务拆小？",
            "请简单总结今天适合做的一项复习。",
        ]
        for index in range(1, turns):
            prompt = topics[(index - 1) % len(topics)]
            result = engine.respond(prompt)
            if not result.ok:
                return CheckResult(False, f"第 {index + 1} 轮模型请求失败：{result.error!r}")
            print(f"第 {index + 1}/{turns} 轮完成。")
    finally:
        engine.close()

    # Build fresh modules and restore from persistent SQLite + Chroma state.
    resumed = _new_engine(
        config,
        session_id,
        user_id,
        fake_embeddings=fake_embeddings,
        embedding_cache_dir=embedding_cache_dir,
    )
    try:
        query = resumed.respond("我叫什么名字？请直接回答姓名。")
        if not query.ok:
            return CheckResult(False, f"重启后的回忆请求失败：{query.error!r}")

        fact_found = any(name in line for line in query.retrieved_facts)
        answer_found = re.search(re.escape(name), query.reply, flags=re.IGNORECASE) is not None
        print(f"重启后回答：{query.reply}")
        print(f"长期记忆命中：{'是' if fact_found else '否'}")
        if fact_found and answer_found:
            embedding_note = "；使用假 embedding，仅验证跨重启记忆链路" if fake_embeddings else ""
            return CheckResult(True, f"真实模型完成多轮对话，并在引擎重启后检索、回答出姓名{embedding_note}。")
        return CheckResult(
            False,
            "未满足跨会话回忆判据："
            f"长期事实命中={fact_found}，回答包含姓名={answer_found}。"
            "请检查 data/d7-live-check 中的运行数据和 API 返回。",
        )
    finally:
        resumed.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=20, help="真实模型对话轮数（默认 20）")
    parser.add_argument("--name", default="D7验证用户", help="测试用姓名，不要填写真实隐私")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="使用假向量，仅验证真实 LLM 的多轮与持久化链路；不验证真实语义检索",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="复用已有的 fastembed 模型缓存目录，例如 data/models",
    )
    args = parser.parse_args()

    try:
        result = run_live_check(
            args.turns,
            args.name,
            fake_embeddings=args.fake_embeddings,
            embedding_cache_dir=args.cache_dir,
        )
    except Exception as error:
        print(f"D7 验证失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"D7 验证{'通过' if result.ok else '未通过'}：{result.detail}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
