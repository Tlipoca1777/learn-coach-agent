r"""
命令行界面
================================================================

    .\.venv\Scripts\python.exe -m memory_assistant          # 推荐
    .\.venv\Scripts\python.exe -m memory_assistant.cli      # 等价

参数：
    --session <id>    恢复一个已有会话（不传就新建）
    --title <文字>    给新会话起个名字
    --fake            用假模型（不调 API、不花钱、离线可用）
    --no-stream       关掉流式输出（一次性打印完整回答）

内置命令（在对话里输入）：
    /help     看帮助
    /status   会话状态：消息数、摘要、覆盖位置、事实条数、token 占用
    /memory   查看当前注入 prompt 的全部记忆内容
    /prompt   查看本次实际发给模型的消息列表（调试神器）
    /forget   忘掉这个会话的一切（消息 + 摘要 + 事实）
    /exit     退出

--------------------------------------------------------------------
--fake 模式有什么用？
--------------------------------------------------------------------
它让整条链路（记忆、落库、摘要、检索、组装 prompt）都能在**不联网、
不花钱**的情况下跑起来。用途：

    · 演示给别人看（不怕网络抖动，也不会烧 token）
    · 调试记忆逻辑（可以反复重放同一段对话）
    · 在没有 API Key 的环境里开发

代价是回答内容是假的，但**记忆的行为是真的** ——
你可以清楚看到 /status 里的消息数、摘要、事实条数怎么随对话增长。

--------------------------------------------------------------------
为什么这个文件也是我提供的？
--------------------------------------------------------------------
它是纯管道：解析参数、调引擎、打印结果。
真正的逻辑都在 engine.py 和四个记忆模块里。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BANNER = r"""
==============================================================
 有记忆的私人助理
==============================================================
"""

HELP_TEXT = """
内置命令：
  /help     显示这份帮助
  /status   会话状态（消息数、摘要、覆盖位置、事实条数、token 占用）
  /memory   查看当前注入 prompt 的全部记忆
  /prompt   查看本次实际发给模型的消息列表
  /forget   忘掉这个会话的一切
  /exit     退出

学习教练：要求模型出一道题并回答后，可问「我有哪些薄弱知识点？」；
判题记录由 record_answer 保存，查询由 get_weak_topics 完成。
"""


# ====================================================================
# 未实现时的友好提示
# ====================================================================
# 如果某个记忆模块还没实现，直接抛 NotImplementedError 会是一堆看不懂的栈；
# 这里把它翻译成人话，并告诉使用者该跑哪个测试文件。
#
# 注：这里**不引用任何文档路径**。因为项目文档里没有"怎么一步步实现"的内容，
#     引用一个仓库里不存在的路径只会让人困惑。规格以模块本身的文档字符串为准。
MODULE_NAMES = {
    "short_term": "ShortTermMemory",
    "summary": "SummaryMemory",
    "long_term": "LongTermMemory",
    "extraction": "FactExtractor",
}

TEST_FILES = {
    "short_term": "tests/test_short_term.py",
    "summary": "tests/test_summary.py",
    "long_term": "tests/test_long_term.py",
    "extraction": "tests/test_extraction.py",
}


def probe_modules() -> list[str]:
    """
    逐个**真正调用一次**四个记忆模块，返回还没实现的那几个。

    ⚠️ 这里有个容易做错的地方：
       不能只看"能不能构造"。因为四个模块的 `__init__` 都是实现好的
       （参数校验那部分），所以构造一定会成功 —— 检测不出问题。
       必须真的调一个**业务方法**，让它有机会抛 NotImplementedError。

    这也是一个通用的调试原则：
       **"能创建"不等于"能用"。验证要打到真正的执行路径上。**
    """
    from memory_assistant.embeddings import FakeEmbeddings
    from memory_assistant.llm import FakeLLM
    from memory_assistant.memory.extraction import FactExtractor
    from memory_assistant.memory.long_term import LongTermMemory
    from memory_assistant.memory.short_term import ShortTermMemory
    from memory_assistant.memory.summary import SummaryMemory

    def probe_short_term() -> None:
        len(ShortTermMemory(max_tokens=100))

    def probe_summary() -> None:
        SummaryMemory(FakeLLM()).get_summary()

    def probe_long_term() -> None:
        LongTermMemory(FakeEmbeddings()).count()

    def probe_extraction() -> None:
        FactExtractor(FakeLLM()).parse_response("[]")

    checks = [
        ("short_term", probe_short_term),
        ("summary", probe_summary),
        ("long_term", probe_long_term),
        ("extraction", probe_extraction),
    ]

    unimplemented: list[str] = []
    for name, probe in checks:
        try:
            probe()
        except NotImplementedError:
            unimplemented.append(name)
        except Exception:
            # 别的异常不在这里处理，交给主流程报错
            pass

    return unimplemented


def print_unimplemented(unimplemented: list[str]) -> None:
    print("⚠️ 还不能启动：下面几个记忆模块还没实现\n")
    for name in unimplemented:
        class_name = MODULE_NAMES[name]
        test_file = TEST_FILES[name]
        print(f"  · {class_name}")
        print(f"      规格：  src/memory_assistant/memory/{name}.py 的模块文档字符串")
        print(f"      测试：  .\\.venv\\Scripts\\python.exe -m pytest {test_file} -v")
        print()
    print("实现完之后回来再跑一次就行。")
    print("=" * 62)


# ====================================================================
# 打印工具
# ====================================================================
def print_status(status: dict) -> None:
    print()
    print("─" * 62)
    print(f"  会话 id       : {status['session_id']}")
    print(f"  数据库消息数  : {status['messages']}")
    print(f"  内存历史条数  : {status['history']}")
    print(f"  短期记忆存了  : {status['stored_messages']} 条")
    print(f"  实际发给模型  : {status['window_messages']} 条"
          f" / {status['window_tokens']} token   ← 受预算裁剪后的")
    print(f"  中期摘要      : {status['summary_tokens']} token"
          f"，已覆盖到消息 id={status['covered_until']}")
    print(f"  待压缩消息    : {status['pending_evicted']} 条")
    print(f"  长期事实      : {status['facts']} 条")
    if status["last_error"]:
        print(f"  ⚠️ 最近错误   : {status['last_error']}")
    print("─" * 62)
    print()


def print_memory(engine) -> None:
    """展示当前会注入 prompt 的全部记忆内容。"""
    print()
    print("=" * 62)
    print(" 当前注入 prompt 的记忆")
    print("=" * 62)

    facts = engine.long_term.search(
        "用户", top_k=engine.retrieve_top_k, user_id=engine.user_id, record_hits=False
    )
    print(f"\n【长期事实】共 {engine.long_term.count(user_id=engine.user_id)} 条，"
          f"本次检索命中 {len(facts)} 条：")
    if facts:
        for fact in facts:
            print(f"  · {fact['text']}")
            print(f"      相似度 {fact['similarity']:.3f}  衰减 {fact['decay']:.3f}"
                  f"  综合 {fact['score']:.3f}  命中 {fact['hit_count']} 次")
    else:
        print("  （空）")

    summary_text = engine.summary.get_summary()
    print(f"\n【中期摘要】")
    print(f"  {summary_text}" if summary_text else "  （还没有摘要）")

    window = [m for m in engine.short_term.get_window() if m["role"] != "system"]
    print(f"\n【近期对话窗口】{len(window)} 条：")
    for index, message in enumerate(window):
        preview = message["content"][:50].replace("\n", " ")
        print(f"  {index:>2}. [{message['role']}] {preview}")

    print()
    print("=" * 62)
    print()


def print_prompt(result) -> None:
    """展示本次实际发给模型的消息列表。"""
    print()
    print("=" * 62)
    print(" 本次发给模型的消息列表")
    print("=" * 62)
    for index, message in enumerate(result.prompt):
        print(f"\n[{index}] role={message['role']}")
        print("─" * 62)
        print(message["content"])
    print()
    print("─" * 62)
    print(" token 占用：")
    for key, value in result.budget.items():
        if key == "usage":
            print(f"   {key:<12}: {value}%")
        else:
            print(f"   {key:<12}: {value}")
    print("=" * 62)
    print()


# ====================================================================
# 主流程
# ====================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m memory_assistant",
        description="有记忆的私人助理（命令行）",
    )
    parser.add_argument("--session", default=None, help="恢复指定会话 id")
    parser.add_argument("--title", default=None, help="新会话的标题")
    parser.add_argument(
        "--fake", action="store_true", help="用假模型，不调 API、不花钱、可离线"
    )
    parser.add_argument("--no-stream", action="store_true", help="关掉流式输出")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(BANNER, end="")

    unimplemented = probe_modules()
    if unimplemented:
        print_unimplemented(unimplemented)
        return 2

    # ---------- 构造引擎 ----------
    from memory_assistant.config import Config
    from memory_assistant.engine import ConversationEngine

    try:
        config = Config.from_env(require_key=not args.fake)
    except RuntimeError as error:
        print(error)
        return 2

    overrides = {}
    if args.fake:
        import json
        import re

        from memory_assistant.embeddings import FakeEmbeddings
        from memory_assistant.llm import FakeLLM
        from memory_assistant.memory import LongTermMemory

        def fake_reply(messages):
            prompt_text = "\n".join(message.get("content", "") for message in messages)
            if "信息抽取器" in prompt_text:
                conversation = prompt_text.split("需要抽取的对话：", 1)[-1]
                user_lines = re.findall(r"^user: (.*)$", conversation, re.MULTILINE)
                facts = []
                patterns = (
                    (r"我叫([^，。！？\s]+)", "姓名是"),
                    (r"我住在([^，。！？\s]+)", "住在"),
                    (r"我喜欢([^，。！？\s]+)", "喜欢"),
                    (r"我对([^，。！？\s]+?)过敏", "对……过敏"),
                )
                for line in user_lines:
                    for pattern, predicate in patterns:
                        for value in re.findall(pattern, line):
                            facts.append({
                                "subject": "user",
                                "predicate": predicate,
                                "object": value,
                                "confidence": 0.95,
                                "evidence": line,
                            })
                return json.dumps(facts, ensure_ascii=False)
            if "摘要器" in prompt_text:
                return "已压缩较早的对话内容。"
            return "（假模型回答）我记下了。"

        print("⚠️ 假模型模式：回答是假的，但记忆行为是真的\n")
        overrides["llm"] = FakeLLM(reply_fn=fake_reply)
        overrides["long_term"] = LongTermMemory(
            FakeEmbeddings(), persist_dir=config.data_dir / "chroma-fake"
        )

    try:
        engine = ConversationEngine(
            config, session_id=args.session, title=args.title, **overrides
        )
        engine.start()
    except Exception as error:
        print(f"❌ 启动失败：{type(error).__name__}: {error}")
        return 1

    print(f"会话 id：{engine.session_id}")
    print(f"模型    ：{'（假模型）' if args.fake else config.model}")
    print(f"数据目录：{config.data_dir}")
    print()
    print(HELP_TEXT)

    # ---------- 对话循环 ----------
    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("再见！")
            break

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/status":
            print_status(engine.status())
            continue

        if user_input == "/memory":
            print_memory(engine)
            continue

        if user_input == "/forget":
            confirm = input("确认要忘掉这个会话的一切吗？(输入 yes 确认) > ").strip()
            if confirm.lower() == "yes":
                engine.forget_all()
                print("✅ 已清空消息、摘要和事实。\n")
            else:
                print("已取消。\n")
            continue

        # ---------- 正常对话 ----------
        streaming = not args.no_stream
        if streaming:
            print("AI > ", end="", flush=True)

        result = engine.respond(
            user_input,
            on_token=(lambda piece: print(piece, end="", flush=True)) if streaming else None,
        )

        if streaming:
            print()
        else:
            print(f"AI > {result.reply}")

        if not result.ok:
            print(f"\n⚠️ 这一轮出错了：{type(result.error).__name__}: {result.error}")
            print("   （你的消息已经存下来了，可以直接重试）")

        if result.new_summary:
            print(f"\n[已生成新摘要，覆盖到消息 id={engine._covered_until}]")
        if result.extracted_facts:
            preview = "、".join(f.text for f in result.extracted_facts[:3])
            print(f"[本轮记住：{preview}]")
        print()

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
