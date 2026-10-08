r"""
对比实验：全量历史 vs 朴素条数截断 vs token 预算裁剪
================================================================

这是项目里第一个"用数据说话"的实验。

它回答一个面试必问的问题：
    「你为什么不直接把最近 10 轮对话发过去，非要自己写记忆管理？」

跑完你就有答案了，而且**有数据**。

--------------------------------------------------------------------
运行前提
--------------------------------------------------------------------
必须先实现 ShortTermMemory（见 src/memory_assistant/memory/short_term.py）。
没实现的话脚本会友好提示，不会报一堆看不懂的错。

--------------------------------------------------------------------
运行命令
--------------------------------------------------------------------
    # 默认场景
    .\.venv\Scripts\python.exe scripts\step2_budget_experiment.py

    # 自己调参数玩
    .\.venv\Scripts\python.exe scripts\step2_budget_experiment.py --turns 120 --budget 6000
    .\.venv\Scripts\python.exe scripts\step2_budget_experiment.py --no-long-docs

    # 顺便算钱（单价单位：元 / 每百万 token）
    .\.venv\Scripts\python.exe scripts\step2_budget_experiment.py --price-in 2 --price-out 8

--------------------------------------------------------------------
三种策略是什么
--------------------------------------------------------------------
    ① 全量历史       —— 全部消息发过去。简单，但 token 无限增长。
    ② 朴素条数截断   —— 只保留最近 N 条。大多数教学 Demo 的做法。
    ③ token 预算裁剪 —— 我们的方案。从最新的往回装，装不下就停。

这个实验要证明 ② 的致命问题：
    **「条数」和「token 数」根本不是一回事。**
    用户粘贴一份长文档进来，1 条就可能吃掉几千 token。
    只按条数截断，预算会瞬间失控，然后你的 API 调用直接失败。

--------------------------------------------------------------------
为什么场景要这样设计
--------------------------------------------------------------------
我们刻意构造了一个「长度分布极不均匀」的对话：
    - 大部分是几十 token 的日常对话
    - 中间夹着两份几千 token 的粘贴文档

因为**均匀长度的对话无法暴露问题**：
如果每条消息都一样长，按条数和按 token 裁出来的结果差不多，
你就体会不到为什么要自己写记忆管理。
真实用户的输入长度方差极大，这才是问题的本质。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory_assistant.llm import estimate_messages_tokens, estimate_tokens  # noqa: E402
from memory_assistant.memory import ShortTermMemory  # noqa: E402

SYSTEM_PROMPT = "你是一个有记忆的私人助理，回答简洁但信息完整。"


# ====================================================================
# 一、构造一段"像真的"对话
# ====================================================================
# 为什么要造数据，而不是让你手动聊 80 轮？
#   因为实验必须**可复现**。今天跑和下周跑要得到同样的数字，
#   否则你没法把数据写进简历 —— "我实测省了 70%" 这种话必须有据可查。

# 一组真实感的问答对。注意 assistant 的回答长度刻意做得不一样，
# 因为真实的对话里，回答本来就是长短不齐的。
CONVERSATION_PAIRS: list[tuple[str, str]] = [
    ("你好，在吗？",
     "在的！我可以帮你记事、算数、管理待办、查资料。有什么需要直接说就行。"),
    ("我叫小明，在杭州做后端开发。",
     "记住了：你是小明，在杭州做后端开发。以后聊到技术问题我会按后端的视角来回答。"),
    ("我平时喜欢喝手冲咖啡，不加糖。",
     "好的，手冲不加糖。以后给你推荐咖啡馆或者聊到饮品时我会注意这一点。"),
    ("帮我记一下，下周三要交季度报告。",
     "已记录：下周三交季度报告。需要我在下周一提醒你一次吗？"
     "如果要的话告诉我提醒时间，我把它加到待办里。"),
    ("我最近在学 Python 和 LangChain。",
     "不错，这两个方向很实用。Python 是当前 AI 应用开发的主力语言，"
     "LangChain 则提供了现成的抽象。建议你先把它俩的基础打牢，再去看更复杂的框架。"),
    ("我养了一只猫，叫豆豆，是只橘猫。",
     "记住了，橘猫豆豆。橘猫普遍比较能吃，注意控制体重，"
     "成年橘猫超过 6 公斤就要考虑减粮了。"),
    ("我对花生过敏，点菜的时候要特别注意。",
     "已记录：花生过敏。这个信息很重要，我会在涉及饮食的场景主动提醒你。"
     "注意很多酱料和零食里都含花生成分，不只是花生本身。"),
    ("周末我一般会去爬山。",
     "好的，周末爬山。杭州周边可选的路线不少，"
     "如果哪天你想让我帮你规划路线或者看天气，直接说。"),
    ("我女朋友叫小雨，她不吃辣。",
     "记住了：小雨，不吃辣。以后一起点餐或者推荐餐厅时我会避开川湘菜系。"),
    ("我们打算明年去日本旅行。",
     "明年日本旅行，已记下。等你确定月份和城市之后，"
     "我可以帮你整理签证材料清单和大致的行程框架。"),
    ("我现在用的电脑是 MacBook Pro。",
     "好的，MacBook Pro。以后涉及到命令行操作的建议我会按 macOS 来给，"
     "不会给你 Windows 或 Linux 的命令。"),
    ("我更喜欢用 VS Code 写代码，不喜欢 PyCharm。",
     "记住了，偏好 VS Code。这个偏好我会一直保留，"
     "以后推荐插件或者配置时都按 VS Code 来。"),
    ("每天早上 7 点起床，晚上 11 点睡。",
     "作息挺规律的。已记录，以后安排提醒或者日程的时候我会避开你的睡眠时段。"),
    ("我在读《设计数据密集型应用》这本书。",
     "好书，是分布式系统领域的经典。已记录，"
     "等你读完之后如果想聊里面的某个章节，我记得你在读这本。"),
    ("我最近在减脂，晚上不吃主食。",
     "好的，晚上不吃主食。已记录。减脂期注意蛋白质摄入不要跟着一起减，"
     "否则容易掉肌肉。"),
]

# ---------- 模拟「用户粘贴进来的长文档」 ----------
# 这里用程序生成，而不是把两千字硬写死在代码里。
# 好处是：长度可控、可以按参数调节，代码也不会变成一坨文本。
_DOC_SECTIONS = [
    "【项目背景】需要为内部知识库搭建一套检索增强生成系统，"
    "目标是让员工用自然语言查询公司内部的规章制度、技术文档和历史项目资料，"
    "降低新人的信息获取成本。",
    "【功能需求】一、支持上传 PDF、Word、Markdown 三种格式，"
    "单文件不超过 50MB，单次批量不超过 100 个文件。"
    "二、解析后自动切分成不超过 500 字的片段，相邻片段保留 50 字重叠，避免语义被硬切断。"
    "三、为每个片段生成向量并写入向量库，同时保留原文位置信息以支持溯源。"
    "四、问答时先做向量检索取回 top-8 片段，再交给大模型生成回答，回答必须标注引用来源。",
    "【非功能需求】一、单次查询响应时间不超过 3 秒。"
    "二、文档更新后 5 分钟内完成索引重建。"
    "三、系统需支持至少 50 个并发用户。"
    "四、所有数据必须留在内网，不得调用外部 API。"
    "五、需要保留完整的查询审计日志，保存期不少于 180 天。",
    "【技术约束】现有技术栈是 Python 3.11 + FastAPI + PostgreSQL，"
    "希望尽量复用，不要引入过重的新框架。团队目前只有两名后端工程师，"
    "开发周期为两个月，且两人还要兼顾线上问题的排查，实际投入大约只有一半人力。",
    "【风险提示】一、文档格式复杂，PDF 里的表格和双栏排版解析成功率可能不足 80%。"
    "二、内网部署意味着不能用云端向量服务，需要自行评估开源向量库的运维成本。"
    "三、50 并发对单机部署是不小的压力，需要提前做压测。"
    "四、两名工程师两个月的时间可能偏紧，建议砍掉非核心需求。",
    "【待确认事项】一、是否需要支持扫描版 PDF 的 OCR？这会影响解析方案的选型。"
    "二、权限模型是全部文档对所有员工可见，还是需要按部门隔离？"
    "三、向量库的更新是实时还是定时批处理？"
    "四、是否需要支持多轮追问，还是每次查询独立？",
]


def make_long_document(title: str, target_tokens: int, variant: int = 0) -> str:
    """
    生成一份约 target_tokens 长度的长文档，模拟用户粘贴进来的内容。

    参数 variant：
        起始段落的偏移量。让两份文档看起来内容不同
        （否则完全一样的两份文档一眼就假）。

    实现方式：循环拼接文档段落，直到长度达标。
    这是很常用的「造测试数据」手法 —— 用程序生成，而不是把两千字硬编码进代码。
    """
    parts = [title]
    index = 0
    # index < 200 是防止参数写错（比如 target_tokens 给了个极大值）时死循环
    while estimate_tokens("\n".join(parts)) < target_tokens and index < 200:
        section = _DOC_SECTIONS[(index + variant) % len(_DOC_SECTIONS)]
        parts.append(f"\n===== 第 {index + 1} 节 =====\n{section}")
        index += 1
    return "\n".join(parts)


def build_conversation(
    turns: int,
    long_count: int = 2,
    long_tokens: int = 2000,
) -> list[dict]:
    """
    构造一个有 turns 条消息的对话。

    参数：
        turns        消息条数
        long_count   插入几份长文档
        long_tokens  每份长文档大约多少 token

    长文档的插入位置是刻意设计的：
        放在**最后 10 条之内**，这样"保留最近 10 条"的朴素方案
        一定会把它们全部带上，于是预算必然爆掉。

    ⚠️ 一个必须遵守的约束：只替换 user 消息（偶数下标）。
       如果替换到 assistant 的位置，就会出现"连续两条 user 消息"，
       对话数据一眼就假，实验结论也不可信。
       代价是每隔 4 条才有一个可用位置，所以最近 10 条里最多塞 3 份。
    """
    messages: list[dict] = []

    # 用问答对循环填充，直到凑够条数
    index = 0
    while len(messages) < turns:
        user_text, assistant_text = CONVERSATION_PAIRS[index % len(CONVERSATION_PAIRS)]
        messages.append({"role": "user", "content": user_text})
        if len(messages) < turns:
            messages.append({"role": "assistant", "content": assistant_text})
        index += 1
    messages = messages[:turns]

    # ---------- 插入长文档 ----------
    # 只替换 user 消息（偶数下标），因为真实对话一定是 user/assistant 交替的。
    # 如果替换到 assistant 的位置，就会出现「连续两条 user 消息」这种假数据，
    # 实验结论也就不再可信 —— 造测试数据同样要讲真实性。
    if long_count > 0 and turns >= 6:
        # 从倒数第 2 条往前找最近的 user 位置，
        # 这样最后一条仍然是 assistant 的回答，对话读起来是自然的
        base = len(messages) - 2
        if base % 2 != 0:
            base -= 1

        for order in range(long_count):
            # 每次往前退 4 个位置（仍然是偶数），保证都落在"最近 10 条"里面
            position = base - order * 4
            if position < 0:
                break
            document = make_long_document(
                f"帮我看一下这份需求文档（第 {order + 1} 份）：",
                long_tokens,
                variant=order * 2,
            )
            messages[position] = {"role": "user", "content": document}

    return messages


# ====================================================================
# 二、三种策略
# ====================================================================


def strategy_full(messages: list[dict]) -> list[dict]:
    """策略 ①：全量历史，一条不丢。"""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)


def strategy_naive(messages: list[dict], keep: int) -> list[dict]:
    """策略 ②：只保留最近 keep 条。"""
    # 切片 [-keep:] 取最后 keep 条，这是最直观的写法，也是大多数 Demo 的写法
    return [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages[-keep:])


def strategy_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    """策略 ③：我们自己的 token 预算裁剪。"""
    memory = ShortTermMemory(max_tokens=max_tokens, system_prompt=SYSTEM_PROMPT)
    for message in messages:
        memory.add(message["role"], message["content"])
    return memory.get_window()


# ====================================================================
# 三、打印
# ====================================================================


def make_bar(value: int, total: int, width: int = 12) -> str:
    """返回一个进度条，一眼看出有没有超预算。"""
    if total <= 0:
        return "⬜" * width
    filled = min(width, round(width * value / total))
    return "🟩" * filled + "⬜" * (width - filled)


def print_message_detail(messages: list[dict], window: list[dict]) -> None:
    """打印消息明细表：每条消息占多少 token，有没有被保留。"""
    # 我们的算法保证"保留的是最近连续的一段"，
    # 所以被保留的条数 = 窗口总条数 - system 条数
    system_count = 1 if window and window[0]["role"] == "system" else 0
    kept_count = len(window) - system_count

    # 防御性检查：验证确实是一段连续的后缀。
    # 如果不是，说明 ShortTermMemory 的实现和规格不一致，要提醒用户。
    if kept_count > 0:
        expected = messages[-kept_count:]
        actual = window[system_count:]
        if [m["content"] for m in expected] != [m["content"] for m in actual]:
            print("⚠️ 警告：被保留的消息不是「最近连续的一段」，")
            print("   这和 docs/记忆架构设计.md 的规格不符，请检查 get_window() 的实现。")
            print()

    first_kept_index = len(messages) - kept_count

    print(f"{'#':>4} {'角色':<10} {'token':>7}  {'状态':<8} 内容")
    print("-" * 78)

    for index, message in enumerate(messages):
        kept = index >= first_kept_index
        status = "✅ 保留" if kept else "❌ 裁掉"
        role = message["role"]
        tokens = estimate_tokens(message["content"])
        preview = message["content"][:26].replace("\n", " ").replace("=", " ")

        if index == first_kept_index and first_kept_index > 0:
            print("-" * 78 + "  ← 预算边界，这行以下被保留")

        print(f"{index:>4} {role:<10} {tokens:>7}  {status:<8} {preview}...")

    print("-" * 78)
    if first_kept_index == 0:
        print("（本场景没有触发裁剪 —— 说明预算给得比较宽松）")
    else:
        print(f"（裁掉了前 {first_kept_index} 条，保留了后 {kept_count} 条）")


def print_comparison(
    messages: list[dict],
    full: list[dict],
    naive: list[dict],
    budget: list[dict],
    max_tokens: int,
    naive_keep: int,
    price_in: float | None,
    price_out: float | None,
) -> None:
    """打印三种策略的对比表，以及结论。"""
    full_tokens = estimate_messages_tokens(full)
    naive_tokens = estimate_messages_tokens(naive)
    budget_tokens = estimate_messages_tokens(budget)

    # 提前算好，避免后面变量未定义
    saved = (1 - budget_tokens / full_tokens) * 100 if full_tokens else 0.0
    scale = max(full_tokens, naive_tokens, budget_tokens, max_tokens)

    print()
    print("=" * 78)
    print(" 三方对比")
    print("=" * 78)
    print(f"{'策略':<24} {'消息数':>6} {'输入token':>10} {'占预算':>8}  相对规模")
    print("-" * 78)

    rows = [
        ("① 全量历史（不裁剪）", full, full_tokens),
        (f"② 朴素截断（最近{naive_keep}条）", naive, naive_tokens),
        (f"③ token 预算裁剪", budget, budget_tokens),
    ]

    for name, window, tokens in rows:
        ratio = f"{tokens / max_tokens * 100:.0f}%" if max_tokens else "—"
        flag = "  ⚠️ 超预算" if tokens > max_tokens else ""
        print(
            f"{name:<24} {len(window):>6} {tokens:>10} {ratio:>8}  "
            f"{make_bar(tokens, scale)}{flag}"
        )

    print("-" * 78)
    print(f"{'预算上限':<24} {'':>6} {max_tokens:>10}")
    print(f"{'模型上下文窗口（参考）':<24} {'':>6} {64000:>10}")

    # ---------- 结论 ----------
    print()
    print("=" * 78)
    print(" 结论")
    print("=" * 78)
    print()
    utilization = budget_tokens / max_tokens * 100 if max_tokens else 0.0
    longest = max((estimate_tokens(m["content"]) for m in messages), default=0)
    # 判断本次到底有没有触发裁剪
    system_count = 1 if budget and budget[0]["role"] == "system" else 0
    trimmed = (len(budget) - system_count) < len(messages)

    print("· ① 省了多少 token")
    print(f"    相比全量历史，预算裁剪节省 {saved:.1f}%（{full_tokens} → {budget_tokens}）")
    print()

    print("· ② 朴素截断的致命问题")
    if naive_tokens > max_tokens:
        print(f"    超预算 {naive_tokens - max_tokens} token（{naive_tokens} > {max_tokens}）")
        print("    原因：最近 10 条里夹着长文档。它「以为」自己只留了 10 条很克制，")
        print(f"          实际上留了 {naive_tokens} token —— 比整个预算还多。")
    else:
        print(f"    本场景没超预算（{naive_tokens} ≤ {max_tokens}）—— 但这只是运气。")
        print("    它会不会超，完全取决于「预算」和「最近 N 条里有多少长消息」的相对大小，")
        print("    而这两件事它都感知不到。试着调小 --budget 或调大 --long-tokens。")
    print("    —— 『条数』和『token 数』之间没有换算关系。")
    print("       后果很实际：API 调用会因超出上下文窗口直接失败，或者账单远超预期。")
    print("       而这种 bug 在测试环境几乎发现不了，因为你自己测试时不会粘贴长文档。")
    print()

    tight = strategy_naive(messages, keep=3)
    tight_tokens = estimate_messages_tokens(tight)
    print(f"    那改成「最近 3 条」保险一点？本场景是 {tight_tokens} token。")
    print("    如果这 3 条刚好都是长文档，照样爆；如果都是闲聊，你又白白浪费预算、丢了上下文。")
    print("    固定条数永远无法同时解决这两个问题 —— 因为它根本不知道自己的成本是多少。")
    print()

    if not trimmed:
        print("· ③ 本次没有触发裁剪")
        print(f"    对话总量（{full_tokens} token）还没到预算（{max_tokens}），所以窗口全放下了。")
        print("    这本身是个有用的信息：说明这个预算对当前对话长度偏宽松。")
        print("    真实系统里预算应该按「最坏情况」定，而不是按「平均情况」——")
        print("    因为你无法控制用户会不会突然粘贴一份三千字的需求文档。")
    else:
        print("· ③ 我们的方案也不是没代价（这才是诚实的地方）")
        print(f"    预算利用率只有 {utilization:.0f}%（用了 {budget_tokens} / {max_tokens} token）")
        print(f"    原因是场景里最长的一条消息本身就有 {longest} token。")
        print("    规格要求「装不下就停、不许跳过」，所以碰到一条超长消息时，")
        print("    它更老的（可能更重要的）对话就全部被放弃了。")
        print()
        print("    这个代价可以接受，理由在架构分工：")
        print("      · 短期记忆只负责「对话连贯性」—— 最近在聊什么")
        print("      · 「你是谁、住哪、喜欢什么」由 用户画像 + 长期记忆 负责")
        print("    所以短期窗口丢掉早期对话并不可怕，那些信息本来就不该指望它记住。")
        print("    （这正是本项目要做四层记忆的原因，见 docs/记忆架构设计.md）")
        print()
        print("    想提高利用率可以挑战 short_term.py 末尾的「进阶 2」：")
        print("    不整条丢弃，而是把超长消息截断到预算内。代价是可能截到句子中间，引入噪声。")

    # ---------- 成本 ----------
    if price_in is not None and price_out is not None:
        print()
        print("=" * 78)
        print(" 成本估算（按你提供的单价）")
        print("=" * 78)
        output_tokens = 200
        for name, tokens in [
            ("① 全量历史", full_tokens),
            ("② 朴素截断", naive_tokens),
            ("③ token 预算", budget_tokens),
        ]:
            cost = tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out
            print(f"{name:<16} 单次 {cost:.6f} 元    1000 次 {cost * 1000:.3f} 元")
        print()
        print("（输出统一按 200 token 估算；单价请以官网为准，随时可能调整）")
    else:
        print()
        print("💡 想看成本对比，加参数：--price-in 单价 --price-out 单价")
        print("   单价单位是「元 / 每百万 token」，到模型官网查当前价格。")

    # ---------- 可写进简历的一句话 ----------
    print()
    print("=" * 78)
    print(" 可以写进 README / 简历的一句话")
    print("=" * 78)
    print()
    print(
        f"「在 {len(messages)} 条消息、含 {sum(1 for m in messages if estimate_tokens(m['content']) > 500)} "
        f"份长文档的场景下，"
        f"我实现的 token 预算裁剪把单次请求输入从\n"
        f"  {full_tokens} token 降到 {budget_tokens} token（-{saved:.0f}%），"
        f"且始终不超预算；\n"
        f"  而按条数截断的方案在同样预算下超出 {max(0, naive_tokens - max_tokens)} token。」"
    )
    print()


# ====================================================================
# 四、主流程
# ====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对比三种对话历史策略的 token 消耗",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--turns", type=int, default=80, help="对话消息条数（默认 80）")
    parser.add_argument("--budget", type=int, default=4000, help="token 预算（默认 4000）")
    parser.add_argument(
        "--naive-keep", type=int, default=10, help="朴素方案保留最近几条（默认 10）"
    )
    parser.add_argument("--long-count", type=int, default=2, help="插入几份长文档（默认 2）")
    parser.add_argument(
        "--long-tokens", type=int, default=2000, help="每份长文档约多少 token（默认 2000）"
    )
    parser.add_argument(
        "--no-long-docs", action="store_true", help="不插入长文档，观察均匀长度的对比"
    )
    parser.add_argument("--price-in", type=float, default=None, help="输入单价，元/百万 token")
    parser.add_argument("--price-out", type=float, default=None, help="输出单价，元/百万 token")
    args = parser.parse_args()

    long_count = 0 if args.no_long_docs else args.long_count

    print("=" * 78)
    print(" 对比实验：全量历史 vs 朴素条数截断 vs token 预算裁剪")
    print("=" * 78)
    print()
    print(f"场景：{args.turns} 条消息", end="")
    if long_count:
        print(f"，其中包含 {long_count} 份约 {args.long_tokens} token 的长文档")
    else:
        print("（长度分布均匀，无长文档）")
    print(f"预算：{args.budget} token")
    print(f"朴素方案：保留最近 {args.naive_keep} 条")

    # 长文档的放置位置受"角色必须交替"的约束：
    # 只能落在偶数下标（user）上，也就是每隔 4 条才有一个可用位置。
    # 所以在最近 10 条里，最多只能塞 3 份长文档。
    if long_count > 3:
        print()
        print(f"提示：为了让「朴素截断」必然超预算，长文档都尽量放在最后 10 条内。")
        print(f"      但受角色交替限制，最后 10 条里最多只有 3 个可用位置，")
        print(f"      所以第 4 份及以后会落到对比窗口之外，不影响本次结论。")

    messages = build_conversation(
        args.turns, long_count=long_count, long_tokens=args.long_tokens
    )
    print(f"实际生成：{len(messages)} 条消息，共 {estimate_messages_tokens(messages)} token")
    print()

    # ---------- 策略 ③ 依赖用户的代码，单独 try ----------
    try:
        budget_window = strategy_budget(messages, args.budget)
    except NotImplementedError as error:
        print("=" * 78)
        print(" ⚠️ 还没法跑这个实验")
        print("=" * 78)
        print()
        print(f"原因：ShortTermMemory 还没实现（{error}）")
        print()
        print("这个实验要用到你写的 get_window()。请先完成：")
        print("    src/memory_assistant/memory/short_term.py")
        print()
        print("完成后确认测试全绿：")
        print(r"    .\.venv\Scripts\python.exe -m pytest tests/ -v")
        print()
        print("看到 19 passed 之后，再回来跑本脚本。")
        print()
        # 注意：这不是错误，是"还没到时候"，所以退出码是 0
        return 0
    except Exception as error:
        print(f"❌ 运行 ShortTermMemory 时出错：{type(error).__name__}: {error}")
        print()
        print("这通常是实现里有 bug。建议先用测试定位到具体哪一条：")
        print(r"    .\.venv\Scripts\python.exe -m pytest tests/ -v")
        return 1

    # ---------- 明细 ----------
    print("=" * 78)
    print(" 消息明细（策略 ③ 视角）")
    print("=" * 78)
    print()
    print_message_detail(messages, budget_window)

    # ---------- 对比 ----------
    print_comparison(
        messages=messages,
        full=strategy_full(messages),
        naive=strategy_naive(messages, args.naive_keep),
        budget=budget_window,
        max_tokens=args.budget,
        naive_keep=args.naive_keep,
        price_in=args.price_in,
        price_out=args.price_out,
    )

    print("=" * 78)
    print(" 下一步")
    print("=" * 78)
    print()
    print("1. 把上面的数字记下来（README 的「评测数据」章节有一张现成的表）")
    print("2. 换参数再跑几组，观察趋势：")
    print("     --turns 160          对话更长，看节省比例怎么变")
    print("     --budget 2000        预算更紧，看裁剪掉多少")
    print("     --no-long-docs       没有长文档时，朴素方案是不是就没问题了？")
    print("3. 想清楚一个问题：预算设成多少最合适？")
    print("   太大费钱，太小会丢掉关键信息（比如用户的名字）。")
    print("   这个权衡就是第 6 周「token 预算表」要解决的问题。")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
