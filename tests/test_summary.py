r"""
SummaryMemory（中期记忆）的验收测试
================================================================

这些测试现在**全部是失败的**。让它们变绿，就是你的第 4 周作业。

运行：
    # 全部
    .\.venv\Scripts\python.exe -m pytest tests/test_summary.py -v

    # 只跑某一组（比如你正在做的）
    .\.venv\Scripts\python.exe -m pytest tests/test_summary.py -k "add_evicted" -v

--------------------------------------------------------------------
这个测试文件教你的三件事
--------------------------------------------------------------------
【1】怎么测"和模型的交互"而不花一分钱、不连一次网
    答案是用 FakeLLM。它不是"假装通过测试"，
    而是让你能**精确控制模型的返回值**，从而测出各种边界：
    返回空字符串、返回超长文本、直接抛异常……
    这些情况用真模型很难稳定复现，但它们恰恰是最容易出 bug 的地方。

【2】怎么测"失败路径"
    注意 `BrokenLLM` 和 `FlakyLLM` 这两个小类。
    "模型挂了会怎样"比"模型正常会怎样"更值得测 ——
    因为正常路径你手动跑一次就发现了，失败路径往往等线上出事才暴露。

【3】为什么 token_counter 要注入假的
    和短期记忆一样：用 `lambda s: len(s)`（1 字符 = 1 token），
    所有数字都能手算，测试不依赖估算算法的细节。
    将来我们优化了 estimate_tokens，这些测试也不会莫名其妙地挂。
"""

import pytest

from memory_assistant.llm import FakeLLM
from memory_assistant.memory.summary import SummaryMemory


# ====================================================================
# 测试辅助
# ====================================================================


def count_chars(text: str) -> int:
    """确定性的假 token 计数器：1 个字符 = 1 个 token。"""
    return len(text)


def make_memory(llm, *, max_summary_tokens: int = 50, trigger_tokens: int = 10):
    """创建使用假计数器的 SummaryMemory，省得每个测试都写一遍。"""
    return SummaryMemory(
        llm,
        max_summary_tokens=max_summary_tokens,
        trigger_tokens=trigger_tokens,
        token_counter=count_chars,
    )


def prompt_text(fake_llm: FakeLLM, index: int = -1) -> str:
    """
    把 FakeLLM 收到的那次 prompt 拼成一个字符串，方便检查内容。

    FakeLLM.calls 里存的是每次传入的 messages 列表（已经复制过，不会被污染）。
    """
    messages = fake_llm.calls[index]
    return "\n".join(str(m.get("content", "")) for m in messages)


class BrokenLLM:
    """一个总是失败的假模型，用来测试错误处理路径。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("模拟网络故障")
        self.call_count = 0

    def chat(self, messages, **kwargs):
        self.call_count += 1
        raise self.error


class FlakyLLM:
    """第一次调用失败，之后成功。用来测试"失败了能不能重试"。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def chat(self, messages, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("第一次调用失败（模拟网络抖动）")
        return self._responses.pop(0)


@pytest.fixture
def fake_llm():
    """默认的假模型，默认回答一句像摘要的话。"""
    return FakeLLM(default_response="用户叫小明，在杭州做后端开发。")


@pytest.fixture
def memory(fake_llm):
    return make_memory(fake_llm)


def evicted(*contents, start_id=1) -> list[dict]:
    """构造一批"被挤出的消息"，方便测试。"""
    messages = []
    for offset, text in enumerate(contents):
        role = "user" if offset % 2 == 0 else "assistant"
        messages.append({"id": start_id + offset, "role": role, "content": text})
    return messages


# ====================================================================
# 组 1：构造与参数校验（规格 S1）
# ====================================================================


def test_rejects_non_positive_limits(fake_llm):
    with pytest.raises(ValueError):
        make_memory(fake_llm, max_summary_tokens=0)
    with pytest.raises(ValueError):
        make_memory(fake_llm, trigger_tokens=0)
    with pytest.raises(ValueError):
        make_memory(fake_llm, trigger_tokens=-5)


def test_initial_state_is_empty(memory):
    assert memory.get_summary() is None
    assert memory.pending_count == 0
    assert memory.pending_tokens == 0
    assert memory.covered_until == 0
    assert memory.last_error is None
    assert memory.should_compress() is False


# ====================================================================
# 组 2：add_evicted（规格 S2）
# ====================================================================


def test_add_evicted_returns_accepted_count(memory):
    accepted = memory.add_evicted(evicted("你好", "你好呀"))
    assert accepted == 2
    assert memory.pending_count == 2


def test_add_evicted_ignores_system_messages(memory):
    """系统提示词是永久的，不该被摘要掉。"""
    messages = [
        {"id": 1, "role": "system", "content": "你是一个助手"},
        {"id": 2, "role": "user", "content": "你好"},
    ]
    accepted = memory.add_evicted(messages)

    assert accepted == 1
    assert memory.pending_count == 1


def test_add_evicted_ignores_empty_content(memory):
    messages = [
        {"id": 1, "role": "user", "content": "有内容"},
        {"id": 2, "role": "assistant", "content": ""},
        {"id": 3, "role": "user", "content": "   "},
    ]
    accepted = memory.add_evicted(messages)

    assert accepted == 1
    assert memory.pending_count == 1


def test_add_evicted_tracks_covered_until(memory):
    """covered_until 要跟着消息 id 走，它是"摘要覆盖到哪"的标记。"""
    memory.add_evicted(evicted("一", "二", start_id=5))     # id 5、6
    assert memory.covered_until == 6

    memory.add_evicted(evicted("三", start_id=10))          # id 10
    assert memory.covered_until == 10


def test_covered_until_never_goes_backwards(memory):
    """即使消息乱序送来，covered_until 也只能前进不能后退。"""
    memory.add_evicted(evicted("一", start_id=100))
    memory.add_evicted(evicted("二", start_id=3))
    assert memory.covered_until == 100


def test_messages_without_id_do_not_break_covered_until(memory):
    """没有 id 字段的消息也要能接收（比如手工构造的数据）。"""
    accepted = memory.add_evicted(
        [{"role": "user", "content": "没有 id 的消息"}]
    )
    assert accepted == 1
    assert memory.covered_until == 0


def test_pending_tokens_accumulates(memory):
    memory.add_evicted(evicted("12345", "67890"))
    assert memory.pending_tokens == 10


def test_add_evicted_returns_zero_for_empty_input(memory):
    assert memory.add_evicted([]) == 0
    assert memory.pending_count == 0


# ====================================================================
# 组 3：should_compress 阈值（规格 S3）
# ====================================================================


def test_should_not_compress_below_threshold(fake_llm):
    """⭐ 这是本模块最重要的设计点：攒够了才压，不是每轮都调模型。"""
    memory = make_memory(fake_llm, trigger_tokens=100)

    memory.add_evicted(evicted("短消息"))
    assert memory.should_compress() is False

    memory.add_evicted(evicted("又一条短消息"))
    assert memory.should_compress() is False


def test_should_compress_when_threshold_reached(fake_llm):
    memory = make_memory(fake_llm, trigger_tokens=10)

    memory.add_evicted(evicted("123456789"))   # 9 个 token，还不够
    assert memory.should_compress() is False

    memory.add_evicted(evicted("0"))            # 累计 10 个，刚好达到
    assert memory.should_compress() is True


# ====================================================================
# 组 4：compress 的核心行为（规格 S4）
# ====================================================================


def test_compress_calls_llm_and_sets_summary(memory, fake_llm):
    memory.add_evicted(evicted("我叫小明"))

    result = memory.compress()

    assert result == "用户叫小明，在杭州做后端开发。"
    assert memory.get_summary() == result
    assert fake_llm.call_count == 1


def test_compress_clears_pending_but_keeps_covered_until(memory):
    memory.add_evicted(evicted("我一", "你二", start_id=7))
    assert memory.pending_count == 2

    memory.compress()

    assert memory.pending_count == 0
    assert memory.pending_tokens == 0
    # 队列清了，但"已经覆盖到哪"这个标记必须留着
    assert memory.covered_until == 8


def test_compress_returns_none_when_nothing_pending(memory, fake_llm):
    assert memory.compress() is None
    # 队列为空时，一次模型都不该调 —— 这是在省钱
    assert fake_llm.call_count == 0


def test_compress_twice_without_new_messages_is_noop(memory, fake_llm):
    memory.add_evicted(evicted("我叫小明"))
    memory.compress()
    memory.compress()

    assert fake_llm.call_count == 1


# ====================================================================
# 组 5：递归增量（规格 S5）—— 这是"递归"的关键
# ====================================================================


def test_first_compression_prompt_has_no_previous_summary():
    """首次压缩时，prompt 里不该出现"已有摘要"的段落。"""
    llm = FakeLLM(["第一版摘要"])
    memory = make_memory(llm)

    memory.add_evicted(evicted("我叫小明"))
    memory.compress()

    text = prompt_text(llm)
    assert "已有摘要" not in text
    assert "增量更新" not in text


def test_second_compression_includes_previous_summary():
    """
    ⭐ 递归的核心：第二次压缩必须把第一次的摘要带上。

    如果这条测试挂了，说明你每次都在"从原始对话重新压缩"，
    那就退化成了方案 A，早期信息会被反复稀释。
    """
    llm = FakeLLM(["第一版摘要", "第二版摘要"])
    memory = make_memory(llm)

    memory.add_evicted(evicted("我叫小明"))
    memory.compress()

    memory.add_evicted(evicted("我在杭州", start_id=10))
    result = memory.compress()

    assert result == "第二版摘要"

    text = prompt_text(llm)
    assert "第一版摘要" in text, "prompt 里必须包含旧摘要，否则不是递归压缩"
    assert "增量更新" in text, "必须明确告诉模型这是增量更新"


# ====================================================================
# 组 6：prompt 内容（规格 S5）
# ====================================================================


def test_prompt_contains_retention_instructions(memory, fake_llm):
    """prompt 必须明确要求"保留什么、丢弃什么"。"""
    memory.add_evicted(evicted("我叫小明"))
    memory.compress()

    text = prompt_text(fake_llm)

    # 必须保留的东西
    for keyword in ["偏好", "决定"]:
        assert keyword in text, f"prompt 里缺少「必须保留」清单中的：{keyword}"

    # 可以丢弃的东西
    assert "寒暄" in text, "prompt 里应该说明可以丢弃寒暄客套"


def test_prompt_mentions_length_limit(memory, fake_llm):
    """prompt 里要写清楚摘要的长度上限。"""
    memory.add_evicted(evicted("我叫小明"))
    memory.compress()

    text = prompt_text(fake_llm)
    assert str(memory.max_summary_tokens) in text


def test_prompt_contains_pending_conversation(memory, fake_llm):
    """待压缩的对话原文必须出现在 prompt 里。"""
    memory.add_evicted(evicted("我养了一只橘猫叫豆豆", "记住了"))
    memory.compress()

    text = prompt_text(fake_llm)
    assert "我养了一只橘猫叫豆豆" in text
    assert "记住了" in text


def test_prompt_uses_message_list_format(memory, fake_llm):
    """prompt 应该是一个 messages 列表，符合 API 格式。"""
    memory.add_evicted(evicted("我叫小明"))
    memory.compress()

    prompt = fake_llm.calls[-1]
    assert isinstance(prompt, list)
    assert all("role" in m and "content" in m for m in prompt)
    assert all(m["role"] in ("system", "user", "assistant") for m in prompt)


# ====================================================================
# 组 7：健壮性（规格 S6）—— 这一组最重要
# ====================================================================


def test_empty_response_does_not_overwrite_existing_summary():
    """
    ⭐ 最危险的情况：模型抽风返回空字符串。

    如果这时你把旧摘要清掉了，等于把用户的历史记忆全删了。
    正确行为：视为失败，旧摘要原样保留。
    """
    llm = FakeLLM(["有效的老摘要", "   "])
    memory = make_memory(llm)

    memory.add_evicted(evicted("第一批"))
    memory.compress()
    assert memory.get_summary() == "有效的老摘要"

    memory.add_evicted(evicted("第二批", start_id=10))
    result = memory.compress()

    assert result is None
    assert memory.get_summary() == "有效的老摘要", "旧摘要绝对不能被空字符串覆盖"
    assert memory.last_error is not None


def test_empty_response_keeps_pending_for_retry():
    """失败时待压缩队列要保留，否则这批消息就永久丢了。"""
    llm = FakeLLM(["有效摘要", ""])
    memory = make_memory(llm)

    memory.add_evicted(evicted("第一批"))
    memory.compress()

    memory.add_evicted(evicted("第二批", start_id=10))
    memory.compress()

    assert memory.pending_count == 1, "失败后队列必须保留，等下次重试"


def test_llm_failure_keeps_state_and_records_error():
    """模型抛异常时：不丢摘要、不丢队列、记录错误、返回 None。"""
    llm = BrokenLLM(RuntimeError("模拟网络故障"))
    memory = make_memory(llm)

    memory.add_evicted(evicted("我叫小明"))
    result = memory.compress()

    assert result is None
    assert memory.get_summary() is None          # 本来就没有，现在也不该有
    assert memory.pending_count == 1              # 队列保留
    assert memory.covered_until == 1              # 覆盖标记保留
    assert isinstance(memory.last_error, RuntimeError)
    assert "模拟网络故障" in str(memory.last_error)


def test_failure_does_not_lose_existing_summary():
    """已经有摘要的情况下，模型挂了也不能把摘要弄丢。"""
    llm = FakeLLM(["老摘要"])
    memory = make_memory(llm)

    memory.add_evicted(evicted("第一批"))
    memory.compress()
    assert memory.get_summary() == "老摘要"

    # 换成一个总是失败的模型
    memory.llm = BrokenLLM()
    memory.add_evicted(evicted("第二批", start_id=10))
    memory.compress()

    assert memory.get_summary() == "老摘要"
    assert memory.pending_count == 1


def test_last_error_is_reset_after_success():
    """成功后 last_error 要清零，它的语义是"最近一次尝试的结果"。"""
    llm = BrokenLLM()
    memory = make_memory(llm)

    memory.add_evicted(evicted("我叫小明"))
    memory.compress()
    assert memory.last_error is not None

    # 换成正常的模型再压一次
    memory.llm = FakeLLM(["成功了"])
    memory.compress()

    assert memory.last_error is None


def test_retry_after_transient_failure_succeeds():
    """一次失败之后重试应该能成功 —— 模拟网络抖动。"""
    llm = FlakyLLM(["重试后的摘要"])
    memory = make_memory(llm)

    memory.add_evicted(evicted("我养了只猫"))

    assert memory.compress() is None            # 第一次失败
    assert memory.pending_count == 1

    result = memory.compress()                  # 第二次成功
    assert result == "重试后的摘要"
    assert memory.get_summary() == "重试后的摘要"
    assert memory.pending_count == 0


def test_overlong_summary_is_truncated():
    """
    模型不听话返回了超长摘要时，要截断到上限。

    这是安全网：正常情况不该触发，因为 prompt 里已经要求了长度。
    """
    llm = FakeLLM(["很长" * 500])
    memory = make_memory(llm, max_summary_tokens=20)

    memory.add_evicted(evicted("我叫小明"))
    result = memory.compress()

    assert result is not None
    assert len(result) <= memory.max_summary_tokens + 1   # +1 是省略号
    assert result.endswith("…")
    assert memory.get_summary() == result


def test_normal_length_summary_is_not_truncated(memory):
    """正常长度的摘要不该被动过。"""
    memory.add_evicted(evicted("我叫小明"))
    result = memory.compress()

    assert not result.endswith("…")
    assert result == "用户叫小明，在杭州做后端开发。"


# ====================================================================
# 组 8：maybe_compress（规格 S7）
# ====================================================================


def test_maybe_compress_skips_when_below_threshold(fake_llm):
    """⭐ 未达阈值时必须直接跳过，一次模型都不许调。"""
    memory = make_memory(fake_llm, trigger_tokens=1000)
    memory.add_evicted(evicted("很短的一句话"))

    assert memory.maybe_compress() is None
    assert fake_llm.call_count == 0, "没到阈值就调模型 = 烧钱"
    assert memory.pending_count == 1


def test_maybe_compress_runs_when_threshold_reached(fake_llm):
    memory = make_memory(fake_llm, trigger_tokens=5)
    memory.add_evicted(evicted("这句话超过了五个字"))

    assert memory.maybe_compress() == "用户叫小明，在杭州做后端开发。"
    assert fake_llm.call_count == 1
    assert memory.pending_count == 0


def test_maybe_compress_is_noop_without_pending(memory, fake_llm):
    assert memory.maybe_compress() is None
    assert fake_llm.call_count == 0


# ====================================================================
# 组 9：其它（规格 S8）
# ====================================================================


def test_clear_resets_everything(memory):
    """clear 的语义是"彻底忘掉"，所以摘要也要清掉。"""
    memory.add_evicted(evicted("我叫小明"))
    memory.compress()
    assert memory.get_summary() is not None

    memory.clear()

    assert memory.get_summary() is None
    assert memory.pending_count == 0
    assert memory.pending_tokens == 0
    assert memory.covered_until == 0
    assert memory.last_error is None


def test_clear_keeps_memory_usable():
    """
    规格只要求 clear 清掉"数据"（摘要、队列、覆盖标记、错误），
    **不要求**重置 max_summary_tokens / trigger_tokens 这类配置。
    两种做法都可以，这里只要求 clear 之后还能继续正常使用。
    """
    # 注意给了两条预设回答：第一条会被上面的 compress() 消费掉
    llm = FakeLLM(["第一批的摘要", "清空后的新摘要"])
    memory = make_memory(llm, trigger_tokens=5)

    memory.add_evicted(evicted("第一批"))
    memory.compress()
    memory.clear()

    memory.add_evicted(evicted("清空后的内容", start_id=10))
    assert memory.maybe_compress() == "清空后的新摘要"


def test_repr_is_informative(memory):
    memory.add_evicted(evicted("我叫小明"))
    text = repr(memory)

    assert "SummaryMemory" in text
    assert "1" in text                     # 待压缩条数
    assert str(memory.trigger_tokens) in text


# ====================================================================
# 组 9b：恢复状态（规格 S9）
# ====================================================================


def test_restore_sets_summary_and_covered_until():
    memory = make_memory(FakeLLM())

    memory.restore("用户叫小明，在杭州做后端开发。", covered_until=50)

    assert memory.get_summary() == "用户叫小明，在杭州做后端开发。"
    assert memory.covered_until == 50


def test_restore_with_none_summary_only_sets_position():
    """只恢复覆盖位置、没有摘要，是合法的（比如从没压缩过就重启了）。"""
    memory = make_memory(FakeLLM())
    memory.add_evicted(evicted("一批还没压缩的消息"))

    memory.restore(None, covered_until=7)

    assert memory.get_summary() is None
    assert memory.covered_until == 7
    assert memory.pending_count == 1, "restore 不该动待压缩队列"


def test_restore_treats_blank_summary_as_none():
    memory = make_memory(FakeLLM())

    memory.restore("   ", covered_until=3)

    assert memory.get_summary() is None
    assert memory.covered_until == 3


def test_restore_does_not_touch_last_error():
    memory = make_memory(BrokenLLM())
    memory.add_evicted(evicted("一些消息"))
    memory.compress()
    assert memory.last_error is not None

    memory.restore("恢复出来的摘要", covered_until=5)

    assert memory.last_error is not None, "restore 只负责恢复摘要状态"


def test_restore_then_compression_keeps_old_summary():
    """
    ⭐⭐ 这条测试是 restore() 存在的全部理由。请务必理解它。

    场景：程序重启。
        第 1 天：聊了 50 轮，摘要覆盖到 id=50，存进了数据库。
        第 2 天：程序重新启动。

    如果重启后不把旧摘要恢复回来，那么下一次压缩会变成
        "新摘要 = LLM(空 + 新消息)"
    结果就是**前 50 轮的记忆被彻底丢掉** —— 而且不报任何错，
    用户只会发现"助手怎么突然失忆了"。

    正确做法：启动时先 restore 旧摘要，再继续追加新消息。
    这样新摘要 = LLM(旧摘要 + 新消息)，历史得以保留。
    """
    llm = FakeLLM(["合并后的新摘要"])
    memory = make_memory(llm, trigger_tokens=5)

    # ---- 模拟"第 2 天启动"：先恢复 ----
    memory.restore("第一天：用户叫小明，在杭州做后端开发。", covered_until=50)
    assert memory.covered_until == 50

    # ---- 继续聊，触发压缩 ----
    memory.add_evicted(evicted("第二天聊的新内容", start_id=51))
    result = memory.maybe_compress()

    assert result == "合并后的新摘要"

    sent = prompt_text(llm)
    assert "第一天：用户叫小明，在杭州做后端开发。" in sent, (
        "prompt 里没有旧摘要 —— 说明压缩是「从零开始」的，"
        "前 50 轮的记忆会全部丢失"
    )
    assert "第二天聊的新内容" in sent


# ====================================================================
# 组 10：集成 —— 和短期记忆的 on_evict 接起来
# ====================================================================


def test_integrates_with_short_term_eviction():
    """
    中期记忆是被短期记忆"喂"出来的：窗口挤掉的消息交给它。

    这个测试手工模拟那个流程（第 4 周你会用 on_evict 回调真正接起来）：
        短期记忆装不下 → 消息被挤出 → add_evicted → 攒够 → 压缩

    字符数核对（假计数器下 1 字符 = 1 token）：
        第一批："我叫小明"(4) + "记住了"(3) = 7，阈值 15 → 还不够
        第二批："我在杭州做后端开发"(9) + "好的"(2) = 11，累计 18 ≥ 15 → 触发
    """
    llm = FakeLLM(["用户叫小明，在杭州做后端开发，偏好手冲咖啡。", "更新后的摘要"])
    memory = make_memory(llm, trigger_tokens=15)

    # 第一轮：挤出几条消息，还没到阈值
    memory.add_evicted(evicted("我叫小明", "记住了", start_id=1))
    assert memory.should_compress() is False
    assert memory.maybe_compress() is None

    # 第二轮：又挤出几条，累计够了
    memory.add_evicted(evicted("我在杭州做后端开发", "好的", start_id=4))
    summary = memory.maybe_compress()

    assert summary == "用户叫小明，在杭州做后端开发，偏好手冲咖啡。"
    assert memory.covered_until == 5
    assert memory.pending_count == 0
