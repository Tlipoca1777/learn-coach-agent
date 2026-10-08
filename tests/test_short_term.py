r"""
ShortTermMemory 的验收测试
================================================================

这些测试现在**全部是失败的**。让它们变绿，就是你的作业。

--------------------------------------------------------------------
怎么用
--------------------------------------------------------------------
在项目根目录 D:\Agent开发 下执行：

    # 跑全部测试
    .\.venv\Scripts\python.exe -m pytest tests/ -v

    # 只跑某一个测试（比如你正在做的那个）
    .\.venv\Scripts\python.exe -m pytest tests/test_short_term.py::test_add_and_len -v

    # 失败时显示详细的变量值，调试用
    .\.venv\Scripts\python.exe -m pytest tests/ -v --tb=long

--------------------------------------------------------------------
测试结果怎么看
--------------------------------------------------------------------
    PASSED  ✅ 通过
    FAILED  ❌ 失败，下面会告诉你"期望什么、实际得到什么"
    ERROR   💥 连测试都跑不起来（通常是 import 出错或构造函数报错）

--------------------------------------------------------------------
为什么先写测试、再写实现？（测试驱动开发 TDD）
--------------------------------------------------------------------
因为在写测试的时候，你被迫先想清楚"这个函数到底应该做什么"。
很多新手一上来就写实现，写到一半才发现边界情况没想明白，只能推倒重来。
先定验收标准，再写代码，返工率会低很多，这也是专业团队的标准做法。

--------------------------------------------------------------------
关于 fake_counter
--------------------------------------------------------------------
注意下面用的 token 计数器是**假的**：
    1 个字符 = 1 个 token，不带任何额外开销。

为什么不用真实的 estimate_tokens？
因为真实估算是启发式的，将来我们优化了算法，它的输出会变，
所有依赖具体数字的测试就会莫名其妙地失败。
用确定性的假计数器，测试只验证**裁剪逻辑**对不对，与估算算法解耦。
这个技巧叫"测试替身"，是写出稳定测试的关键。
"""

import pytest

from memory_assistant.memory import ShortTermMemory


def fake_counter(messages: list[dict]) -> int:
    """
    确定性的假 token 计数器：1 个字符算 1 个 token。

    这样测试里的数字都是我们手算得出来的，非常直观。
    """
    return sum(len(m.get("content", "")) for m in messages)


def make_memory(max_tokens: int = 100, system_prompt: str | None = None, **kwargs):
    """创建一个使用假计数器的记忆对象，省得每个测试都写一遍。"""
    return ShortTermMemory(
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        token_counter=fake_counter,
        **kwargs,
    )


# ====================================================================
# 组 1：构造与参数校验（规格 S1）
# ====================================================================


def test_rejects_non_positive_max_tokens():
    """max_tokens 必须是正数。"""
    with pytest.raises(ValueError):
        make_memory(max_tokens=0)

    with pytest.raises(ValueError):
        make_memory(max_tokens=-10)


def test_rejects_non_positive_max_messages():
    """max_messages 给了的话，也必须是正数。"""
    with pytest.raises(ValueError):
        make_memory(max_messages=0)


# ====================================================================
# 组 2：添加消息与长度（规格 S2、S7）
# ====================================================================


def test_add_and_len():
    """添加后，长度要跟着变。"""
    memory = make_memory()
    assert len(memory) == 0

    memory.add("user", "你好")
    assert len(memory) == 1

    memory.add("assistant", "你好！")
    assert len(memory) == 2


def test_rejects_invalid_role():
    """role 只能是 system / user / assistant。"""
    memory = make_memory()

    with pytest.raises(ValueError):
        memory.add("tool", "这是一个非法的角色")

    with pytest.raises(ValueError):
        memory.add("User", "大小写敏感，这也是非法的")

    # 出错之后不应该被污染
    assert len(memory) == 0


def test_system_role_replaces_system_prompt():
    """add('system', ...) 应该替换 system_prompt，而不是进入消息列表。"""
    memory = make_memory(system_prompt="原始提示")
    memory.add("system", "新的提示")

    # system 不算作消息
    assert len(memory) == 0
    assert memory.system_prompt == "新的提示"

    window = memory.get_window()
    assert window == [{"role": "system", "content": "新的提示"}]


# ====================================================================
# 组 3：get_window 的基本行为（规格 S3、S6）
# ====================================================================


def test_empty_memory_with_system():
    """空记忆 + 有 system prompt 时，只返回 system。"""
    memory = make_memory(system_prompt="你是一个助手")
    assert memory.get_window() == [{"role": "system", "content": "你是一个助手"}]


def test_empty_memory_without_system():
    """空记忆 + 没有 system prompt 时，返回空列表。"""
    memory = make_memory()
    assert memory.get_window() == []


def test_window_excludes_system_when_not_set():
    """没有 system prompt 时，结果里不应该出现 system 角色的消息。"""
    memory = make_memory()
    memory.add("user", "你好")
    memory.add("assistant", "你好")

    window = memory.get_window()
    assert all(m["role"] != "system" for m in window)
    assert len(window) == 2


def test_window_keeps_chronological_order():
    """顺序必须是"从旧到新"，这一点对模型效果影响很大。"""
    memory = make_memory(system_prompt="SYS")
    memory.add("user", "第一句")
    memory.add("assistant", "第二句")
    memory.add("user", "第三句")

    window = memory.get_window()
    contents = [m["content"] for m in window]
    assert contents == ["SYS", "第一句", "第二句", "第三句"]


# ====================================================================
# 组 4：token 预算裁剪 —— 本作业的核心（规格 S3、S4）
# ====================================================================


def test_trims_oldest_messages_when_over_budget():
    """
    超预算时，从最老的开始丢掉。

    假计数器下 1 字符 = 1 token，所以：
        "aaa"   = 3
        "bbbb"  = 4
        "ccccc" = 5
        "dd"    = 2
    预算 10：最新的 "dd"(2) + "ccccc"(5) = 7 ✅
             再加 "bbbb" 就是 11 > 10 ❌ 停止
    所以结果应该只有 "ccccc" 和 "dd"。
    """
    memory = make_memory(max_tokens=10)
    memory.add("user", "aaa")
    memory.add("assistant", "bbbb")
    memory.add("user", "ccccc")
    memory.add("assistant", "dd")

    window = memory.get_window()
    contents = [m["content"] for m in window]

    assert contents == ["ccccc", "dd"]


def test_stops_at_first_message_that_does_not_fit():
    """
    ⚠️ 这是最容易做错的一条。

    装不下的时候必须**停止**，不能"跳过它继续往前找更短的"。
    因为对话是连续的，跳过中间一条会让上下文断裂。

    构造：
        m1 = "aaaa"       (4)
        m2 = "b" * 100    (100)  ← 装不下
        m3 = "cc"         (2)    ← 最新
    预算 10。
    正确：从 m3 开始，"cc"(2) 装得下；再算 m2 就是 102 > 10，停止。
          结果是 ["cc"]。
    错误：跳过 m2 继续看 m1，发现 2+4=6 ≤ 10，于是把 m1 也加进来。
          结果是 ["aaaa", "cc"]，上下文断裂。
    """
    memory = make_memory(max_tokens=10)
    memory.add("user", "aaaa")
    memory.add("assistant", "b" * 100)
    memory.add("user", "cc")

    window = memory.get_window()
    contents = [m["content"] for m in window]

    assert contents == ["cc"], (
        f"期望只保留最新的 'cc'，实际得到 {contents}。"
        "提示：装不下时要 break，不要 continue。"
    )


def test_always_keeps_newest_message_even_if_over_budget():
    """
    最新的一条自己就超预算时，也必须保留（规格 S4）。

    否则用户刚问的问题被丢掉，模型就没东西可答，功能直接坏掉。
    代价是 token_count 可能超过 max_tokens —— 这是有意为之，
    所以上层预算必须留安全余量。
    """
    memory = make_memory(max_tokens=5)
    memory.add("user", "这是一条非常长的消息" * 20)

    window = memory.get_window()

    assert len(window) == 1
    assert window[0]["content"].startswith("这是一条非常长的消息")
    # 明确记录这个"故意超预算"的行为
    assert memory.token_count > memory.max_tokens


def test_system_prompt_is_not_counted_against_budget():
    """
    max_tokens 只约束对话消息，不约束 system prompt（规格 S3 的注意事项 c）。

    system 很长 + 消息刚好用满预算时，消息不应该被裁掉。
    """
    long_system = "系" * 500
    memory = make_memory(max_tokens=10, system_prompt=long_system)
    memory.add("user", "12345")
    memory.add("assistant", "67890")

    window = memory.get_window()
    contents = [m["content"] for m in window]

    # 两条消息共 10 token，刚好等于预算，应该都保留
    assert contents == [long_system, "12345", "67890"]


# ====================================================================
# 组 5：条数上限（规格 S2、S7）
# ====================================================================


def test_max_messages_cap_drops_oldest():
    """token 预算很宽松时，条数上限也要生效。"""
    memory = make_memory(max_tokens=1000, max_messages=2)
    memory.add("user", "一")
    memory.add("assistant", "二")
    memory.add("user", "三")

    assert len(memory) == 2
    contents = [m["content"] for m in memory.get_window()]
    assert contents == ["二", "三"]


# ====================================================================
# 组 6：防御性拷贝（规格 S6）
# ====================================================================


def test_get_window_returns_a_copy():
    """
    外面修改返回的列表，不能影响内部状态。

    如果实现里写的是 return self._messages，这个测试就会挂。
    """
    memory = make_memory(system_prompt="SYS")
    memory.add("user", "原始消息")

    window = memory.get_window()
    window.append({"role": "user", "content": "外部偷偷加的消息"})
    window[0]["content"] = "外部改掉的系统提示"

    # 再取一次，应该完全不受影响
    fresh = memory.get_window()
    contents = [m["content"] for m in fresh]
    assert contents == ["SYS", "原始消息"]
    assert memory.system_prompt == "SYS"


# ====================================================================
# 组 7：clear（规格 S5）
# ====================================================================


def test_clear_removes_messages_but_keeps_system_prompt():
    """清空消息，但 system prompt 要保留。"""
    memory = make_memory(system_prompt="我是助手")
    memory.add("user", "你好")
    memory.add("assistant", "你好！")

    memory.clear()

    assert len(memory) == 0
    assert memory.get_window() == [{"role": "system", "content": "我是助手"}]
    assert memory.system_prompt == "我是助手"


# ====================================================================
# 组 8：token_count 与 repr（规格 S7）
# ====================================================================


def test_token_count_matches_window():
    """token_count 应该等于 get_window() 的 token 数。"""
    memory = make_memory(max_tokens=100, system_prompt="SYS")
    memory.add("user", "12345")

    expected = fake_counter(memory.get_window())
    assert memory.token_count == expected


def test_repr_is_informative():
    """repr 要能一眼看出状态，方便调试。"""
    memory = make_memory(max_tokens=1000)
    memory.add("user", "你好")

    text = repr(memory)

    assert "ShortTermMemory" in text
    assert "1" in text       # 条数
    assert "1000" in text    # 预算


# ====================================================================
# 组 9：集成测试 —— 换成真实的 token 估算函数
# ====================================================================


def test_works_with_real_token_estimator():
    """
    不注入假计数器，直接用默认的 estimate_messages_tokens。

    这个测试不检查精确数字（因为估算是启发式的），
    只检查"不管怎样都不会失控"这个性质。
    """
    from memory_assistant.llm import estimate_messages_tokens

    memory = ShortTermMemory(max_tokens=200, system_prompt="你是一个助手")

    # 塞进去 50 条消息，远超预算
    for i in range(50):
        memory.add("user", f"这是第 {i} 条消息，用来测试裁剪是否生效。")

    window = memory.get_window()

    # 1. 数量必须被裁下来
    assert len(window) < 50
    # 2. 最新的那条必须在（规格 S4 的保证）
    assert window[-1]["content"] == "这是第 49 条消息，用来测试裁剪是否生效。"
    # 3. 顺序必须是从旧到新
    assert window[0]["role"] == "system"
    # 4. 真实 token 数不应该离谱地超出预算
    #    允许超一点是因为最后一条消息可能本身就很大
    assert estimate_messages_tokens(window) < 200 * 1.5
