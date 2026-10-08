r"""
短期记忆 —— 滑动窗口 + token 预算裁剪
================================================================

⭐⭐ 这是你的第一个正式作业。下面的"规格"已经写清楚了，
    代码留给你自己实现。验收标准在 tests/test_short_term.py。

--------------------------------------------------------------------
为什么要做这个？（先想清楚问题，再写代码）
--------------------------------------------------------------------
大模型没有记忆，所以每次请求都要把历史对话重新发过去。
但历史不能无限增长：
    - 模型有上下文窗口上限
    - token 越多越贵
    - 无关的旧内容还会干扰模型判断

最粗糙的方案是"只保留最近 10 条消息"。它的问题是：
    一条 5 字的消息和一条 5000 字的消息，都算"1 条"。
    用户贴了一篇长文章进来，10 条就爆了。

正确做法：**按 token 预算裁剪，从最新的往旧的装，装不下就停。**

--------------------------------------------------------------------
你要实现的接口
--------------------------------------------------------------------

    memory = ShortTermMemory(max_tokens=1000, system_prompt="你是一个助手")

    memory.add("user", "你好")
    memory.add("assistant", "你好！")
    memory.add("user", "我叫小明")

    memory.get_window()
    # -> [
    #      {"role": "system", "content": "你是一个助手"},
    #      {"role": "user", "content": "你好"},
    #      {"role": "assistant", "content": "你好！"},
    #      {"role": "user", "content": "我叫小明"},
    #    ]

--------------------------------------------------------------------
详细规格（请逐条对照实现）
--------------------------------------------------------------------

【S1】构造函数
    __init__(self, max_tokens: int, system_prompt: str | None = None,
             max_messages: int | None = None,
             token_counter: Callable[[list[dict]], int] | None = None)

    - max_tokens：这条记忆能占的最大 token 数（不含 system prompt 的开销）。
      ⚠️ 必须校验 > 0，否则抛 ValueError。
    - system_prompt：系统提示词。None 表示没有。
    - max_messages：消息条数的额外上限，None 表示不限制。
      （为什么要两个限制？见下方"设计说明"）
    - token_counter：token 计数函数，默认用 estimate_messages_tokens。
      **允许外部注入是刻意设计的**：测试时传一个确定性的假计数器，
      测试就永远稳定，不会因为估算算法的细微变化而失败。

【S2】add(self, role: str, content: str) -> None
    - 追加一条消息。
    - role 只能是 "system" / "user" / "assistant"，其它值抛 ValueError。
      （在入口就挡住脏数据，比在出口处理省事得多）
    - role == "system" 时不要塞进消息列表，而是替换 self.system_prompt。
      为什么？因为 system 必须永远在最前面且不被裁剪，混在列表里会很乱。
    - 追加后如果条数超过 max_messages，从最老的开始丢，直到不超。

【S3】get_window(self) -> list[dict]
    这是最重要的方法。返回"这次要发给模型的消息列表"。

    算法：
        1. 结果列表 result = []
        2. 如果 system_prompt 不为 None，先放进去：
               result.append({"role": "system", "content": self.system_prompt})
        3. 从**最新的消息往回**遍历（也就是 reversed）：
               - 把当前消息临时加入候选
               - 用 token_counter 算候选的 token 总量
               - 如果没超 max_tokens → 接受这条，继续往前
               - 如果超了 → **停止**（break），不要再往前看了
        4. 最后把收集到的消息**反转回时间顺序**（从旧到新）
        5. 返回 result

    ⚠️ 三个容易做错的点：
        (a) 必须是"装不下就停"，不能"装不下就跳过继续往前找更短的"。
            因为对话是连续的，跳过中间一条会让上下文断裂，模型会困惑。
        (b) 最终顺序必须是**从旧到新**（时间顺序），
            大模型对消息顺序很敏感，倒序会严重影响效果。
        (c) token 计算时，**不要**把 system prompt 算进 max_tokens 里，
            max_tokens 只约束对话消息。system 是固定开销，单独对待。

【S4】边界情况：最新的一条自己就超预算怎么办？
    规格：**无论如何都要保留最新的一条消息**。
    理由：最新消息通常是用户刚问的问题。把它丢掉，模型就没有问题可答了，
          功能直接坏掉。宁可超预算，也不能让这一轮无法回答。
    后果：token_count 可能大于 max_tokens。这是有意为之，
          所以上层的总预算一定要留安全余量。

【S5】clear(self) -> None
    清空所有消息，但**保留 system_prompt**。

    建议用 `self._messages.clear()`，而不是 `self._messages = []`。

    ⚠️ 说明一下，我原来写的理由是错的，更正：
       在这个类里，**两种写法测试都能过** ——
       因为没有任何外部代码持有这个列表的引用，重新赋值也没人受影响。

       但 `.clear()` 仍然是更好的习惯，理由是**将来**：
       如果以后有别的地方引用了这个列表（比如传给了回调、存进了另一个对象），
       重新赋值会让那个引用仍然指向**旧列表**，
       于是出现"我明明清空了，别人还看得到旧数据"这种极难排查的 bug。
       —— 和 S6 的"防御性拷贝"是同一个思路：**谁拥有数据，谁负责它的生命周期。**

       （这条改动本身也值得记一下：写文档时我以为的原因，和实际验证出来的原因
         不一样。**能跑通的代码 + 想清楚的原因，才是真的懂。**）

【S6】返回的列表必须是副本
    get_window() 返回的列表，外面改了不能影响内部状态。
    所以不能直接 return self._messages。
    （这叫"防御性拷贝"。不这么做的话，调用方一个 append 就把你的内部状态污染了，
      这种 bug 极难排查。）

【S7】其它
    - __len__()：返回存储的消息条数（不含 system prompt）。
      这样 len(memory) 能工作。
    - token_count 属性：返回 get_window() 占用的 token 数。
    - __repr__()：返回类似
      <ShortTermMemory 3 条消息 / 对话 45 token（预算 1000）/ 总计 55 token>
      ⚠️ 注意要把**对话部分**和**总计**分开显示。
        因为 max_tokens 只约束对话部分，如果只显示总计，
        就会出现"总计 55 / 预算 1000 之外的错觉"，或者反过来
        让人以为"超过预算了"（当 system prompt 很长时）。
      调试时一眼就能看到状态，非常实用。

--------------------------------------------------------------------
设计说明：为什么同时限制"token"和"条数"？
--------------------------------------------------------------------
    - token 上限控制成本和上下文长度（主要约束）
    - 条数上限防止"很多条极短消息"导致的另一个问题：
      每条消息都有固定的格式开销（约 4 token），
      而且消息条数太多会让模型注意力分散。
    两层限制互补，实际产品中很常见。

--------------------------------------------------------------------
怎么开始
--------------------------------------------------------------------
    1. 先读一遍 tests/test_short_term.py，那是你的验收标准
    2. 运行测试，看它们全部失败（这叫"红"）
        .\.venv\Scripts\python.exe -m pytest tests/ -v
    3. 一条一条实现，每实现一条就跑一次测试
    4. 全部变绿（passed）就算完成
    5. 完成后可以试着挑战"进阶任务"（见文件末尾）

遇到困难时的求助顺序：
    ① 看测试里的报错信息（pytest 会告诉你是哪一行、期望什么、实际得到什么）
    ② 看本文件的规格说明
    ③ 自己想 15 分钟
    ④ 再问人（这样学到的才扎实）
"""

from typing import Callable

from memory_assistant.llm import estimate_messages_tokens

VALID_ROLES = {"system", "user", "assistant"}


class ShortTermMemory:
    """短期记忆：滑动窗口 + token 预算裁剪。规格见模块文档字符串。"""

    def __init__(
        self,
        max_tokens: int,
        system_prompt: str | None = None,
        max_messages: int | None = None,
        token_counter: Callable[[list[dict]], int] | None = None,
    ) -> None:
        # ---- 这里给你一个开头，剩下的自己写 ----

        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须大于 0，收到的是 {max_tokens}")

        if max_messages is not None and max_messages <= 0:
            raise ValueError(f"max_messages 必须大于 0，收到的是 {max_messages}")

        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        # `or` 的妙用：如果传进来的 token_counter 是 None，就用默认的。
        # 这是 Python 里非常常见的默认值写法。
        self._token_counter = token_counter or estimate_messages_tokens

        # 消息存在这里。下划线开头表示"内部使用，外面别碰"。
        self._messages: list[dict] = []

    # ==================================================================
    # 实现
    # ==================================================================

    def add(self, role: str, content: str) -> None:
        """
        追加一条消息。【S2】

        分三步：校验 role → system 走特殊通道 → 普通消息追加并检查条数上限。
        """
        # ---------- 第 1 步：在入口挡住脏数据 ----------
        # 这是"快速失败"原则：数据刚进系统时就校验，
        # 而不是等到组装 prompt 的时候才发现 role 是"机器人"。
        # Java 里的写法是 if (!VALID_ROLES.contains(role)) throw new IllegalArgumentException(...)
        if role not in VALID_ROLES:
            # {role!r} 里的 !r 表示"用 repr() 输出"。
            # repr 会给字符串加上引号，所以能看到 "User" 和 user 的区别 ——
            # 调大小写错误时，这个引号能帮你一眼看出问题。
            raise ValueError(
                f"非法的 role：{role!r}，只能是 {sorted(VALID_ROLES)} 之一"
            )

        # ---------- 第 2 步：system 走特殊通道 ----------
        # 为什么不和普通消息放同一个列表？
        #   因为 system 必须永远排在最前面，而且**永远不能被裁剪**。
        #   如果混在列表里，每次裁剪都得特殊处理它，很容易漏；
        #   单独存一个属性之后，"不可裁剪"这件事就由数据结构本身保证了，
        #   而不是靠调用方记得。
        #
        # 这是一个通用的设计思路：**用数据结构表达约束，而不是用注释和纪律。**
        if role == "system":
            self.system_prompt = content
            return

        # ---------- 第 3 步：追加，然后看条数有没有超上限 ----------
        self._messages.append({"role": role, "content": content})

        # 条数上限是 token 预算之外的**第二道闸**。为什么需要两道？
        #   · token 上限控制成本和上下文长度（这是主要约束，在 get_window 里生效）
        #   · 条数上限防的是另一种情况：很多条极短消息。每条消息有约 4 个 token
        #     的格式开销，条数太多还会让模型注意力分散。
        if self.max_messages is not None:
            # 用 while 不用 if：万一一次超了好几条（比如上限被调小），
            # while 才能一直丢到合规为止。
            while len(self._messages) > self.max_messages:
                # pop(0) 丢掉最老的那条。
                # 注意这是 O(n) 操作（后面所有元素要前移）；数据量大时应该用
                # collections.deque 代替 list。这里消息最多几十条，不值得优化。
                self._messages.pop(0)

    def get_window(self) -> list[dict]:
        """
        返回这次要发给模型的消息列表。【S3】【S4】【S6】

        **这是整个模块的核心。** 算法分四步，见下面逐段注释。
        """
        # ---------- 第 1 步：system prompt 永远在最前，且不计入预算 ----------
        # 为什么不计入 max_tokens？
        #   因为它是固定开销。压缩系统提示词等于改变 Agent 的性格和规则，
        #   这比少放几条对话严重得多。所以宁可裁对话，也不动 system。
        result: list[dict] = []
        if self.system_prompt is not None:
            result.append({"role": "system", "content": self.system_prompt})

        # ---------- 第 2 步：从最新往回装，装不下就停 ----------
        collected: list[dict] = []
        for message in reversed(self._messages):
            # 候选集合 = 这条消息 + 已经收下的那些
            # 顺序对 token 求和没影响，所以直接拼起来算总量。
            candidate = [message, *collected]

            if self._token_counter(candidate) <= self.max_tokens:
                # 装得下 → 插到最前面，这样 collected 始终是"从旧到新"的顺序
                collected.insert(0, message)
            else:
                # ⚠️ 装不下就 **break（停止）**，不是 continue（跳过）。
                #
                # 为什么不能跳过？因为对话是**连续的**。假设我们有：
                #     [用户: 我在杭州] [助手: 好的] [用户: 贴了一大段文档] [用户: 帮我看看]
                # 预算装不下"那一大段文档"，如果跳过它继续往前找，
                # 就会把"我在杭州"也放进来，而中间隔着一大段被丢掉的内容 ——
                # 模型看到的是断裂的上下文，反而更容易困惑。
                #
                # **宁可少给一点连贯的内容，也不要给断裂的上下文。**
                # 这是本模块唯一真正需要想清楚的设计决策，也是面试常问的点。
                break

        # ---------- 第 3 步：例外处理 —— 最新一条无条件保留（S4）----------
        # 场景：用户粘贴了一篇 5000 字的文章，它自己就超过了整个预算。
        # 如果没有这个例外，collected 会是空的 → 模型收到一个"没有问题"的 prompt
        # → 功能直接坏掉。
        # 所以：**宁可超预算，也不能让这一轮无法回答。**
        if not collected and self._messages:
            collected = [self._messages[-1]]

        # ---------- 第 4 步：返回副本（S6 防御性拷贝）----------
        # 为什么必须 copy？
        #   Python 的 list 和 dict 都是**引用语义**（和 Java 的数组/对象一样）。
        #   如果直接 `return self._messages`，外面一句 append 就把内部状态污染了，
        #   而且这类 bug 极难排查 —— 你会在完全无关的地方发现数据莫名其妙变多了。
        #
        #   dict(message) 是浅拷贝。这里够用，因为消息字典里只有 str，没有嵌套对象。
        #   如果 content 是嵌套结构，就得用 copy.deepcopy 了。
        result.extend(dict(message) for message in collected)
        return result

    def clear(self) -> None:
        """
        清空消息，但保留 system_prompt。【S5】

        clear 的语义是"忘掉刚才聊的"，不是"忘掉你是谁"。
        所以只清消息列表，不动 system_prompt。
        （真正"彻底忘掉"是 LongTermMemory.delete_user 那种，见长期记忆模块。）
        """
        # 用 .clear() 而不是重新赋值 self._messages = []。
        # 在这个类里两种写法测试都能过（没有外部代码持有这个列表的引用），
        # 但 .clear() 是更好的习惯：将来如果别处引用了这个列表，
        # 重新赋值会让那个引用继续指向**旧列表**，
        # 于是出现"我明明清空了，别人还看得到旧数据"这种极难排查的 bug。
        # —— 和上面第 4 步的"防御性拷贝"是同一个思路：谁拥有数据，谁负责它的生命周期。
        self._messages.clear()

    def __len__(self) -> int:
        """
        返回消息条数（不含 system prompt）。【S7】

        实现 __len__ 之后，len(memory) 就能用了。
        注意它返回的是**存着的**消息数，不是窗口里的条数 ——
        窗口大小要用 len(memory.get_window()) 算。
        """
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """
        当前**窗口**占用的 token 数。【S7】

        @property 装饰器的作用：让方法可以像属性一样访问。
            memory.token_count      ← 用起来像个字段
        而不是
            memory.token_count()    ← 没有 @property 就得这样写

        Java 里的对应物是 getter 方法，但 Python 的 @property 更彻底：
        调用方完全看不出这是算出来的还是存着的，将来把字段改成计算属性
        也不会破坏调用方代码。这叫"统一访问原则"。
        """
        return self._token_counter(self.get_window())

    def __repr__(self) -> str:
        """
        调试用的字符串表示。【S7】

        __repr__ 和 __str__ 的区别：
            __str__  → print(obj) 时用，给人看
            __repr__ → 在解释器里直接敲 obj、或放进列表里显示时用，给开发者看
        一个信息量足够的 __repr__ 能省下大量 print 调试的时间：
        出问题时你直接 print(memory) 就能看到状态。
        """
        window = self.get_window()
        total = self._token_counter(window)

        # 把"对话部分"和"system 部分"分开显示。
        #
        # 为什么要分开？因为 max_tokens 只约束**对话部分**（见 S3），
        # 所以"总计 26 token / 预算 20"看起来像超了，实际对话只用了 16 ——
        # system prompt 那 10 个 token 本来就不算在预算里。
        #
        # 一个**会误导人的调试输出比没有更糟**：它会让你花时间去查一个不存在的问题。
        # 这条也是我实测跑出来才发现的（第一版 repr 就是写成了"总计 / 预算"）。
        system_tokens = (
            self._token_counter(window[:1]) if self.system_prompt is not None else 0
        )
        conversation_tokens = total - system_tokens

        return (
            f"<ShortTermMemory {len(self._messages)} 条消息"
            f" / 对话 {conversation_tokens} token（预算 {self.max_tokens}）"
            f" / 总计 {total} token>"
        )


# ====================================================================
# 进阶任务（做完上面的再看，都是真实项目里会遇到的问题）
# ====================================================================
#
# 【进阶 1】性能优化
#     现在的 get_window() 每加一条消息就要重算一次整个列表的 token 数，
#     消息多的时候是 O(n²)。
#     优化思路：给每条消息单独缓存 token 数，或者维护一个累计和。
#     完成后写个 benchmark 对比前后耗时，记下来 —— 面试可以讲。
#
# 【进阶 2】从中间截断超长消息
#     现在如果一条消息特别长（比如用户贴了一篇 5000 字的文章），
#     我们要么整条保留（超预算），要么整条丢弃（丢信息）。
#     更好的方案：把它截断到预算内，并在末尾加上 "[内容已截断]"。
#     想想这会带来什么新问题？（提示：截断到一半的句子可能改变语义）
#
# 【进阶 3】加一个 on_evict 回调
#     当消息被挤出窗口时，我们希望把它交给中期记忆（第 4 周的 SummaryMemory）
#     去做摘要，而不是直接丢掉。
#     实现：构造函数接受 on_evict: Callable[[list[dict]], None] | None，
#           窗口装不下的时候，把被挤出的消息传给这个回调。
#     这个设计叫"观察者模式"，是连接短期记忆和中期记忆的桥梁。
