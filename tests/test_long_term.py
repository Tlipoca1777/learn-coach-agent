r"""
LongTermMemory（长期记忆）的验收测试
================================================================

这些测试现在**全部是失败的**。让它们变绿，就是你的第 5 周作业。

运行：
    # 全部
    .\.venv\Scripts\python.exe -m pytest tests/test_long_term.py -v

    # 只跑某一组
    .\.venv\Scripts\python.exe -m pytest tests/test_long_term.py -k "dedup" -v

--------------------------------------------------------------------
这个测试文件教你的五件事
--------------------------------------------------------------------
【1】怎么用"脚本化向量"精确控制相似度
    注意下面 `ScriptedEmbeddings` 和 `unit()` 这两个辅助。
    它们允许你直接指定「这段文本 → 这个向量」，
    于是你可以**精确构造出余弦相似度恰好是 0.96 的两段文本**。

    为什么需要这个？
    因为如果用真实的向量模型，你根本不知道两句话的相似度是多少，
    测试就只能写成"大概能命中吧"这种含糊的东西。
    有了脚本化向量，阈值逻辑、去重逻辑、排序逻辑全都能精确验证。
    这是"控制变量"思想在测试上的体现。

【2】为什么需要"不同平面"的向量（axis 参数）
    一开始我把所有向量都放在同一个平面上，结果测试写不通：
        unit(0.6) 和 unit(0.7) 的互相相似度是 0.991，
        高过 0.95 的去重阈值 → 第二条事实直接被合并了。
    但我想测的是"两条**不同**事实的排序"。

    真实 embedding 里这个问题不存在：高维空间里，
    "两个向量都和查询很像"并不意味着"它们俩互相很像"。
    所以 unit() 加了 axis 参数，把向量放到不同的坐标平面上，
    用来模拟真实高维空间的这种性质。
    —— 造测试数据时想清楚"几何直觉"，能省下很多 debug 时间。

【3】为什么不在阈值上做"卡边"断言
    你可能想测"相似度刚好等于 0.95 时应该合并"。
    这个测试**不能写**，因为浮点运算有误差：
    数学上刚好 0.95，实际算出来可能是 0.9499999999999998，
    于是测试随机通过或失败。
    正确做法是留出安全余量（用 0.96 和 0.90 测 0.95 的阈值），
    另外用"自定义阈值 + 明显差异"验证参数确实生效。
    这是数值测试的通用原则。

【4】怎么测时间相关的逻辑
    衰减逻辑如果依赖"现在几点"，测试就没法复现。
    所以 `add` / `search` / `invalidate` 都接受一个 `now` 参数，
    测试传固定时间进去，结果就完全确定了。
    这和 `token_counter` 注入是同一个思路。

【5】为什么要检查"读操作有没有偷偷写"
    注意 `test_add_does_not_record_hits`。
    它验证"添加事实时内部的去重检索不能记录命中次数"。
    这类"看不见的副作用"是最难排查的 bug ——
    不报错，但数据慢慢就不对了。
"""

import itertools
import math
from datetime import timedelta

import pytest

from memory_assistant.memory.long_term import (
    LongTermMemory,
    days_between,
    parse_iso,
    to_iso,
    utc_now,
)

# ====================================================================
# 测试辅助
# ====================================================================

_collection_counter = itertools.count()


def unit(cos_with_query: float, axis: int = 1) -> list[float]:
    """
    构造一个 4 维单位向量：它与查询向量 [1,0,0,0] 的余弦相似度**恰好**是
    cos_with_query，并且落在"第 0 维 + 第 axis 维"构成的平面上。

    原理：平面上的单位向量可以写成 [cos θ, ..., sin θ, ...]，
    与 [1,0,0,0] 的点积就是 cos θ。

    为什么要有 axis 参数？见文件头说明【2】。
    两个落在**不同平面**上的向量，互相相似度是 cos1 * cos2，
    而不是接近 1 —— 这才像真实的高维 embedding。
    """
    vector = [0.0] * 4
    vector[0] = cos_with_query
    vector[axis] = math.sqrt(max(0.0, 1.0 - cos_with_query**2))
    return vector


class ScriptedEmbeddings:
    """
    脚本化假向量：可以精确指定"某段文本 → 某个向量"。

    没在 mapping 里的文本会得到一个固定的后备向量 [0,0,0,1]，
    它与用 unit() 构造的向量都正交（余弦相似度 0），
    所以不会意外干扰测试。
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None, dimension: int = 4):
        self.mapping = dict(mapping or {})
        self.dimension = dimension
        self.model_name = "scripted"
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        if text in self.mapping:
            return list(self.mapping[text])
        fallback = [0.0] * self.dimension
        fallback[-1] = 1.0
        return fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


def make_memory(mapping=None, **kwargs) -> LongTermMemory:
    """创建一个用脚本化向量、每次用独立集合的长期记忆（内存态）。"""
    kwargs.setdefault("collection_name", f"test_{next(_collection_counter)}")
    return LongTermMemory(ScriptedEmbeddings(mapping), **kwargs)


# ====================================================================
# 组 1：构造与参数校验（规格 S1）
# ====================================================================


def test_rejects_invalid_dedup_threshold():
    with pytest.raises(ValueError):
        make_memory(dedup_threshold=0.0)
    with pytest.raises(ValueError):
        make_memory(dedup_threshold=1.5)
    with pytest.raises(ValueError):
        make_memory(dedup_threshold=-0.1)


def test_rejects_invalid_half_life():
    with pytest.raises(ValueError):
        make_memory(half_life_days=0)
    with pytest.raises(ValueError):
        make_memory(half_life_days=-30)


def test_rejects_negative_hit_boost():
    with pytest.raises(ValueError):
        make_memory(hit_boost=-0.1)


def test_default_is_in_memory():
    """不传 persist_dir 时应该是内存态，不产生任何文件。"""
    memory = make_memory()
    assert memory.count() == 0


# ====================================================================
# 组 2：add 与 get（规格 S2、S4）
# ====================================================================


def test_add_creates_fact():
    memory = make_memory({"user 住在 杭州": unit(1.0)})

    result = memory.add("住在", "杭州")

    assert result["action"] == "created"
    assert isinstance(result["id"], str)
    assert result["similarity"] == 0.0        # 库里本来什么都没有
    assert memory.count() == 1


def test_add_renders_text_as_subject_predicate_object():
    """
    事实的文本表示是 "{subject} {predicate} {object}"。

    这个渲染规则很重要：它既是向量化的输入，也是展示给用户看的内容。
    """
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact_id = memory.add("住在", "杭州")["id"]

    fact = memory.get(fact_id)
    assert fact["text"] == "user 住在 杭州"
    assert fact["subject"] == "user"
    assert fact["predicate"] == "住在"
    assert fact["object"] == "杭州"


def test_add_with_custom_subject():
    """subject 不一定是 user —— 也支持"我老婆喜欢辣"这种情况。"""
    memory = make_memory({"老婆 喜欢 辣": unit(1.0)})
    fact_id = memory.add("喜欢", "辣", subject="老婆")["id"]

    assert memory.get(fact_id)["text"] == "老婆 喜欢 辣"
    assert memory.get(fact_id)["subject"] == "老婆"


def test_add_stores_metadata():
    memory = make_memory({"user 对花生过敏 严重": unit(1.0)})

    fact_id = memory.add("对花生过敏", "严重", confidence=0.85, source_message_id=42)["id"]
    fact = memory.get(fact_id)

    assert fact["confidence"] == pytest.approx(0.85)
    assert fact["source_message_id"] == 42
    assert fact["user_id"] == "default"


def test_add_initial_state():
    """新添加的事实应该是：有效、命中 0 次、从未命中、未失效。"""
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact = memory.get(memory.add("住在", "杭州")["id"])

    assert fact["invalid_at"] is None
    assert fact["hit_count"] == 0
    assert fact["last_hit_at"] is None
    assert fact["created_at"] == fact["updated_at"]


def test_get_missing_returns_none():
    memory = make_memory()
    assert memory.get("根本不存在的id") is None


# ====================================================================
# 组 3：search 基础（规格 S3）
# ====================================================================


def test_search_orders_by_similarity():
    """三个事实放到不同平面上，避免它们之间被误判成重复。"""
    memory = make_memory(
        {
            "user 住在 杭州": unit(0.9, 1),
            "user 喜欢 咖啡": unit(0.6, 2),
            "user 养了 猫": unit(0.3, 3),
            "查询": unit(1.0),
        }
    )
    memory.add("住在", "杭州")
    memory.add("喜欢", "咖啡")
    memory.add("养了", "猫")

    results = memory.search("查询")

    assert [f["text"] for f in results] == [
        "user 住在 杭州",
        "user 喜欢 咖啡",
        "user 养了 猫",
    ]


def test_search_respects_top_k():
    memory = make_memory(
        {
            "user A A": unit(0.9, 1),
            "user B B": unit(0.7, 2),
            "user C C": unit(0.5, 3),
            "查询": unit(1.0),
        }
    )
    memory.add("A", "A")
    memory.add("B", "B")
    memory.add("C", "C")

    assert len(memory.search("查询", top_k=2)) == 2
    assert len(memory.search("查询", top_k=10)) == 3


def test_search_on_empty_memory_returns_empty_list():
    """空库要返回空列表，不能返回 None —— 调用方会直接 for 遍历它。"""
    memory = make_memory({"查询": unit(1.0)})
    assert memory.search("查询") == []


def test_search_result_contains_score_fields():
    memory = make_memory({"user 住在 杭州": unit(0.8, 1), "查询": unit(1.0)})
    memory.add("住在", "杭州")

    fact = memory.search("查询")[0]

    assert fact["similarity"] == pytest.approx(0.8, abs=0.01)
    assert "decay" in fact
    assert "score" in fact
    assert fact["score"] == pytest.approx(
        fact["similarity"] * fact["decay"] * (1 + math.log1p(fact["hit_count"]) * 0.1),
        rel=1e-6,
    )


def test_get_does_not_contain_score_fields():
    """只有 search 结果才带 similarity/decay/score。"""
    memory = make_memory({"user 住在 杭州": unit(0.8, 1)})
    fact = memory.get(memory.add("住在", "杭州")["id"])

    assert "similarity" not in fact
    assert "decay" not in fact
    assert "score" not in fact


def test_search_isolates_users():
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    memory.add("住在", "杭州", user_id="alice")
    memory.add("住在", "杭州", user_id="bob")

    alice = memory.search("查询", user_id="alice")
    bob = memory.search("查询", user_id="bob")
    nobody = memory.search("查询", user_id="carol")

    assert len(alice) == 1 and alice[0]["user_id"] == "alice"
    assert len(bob) == 1 and bob[0]["user_id"] == "bob"
    assert nobody == []


# ====================================================================
# 组 4：去重（规格 S2）
# ====================================================================


def test_identical_fact_is_merged():
    memory = make_memory({"user 住在 杭州": unit(1.0)})

    first = memory.add("住在", "杭州")
    second = memory.add("住在", "杭州")

    assert first["action"] == "created"
    assert second["action"] == "merged"
    assert second["id"] == first["id"]
    assert memory.count() == 1


def test_similar_above_threshold_is_merged():
    """
    相似度 0.96 > 阈值 0.95 → 合并。

    为什么用 0.96 而不是"刚好 0.95"：浮点误差会让卡边测试不稳定。
    详见文件头说明【3】。
    """
    memory = make_memory({"user 住在 杭州": unit(1.0), "user 住在 杭州市": unit(0.96)})
    memory.add("住在", "杭州")

    result = memory.add("住在", "杭州市")

    assert result["action"] == "merged"
    assert result["similarity"] == pytest.approx(0.96, abs=0.01)
    assert memory.count() == 1


def test_similar_below_threshold_creates_new():
    """相似度 0.90 < 阈值 0.95 → 新事实。"""
    memory = make_memory({"user 住在 杭州": unit(1.0), "user 住在 上海": unit(0.90)})
    memory.add("住在", "杭州")

    result = memory.add("住在", "上海")

    assert result["action"] == "created"
    assert memory.count() == 2


def test_dedup_respects_custom_threshold():
    """阈值调低之后，原本不合并的就该合并了。"""
    memory = make_memory(
        {"user 住在 杭州": unit(1.0), "user 住在 上海": unit(0.90)},
        dedup_threshold=0.5,
    )
    memory.add("住在", "杭州")

    assert memory.add("住在", "上海")["action"] == "merged"


def test_merge_increments_hit_count():
    """重复说同一件事，说明这件事重要 —— 用 hit_count 记录下来。"""
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact_id = memory.add("住在", "杭州")["id"]
    assert memory.get(fact_id)["hit_count"] == 0

    memory.add("住在", "杭州")
    memory.add("住在", "杭州")

    assert memory.get(fact_id)["hit_count"] == 2


def test_merge_does_not_change_created_at():
    """created_at 记录"第一次出现"的时间，合并时绝不能改。"""
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    t1 = utc_now()
    fact_id = memory.add("住在", "杭州", now=t1)["id"]
    created = memory.get(fact_id)["created_at"]

    memory.add("住在", "杭州", now=t1 + timedelta(days=365))

    assert memory.get(fact_id)["created_at"] == created
    assert memory.get(fact_id)["updated_at"] != created


def test_merge_takes_max_confidence():
    """合并时置信度取较大的那个 —— 更确定的信息应该保留下来。"""
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact_id = memory.add("住在", "杭州", confidence=0.6)["id"]

    memory.add("住在", "杭州", confidence=0.9)
    assert memory.get(fact_id)["confidence"] == pytest.approx(0.9)

    memory.add("住在", "杭州", confidence=0.3)
    assert memory.get(fact_id)["confidence"] == pytest.approx(0.9)


def test_dedup_ignores_invalidated_facts():
    """
    失效的事实不参与去重。

    场景：用户说"我住在杭州" → 后来说"我搬到上海了"（杭州那条失效）
          → 再后来说"我又搬回杭州了"
    这时应该**新建**一条有效的杭州事实，而不是合并到失效的那条上。
    """
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    first_id = memory.add("住在", "杭州")["id"]
    memory.invalidate(first_id)

    result = memory.add("住在", "杭州")

    assert result["action"] == "created"
    assert result["id"] != first_id
    assert memory.count(include_invalid=True) == 2


def test_dedup_is_per_user():
    """不同用户之间不能互相当成重复。"""
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    memory.add("住在", "杭州", user_id="alice")

    result = memory.add("住在", "杭州", user_id="bob")

    assert result["action"] == "created"
    assert memory.count() == 2


def test_add_does_not_record_hits():
    """
    ⭐ 检查"看不见的副作用"。

    add() 内部要做一次去重检索，但那次检索**不能记录命中次数**。
    否则你每添加一条事实，库里已有事实的 hit_count 就会莫名其妙地涨，
    衰减和加权全部被污染 —— 而且不会有任何报错。

    构造：两条相似度 0.90 的事实（低于阈值，不合并），
          添加第二条之后，第一条的 hit_count 必须还是 0。
    """
    memory = make_memory({"user 住在 杭州": unit(1.0), "user 住在 上海": unit(0.90)})
    first_id = memory.add("住在", "杭州")["id"]
    memory.add("住在", "上海")

    assert memory.get(first_id)["hit_count"] == 0, (
        "add() 内部的去重检索记录了命中次数 —— 记得给它传 record_hits=False"
    )


# ====================================================================
# 组 5：时间衰减（规格 S3）
# ====================================================================


def test_fresh_fact_has_full_decay():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    memory.add("住在", "杭州", now=now)

    fact = memory.search("查询", now=now)[0]

    assert fact["decay"] == pytest.approx(1.0, abs=1e-6)


def test_decay_halves_after_one_half_life():
    """
    半衰期的定义：过了 half_life_days 天，权重降到一半。
    用注入的 now 精确构造"30 天前创建"的场景。
    """
    created = utc_now()
    memory = make_memory(
        {"user 住在 杭州": unit(1.0), "查询": unit(1.0)}, half_life_days=30.0
    )
    memory.add("住在", "杭州", now=created)

    fact = memory.search("查询", now=created + timedelta(days=30))[0]

    assert fact["decay"] == pytest.approx(0.5, abs=1e-6)


def test_decay_uses_last_hit_at_when_available():
    """
    衰减的参考时间是 last_hit_at（如果有），不是 created_at。

    原因：半年前创建、但昨天刚被用到的偏好，比半年前创建、
    再也没提过的偏好更值得注入 prompt。
    """
    created = utc_now()
    memory = make_memory(
        {"user 住在 杭州": unit(1.0), "查询": unit(1.0)}, half_life_days=30.0
    )
    fact_id = memory.add("住在", "杭州", now=created)["id"]

    day100 = created + timedelta(days=100)
    memory.search("查询", now=day100)
    assert memory.get(fact_id)["last_hit_at"] is not None

    day101 = day100 + timedelta(days=1)
    fact = memory.search("查询", now=day101)[0]

    assert fact["decay"] > 0.9, (
        f"decay={fact['decay']:.4f}，说明参考时间用的是 created_at 而不是 last_hit_at"
    )


def test_future_timestamp_is_clamped():
    """如果 now 早于 created_at（数据异常），decay 必须被 clamp 到 1，不能大于 1。"""
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    memory.add("住在", "杭州", now=now)

    fact = memory.search("查询", now=now - timedelta(days=365))[0]

    assert fact["decay"] == pytest.approx(1.0, abs=1e-6)


def test_old_fact_ranks_below_fresh_fact():
    """
    相似度相同时，新事实应该排在旧事实前面。

    两个事实放在不同平面上：它们对查询的相似度都是 0.8，
    但它们互相的相似度只有 0.64，所以不会被去重合并。
    """
    now = utc_now()
    memory = make_memory(
        {
            "user 旧 事": unit(0.8, 1),
            "user 新 事": unit(0.8, 2),
            "查询": unit(1.0),
        },
        half_life_days=30.0,
    )
    memory.add("旧", "事", now=now - timedelta(days=365))
    memory.add("新", "事", now=now)

    results = memory.search("查询", now=now, record_hits=False)

    assert results[0]["text"] == "user 新 事"


# ====================================================================
# 组 6：命中记录与加权（规格 S3）
# ====================================================================


def test_search_records_hits_by_default():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    memory.search("查询", now=now)
    assert memory.get(fact_id)["hit_count"] == 1

    memory.search("查询", now=now)
    assert memory.get(fact_id)["hit_count"] == 2
    assert memory.get(fact_id)["last_hit_at"] == to_iso(now)


def test_record_hits_false_does_not_write():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    memory.search("查询", now=now, record_hits=False)

    assert memory.get(fact_id)["hit_count"] == 0
    assert memory.get(fact_id)["last_hit_at"] is None


def test_hit_count_boosts_ranking():
    """
    命中次数多的记忆会获得排序加成。

    构造：A 的相似度(0.70) 比 B(0.75) 低，但 A 被反复命中。
          加权公式是 1 + log1p(hit_count) * 0.1，
          hit_count=5 时加成约 1.18 倍，足以让 A 反超 B：
              0.70 * 1.18 = 0.826  >  0.75

    A 和 B 放在不同平面上，互相相似度只有 0.525，不会被去重合并。
    """
    now = utc_now()
    memory = make_memory(
        {
            "user A 事": unit(0.70, 1),
            "user B 事": unit(0.75, 2),
            "只命中A": unit(0.70, 1),      # 这个查询和 A 的向量完全一样
            "泛查询": unit(1.0),
        }
    )
    a_id = memory.add("A", "事", now=now)["id"]
    memory.add("B", "事", now=now)

    # 用"只命中 A"的查询把 A 的 hit_count 堆到 5
    for _ in range(5):
        top = memory.search("只命中A", top_k=1, now=now)
        assert top[0]["text"] == "user A 事"
    assert memory.get(a_id)["hit_count"] == 5

    results = memory.search("泛查询", now=now, record_hits=False)
    by_text = {f["text"]: f for f in results}

    assert by_text["user A 事"]["score"] > by_text["user B 事"]["score"], (
        f"A 的 score={by_text['user A 事']['score']:.4f}"
        f"（相似度 0.70 + 5 次命中），"
        f"B 的 score={by_text['user B 事']['score']:.4f}（相似度 0.75）\n"
        "命中加权没生效？检查公式是不是 1 + log1p(hit_count) * hit_boost"
    )


# ====================================================================
# 组 7：失效、删除、统计（规格 S5~S9）
# ====================================================================


def test_invalidate_excludes_from_search():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    assert memory.invalidate(fact_id, now=now) is True
    assert memory.search("查询", now=now) == []


def test_include_invalid_returns_invalidated_facts():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]
    memory.invalidate(fact_id, now=now)

    results = memory.search("查询", now=now, include_invalid=True)

    assert len(results) == 1
    assert results[0]["id"] == fact_id


def test_invalidate_sets_invalid_at():
    """失效是"标记"而不是"删除" —— 要能回溯"什么时候改的"。"""
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    memory.invalidate(fact_id, now=now)

    fact = memory.get(fact_id)
    assert fact is not None, "失效不应该物理删除数据"
    assert fact["invalid_at"] == to_iso(now)


def test_invalidate_missing_returns_false():
    memory = make_memory()
    assert memory.invalidate("不存在的id") is False


def test_invalidate_twice_returns_false():
    """第二次失效没有产生状态变化，应该返回 False。"""
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    assert memory.invalidate(fact_id, now=now) is True
    assert memory.invalidate(fact_id, now=now) is False


def test_delete_removes_fact():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0), "查询": unit(1.0)})
    fact_id = memory.add("住在", "杭州", now=now)["id"]

    assert memory.delete(fact_id) is True
    assert memory.get(fact_id) is None
    assert memory.search("查询", now=now, include_invalid=True) == []


def test_delete_missing_returns_false():
    memory = make_memory()
    assert memory.delete("不存在的id") is False


def test_delete_user_removes_all_their_facts():
    memory = make_memory(
        {"user 住在 杭州": unit(1.0, 1), "user 喜欢 咖啡": unit(0.5, 2)}
    )
    memory.add("住在", "杭州", user_id="alice")
    memory.add("喜欢", "咖啡", user_id="alice")
    memory.add("住在", "杭州", user_id="bob")

    removed = memory.delete_user("alice")

    assert removed == 2
    assert memory.count(user_id="alice") == 0
    assert memory.count(user_id="bob") == 1


def test_count_with_filters():
    now = utc_now()
    memory = make_memory({"user 住在 杭州": unit(1.0, 1), "user 喜欢 咖啡": unit(0.5, 2)})
    first = memory.add("住在", "杭州")["id"]
    memory.add("喜欢", "咖啡")
    memory.invalidate(first, now=now)

    assert memory.count() == 1                          # 默认不含失效
    assert memory.count(include_invalid=True) == 2
    assert memory.count(user_id="default") == 1
    assert memory.count(user_id="nobody") == 0


def test_clear():
    memory = make_memory({"user 住在 杭州": unit(1.0)})
    memory.add("住在", "杭州")
    assert memory.count() == 1

    memory.clear()

    assert memory.count() == 0


# ====================================================================
# 组 8：持久化（规格 S1）
# ====================================================================


def test_persists_across_reopen(tmp_path):
    """给了 persist_dir 之后，关掉再打开数据还在 —— 这是长期记忆的立身之本。"""
    mapping = {"user 住在 杭州": unit(1.0), "查询": unit(1.0)}
    now = utc_now()

    first = LongTermMemory(
        ScriptedEmbeddings(mapping),
        persist_dir=tmp_path,
        collection_name="persist_test",
    )
    fact_id = first.add("住在", "杭州", now=now)["id"]
    assert first.count() == 1

    # 模拟程序退出后重新打开
    second = LongTermMemory(
        ScriptedEmbeddings(mapping),
        persist_dir=tmp_path,
        collection_name="persist_test",
    )

    assert second.count() == 1
    assert second.get(fact_id)["text"] == "user 住在 杭州"
    assert second.search("查询", now=now)[0]["similarity"] == pytest.approx(1.0, abs=0.01)


# ====================================================================
# 组 9：已提供的辅助函数
# ====================================================================


def test_days_between_basic():
    result = days_between(
        "2025-01-01T00:00:00+00:00", parse_iso("2025-01-11T00:00:00+00:00")
    )
    assert result == pytest.approx(10.0)


def test_days_between_clamps_negative():
    """负数必须被 clamp 成 0，否则时间倒流会让记忆变得更"重要"。"""
    result = days_between(
        "2025-06-01T00:00:00+00:00", parse_iso("2025-01-01T00:00:00+00:00")
    )
    assert result == 0.0


def test_days_between_handles_empty():
    assert days_between("", utc_now()) == 0.0


def test_iso_roundtrip():
    now = utc_now()
    assert parse_iso(to_iso(now)) == now.replace(microsecond=0)


def test_parse_iso_handles_naive_string():
    """没有时区的字符串要当成 UTC，不能直接当本地时间。"""
    moment = parse_iso("2025-01-01T00:00:00")
    assert moment.tzinfo is not None


# ====================================================================
# 组 10：repr（规格 S10）
# ====================================================================


def test_repr_is_informative():
    memory = make_memory({"user 住在 杭州": unit(1.0)}, half_life_days=7)
    memory.add("住在", "杭州")

    text = repr(memory)

    assert "LongTermMemory" in text
    assert "1" in text          # 事实条数
    assert "7" in text          # 半衰期
