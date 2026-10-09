r"""
长期记忆 —— 向量库里的结构化事实
================================================================

⭐⭐ 这是长期记忆模块的规格说明。D3 已实现基础存取能力，D4 继续完成评分、去重与失效管理。
    验收标准在 tests/test_long_term.py。

--------------------------------------------------------------------
先想清楚：长期记忆和短期、中期到底有什么不同？
--------------------------------------------------------------------
    短期记忆：最近几轮原文          —— 只管"刚才说了什么"
    中期记忆：被挤出对话的摘要       —— 只管"这一段聊了什么"
    长期记忆：关于用户的**结构化事实** —— 管"你是一个什么样的人"

关键区别在于：**长期记忆是跨会话、无期限的**。
关掉程序一个月后再打开，它还得记得你对花生过敏。

--------------------------------------------------------------------
为什么存"抽取的事实"而不是"原始对话"？（面试必问）
--------------------------------------------------------------------
| 维度       | 存原始对话              | 存抽取的事实（本项目）    |
|-----------|------------------------|-------------------------|
| 检索精度   | 低（噪声大）             | 高（一条事实一个语义单元） |
| token 效率 | 差（找回一段话只有一句有用）| 好                      |
| 可去重     | 几乎不可能               | 可以（同一事实合并）      |
| 可冲突消解 | 不可能                  | 可以（新旧事实标记生效期） |
| 可解释     | 差                      | 好（能说"我记得你说过 X"）|
| 成本       | 低（不调 LLM）           | 高（抽取要调 LLM）        |

结论：精度和可控性的收益远大于额外的抽取成本。

--------------------------------------------------------------------
⚠️ 写这个模块之前，我把 Chroma 1.5.9 的实际行为全测了一遍。
   下面四条约束会直接决定你怎么写代码，**请务必先看懂**。
--------------------------------------------------------------------

【约束 1】绝不能让 Chroma 自己算向量
    如果你调用 `get_or_create_collection()` 时不传 embedding_function，
    Chroma 会**自动实例化它自带的 `DefaultEmbeddingFunction`** ——
    那个模型是英文的 all-MiniLM-L6-v2，在中文上基本不可用
    （实测数据见 docs/记忆架构设计.md 第 12 节）。

    正确做法：**我们自己算好向量，显式传给 Chroma**：
        collection.add(ids=[...], embeddings=[[...], [...]], documents=[...])
    实测确认：显式传 embeddings 可以完全绕过集合自带的 EF。

    这样 Chroma 就退化成一个纯粹的"向量索引"，向量只有一个来源（我们的
    EmbeddingProvider），不会出现"入库用中文模型、查询用英文模型"这种事。

【约束 2】metadata 里**不能放 None**
    实测报错：`TypeError: Cannot convert Python object to MetadataValue`

    这直接影响设计：我们不能把 `invalid_at = None` 存进去。
    解决办法是**多存一个布尔字段**：
        "is_valid": True / False      ← 用它做过滤（实测 where 支持 bool）
        "invalid_at": ""              ← 用空字符串代替 None
    空字符串是允许的（实测确认）。

【约束 3】Chroma 返回的是**距离**，不是相似度，而且默认空间是平方 L2
    实测：两个正交单位向量的距离是 2.0（正好等于平方 L2），
    而我们要的是余弦相似度。如果不处理，排序公式就会用错，
    而且**不会报错**，只是"结果不太对"，极难发现。

    解决办法：建集合时显式指定**余弦空间**（实测两种写法都可用）：
        configuration={"hnsw": {"space": "cosine"}}
    余弦空间下距离与相似度的换算非常干净：
        similarity = 1 - distance
    （实测验证：自己和自己的距离是 0，正交向量的距离是 1）

【约束 4】维度必须一致
    实测报错：`Collection expecting embedding with dimension of 8, got 16`
    这条其实是好事 —— 它帮你挡住了"换了向量模型但没重建库"这种事故。
    （如果把 512 维的模型换成 768 维的，你会立刻收到明确的报错，
      而不是得到一堆无意义的检索结果。）

--------------------------------------------------------------------
你要实现的接口
--------------------------------------------------------------------

    from memory_assistant.embeddings import create_embeddings
    from memory_assistant.memory.long_term import LongTermMemory

    embedder = create_embeddings(fake=True)          # 测试用假向量
    memory = LongTermMemory(embedder)                # 默认内存态

    memory.add("住在", "杭州")
    memory.add("喜欢", "手冲咖啡，不加糖")
    memory.add("对……过敏", "花生")

    for fact in memory.search("我平时喝什么？", top_k=2):
        print(fact["text"], fact["score"])

--------------------------------------------------------------------
详细规格
--------------------------------------------------------------------

【S1】构造函数
    __init__(self, embedder, persist_dir=None, collection_name="facts", *,
             client=None, dedup_threshold=0.95,
             half_life_days=30.0, hit_boost=0.1)

    - embedder：任何满足 embeddings.EmbeddingProvider 接口的对象。
      **测试时必须传 FakeEmbeddings 或自己写的假向量**
      （不然每次跑测试都要加载 90MB 的真模型，你不会想跑测试的）。
    - persist_dir：Chroma 的持久化目录。
        None 且 client 也是 None  → 用 EphemeralClient（内存，进程退出就没了）
        给了 persist_dir          → 用 PersistentClient(path)
      这个"默认内存、按需持久化"的约定，让测试可以零成本地跑。
    - client：允许直接注入一个已经建好的 Chroma 客户端（测试用）。
      给了 client 就忽略 persist_dir。
    - dedup_threshold：去重阈值，默认 0.95。必须满足 0 < 阈值 <= 1。
    - half_life_days：时间衰减的半衰期，默认 30 天。必须 > 0。
    - hit_boost：命中加权的系数，默认 0.1。必须 >= 0。

    ⚠️ 参数不合法一律抛 ValueError。

    建集合时的要求（对应约束 1、3）：
        · 显式指定余弦空间
        · 绝对不要传 embedding_function（我们用不到它）
        · 如果集合已经存在，就直接取出来用，不要试图改它的配置

【S2】add(self, predicate, object, *, subject="user", user_id="default",
          confidence=1.0, source_message_id=None, now=None) -> dict

    添加一条事实。返回一个字典：
        {"id": 事实id, "action": "created" 或 "merged", "similarity": 与已有事实的最高相似度}

    action 的含义：
        "created" —— 新事实，写进库
        "merged"  —— 库里已有高度相似的事实，合并而不是新增

    事实的文本表示（用于向量化和显示）：
        text = f"{subject} {predicate} {object}"
    例如 subject="user", predicate="住在", object="杭州" → "user 住在 杭州"

    去重逻辑（这是 add 的核心）：
        1. 用 text 算向量
        2. 在**同一个 user_id** 的**有效事实**里找最相似的一条
           （注意：失效的事实不参与去重 —— 否则用户改口后又改回来就加不进去了）
        3. 最高相似度 >= dedup_threshold →
               合并：更新 updated_at、hit_count += 1、
               confidence 取 max(旧的, 新的)，返回 action="merged"
        4. 否则 → 新建，返回 action="created"

    ⚠️ 四个容易漏的点：
        (a) 返回的 similarity 在没有任何已有事实时应该是 0.0（不是 None）
        (b) 合并时**不要**修改 created_at —— 那是这条事实"第一次出现"的时间
        (c) 判断去重要看**相似度 similarity**，不是 score。
            因为 score 里混了时间衰减和命中加权，
            用它去重会导致"很久以前重复说过的事实"躲过去重判断，
            库里攒下一堆几乎一样的记录。
        (d) 内部这次去重检索**必须关掉命中记录**（record_hits=False）。
            否则"添加一条事实"会意外地把已有事实的 hit_count 加 1，
            污染衰减和加权 —— 而且这个 bug 非常隐蔽，
            你会看到"明明没检索过，命中次数却在涨"。

【S3】search(self, query, *, top_k=5, user_id="default",
             include_invalid=False, record_hits=True, now=None) -> list[dict]

    语义检索。返回按 **score 从高到低**排序的事实列表。

    评分公式（三个因子相乘）：
        similarity = 1 - distance                    （余弦空间，见约束 3）
        decay      = 0.5 ** (days / half_life_days)   （时间衰减）
        boost      = 1 + log1p(hit_count) * hit_boost （常用记忆加权）
        score      = similarity * decay * boost

    decay 里的 days 怎么算（**这是最容易做错的地方**）：
        参考时间 = last_hit_at（如果有）否则 created_at
        days = 参考时间到 now 之间的天数，**负数按 0 处理**
    为什么用 last_hit_at 而不是 created_at？
        因为"最近被用到过的记忆"应该重新变得重要。
        一个半年前创建、但昨天刚被用到的偏好，比一个半年前创建、
        再也没提过的偏好更值得注入 prompt。
    为什么天数要 clamp 到 0？
        因为测试或数据修复时可能传入"未来"的时间戳，
        负天数会让 decay 大于 1，把评分逻辑弄乱。

    返回的每条事实是一个字典，包含：
        id / text / subject / predicate / object / user_id / confidence /
        source_message_id / created_at / updated_at / invalid_at /
        hit_count / last_hit_at          ← 这些是存储的字段
        similarity / decay / score       ← 这三个只在 search 结果里出现

    参数说明：
        top_k            最多返回几条
        include_invalid  True 时连已失效的事实也返回（调试和"你改过什么"很有用）
        record_hits      True 时把命中的事实 hit_count += 1、last_hit_at 更新为 now
                         ⚠️ 这会让"读"操作产生"写"。默认开启是为了让衰减机制生效，
                            但你要知道它的代价：检索从纯读变成了读写。
                            高并发场景下这会成为瓶颈（第 8 周做 Web 服务时会讨论）。
        now              注入当前时间，**测试时间衰减必须用它**
                         （和 token_counter 注入是同一个思路：让测试可复现）

    ⚠️ 如果库里没有任何匹配的事实，返回**空列表**，不要返回 None。

    📌 一个已知的简化（想清楚了可以当进阶任务做）：
        top_k 是直接交给向量库的候选数，也就是"按**向量相似度**先取前 top_k 条"，
        然后我们再按 score 重排返回。
        严格来说这不完全等价于"按 score 取前 top_k 条"——
        因为 score 里混了衰减和命中加权，可能存在"相似度排第 6 但 score 应该进前 5"
        的事实被提前丢掉。
        更好的做法是先多取一些候选（比如 top_k * 3），都算完 score 再截断。
        这里为了把逻辑讲清楚，先保持简化版。

【S4】get(self, fact_id) -> dict | None
    按 id 取一条事实（不带 similarity / decay / score）。不存在返回 None。

【S5】invalidate(self, fact_id, *, now=None) -> bool
    把一条事实标记为失效（冲突消解用）。
        · 把 is_valid 设为 False
        · 把 invalid_at 设为 now 的 ISO 字符串
        · **不要物理删除** —— 要能回溯"你之前说过 X，什么时候改的"
    成功返回 True；id 不存在返回 False。
    已经是失效状态再调一次，返回 True（幂等）还是 False？
    规格选：**返回 False**，表示"这次调用没有产生状态变化"。

【S6】delete(self, fact_id) -> bool
    物理删除。删除后 get() 返回 None、search() 不再返回它。
    不存在时返回 False。

【S7】delete_user(self, user_id) -> int
    删除某个用户的**全部**事实，返回删掉的条数。
    这是"被遗忘权"的实现：用户说"忘掉我的一切"，必须真的删干净。

【S8】count(self, *, user_id=None, include_invalid=False) -> int
    统计事实条数。user_id 为 None 表示不按用户过滤。
    （实现提示：Chroma 的 collection.get() 可以带 where 条件取回全部匹配项，
      条数就是返回 id 列表的长度。数据量大时这会很慢，
      真实项目应该单独维护计数 —— 这里够用就行。）

【S9】clear(self) -> None
    清空整个集合的所有事实。

【S10】__repr__
    类似 <LongTermMemory 12 条事实 / 集合=facts / 半衰期=30天>

--------------------------------------------------------------------
模块里已经给你的辅助函数（直接用，不用自己写）
--------------------------------------------------------------------
    to_iso(dt)            datetime → ISO 字符串
    parse_iso(text)       ISO 字符串 → datetime（带 UTC 时区）
    days_between(start_iso, end_dt) -> float
                          两个时间之间相差几天，**负数会被 clamp 成 0**
    utc_now() -> datetime 当前 UTC 时间

这几行逻辑不难，但很容易在时区上栽跟头（naive vs aware datetime），
所以直接给你，让你把精力放在记忆逻辑本身。

--------------------------------------------------------------------
怎么开始
--------------------------------------------------------------------
    1. 先读 tests/test_long_term.py
    2. 跑测试看全红：
       .\.venv\Scripts\python.exe -m pytest tests/test_long_term.py -v
    3. 按这个顺序实现：
       辅助属性/__repr__ → add（先不管去重，直接建）→ get
       → search（先不管衰减，只按相似度排）
       → 加 decay → 加 boost/record_hits
       → 去重 → invalidate / delete / delete_user / count / clear
    4. 全绿后挑战进阶任务

⚠️ 这个作业是三个里面最难的，因为它同时涉及：
   · 和外部库（Chroma）打交道
   · 评分公式（三因子相乘）
   · 时间计算
   卡住很正常，超过 30 分钟就来找我。
"""

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ====================================================================
# 已提供的辅助函数（时间处理）
# ====================================================================


def utc_now() -> datetime:
    """当前 UTC 时间（带时区信息）。

    为什么一定要带时区？
        因为不带时区的 datetime（naive）和带时区的（aware）不能直接相减，
        Python 会抛 TypeError。这是个非常常见的坑，统一用 aware 就没事了。
    """
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """datetime → ISO 8601 字符串。"""
    return moment.isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    """ISO 8601 字符串 → datetime。

    兼容两种情况：带时区（'2025-01-01T00:00:00+00:00'）
    和不带时区（'2025-01-01T00:00:00'，Chroma 里手写数据可能这样）。
    不带时区的一律当成 UTC。
    """
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def days_between(start_iso: str, end: datetime) -> float:
    """
    从 start_iso 到 end 之间相差多少天。

    ⚠️ 负数会被 clamp 成 0。为什么？
       因为测试或修数据时可能传进来"未来"的时间戳，
       负天数会让衰减因子大于 1（时间倒流反而让记忆更重要），逻辑就乱了。
    """
    if not start_iso:
        return 0.0
    delta = (end - parse_iso(start_iso)).total_seconds() / 86400.0
    return max(0.0, delta)


DEFAULT_COLLECTION_NAME = "facts"
DEFAULT_DEDUP_THRESHOLD = 0.95
DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_HIT_BOOST = 0.1


class LongTermMemory:
    """长期记忆：向量库里的结构化事实。规格见模块文档字符串。"""

    def __init__(
        self,
        embedder,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        client=None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        hit_boost: float = DEFAULT_HIT_BOOST,
    ) -> None:
        if not 0.0 < dedup_threshold <= 1.0:
            raise ValueError(f"dedup_threshold 必须在 (0, 1] 之间，收到的是 {dedup_threshold}")
        if half_life_days <= 0:
            raise ValueError(f"half_life_days 必须大于 0，收到的是 {half_life_days}")
        if hit_boost < 0:
            raise ValueError(f"hit_boost 不能为负数，收到的是 {hit_boost}")

        self.embedder = embedder
        self.collection_name = collection_name
        self.dedup_threshold = dedup_threshold
        self.half_life_days = half_life_days
        self.hit_boost = hit_boost

        import chromadb

        if client is not None:
            self._client = client
        elif persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=str(Path(persist_dir)))

        # 不传 embedding_function，避免 Chroma 偷偷启用英文默认模型。
        # 显式使用 cosine，后续可用 similarity = 1 - distance。
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    @staticmethod
    def _metadata_to_fact(fact_id: str, metadata: dict, document: str | None = None) -> dict:
        """把 Chroma metadata 还原成对外的事实字典。"""
        source_id = int(metadata.get("source_message_id", -1))
        return {
            "id": fact_id,
            "text": document or metadata.get("text", ""),
            "subject": metadata.get("subject", ""),
            "predicate": metadata.get("predicate", ""),
            "object": metadata.get("object", ""),
            "user_id": metadata.get("user_id", "default"),
            "confidence": float(metadata.get("confidence", 1.0)),
            "source_message_id": None if source_id == -1 else source_id,
            "created_at": metadata.get("created_at", ""),
            "updated_at": metadata.get("updated_at", ""),
            "invalid_at": metadata.get("invalid_at") or None,
            "hit_count": int(metadata.get("hit_count", 0)),
            "last_hit_at": metadata.get("last_hit_at") or None,
        }

    @staticmethod
    def _fact_metadata(*, subject: str, predicate: str, object: str, user_id: str,
                       confidence: float, source_message_id: int | None,
                       created_at: str, updated_at: str, invalid_at: str = "",
                       hit_count: int = 0, last_hit_at: str = "") -> dict:
        """Chroma metadata 不接受 None，用 -1 和空字符串表达缺失值。"""
        return {
            "subject": subject, "predicate": predicate, "object": object,
            "user_id": user_id, "confidence": float(confidence),
            "source_message_id": -1 if source_message_id is None else int(source_message_id),
            "created_at": created_at, "updated_at": updated_at,
            "invalid_at": invalid_at, "is_valid": invalid_at == "",
            "hit_count": int(hit_count), "last_hit_at": last_hit_at,
        }

    def _all_records(self, *, user_id: str | None = None) -> list[dict]:
        """读取匹配记录；当前个人项目规模小，优先保持实现直观。"""
        where = {"user_id": user_id} if user_id is not None else None
        result = self._collection.get(where=where, include=["metadatas", "documents"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])
        return [
            self._metadata_to_fact(
                fact_id, metadatas[index] or {},
                documents[index] if index < len(documents) else None,
            )
            for index, fact_id in enumerate(ids)
        ]

    # ==================================================================
    # 你的任务从这里开始
    # ==================================================================

    def add(
        self,
        predicate: str,
        object: str,
        *,
        subject: str = "user",
        user_id: str = "default",
        confidence: float = 1.0,
        source_message_id: int | None = None,
        now: datetime | None = None,
    ) -> dict:
        """添加一条事实。D3 完成基础写入；D4 在此基础上加入去重。"""
        moment = now or utc_now()
        timestamp = to_iso(moment)
        text = f"{subject} {predicate} {object}"
        embedding = self.embedder.embed_documents([text])[0]
        fact_id = uuid.uuid4().hex
        metadata = self._fact_metadata(
            subject=subject, predicate=predicate, object=object, user_id=user_id,
            confidence=confidence, source_message_id=source_message_id,
            created_at=timestamp, updated_at=timestamp,
        )
        self._collection.add(
            ids=[fact_id], embeddings=[embedding], documents=[text], metadatas=[metadata]
        )
        return {"id": fact_id, "action": "created", "similarity": 0.0}

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        user_id: str = "default",
        include_invalid: bool = False,
        record_hits: bool = True,
        now: datetime | None = None,
    ) -> list[dict]:
        """语义检索。规格见【S3】。"""
        if top_k <= 0:
            return []
        records = self._all_records(user_id=user_id)
        candidates = [record for record in records if include_invalid or record["invalid_at"] is None]
        if not candidates:
            return []
        query_vector = self.embedder.embed_query(query)
        raw = self._collection.get(ids=[record["id"] for record in candidates], include=["embeddings"])
        vectors_by_id = dict(zip(raw["ids"], raw.get("embeddings", [])))
        norm_q = math.sqrt(sum(float(value) ** 2 for value in query_vector))
        scored: list[dict] = []
        for fact in candidates:
            vector = vectors_by_id.get(fact["id"])
            if vector is None:
                continue
            norm_v = math.sqrt(sum(float(value) ** 2 for value in vector))
            dot = sum(float(left) * float(right) for left, right in zip(query_vector, vector))
            similarity = dot / (norm_q * norm_v) if norm_q and norm_v else 0.0
            similarity = max(0.0, min(1.0, similarity))
            fact.update({"similarity": similarity, "decay": 1.0, "score": similarity})
            scored.append(fact)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def get(self, fact_id: str) -> dict | None:
        """按 id 取一条事实。规格见【S4】。"""
        result = self._collection.get(ids=[fact_id], include=["metadatas", "documents"])
        if not result.get("ids"):
            return None
        metadata = (result.get("metadatas") or [{}])[0] or {}
        document = (result.get("documents") or [None])[0]
        return self._metadata_to_fact(result["ids"][0], metadata, document)

    def invalidate(self, fact_id: str, *, now: datetime | None = None) -> bool:
        """把事实标记为失效。规格见【S5】。"""
        raise NotImplementedError("【S5】请实现 invalidate()")

    def delete(self, fact_id: str) -> bool:
        """物理删除一条事实。规格见【S6】。"""
        raise NotImplementedError("【S6】请实现 delete()")

    def delete_user(self, user_id: str) -> int:
        """删除某个用户的全部事实。规格见【S7】。"""
        raise NotImplementedError("【S7】请实现 delete_user()")

    def count(self, *, user_id: str | None = None, include_invalid: bool = False) -> int:
        """统计事实条数。规格见【S8】。"""
        records = self._all_records(user_id=user_id)
        return len(records) if include_invalid else sum(
            1 for record in records if record["invalid_at"] is None
        )

    def clear(self) -> None:
        """清空所有事实。规格见【S9】。"""
        ids = self._collection.get(include=[]).get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    def __repr__(self) -> str:
        """规格见【S10】。"""
        return (
            f"<LongTermMemory {self.count()} 条事实 / 集合={self.collection_name}"
            f" / 半衰期={self.half_life_days:g}天>"
        )


# ====================================================================
# 进阶任务
# ====================================================================
#
# 【进阶 1】把命中记录改成"批量延迟写"
#     record_hits=True 时每次检索都要写库，读变成了读写。
#     改造思路：把命中先记在内存里（一个 dict: fact_id → 命中次数），
#     每积累 N 次或每隔 T 秒再批量刷进 Chroma。
#     注意要想清楚：进程崩溃时丢失的命中记录，可以接受吗？
#
# 【进阶 2】实现混合检索（向量 + 关键词）
#     纯向量检索对**专有名词、人名、编号**效果不好
#     （例如查 "豆豆" 可能召回一堆不相关的句子）。
#     思路：同时做一次关键词匹配（哪怕是简单的子串匹配），
#     然后用 RRF（Reciprocal Rank Fusion）融合两个排名：
#         score = 1/(k + rank_vector) + 1/(k + rank_keyword)
#     这是一个值得做的对比实验：量化"纯向量"和"混合检索"的召回差异。
#
# 【进阶 3】给检索加"最低相似度门槛"
#     现在的 search 无论多不相关都会返回 top_k 条。
#     实际使用中，如果最高相似度只有 0.2，那这些"记忆"全是噪声，
#     注入 prompt 只会干扰模型。
#     加一个 min_similarity 参数（默认 0），低于它的事实直接过滤掉。
#     想清楚默认值该是多少 —— 这需要你跑数据来定，不能拍脑袋。
#
# 【进阶 4】把"冲突消解"的半成品搭起来
#     docs/记忆架构设计.md 里说：相似度在 0.75~0.95 之间时应该交给 LLM 判断
#     "重复 / 补充 / 冲突"。这一层第 10 周才做，但现在可以先把
#     接口留出来：add() 在 similarity 落在中间带时返回
#     action="needs_review"，并把候选事实 id 一起返回。
#     这样将来接入 LLM 时不用改动调用方。
