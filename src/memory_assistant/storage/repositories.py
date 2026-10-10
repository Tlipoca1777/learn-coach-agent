r"""
仓储层（Repository）
================================================================

职责：把"数据怎么存"和"业务怎么用"分开。

--------------------------------------------------------------------
为什么要有这一层？为什么不直接在业务代码里写 SQL？
--------------------------------------------------------------------
假设你的 CLI 里直接写：
    conn.execute("INSERT INTO messages(...) VALUES(...)")

那么：
    · SQL 散落在十几个文件里，改表结构时要全部找出来改
    · 想写单元测试，就得先造一个数据库
    · 想换成 PostgreSQL，等于重写整个项目

仓储模式的做法是：**所有 SQL 只出现在这一层**。
业务代码只说"保存一条消息"，不关心它是存 SQLite 还是存文件。

这在面试里叫"关注点分离"（Separation of Concerns），
是很基础的架构能力，但很多人写小项目时完全不分层。

--------------------------------------------------------------------
小知识：Repository 还是 DAO？
--------------------------------------------------------------------
两者很像。粗略区别：
    DAO（Data Access Object）—— 偏技术，直接映射"哪张表"，
                                 方法名常是 insert / update / delete
    Repository               —— 偏业务，方法名是业务语言，
                                 比如 for_prompt（取出发给模型的消息）

我们用的是 Repository，所以你会看到 `for_prompt` 这种业务化的方法名。

--------------------------------------------------------------------
怎么用
--------------------------------------------------------------------
    from memory_assistant.storage import Database, Store

    with Database("data/assistant.db") as db:
        db.initialize()
        store = Store(db)

        sid = store.sessions.create(title="第一次对话")
        store.messages.append(sid, "user", "你好")
        store.messages.append(sid, "assistant", "你好！")

        print(store.messages.for_prompt(sid))
        # [{'role': 'user', 'content': '你好'},
        #  {'role': 'assistant', 'content': '你好！'}]
"""

import math
import uuid

from memory_assistant.storage.database import Database, utc_now_iso

# 合法的消息角色。
# ⚠️ 这里和数据库里的 CHECK 约束是重复的，为什么还要在 Python 里再查一次？
#    因为数据库报的错是 "CHECK constraint failed"，
#    而 Python 里报的错可以是 "role 必须是 user/assistant，你传了 'robot'"。
#    给开发者看清晰的中文报错，比让他去猜 SQL 错误快得多。
#    这叫"快速失败 + 友好报错"，两层校验不冲突。
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


# ====================================================================
# 会话
# ====================================================================
class SessionRepository:
    """会话（一次连续的对话）的增删查。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, session_id: str | None = None, title: str | None = None) -> str:
        """
        创建一个新会话，返回它的 id。

        参数 session_id 留空时会自动生成一个短的随机 id。
        自动生成的 id 用 uuid4().hex[:12] —— 12 位十六进制，
        碰撞概率极低，而且比完整 uuid 好读。
        """
        session_id = session_id or uuid.uuid4().hex[:12]
        now = utc_now_iso()
        self.db.conn.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        self.db.conn.commit()
        return session_id

    def get(self, session_id: str) -> dict | None:
        """按 id 查会话，不存在返回 None。"""
        row = self.db.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_or_create(self, session_id: str, title: str | None = None) -> str:
        """
        有就返回，没有就创建。

        这个方法是给 CLI 用的：用户可能用 --session abc 指定一个会话，
        第一次用的时候它还不存在，我们希望自动建好而不是报错。
        """
        if self.get(session_id) is None:
            self.create(session_id=session_id, title=title)
        return session_id

    def touch(self, session_id: str) -> None:
        """更新会话的最后活动时间。"""
        self.db.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (utc_now_iso(), session_id),
        )

    def list_recent(self, limit: int = 10) -> list[dict]:
        """列出最近活动的会话，用于给用户展示"继续哪个对话"。"""
        rows = self.db.conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        """
        删除会话。

        注意：它的消息和摘要会被**自动删掉**，
        因为建表时写了 ON DELETE CASCADE。
        这里能体现出【坑 1】（外键默认关闭）的重要性 ——
        如果没开外键，这个 delete 只会删掉会话本身，
        留下几百条孤儿消息永远查不到也删不掉。
        """
        cursor = self.db.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.db.conn.commit()
        return cursor.rowcount > 0


# ====================================================================
# 消息
# ====================================================================
class MessageRepository:
    """消息的增删查。这是短期记忆的持久化载体。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int | None = None,
    ) -> int:
        """
        追加一条消息，返回它的自增 id。

        同时会更新会话的 updated_at —— 这两步必须在同一个事务里，
        否则可能出现"消息存进去了但会话时间没更新"的不一致状态。
        """
        if role not in VALID_ROLES:
            raise ValueError(
                f"非法的 role: {role!r}\n"
                f"只能是：{', '.join(sorted(VALID_ROLES))}"
            )
        if not content:
            raise ValueError("content 不能为空")

        # `with self.db.conn:` 是一个事务块：
        # 正常结束时自动 commit，抛异常时自动 rollback。
        # 用它是为了不忘记 commit，也为了两条语句的原子性。
        with self.db.conn:
            cursor = self.db.conn.execute(
                "INSERT INTO messages(session_id, role, content, token_count, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (session_id, role, content, token_count, utc_now_iso()),
            )
            self.db.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), session_id),
            )
        return int(cursor.lastrowid)

    def list_by_session(
        self,
        session_id: str,
        limit: int | None = None,
        after_id: int = 0,
    ) -> list[dict]:
        """
        取某个会话的消息，按时间从旧到新排列。

        参数：
            limit     最多取几条。**取的是最新的 N 条**，不是最旧的 N 条。
            after_id  只取 id 大于它的消息。
                      这是给中期摘要用的：摘要覆盖到 id=50 了，
                      就只要 id>50 的消息，避免重复加载。
        """
        if limit is None:
            rows = self.db.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC",
                (session_id, after_id),
            ).fetchall()
            return [dict(row) for row in rows]

        # ---------- 「取最新 N 条但保持正序」的经典写法 ----------
        # 不能直接写 ORDER BY id ASC LIMIT ? —— 那取到的是最旧的 N 条。
        # 正确做法是先在子查询里倒序取 N 条，再在外层正序排列。
        # 这个模式在实际开发里出现频率极高。
        rows = self.db.conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM messages
                WHERE session_id = ? AND id > ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, after_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def for_prompt(self, session_id: str, after_id: int = 0) -> list[dict]:
        """
        取出发给大模型的消息列表，格式正好是 API 需要的样子。

        这个方法就是 Repository 和 DAO 的区别：
        它返回的不是"数据库行"，而是"业务需要的数据结构"。
        数据库行里还有 id、created_at、token_count 这些字段，
        但对模型来说只有 role 和 content 有意义，多余的字段要剥掉。
        """
        rows = self.list_by_session(session_id, after_id=after_id)
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def count(self, session_id: str) -> int:
        """这个会话有多少条消息。"""
        row = self.db.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])

    def last_id(self, session_id: str) -> int:
        """
        最后一条消息的 id。会话为空时返回 0。

        什么时候用？写摘要的时候要记录"我覆盖到哪了"：
            covered_until = messages.last_id(session_id)
        """
        row = self.db.conn.execute(
            "SELECT MAX(id) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        # MAX() 在没有匹配行时返回 NULL，所以要兜底成 0
        return int(row["n"]) if row["n"] is not None else 0

    def delete_by_session(self, session_id: str) -> int:
        """删除某会话的全部消息，返回删了几条。"""
        cursor = self.db.conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        self.db.conn.commit()
        return cursor.rowcount

    def delete_older_than(self, session_id: str, keep_last: int) -> int:
        """
        只保留最新的 keep_last 条，其余删掉。

        为什么需要这个？
            因为窗口裁掉的消息如果不落库，重启后就没法重建完整的原始历史了。
            但库也不能无限膨胀，所以需要一个"清理旧数据"的手段。
            真实项目里通常是"归档到冷存储"而不是直接删，我们简化成删除。
        """
        rows = self.db.conn.execute(
            """
            DELETE FROM messages
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM (
                    SELECT id FROM messages WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                )
            )
            """,
            (session_id, session_id, keep_last),
        )
        self.db.conn.commit()
        return rows.rowcount


# ====================================================================
# 中期摘要
# ====================================================================
class SummaryRepository:
    """中期摘要的增删查。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        session_id: str,
        content: str,
        covered_until: int,
        token_count: int | None = None,
    ) -> int:
        """
        存一份新的摘要。

        参数 covered_until：
            这份摘要已经覆盖到哪条消息（含）。
            下次加载历史时，只要取 id > covered_until 的消息即可，
            不会重复把已经压缩过的内容再读一遍。
        """
        cursor = self.db.conn.execute(
            "INSERT INTO summaries(session_id, content, covered_until, token_count, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (session_id, content, covered_until, token_count, utc_now_iso()),
        )
        self.db.conn.commit()
        return int(cursor.lastrowid)

    def latest(self, session_id: str) -> dict | None:
        """取最新的那份摘要（我们用的是"递归增量压缩"，所以只需要最新的一份）。"""
        row = self.db.conn.execute(
            "SELECT * FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def history(self, session_id: str) -> list[dict]:
        """
        取出所有历史摘要。

        平时不用，但调试"摘要怎么一步步演化的"时非常有用 ——
        你可以看到第 1 份摘要说了什么，第 5 份摘要丢了哪些信息。
        这个能力在第 11 周做评测（摘要保真度）时会用到。
        """
        rows = self.db.conn.execute(
            "SELECT * FROM summaries WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_by_session(self, session_id: str) -> int:
        cursor = self.db.conn.execute(
            "DELETE FROM summaries WHERE session_id = ?", (session_id,)
        )
        self.db.conn.commit()
        return cursor.rowcount


class LearningRepository:
    """Persist learning topics, answer attempts and per-user mastery."""

    VALID_VERDICTS = frozenset({"correct", "partial", "wrong"})

    def __init__(self, db: Database) -> None:
        self.db = db

    def record_answer(
        self,
        *,
        topic_name: str,
        question: str,
        user_answer: str,
        verdict: str,
        score: float,
        feedback: str = "",
        user_id: str = "default",
    ) -> dict:
        """Atomically append an attempt and update the topic's mastery score."""
        topic_name = _required_text(topic_name, "topic_name")
        question = _required_text(question, "question")
        user_answer = _required_text(user_answer, "user_answer")
        user_id = _required_text(user_id, "user_id")
        if not isinstance(verdict, str):
            raise ValueError("verdict must be correct, partial, or wrong")
        verdict = verdict.strip().lower()
        if verdict not in self.VALID_VERDICTS:
            raise ValueError("verdict must be correct, partial, or wrong")
        if isinstance(score, bool):
            raise ValueError("score must be a number between 0 and 1")
        try:
            score = float(score)
        except (TypeError, ValueError) as error:
            raise ValueError("score must be a number between 0 and 1") from error
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be a number between 0 and 1")

        now = utc_now_iso()
        with self.db.conn:
            topic = self.db.conn.execute(
                "SELECT id FROM topics WHERE name = ?", (topic_name,)
            ).fetchone()
            if topic is None:
                cursor = self.db.conn.execute(
                    "INSERT INTO topics(name, display_name, category, created_at) "
                    "VALUES(?, ?, 'general', ?)",
                    (topic_name, topic_name, now),
                )
                topic_id = int(cursor.lastrowid)
            else:
                topic_id = int(topic["id"])

            mastery_row = self.db.conn.execute(
                "SELECT * FROM topic_mastery WHERE topic_id = ? AND user_id = ?",
                (topic_id, user_id),
            ).fetchone()
            old_mastery = float(mastery_row["mastery"]) if mastery_row else 0.3
            old_count = int(mastery_row["attempt_count"]) if mastery_row else 0
            if verdict == "correct":
                mastery = old_mastery + (1.0 - old_mastery) * 0.3
            elif verdict == "partial":
                mastery = old_mastery + (1.0 - old_mastery) * 0.15
            else:
                mastery = old_mastery * 0.5

            self.db.conn.execute(
                "INSERT INTO attempts(topic_id, user_id, question, user_answer, "
                "verdict, score, feedback, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (topic_id, user_id, question, user_answer, verdict, score,
                 _optional_text(feedback, "feedback"), now),
            )
            self.db.conn.execute(
                "INSERT INTO topic_mastery(topic_id, user_id, mastery, attempt_count, updated_at) "
                "VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(topic_id, user_id) DO UPDATE SET "
                "mastery=excluded.mastery, attempt_count=excluded.attempt_count, "
                "updated_at=excluded.updated_at",
                (topic_id, user_id, mastery, old_count + 1, now),
            )

        return {
            "topic_name": topic_name,
            "mastery": round(mastery, 4),
            "attempt_count": old_count + 1,
            "status": _mastery_status(mastery),
        }

    def get_weak_topics(self, *, limit: int = 5, user_id: str = "default") -> list[dict]:
        """Return a user's lowest-mastery topics first, with latest feedback."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer between 1 and 50")
        try:
            limit = int(limit)
        except (TypeError, ValueError) as error:
            raise ValueError("limit must be an integer between 1 and 50") from error
        if not 1 <= limit <= 50:
            raise ValueError("limit must be an integer between 1 and 50")
        user_id = _required_text(user_id, "user_id")
        rows = self.db.conn.execute(
            "SELECT t.name AS topic_name, t.display_name, t.category, "
            "m.mastery, m.attempt_count, "
            "(SELECT a.feedback FROM attempts a WHERE a.topic_id=t.id AND a.user_id=m.user_id "
            "ORDER BY a.id DESC LIMIT 1) AS latest_feedback "
            "FROM topic_mastery m JOIN topics t ON t.id=m.topic_id "
            "WHERE m.user_id = ? AND m.mastery < 0.4 "
            "ORDER BY m.mastery ASC, m.attempt_count DESC, t.name ASC "
            "LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "topic_name": row["topic_name"],
                "display_name": row["display_name"],
                "category": row["category"],
                "mastery": round(float(row["mastery"]), 4),
                "attempt_count": int(row["attempt_count"]),
                "status": _mastery_status(float(row["mastery"])),
                "latest_feedback": row["latest_feedback"],
            }
            for row in rows
        ]


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip() or None


def _mastery_status(mastery: float) -> str:
    if mastery < 0.4:
        return "weak"
    if mastery < 0.8:
        return "learning"
    return "mastered"


# ====================================================================
# 把三个仓储打包，方便使用
# ====================================================================
class Store:
    """
    一组仓储的集合。

    为什么要有它？因为业务代码里同时用三个仓储很常见，
    每次都手写三个实例太啰嗦：

        store = Store(db)
        store.sessions.create()
        store.messages.append(...)
        store.summaries.latest(...)

    顺便说一句：`db.conn` 暴露出去让别处直接执行 SQL 是"漏抽象"，
    但我们刻意保留它 —— 因为总会有仓储没覆盖到的临时查询需求。
    工程上这叫"实用主义"，不用为了纯粹性把自己憋死。
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.sessions = SessionRepository(db)
        self.messages = MessageRepository(db)
        self.summaries = SummaryRepository(db)
        self.learning = LearningRepository(db)
        # TODO（第 5 周）：facts 仓储（需要配合 Chroma 向量库）
        # TODO（第 10 周）：profiles 仓储（用户画像）

    def delete_session(self, session_id: str) -> None:
        """
        彻底删除一个会话及其全部痕迹。

        这是"被遗忘权"的实现：用户说"忘掉这一切"，就要真的删干净。
        因为建表时配了级联删除，这里只需要删 sessions 一行，
        messages 和 summaries 会自动跟着走。
        """
        self.sessions.delete(session_id)

    def stats(self, session_id: str) -> dict:
        """返回一个会话的统计信息，CLI 的 /status 命令会用。"""
        return {
            "messages": self.messages.count(session_id),
            "last_message_id": self.messages.last_id(session_id),
            "has_summary": self.summaries.latest(session_id) is not None,
        }

    def __repr__(self) -> str:
        return f"<Store {self.db}>"
