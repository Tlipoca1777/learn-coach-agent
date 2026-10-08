r"""
存储层验收测试
================================================================

这些测试覆盖已经实现好的数据库层和仓储层，
和你的 `ShortTermMemory` 作业没有关系 —— 你可以先不管它们。

运行：
    .\.venv\Scripts\python.exe -m pytest tests/test_storage.py -v

--------------------------------------------------------------------
这里演示了几个值得学的测试技巧
--------------------------------------------------------------------
1. **用内存数据库**（`:memory:`）
   每条测试都新建一个空的库，测试之间完全隔离，
   而且不产生任何文件，速度快到看不出延迟。

2. **fixture 管理生命周期**
   `@pytest.fixture` 用来准备测试需要的东西。
   yield 之前的代码是"准备"，之后的代码是"清理"。
   这样每条测试都能拿到一个干净的库，不用自己写 try/finally。

3. **测边界和异常，而不只测正常流程**
   注意下面有多少条测试是在验证"错误输入会被正确拒绝"。
   真实项目里，异常路径的测试比正常路径更重要 ——
   因为正常路径你手动跑一次就发现了，异常路径往往要等线上出事。
"""

import sqlite3

import pytest

from memory_assistant.storage import (
    MEMORY_DB,
    SCHEMA_VERSION,
    Database,
    Store,
    utc_now_iso,
)


# ====================================================================
# fixture：给每条测试一个干净的数据库
# ====================================================================


@pytest.fixture
def db():
    """内存数据库，建好表，测试结束后关闭。"""
    database = Database(MEMORY_DB)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def store(db):
    """打包好的仓储集合。"""
    return Store(db)


@pytest.fixture
def session_id(store):
    """一个已经创建好的会话 id。"""
    return store.sessions.create(title="测试会话")


# ====================================================================
# 组 1：数据库初始化
# ====================================================================


def test_initialize_creates_all_tables(db):
    """建表语句应该把设计文档里的表都建出来。"""
    expected = {
        "sessions",
        "messages",
        "summaries",
        "facts",
        "profiles",
        "llm_calls",
        "meta",
    }
    assert expected.issubset(set(db.table_names()))


def test_initialize_is_idempotent(db):
    """重复调用 initialize() 不应该报错（用了 IF NOT EXISTS）。"""
    db.initialize()
    db.initialize()
    assert "messages" in db.table_names()


def test_foreign_keys_are_enabled(db):
    """
    【关键】外键必须被显式开启。

    SQLite 默认关闭外键约束，这是一个著名的大坑。
    没开的话，删掉会话后它的消息会变成永远访问不到的孤儿数据。
    """
    assert db.foreign_keys_enabled() is True


def test_memory_database_skips_wal(db):
    """内存数据库不支持 WAL，应该安静地跳过而不是报错。"""
    # 只要 initialize() 没抛异常就算通过
    assert db.journal_mode() in ("memory", "wal")


def test_file_database_enables_wal(tmp_path):
    """文件数据库应该开启 WAL 模式（读写不互相阻塞）。"""
    path = tmp_path / "assistant.db"
    with Database(path) as file_db:
        file_db.initialize()
        assert file_db.journal_mode() == "wal"


def test_database_creates_parent_directory(tmp_path):
    """数据库文件的父目录不存在时应该自动创建。"""
    path = tmp_path / "深层" / "目录" / "assistant.db"
    with Database(path) as file_db:
        file_db.initialize()
    assert path.exists()


def test_schema_version_mismatch_raises(tmp_path):
    """
    数据库版本和代码版本不一致时必须明确报错。

    否则你会遇到 "no such column" 这种极难排查的错误。
    """
    path = tmp_path / "mismatch.db"
    with Database(path) as file_db:
        file_db.initialize()
        file_db.conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
        file_db.conn.commit()

        with pytest.raises(RuntimeError, match="schema 版本不匹配"):
            file_db.initialize()


def test_schema_version_is_recorded(db):
    row = db.conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    assert int(row["value"]) == SCHEMA_VERSION


def test_utc_now_iso_format():
    """时间戳应该是 ISO 8601 且带 UTC 时区。"""
    text = utc_now_iso()
    assert "T" in text
    assert text.endswith("+00:00")


# ====================================================================
# 组 2：约束（异常路径）
# ====================================================================


def test_message_with_unknown_session_is_rejected(db):
    """往不存在的会话里插消息，必须被外键拦住。"""
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES('不存在', 'user', '你好', ?)",
            (utc_now_iso(),),
        )


def test_invalid_role_is_rejected_by_database(db, session_id):
    """数据库的 CHECK 约束是最后一道防线。"""
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES(?, '机器人', '你好', ?)",
            (session_id, utc_now_iso()),
        )


def test_invalid_role_raises_friendly_error(store, session_id):
    """但 Python 层应该先拦住，并给出比 SQL 错误友好得多的提示。"""
    with pytest.raises(ValueError, match="非法的 role"):
        store.messages.append(session_id, "机器人", "你好")


def test_empty_content_is_rejected(store, session_id):
    with pytest.raises(ValueError, match="不能为空"):
        store.messages.append(session_id, "user", "")


def test_duplicate_session_id_is_rejected(store):
    store.sessions.create(session_id="fixed")
    with pytest.raises(sqlite3.IntegrityError):
        store.sessions.create(session_id="fixed")


# ====================================================================
# 组 3：会话仓储
# ====================================================================


def test_create_session_returns_readable_id(store):
    sid = store.sessions.create()
    assert isinstance(sid, str)
    assert len(sid) == 12


def test_get_session(store, session_id):
    session = store.sessions.get(session_id)
    assert session is not None
    assert session["id"] == session_id
    assert session["title"] == "测试会话"


def test_get_missing_session_returns_none(store):
    assert store.sessions.get("不存在的会话") is None


def test_get_or_create(store):
    """不存在就建，存在就复用 —— CLI 里指定会话时用。"""
    sid = store.sessions.get_or_create("my-session")
    assert sid == "my-session"
    assert store.sessions.get("my-session") is not None

    # 再调一次不应该报错，也不应该新建
    before = len(store.sessions.list_recent(100))
    assert store.sessions.get_or_create("my-session") == "my-session"
    assert len(store.sessions.list_recent(100)) == before


def test_list_recent_orders_by_activity(store):
    """
    最近活动的会话排在前面。

    注意：这里靠 updated_at 排序，而时间戳精度是"秒"，
    所以同一个测试里连续创建的几个会话时间戳会一样，
    排序就不稳定。为了测试可靠，我们手动把时间戳改开。
    """
    store.sessions.create(session_id="aaa")
    store.sessions.create(session_id="bbb")
    store.sessions.create(session_id="ccc")

    # 手动设置不同的时间，避免"同秒排序不稳定"这个真实存在的坑
    for index, sid in enumerate(["aaa", "bbb", "ccc"]):
        store.db.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (f"2025-01-0{index + 1}T00:00:00+00:00", sid),
        )
    store.db.conn.commit()

    order = [s["id"] for s in store.sessions.list_recent(10)]
    assert order == ["ccc", "bbb", "aaa"]


def test_delete_session_returns_true_then_false(store, session_id):
    assert store.sessions.delete(session_id) is True
    assert store.sessions.delete(session_id) is False


def test_delete_session_cascades_to_messages(store, session_id):
    """
    【关键】删会话必须连带删掉它的消息和摘要。

    如果这条挂了，说明外键没生效（PRAGMA foreign_keys = ON 没执行），
    数据库里会积累大量永远无法访问的孤儿数据。
    """
    store.messages.append(session_id, "user", "你好")
    store.messages.append(session_id, "assistant", "你好！")
    store.summaries.add(session_id, "一段摘要", covered_until=2)

    assert store.messages.count(session_id) == 2

    store.sessions.delete(session_id)

    assert store.messages.count(session_id) == 0
    assert store.summaries.latest(session_id) is None
    assert store.db.count("messages") == 0
    assert store.db.count("summaries") == 0


# ====================================================================
# 组 4：消息仓储
# ====================================================================


def test_append_returns_increasing_ids(store, session_id):
    first = store.messages.append(session_id, "user", "第一句")
    second = store.messages.append(session_id, "assistant", "第二句")
    assert second > first


def test_append_updates_session_timestamp(store, session_id):
    """追加消息要顺带更新会话的活动时间，否则"最近会话"排序是错的。"""
    store.db.conn.execute(
        "UPDATE sessions SET updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
        (session_id,),
    )
    store.db.conn.commit()

    store.messages.append(session_id, "user", "你好")

    updated = store.sessions.get(session_id)["updated_at"]
    assert updated != "2000-01-01T00:00:00+00:00"


def test_list_by_session_is_chronological(store, session_id):
    store.messages.append(session_id, "user", "一")
    store.messages.append(session_id, "assistant", "二")
    store.messages.append(session_id, "user", "三")

    contents = [m["content"] for m in store.messages.list_by_session(session_id)]
    assert contents == ["一", "二", "三"]


def test_list_with_limit_returns_latest_n_in_order(store, session_id):
    """
    【容易写错】limit 应该是"最新的 N 条"，而且返回时仍是正序。

    如果直接 ORDER BY id ASC LIMIT 2，你会拿到最旧的两条 —— 那就错了。
    """
    for index in range(1, 6):
        store.messages.append(session_id, "user", f"第{index}句")

    contents = [m["content"] for m in store.messages.list_by_session(session_id, limit=2)]
    assert contents == ["第4句", "第5句"]


def test_after_id_filters_older_messages(store, session_id):
    """after_id 是给中期摘要用的：只要 id 大于已覆盖位置的消息。"""
    first = store.messages.append(session_id, "user", "被摘要覆盖的")
    store.messages.append(session_id, "user", "摘要之后的")

    rows = store.messages.list_by_session(session_id, after_id=first)
    assert [m["content"] for m in rows] == ["摘要之后的"]


def test_after_id_combines_with_limit(store, session_id):
    for index in range(1, 7):
        store.messages.append(session_id, "user", f"第{index}句")

    rows = store.messages.list_by_session(session_id, limit=2, after_id=2)
    assert [m["content"] for m in rows] == ["第5句", "第6句"]


def test_for_prompt_strips_database_fields(store, session_id):
    """for_prompt 只保留 role 和 content，可以直接喂给模型。"""
    store.messages.append(session_id, "user", "你好")
    store.messages.append(session_id, "assistant", "你好！")

    prompt = store.messages.for_prompt(session_id)
    assert prompt == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]


def test_count_and_last_id(store, session_id):
    assert store.messages.count(session_id) == 0
    assert store.messages.last_id(session_id) == 0

    last = store.messages.append(session_id, "user", "你好")
    assert store.messages.count(session_id) == 1
    assert store.messages.last_id(session_id) == last


def test_delete_older_than_keeps_newest(store, session_id):
    for index in range(1, 8):
        store.messages.append(session_id, "user", f"第{index}句")

    removed = store.messages.delete_older_than(session_id, keep_last=3)

    assert removed == 4
    remaining = [m["content"] for m in store.messages.list_by_session(session_id)]
    assert remaining == ["第5句", "第6句", "第7句"]


def test_delete_older_than_is_noop_when_few_messages(store, session_id):
    store.messages.append(session_id, "user", "只有一条")
    assert store.messages.delete_older_than(session_id, keep_last=10) == 0
    assert store.messages.count(session_id) == 1


def test_messages_are_isolated_between_sessions(store):
    """两个会话的消息不能互相串。"""
    sid_a = store.sessions.create()
    sid_b = store.sessions.create()

    store.messages.append(sid_a, "user", "A的消息")
    store.messages.append(sid_b, "user", "B的消息")

    assert store.messages.count(sid_a) == 1
    assert [m["content"] for m in store.messages.list_by_session(sid_a)] == ["A的消息"]


def test_delete_by_session_removes_all_messages(store, session_id):
    store.messages.append(session_id, "user", "一")
    store.messages.append(session_id, "user", "二")

    assert store.messages.delete_by_session(session_id) == 2
    assert store.messages.count(session_id) == 0


# ====================================================================
# 组 5：摘要仓储
# ====================================================================


def test_summary_add_and_latest(store, session_id):
    assert store.summaries.latest(session_id) is None

    store.summaries.add(session_id, "第一版摘要", covered_until=4)
    store.summaries.add(session_id, "第二版摘要", covered_until=8)

    latest = store.summaries.latest(session_id)
    assert latest["content"] == "第二版摘要"
    assert latest["covered_until"] == 8


def test_summary_history_preserves_evolution(store, session_id):
    """
    历史摘要要全部保留。

    为什么不覆盖旧的？因为要能调试"摘要怎么一步步演化的"——
    第 5 版丢了信息时，你得能翻回去看是哪一步丢的。
    """
    for version in range(1, 4):
        store.summaries.add(session_id, f"第{version}版", covered_until=version * 2)

    history = store.summaries.history(session_id)
    assert [s["content"] for s in history] == ["第1版", "第2版", "第3版"]


def test_summary_covered_until_works_with_after_id(store, session_id):
    """
    摘要和消息要能配合：摘要覆盖到第 3 条，就只要它之后的消息。

    这是阶段 2「加载历史」的核心逻辑，先在这里验证数据层是通的。
    """
    for index in range(1, 6):
        store.messages.append(session_id, "user", f"第{index}句")

    # 取第 3 条消息的 id 作为"摘要已覆盖到哪"
    all_messages = store.messages.list_by_session(session_id)
    covered = all_messages[2]["id"]
    store.summaries.add(session_id, "前三条的摘要", covered_until=covered)

    remaining = store.messages.for_prompt(session_id, after_id=covered)
    assert [m["content"] for m in remaining] == ["第4句", "第5句"]


# ====================================================================
# 组 6：持久化（跨"重启"）
# ====================================================================


def test_data_survives_reopen(tmp_path):
    """
    关掉数据库再打开，数据还在 —— 这就是"跨会话记忆"的基础。

    这个测试用的是文件数据库，模拟"程序退出再启动"。
    """
    path = tmp_path / "persist.db"

    # 第一次运行
    with Database(path) as db:
        db.initialize()
        store = Store(db)
        sid = store.sessions.create(title="第一次对话")
        store.messages.append(sid, "user", "我叫小明")
        store.messages.append(sid, "assistant", "记住了")

    # 模拟重启：全新连接
    with Database(path) as db:
        db.initialize()
        store = Store(db)

        assert store.sessions.get(sid)["title"] == "第一次对话"
        prompt = store.messages.for_prompt(sid)
        assert prompt == [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "记住了"},
        ]


# ====================================================================
# 组 7：Store 的便捷方法
# ====================================================================


def test_stats(store, session_id):
    stats = store.stats(session_id)
    assert stats["messages"] == 0
    assert stats["last_message_id"] == 0
    assert stats["has_summary"] is False

    store.messages.append(session_id, "user", "你好")
    store.summaries.add(session_id, "摘要", covered_until=1)

    stats = store.stats(session_id)
    assert stats["messages"] == 1
    assert stats["last_message_id"] == 1
    assert stats["has_summary"] is True


def test_delete_session_wipes_everything(store, session_id):
    """被遗忘权：删掉会话后，消息和摘要一条都不剩。"""
    store.messages.append(session_id, "user", "敏感信息")
    store.summaries.add(session_id, "含敏感信息的摘要", covered_until=1)

    store.delete_session(session_id)

    assert store.sessions.get(session_id) is None
    assert store.db.count("messages") == 0
    assert store.db.count("summaries") == 0
