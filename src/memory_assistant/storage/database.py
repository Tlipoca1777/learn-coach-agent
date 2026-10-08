r"""
SQLite 数据库层
================================================================

职责：管理数据库连接、建表、以及那些"所有仓储都要用"的公共逻辑。

--------------------------------------------------------------------
为什么用 SQLite？
--------------------------------------------------------------------
    · Python 标准库自带，不用装任何东西，也不用起服务
    · 整个数据库就是一个文件，拷走就能备份，适合学习和个人项目
    · 支持事务、外键、索引，是真数据库，不是玩具
    · 单机场景性能完全够用（写入可达每秒上万条）

将来要换成 PostgreSQL 也很容易，只要改这个文件。
这也是为什么要专门分出一层：**隔离变化**。

--------------------------------------------------------------------
这个文件里有三个"必须知道"的 SQLite 坑
--------------------------------------------------------------------
【坑 1】外键默认是**关闭**的
    SQLite 出于历史兼容考虑，外键约束默认不生效。
    你必须对**每一个连接**执行 `PRAGMA foreign_keys = ON`。
    不设置的话，你删掉一个会话，它的消息不会跟着删，数据库里会留下
    永远访问不到的垃圾数据（这叫"孤儿记录"）。
    这是一个非常经典、非常隐蔽的坑。

【坑 2】连接不能跨线程共享
    sqlite3 的连接对象默认只能在创建它的线程里用。
    多线程里用会抛 ProgrammingError。
    后面第 8 周做 FastAPI 时（它是多线程的）会碰到这个问题，
    到时候我们的做法是"每个请求开一个连接"。

【坑 3】默认是"自动提交"模式，事务边界很模糊
    我们显式用 `with conn:` 来划定事务边界，
    这样一批操作要么全部成功，要么全部回滚。

--------------------------------------------------------------------
怎么用
--------------------------------------------------------------------
    from memory_assistant.config import Config
    from memory_assistant.storage import Database

    config = Config.from_env()
    db = Database(config.data_dir / "assistant.db")
    db.initialize()          # 建表，可重复执行

    # 或者用 with 语句，离开时自动关闭
    with Database(path) as db:
        db.initialize()
        ...

    自检：
        .\.venv\Scripts\python.exe scripts\check_storage.py
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# 内存数据库的特殊路径。写测试时用它，速度极快且不产生文件。
MEMORY_DB = ":memory:"

# 当前 schema 版本。
# 每次修改表结构就 +1，然后在 _migrate() 里写对应的升级逻辑。
# 为什么需要这个？因为真实项目里数据库是有数据的，
# 你不能直接删库重建，必须能"平滑升级"。
SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    """
    返回当前 UTC 时间的 ISO 8601 字符串，例如 '2025-01-15T08:30:00+00:00'。

    为什么统一用 UTC 而不是本地时间？
        因为本地时间会随夏令时、时区设置变化，
        存进数据库之后你根本不知道那条记录到底是什么时候的。
        **存储一律用 UTC，只在显示给用户时才转成当地时间。**
        这是几乎所有后端系统的标准做法。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ====================================================================
# 建表语句
# ====================================================================
# 完整 schema 见 docs/记忆架构设计.md 第 7 节。
#
# 说明：这里一次性建好了所有的表，包括后面几周才会用到的 facts / profiles。
#      好处是 schema 只有一个地方定义，不用做多次迁移。
#      坏处是有几张表暂时是空的 —— 这在真实项目里很常见，
#      "表设计先行" 是正常的工程实践。
#
# CREATE TABLE IF NOT EXISTS 保证重复执行不会报错，
# 所以 initialize() 可以随便调用多少次。

SCHEMA_SQL = """
-- ============ 会话 ============
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- ============ 消息（短期记忆的持久化）============
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    content     TEXT NOT NULL,
    token_count INTEGER,
    created_at  TEXT NOT NULL
);

-- 索引：我们绝大部分查询都是"取某个会话的最近 N 条消息"，
-- 所以按 (session_id, id) 建索引，让查询走索引而不是全表扫描。
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

-- ============ 中期摘要 ============
CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    -- 这份摘要已经覆盖到哪条消息（含）。
    -- 加载历史时只要取 id > covered_until 的消息，就不会重复加载。
    covered_until INTEGER NOT NULL DEFAULT 0,
    token_count   INTEGER,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON summaries(session_id, id);

-- ============ 长期事实（向量存在 Chroma，结构化字段在这里）============
CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT NOT NULL DEFAULT 'default',
    subject           TEXT NOT NULL DEFAULT 'user',
    predicate         TEXT NOT NULL,
    object            TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 1.0,
    source_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    -- 冲突消解后，旧事实证明失效的时间（NULL 表示仍然有效）
    invalid_at        TEXT,
    expires_at        TEXT,
    hit_count         INTEGER NOT NULL DEFAULT 0,
    last_hit_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_valid
    ON facts(user_id, invalid_at, expires_at);

-- ============ 用户画像 ============
CREATE TABLE IF NOT EXISTS profiles (
    user_id     TEXT PRIMARY KEY,
    data        TEXT NOT NULL,      -- JSON 字符串
    updated_at  TEXT NOT NULL
);

-- ============ 调用日志（成本与可观测）============
CREATE TABLE IF NOT EXISTS llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    -- 用途：chat / summarize / extract / resolve
    -- 有了它才能按用途拆分成本，看出"抽取事实"到底花了多少钱
    purpose             TEXT,
    model               TEXT,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    latency_ms          INTEGER,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created
    ON llm_calls(created_at);

-- ============ 元信息（记录 schema 版本）============
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """一个 SQLite 数据库连接的薄封装。"""

    def __init__(self, path: str | Path = MEMORY_DB, *, verbose: bool = False) -> None:
        """
        参数 path：
            数据库文件路径，或者 ":memory:"（内存数据库，进程退出就没了）。
            内存数据库特别适合写测试：快，而且不会留下垃圾文件。
        """
        self.path = str(path)
        self.verbose = verbose

        # 只有文件数据库才需要建父目录，内存数据库不需要
        if self.path != MEMORY_DB:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # 建立连接
        # ============================================================
        self.conn = sqlite3.connect(self.path)

        # row_factory = sqlite3.Row 之后，查询结果可以像字典一样用列名访问：
        #     row["role"]        ← 推荐，可读性好
        # 而不是只能按下标：
        #     row[2]             ← 可读性差，加一列就全错位
        self.conn.row_factory = sqlite3.Row

        # 【坑 1】外键必须每个连接单独开启，SQLite 默认是关闭的
        self.conn.execute("PRAGMA foreign_keys = ON")

        if self.verbose:
            self.conn.set_trace_callback(print)   # 打印每条实际执行的 SQL

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize(self) -> "Database":
        """
        建表 + 设置。可以重复调用（幂等）。

        顺便做一件重要的事：开启 WAL 日志模式。
        WAL = Write-Ahead Logging，好处是"读写不互相阻塞"：
        一边有人写、一边有人读，不会报 database is locked。
        在 Web 服务场景下几乎必开。
        """
        # WAL 模式对内存数据库无效（会报错），所以要判断一下
        if self.path != MEMORY_DB:
            self.conn.execute("PRAGMA journal_mode = WAL")
            # NORMAL 模式在 WAL 下已经足够安全，而且比默认的 FULL 快很多
            self.conn.execute("PRAGMA synchronous = NORMAL")

        # executescript 可以一次执行多条 SQL 语句
        self.conn.executescript(SCHEMA_SQL)
        self._check_schema_version()
        self.conn.commit()
        return self

    def _check_schema_version(self) -> None:
        """
        检查数据库里的 schema 版本和代码里的是否一致。

        为什么要有这一步？
            如果数据库是旧版本，而代码已经在按新表结构读写，
            你会遇到非常难排查的错误（比如"no such column"）。
            在这里明确报错，问题一眼就能定位。
        """
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()

        if row is None:
            # 全新的数据库，写入当前版本
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return

        found = int(row["value"])
        if found != SCHEMA_VERSION:
            raise RuntimeError(
                f"数据库 schema 版本不匹配：数据库是 v{found}，代码期望 v{SCHEMA_VERSION}。\n"
                f"数据库文件：{self.path}\n"
                "如果这是一个学习项目、数据库里没有重要数据，"
                "最简单的做法是删掉这个文件让它重建。"
            )

    # ------------------------------------------------------------------
    # 事务
    # ------------------------------------------------------------------
    def transaction(self):
        """
        返回一个事务上下文：

            with db.transaction():
                db.conn.execute(...)
                db.conn.execute(...)
            # 到这里才真正提交；中途抛异常会自动回滚

        为什么要显式写事务？
            比如"保存消息"和"更新会话时间"这两步必须一起成功。
            如果只成功了第一步，数据库就处于不一致状态。
            事务保证"要么全做，要么全不做"（原子性）。
        """
        return self.conn

    # ------------------------------------------------------------------
    # 便捷查询
    # ------------------------------------------------------------------
    def table_names(self) -> list[str]:
        """返回所有表名。调试和测试时很有用。"""
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    def count(self, table: str) -> int:
        """返回某张表的行数。日志和测试常用。"""
        # ⚠️ 注意：表名不能用参数占位符（?）传，只能拼字符串。
        #    所以如果表名来自用户输入就会有 SQL 注入风险。
        #    这里表名只来自我们自己的代码，所以是安全的。
        row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"])

    def journal_mode(self) -> str:
        """返回当前的日志模式，测试里用来确认 WAL 生效了。"""
        row = self.conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def foreign_keys_enabled(self) -> bool:
        """返回外键约束是否开启，测试里用来确认 PRAGMA 生效了。"""
        row = self.conn.execute("PRAGMA foreign_keys").fetchone()
        return bool(row[0])

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # 出异常时不要吞掉，直接关闭连接让异常继续往上抛
        self.close()

    def __repr__(self) -> str:
        name = "内存数据库" if self.path == MEMORY_DB else self.path
        return f"<Database {name}>"


# --------------------------------------------------------------------
# 自检
#
# 注意：这里刻意**不写** if __name__ == "__main__" 块。
# 因为如果用 `python -m memory_assistant.storage.database` 运行，
# 包在被 import 时已经加载过一次 database 模块，再当脚本执行会导致
# 模块被加载两次，Python 会给出 RuntimeWarning，容易让初学者困惑。
#
# 想跑自检请用：  .\.venv\Scripts\python.exe scripts\check_storage.py
# --------------------------------------------------------------------
def run_self_check() -> int:
    """检查数据库层是否正常工作。返回 0 表示全部通过。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print(" 数据库层自检")
    print("=" * 60)
    print()

    with Database(MEMORY_DB) as db:
        db.initialize()

        print(f"数据库         : {db}")
        print(f"外键约束       : {'已开启 ✅' if db.foreign_keys_enabled() else '未开启 ❌'}")
        print(f"日志模式       : {db.journal_mode()}（内存库不支持 WAL，正常）")
        print()
        print("已创建的表：")
        for name in db.table_names():
            print(f"  · {name}")
        print()

        # 验证外键真的生效：插一条属于不存在会话的消息，应该被拒绝
        print("验证外键约束是否真的生效...")
        try:
            db.conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES('不存在的会话', 'user', '测试', ?)",
                (utc_now_iso(),),
            )
            print("  ❌ 外键没生效！这条非法数据居然插进去了")
            return 1
        except sqlite3.IntegrityError:
            print("  ✅ 外键生效，非法数据被拒绝")

        # 验证 role 的 CHECK 约束
        print("验证 role 的 CHECK 约束...")
        db.conn.execute(
            "INSERT INTO sessions(id, created_at, updated_at) VALUES('s1', ?, ?)",
            (utc_now_iso(), utc_now_iso()),
        )
        try:
            db.conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES('s1', '机器人', '测试', ?)",
                (utc_now_iso(),),
            )
            print("  ❌ CHECK 约束没生效")
            return 1
        except sqlite3.IntegrityError:
            print("  ✅ CHECK 约束生效，非法 role 被拒绝")

        # 验证级联删除
        print("验证级联删除...")
        db.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES('s1', 'user', '你好', ?)",
            (utc_now_iso(),),
        )
        before = db.count("messages")
        db.conn.execute("DELETE FROM sessions WHERE id = 's1'")
        after = db.count("messages")
        if after == 0 and before == 1:
            print(f"  ✅ 删掉会话后，它的 {before} 条消息被自动清理")
        else:
            print(f"  ❌ 级联删除异常：删除前 {before} 条，删除后 {after} 条")
            return 1

    print()
    print("=" * 60)
    print(" 全部通过 ✅")
    print("=" * 60)
    return 0
