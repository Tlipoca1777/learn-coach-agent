r"""
存储层
================================================================

对外提供三样东西：

    Database  —— 数据库连接与建表
    Store     —— 一组仓储的集合（sessions / messages / summaries）
    各仓储    —— 想单独用也可以直接 import

    from memory_assistant.storage import Database, Store

    with Database("data/assistant.db") as db:
        db.initialize()
        store = Store(db)
        sid = store.sessions.create()
        store.messages.append(sid, "user", "你好")
"""

from memory_assistant.storage.database import (
    MEMORY_DB,
    SCHEMA_VERSION,
    Database,
    utc_now_iso,
)
from memory_assistant.storage.repositories import (
    VALID_ROLES,
    MessageRepository,
    LearningRepository,
    SessionRepository,
    Store,
    SummaryRepository,
)

__all__ = [
    "Database",
    "Store",
    "SessionRepository",
    "MessageRepository",
    "SummaryRepository",
    "LearningRepository",
    "VALID_ROLES",
    "MEMORY_DB",
    "SCHEMA_VERSION",
    "utc_now_iso",
]
