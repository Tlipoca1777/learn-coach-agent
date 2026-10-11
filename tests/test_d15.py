"""D15 regression coverage: profiles, conflict resolution and user isolation."""

from __future__ import annotations

import itertools
import sqlite3

import pytest

from memory_assistant.config import Config
from memory_assistant.embeddings import FakeEmbeddings
from memory_assistant.engine import ConversationEngine, PROFILE_HEADER
from memory_assistant.llm import FakeLLM
from memory_assistant.memory import LongTermMemory
from memory_assistant.storage import Database, Store
from memory_assistant.cli import parse_user_command


_collections = itertools.count()


class StaticExtractor:
    def __init__(self):
        self.facts = []

    def extract(self, messages, *, known_facts=None):
        result, self.facts = self.facts, []
        return result


class Fact:
    def __init__(self, predicate, object):
        self.subject = "user"
        self.predicate = predicate
        self.object = object
        self.confidence = 0.95


@pytest.fixture
def app():
    db = Database(":memory:")
    db.initialize()
    store = Store(db)
    memory = LongTermMemory(
        FakeEmbeddings(), collection_name=f"d15_{next(_collections)}"
    )
    extractor = StaticExtractor()
    config = Config.from_env(require_key=False)
    engine = ConversationEngine(
        config,
        store=store,
        llm=FakeLLM(default_response="记住了。"),
        long_term=memory,
        extractor=extractor,
        user_id="alice",
    )
    yield engine, store, memory, extractor
    db.close()


def test_conflicting_location_invalidates_old_fact_and_search_hides_it(app):
    engine, _, memory, extractor = app
    extractor.facts = [Fact("住在", "杭州")]
    engine.respond("我住在杭州")
    extractor.facts = [Fact("搬到", "上海")]
    engine.respond("我搬到上海了")

    current = memory.search("我住在哪里", user_id="alice", record_hits=False)
    history = memory.search(
        "我住在哪里", user_id="alice", include_invalid=True, record_hits=False
    )

    assert [fact["object"] for fact in current] == ["上海"]
    old = next(fact for fact in history if fact["object"] == "杭州")
    assert old["invalid_at"] is not None
    assert app[1].profiles.get("alice")["住在"] == "上海"


def test_profile_updates_are_user_scoped_and_injected_into_prompt(app):
    engine, store, _, extractor = app
    extractor.facts = [Fact("住在", "杭州"), Fact("喜欢", "咖啡")]
    engine.respond("我住在杭州，也喜欢咖啡")
    assert store.profiles.get("alice") == {"住在": "杭州", "喜欢": "咖啡"}
    engine.respond("继续")
    assert PROFILE_HEADER + "- 住在：杭州\n- 喜欢：咖啡" in [
        item["content"] for item in engine.llm.calls[-1]
    ]

    store.profiles.update("bob", {"住在": "北京"})
    assert store.profiles.get("bob") == {"住在": "北京"}
    assert store.profiles.get("alice")["住在"] == "杭州"


def test_user_switch_restores_isolated_sessions_memory_profile_and_tool_scope(app):
    engine, store, memory, extractor = app
    extractor.facts = [Fact("住在", "杭州")]
    engine.respond("alice 的会话内容")
    alice_session = engine.session_id

    memory.add("住在", "波士顿", user_id="bob")
    store.profiles.update("bob", {"住在": "波士顿"})
    # A distinct, persisted bob session should be selected after switching.
    bob_session = store.sessions.create(user_id="bob", title="bob 会话")
    store.messages.append(bob_session, "user", "bob 的会话内容")
    store.summaries.add(bob_session, "bob 的摘要", covered_until=1)

    engine.switch_user("bob")
    assert engine.session_id == bob_session
    assert [item["content"] for item in engine.short_term.get_window()] == ["bob 的会话内容"]
    assert engine.summary.get_summary() == "bob 的摘要"
    assert engine.long_term.search("住在", user_id="bob", record_hits=False)[0]["object"] == "波士顿"
    assert "recall_memory" in engine.tools
    bob_tool = engine.tools.execute("recall_memory", {"query": "住在"})
    assert "波士顿" in bob_tool
    assert "杭州" not in bob_tool

    engine.switch_user("alice")
    assert engine.session_id == alice_session
    assert [item["content"] for item in engine.short_term.get_window()] == [
        "alice 的会话内容", "记住了。"
    ]
    assert engine.summary.get_summary() is None
    assert engine.store.profiles.get("alice")["住在"] == "杭州"


def test_starting_another_users_session_is_rejected(app):
    engine, store, _, _ = app
    store.sessions.create(session_id="private-to-bob", user_id="bob")
    with pytest.raises(PermissionError, match="其他用户"):
        ConversationEngine(
            engine.config,
            store=store,
            llm=FakeLLM(),
            long_term=engine.long_term,
            extractor=StaticExtractor(),
            user_id="alice",
            session_id="private-to-bob",
        ).start()


def test_user_command_parser_matches_only_exact_command():
    assert parse_user_command("/user alice") == "alice"
    assert parse_user_command("/user   alice ") == "alice"
    assert parse_user_command("/user") == ""
    assert parse_user_command("/username alice") is None


def test_profile_repository_merges_fields_and_deletes_one_user(app):
    _, store, _, _ = app
    store.profiles.update("alice", {"city": "杭州"})
    store.profiles.update("alice", {"hobby": "阅读"})
    store.profiles.update("bob", {"city": "上海"})

    assert store.profiles.get("alice") == {"city": "杭州", "hobby": "阅读"}
    assert store.profiles.delete("alice")
    assert store.profiles.get("alice") == {}
    assert store.profiles.get("bob") == {"city": "上海"}


def test_schema_v2_migration_assigns_existing_sessions_to_default_user(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO sessions VALUES ('legacy-session', 'legacy', '2026-01-01', '2026-01-01');
        INSERT INTO meta VALUES ('schema_version', '2');
        """
    )
    connection.commit()
    connection.close()

    db = Database(path).initialize()
    session = Store(db).sessions.get("legacy-session")
    version = db.conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()["value"]

    assert session["user_id"] == "default"
    assert version == "3"
    db.close()

