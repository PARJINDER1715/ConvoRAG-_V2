import sqlite3
import hashlib
import json
import os
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "convorag.db"
))


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id   TEXT PRIMARY KEY,
                persona_json TEXT NOT NULL DEFAULT '{}',
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES chat_sessions(session_id),
                role         TEXT NOT NULL,   -- 'user' | 'assistant' | 'system'
                content      TEXT NOT NULL,
                created_at   INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id, id);

            -- Groq call cache: key = sha256(model+prompt), value = response text
            CREATE TABLE IF NOT EXISTS groq_cache (
                cache_key    TEXT PRIMARY KEY,
                model        TEXT NOT NULL,
                prompt_hash  TEXT NOT NULL,
                response     TEXT NOT NULL,
                created_at   INTEGER NOT NULL,
                hit_count    INTEGER NOT NULL DEFAULT 0
            );

            -- Editable contradiction pairs for ConflictResolver
            CREATE TABLE IF NOT EXISTS contradiction_pairs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                pos_words    TEXT NOT NULL,   -- JSON array
                neg_words    TEXT NOT NULL,   -- JSON array
                label        TEXT NOT NULL DEFAULT '',
                active       INTEGER NOT NULL DEFAULT 1,
                created_at   INTEGER NOT NULL
            );
        """)
        _seed_contradiction_pairs(con)


def _seed_contradiction_pairs(con: sqlite3.Connection):
    """Insert default pairs only when the table is empty."""
    row = con.execute("SELECT COUNT(*) FROM contradiction_pairs").fetchone()
    if row[0] > 0:
        return

    defaults = [
        (["love","like","enjoy","happy","good"],    ["hate","dislike","upset","bad","angry"],        "sentiment"),
        (["close","best friend","love"],             ["fight","broke up","argument","distant"],       "relationship"),
        (["well","fine","okay","healthy"],           ["sick","ill","hospital","hurt","pain"],         "health"),
        (["together","married","dating"],            ["divorced","separated","broke up","single"],    "relationship-status"),
        (["alive","healthy"],                        ["died","passed away","gone","lost"],            "existence"),
    ]
    now = int(time.time())
    con.executemany(
        "INSERT INTO contradiction_pairs (pos_words, neg_words, label, active, created_at) VALUES (?,?,?,1,?)",
        [(json.dumps(p), json.dumps(n), lbl, now) for p, n, lbl in defaults]
    )


# ── Session API ───────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> dict:
    now = int(time.time())
    with _conn() as con:
        con.execute("""
            INSERT OR IGNORE INTO chat_sessions (session_id, persona_json, created_at, updated_at)
            VALUES (?, '{}', ?, ?)
        """, (session_id, now, now))
        row = con.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row)


def save_message(session_id: str, role: str, content: str):
    get_or_create_session(session_id)
    now = int(time.time())
    with _conn() as con:
        con.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, now)
        )
        con.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )


def get_messages(session_id: str, limit: int = 40) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT role, content FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_persona(session_id: str, persona: dict):
    get_or_create_session(session_id)
    now = int(time.time())
    with _conn() as con:
        con.execute(
            "UPDATE chat_sessions SET persona_json = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(persona), now, session_id)
        )


def get_persona(session_id: str) -> dict:
    row = get_or_create_session(session_id)
    try:
        return json.loads(row["persona_json"])
    except Exception:
        return {}


def list_sessions(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
            SELECT s.session_id, s.updated_at,
                   COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    with _conn() as con:
        con.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        con.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))


# ── Groq idempotency cache ────────────────────────────────────────────────────

def _cache_key(model: str, prompt: str) -> str:
    raw = f"{model}||{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def groq_cache_get(model: str, prompt: str) -> str | None:
    key = _cache_key(model, prompt)
    with _conn() as con:
        row = con.execute(
            "SELECT response FROM groq_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row:
            con.execute(
                "UPDATE groq_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (key,)
            )
            return row["response"]
    return None


def groq_cache_set(model: str, prompt: str, response: str):
    key = _cache_key(model, prompt)
    now = int(time.time())
    with _conn() as con:
        con.execute("""
            INSERT OR REPLACE INTO groq_cache
                (cache_key, model, prompt_hash, response, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (key, model, key[:16], response, now))


def groq_cache_stats() -> dict:
    with _conn() as con:
        row = con.execute("""
            SELECT COUNT(*) as total,
                   SUM(hit_count) as total_hits,
                   MAX(created_at) as last_write
            FROM groq_cache
        """).fetchone()
    return dict(row)


# ── Contradiction pairs API ───────────────────────────────────────────────────

def get_contradiction_pairs() -> list[tuple[set, set]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT pos_words, neg_words FROM contradiction_pairs WHERE active = 1"
        ).fetchall()
    return [(set(json.loads(r["pos_words"])), set(json.loads(r["neg_words"]))) for r in rows]


def add_contradiction_pair(pos_words: list[str], neg_words: list[str], label: str = "") -> int:
    now = int(time.time())
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO contradiction_pairs (pos_words, neg_words, label, active, created_at) VALUES (?,?,?,1,?)",
            (json.dumps(pos_words), json.dumps(neg_words), label, now)
        )
    return cur.lastrowid


def list_contradiction_pairs() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, pos_words, neg_words, label, active FROM contradiction_pairs ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def toggle_contradiction_pair(pair_id: int, active: bool):
    with _conn() as con:
        con.execute(
            "UPDATE contradiction_pairs SET active = ? WHERE id = ?",
            (1 if active else 0, pair_id)
        )
