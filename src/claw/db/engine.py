"""SQLite database engine for CLAW.

Manages async connections via aiosqlite and handles schema initialization.
All queries flow through this engine; the Repository class builds on top.
WAL mode is enabled on connect for concurrent read/write performance.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import aiosqlite
import sqlite_vec

from claw.core.config import DatabaseConfig
from claw.core.exceptions import ConnectionError, DatabaseError, SchemaInitError

logger = logging.getLogger("claw.db")

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DatabaseEngine:
    """SQLite engine wrapping aiosqlite.

    Usage:
        engine = DatabaseEngine(config)
        await engine.connect()
        rows = await engine.fetch_all("SELECT * FROM tasks WHERE status = ?", ["PENDING"])

        async with engine.transaction():
            await engine.execute("INSERT INTO tasks ...")
            await engine.execute("UPDATE projects ...")
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open the SQLite connection with WAL mode and dict row factory."""
        try:
            db_path = Path(self.config.db_path)
            if self.config.db_path != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = await aiosqlite.connect(str(db_path))
            self._conn.row_factory = aiosqlite.Row

            # Load sqlite-vec extension for vector search
            # Must run in aiosqlite's thread since sqlite3 objects are thread-bound
            def _load_vec(conn):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

            await self._conn._execute(_load_vec, self._conn._conn)

            # Enable WAL mode for concurrent reads
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.execute("PRAGMA busy_timeout=5000")

            logger.info("Connected to SQLite at %s (sqlite-vec loaded)", self.config.db_path)
        except Exception as e:
            if self._conn is not None:
                try:
                    await self._conn.close()
                except Exception:
                    pass
                self._conn = None
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise ConnectionError("Database not connected. Call connect() first.")
        return self._conn

    async def initialize_schema(self) -> None:
        """Run schema.sql to create all tables and indexes."""
        if not SCHEMA_PATH.exists():
            raise SchemaInitError(f"Schema file not found: {SCHEMA_PATH}")

        sql = SCHEMA_PATH.read_text()
        try:
            await self.conn.executescript(sql)
            await self.conn.commit()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            raise SchemaInitError(f"Failed to initialize schema: {e}") from e

    async def apply_migrations(self) -> None:
        """Apply incremental schema migrations idempotently.

        Each migration checks whether the target change already exists before
        applying it, so this method is safe to call on every startup.
        Called before initialize_schema() so existing DBs get new columns
        before schema.sql tries to create indexes on them.
        """
        # Guard: skip migrations if methodologies table doesn't exist yet (fresh DB)
        row = await self.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='methodologies'"
        )
        tables_exist = row and row["cnt"] > 0

        if tables_exist:
            # Migration 1: add prism_data column to methodologies
            row = await self.fetch_one(
                "SELECT COUNT(*) as cnt FROM pragma_table_info('methodologies') WHERE name = 'prism_data'"
            )
            if row and row["cnt"] == 0:
                await self.conn.execute(
                    "ALTER TABLE methodologies ADD COLUMN prism_data TEXT"
                )
                await self.conn.commit()
                logger.info("Migration applied: methodologies.prism_data column added")

        # Migration 2: create governance_log table (safe even on fresh DB)
        row = await self.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='governance_log'"
        )
        if row and row["cnt"] == 0:
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS governance_log (
                    id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    methodology_id TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_governance_log_action ON governance_log(action_type);
                CREATE INDEX IF NOT EXISTS idx_governance_log_created ON governance_log(created_at DESC);
            """)
            await self.conn.commit()
            logger.info("Migration applied: governance_log table created")

        if tables_exist:
            # Migration 3: add capability_data column to methodologies
            row = await self.fetch_one(
                "SELECT COUNT(*) as cnt FROM pragma_table_info('methodologies') WHERE name = 'capability_data'"
            )
            if row and row["cnt"] == 0:
                await self.conn.execute(
                    "ALTER TABLE methodologies ADD COLUMN capability_data TEXT"
                )
                await self.conn.commit()
                logger.info("Migration applied: methodologies.capability_data column added")

        # Migration 4: create synergy_exploration_log table (safe even on fresh DB)
        row = await self.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='synergy_exploration_log'"
        )
        if row and row["cnt"] == 0:
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS synergy_exploration_log (
                    id TEXT PRIMARY KEY,
                    cap_a_id TEXT NOT NULL,
                    cap_b_id TEXT NOT NULL,
                    explored_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    result TEXT NOT NULL DEFAULT 'pending'
                        CHECK (result IN ('pending','synergy','no_match','error','stale')),
                    synergy_score REAL,
                    synergy_type TEXT,
                    edge_id TEXT,
                    exploration_method TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(cap_a_id, cap_b_id)
                );
                CREATE INDEX IF NOT EXISTS idx_synergy_log_cap_a ON synergy_exploration_log(cap_a_id);
                CREATE INDEX IF NOT EXISTS idx_synergy_log_cap_b ON synergy_exploration_log(cap_b_id);
                CREATE INDEX IF NOT EXISTS idx_synergy_log_result ON synergy_exploration_log(result);
            """)
            await self.conn.commit()
            logger.info("Migration applied: synergy_exploration_log table created")

        if tables_exist:
            # Migration 5: add novelty_score and potential_score columns to methodologies
            row = await self.fetch_one(
                "SELECT COUNT(*) as cnt FROM pragma_table_info('methodologies') WHERE name = 'novelty_score'"
            )
            if row and row["cnt"] == 0:
                await self.conn.execute(
                    "ALTER TABLE methodologies ADD COLUMN novelty_score REAL"
                )
                await self.conn.execute(
                    "ALTER TABLE methodologies ADD COLUMN potential_score REAL"
                )
                await self.conn.executescript(
                    "CREATE INDEX IF NOT EXISTS idx_meth_novelty ON methodologies(novelty_score DESC);"
                )
                await self.conn.commit()
                logger.info("Migration applied: methodologies.novelty_score + potential_score columns added")

    async def execute(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> None:
        """Execute a query without returning results."""
        try:
            await self.conn.execute(query, params or [])
            await self.conn.commit()
        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    async def execute_returning_lastrowid(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> int:
        """Execute an INSERT and return lastrowid."""
        try:
            cursor = await self.conn.execute(query, params or [])
            await self.conn.commit()
            return cursor.lastrowid or 0
        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    async def fetch_one(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Execute a query and return the first row as a dict, or None."""
        try:
            cursor = await self.conn.execute(query, params or [])
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    async def fetch_all(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        try:
            cursor = await self.conn.execute(query, params or [])
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    @asynccontextmanager
    async def transaction(self):
        """Context manager for explicit transactions."""
        await self.conn.execute("BEGIN")
        try:
            yield self
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")
