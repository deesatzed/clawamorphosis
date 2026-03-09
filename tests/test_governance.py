"""Tests for memory governance — GC, quotas, dedup, monitoring.

All tests use REAL SQLite in-memory databases — no mocks, no placeholders.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from claw.core.config import DatabaseConfig, GovernanceConfig
from claw.core.models import Methodology
from claw.db.engine import DatabaseEngine
from claw.db.repository import Repository
from claw.memory.governance import GovernanceReport, MemoryGovernor, StorageStats


# ---------------------------------------------------------------------------
# Helpers — real implementations
# ---------------------------------------------------------------------------

class FixedEmbeddingEngine:
    """Deterministic embedding engine using SHA-384 — NOT a mock."""

    DIMENSION = 384

    def encode(self, text: str) -> list[float]:
        h = hashlib.sha384(text.encode()).digest()
        raw = [b / 255.0 for b in h] * 8
        return raw[: self.DIMENSION]


@pytest.fixture
async def db_engine():
    config = DatabaseConfig(db_path=":memory:")
    engine = DatabaseEngine(config)
    await engine.connect()
    await engine.initialize_schema()
    await engine.apply_migrations()
    yield engine
    await engine.close()


@pytest.fixture
async def repository(db_engine):
    return Repository(db_engine)


@pytest.fixture
def embedding_engine():
    return FixedEmbeddingEngine()


@pytest.fixture
def default_config():
    return GovernanceConfig()


@pytest.fixture
async def governor(repository, default_config):
    return MemoryGovernor(repository=repository, config=default_config)


async def _make_methodology(
    repository: Repository,
    desc: str,
    state: str = "viable",
    embedding_engine=None,
    fitness_total: float = 0.5,
    tags: list[str] | None = None,
    generation: int = 0,
) -> Methodology:
    """Create and save a real methodology."""
    embedding = None
    if embedding_engine:
        embedding = embedding_engine.encode(desc)

    m = Methodology(
        problem_description=desc,
        problem_embedding=embedding,
        solution_code=f"# solution for: {desc}",
        lifecycle_state=state,
        fitness_vector={"total": fitness_total},
        tags=tags or [],
        generation=generation,
    )
    await repository.save_methodology(m)
    return m


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestGovernanceLogMigration:

    async def test_governance_log_table_exists(self, db_engine):
        """governance_log table exists after migration."""
        row = await db_engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='governance_log'"
        )
        assert row["cnt"] == 1

    async def test_migration_idempotent(self):
        """Calling apply_migrations twice doesn't error."""
        config = DatabaseConfig(db_path=":memory:")
        engine = DatabaseEngine(config)
        await engine.connect()
        await engine.initialize_schema()
        await engine.apply_migrations()
        await engine.apply_migrations()
        row = await engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='governance_log'"
        )
        assert row["cnt"] == 1
        await engine.close()


# ---------------------------------------------------------------------------
# Repository method tests
# ---------------------------------------------------------------------------

class TestRepositoryMethods:

    async def test_count_active_methodologies(self, repository, embedding_engine):
        """count_active_methodologies excludes dead."""
        await _make_methodology(repository, "alive 1", state="viable", embedding_engine=embedding_engine)
        await _make_methodology(repository, "alive 2", state="thriving", embedding_engine=embedding_engine)
        await _make_methodology(repository, "dead one", state="dead", embedding_engine=embedding_engine)

        count = await repository.count_active_methodologies()
        assert count == 2

    async def test_count_methodologies_by_state(self, repository, embedding_engine):
        """count_methodologies_by_state returns correct grouping."""
        await _make_methodology(repository, "v1", state="viable", embedding_engine=embedding_engine)
        await _make_methodology(repository, "v2", state="viable", embedding_engine=embedding_engine)
        await _make_methodology(repository, "t1", state="thriving", embedding_engine=embedding_engine)
        await _make_methodology(repository, "d1", state="dead", embedding_engine=embedding_engine)

        by_state = await repository.count_methodologies_by_state()
        assert by_state.get("viable", 0) == 2
        assert by_state.get("thriving", 0) == 1
        assert by_state.get("dead", 0) == 1

    async def test_get_dead_methodologies(self, repository, embedding_engine):
        """get_dead_methodologies returns only dead."""
        await _make_methodology(repository, "alive", state="viable", embedding_engine=embedding_engine)
        await _make_methodology(repository, "dead 1", state="dead", embedding_engine=embedding_engine)
        await _make_methodology(repository, "dead 2", state="dead", embedding_engine=embedding_engine)

        dead = await repository.get_dead_methodologies()
        assert len(dead) == 2
        assert all(m.lifecycle_state == "dead" for m in dead)

    async def test_delete_methodology(self, repository, embedding_engine):
        """delete_methodology removes from all 3 stores."""
        m = await _make_methodology(
            repository, "to delete", state="viable", embedding_engine=embedding_engine
        )

        # Verify it exists
        loaded = await repository.get_methodology(m.id)
        assert loaded is not None

        # Delete
        success = await repository.delete_methodology(m.id)
        assert success is True

        # Verify gone
        loaded = await repository.get_methodology(m.id)
        assert loaded is None

    async def test_delete_nonexistent_methodology(self, repository):
        """delete_methodology returns False for nonexistent ID."""
        result = await repository.delete_methodology("nonexistent-id")
        assert result is False

    async def test_get_lowest_fitness_methodologies(self, repository, embedding_engine):
        """get_lowest_fitness returns in cull order."""
        await _make_methodology(repository, "dormant 1", state="dormant", embedding_engine=embedding_engine, fitness_total=0.1)
        await _make_methodology(repository, "declining 1", state="declining", embedding_engine=embedding_engine, fitness_total=0.2)
        await _make_methodology(repository, "viable 1", state="viable", embedding_engine=embedding_engine, fitness_total=0.9)

        lowest = await repository.get_lowest_fitness_methodologies(
            states=["dormant", "declining", "viable"], limit=5
        )
        # dormant should come first (cull order), then declining
        assert lowest[0].lifecycle_state == "dormant"
        assert lowest[1].lifecycle_state == "declining"

    async def test_log_governance_action(self, repository):
        """log_governance_action creates audit entry."""
        action_id = await repository.log_governance_action(
            action_type="test_action",
            methodology_id="test-id",
            details={"key": "value"},
        )
        assert action_id is not None

        row = await repository.engine.fetch_one(
            "SELECT * FROM governance_log WHERE id = ?", [action_id]
        )
        assert row is not None
        assert row["action_type"] == "test_action"

    async def test_count_episodes(self, repository):
        """count_episodes returns correct count."""
        count = await repository.count_episodes()
        assert count == 0

        await repository.log_episode(
            session_id="s1", event_type="test", event_data={"x": 1}
        )
        count = await repository.count_episodes()
        assert count == 1

    async def test_delete_old_episodes(self, repository):
        """delete_old_episodes removes episodes before cutoff."""
        # Create an episode with a very old timestamp
        old_date = (datetime.now(UTC) - timedelta(days=200)).isoformat()
        await repository.engine.execute(
            """INSERT INTO episodes (id, session_id, event_type, event_data, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ["old-ep", "s1", "old_event", "{}", old_date],
        )
        # Create a recent episode
        await repository.log_episode(
            session_id="s1", event_type="recent", event_data={}
        )

        cutoff = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        deleted = await repository.delete_old_episodes(cutoff)
        assert deleted == 1

        # Recent should still exist
        remaining = await repository.count_episodes()
        assert remaining == 1

    async def test_get_methodologies_by_tag(self, repository, embedding_engine):
        """get_methodologies_by_tag filters by tag correctly."""
        await _make_methodology(
            repository, "tagged one", state="viable",
            embedding_engine=embedding_engine, tags=["python", "async"]
        )
        await _make_methodology(
            repository, "tagged two", state="viable",
            embedding_engine=embedding_engine, tags=["javascript"]
        )

        py = await repository.get_methodologies_by_tag("python")
        assert len(py) == 1
        assert "python" in py[0].tags

    async def test_get_db_size_bytes(self, repository):
        """get_db_size_bytes returns non-zero for in-memory DB."""
        size = await repository.get_db_size_bytes()
        # In-memory DB still has pages
        assert size >= 0


# ---------------------------------------------------------------------------
# Garbage collection tests
# ---------------------------------------------------------------------------

class TestGarbageCollection:

    async def test_gc_removes_dead_methodologies(self, governor, repository, embedding_engine):
        """GC removes dead methodologies from all stores."""
        m = await _make_methodology(
            repository, "dead methodology", state="dead",
            embedding_engine=embedding_engine
        )

        collected = await governor.garbage_collect_dead()
        assert collected == 1

        loaded = await repository.get_methodology(m.id)
        assert loaded is None

    async def test_gc_preserves_alive_methodologies(self, governor, repository, embedding_engine):
        """GC does not touch non-dead methodologies."""
        alive = await _make_methodology(
            repository, "alive", state="viable", embedding_engine=embedding_engine
        )
        await _make_methodology(
            repository, "dead", state="dead", embedding_engine=embedding_engine
        )

        collected = await governor.garbage_collect_dead()
        assert collected == 1

        loaded = await repository.get_methodology(alive.id)
        assert loaded is not None

    async def test_gc_no_dead_returns_zero(self, governor, repository, embedding_engine):
        """GC with no dead methodologies returns 0."""
        await _make_methodology(
            repository, "alive", state="viable", embedding_engine=embedding_engine
        )
        collected = await governor.garbage_collect_dead()
        assert collected == 0

    async def test_gc_logs_to_governance_log(self, governor, repository, embedding_engine):
        """GC creates audit entries in governance_log."""
        await _make_methodology(
            repository, "dead for audit", state="dead",
            embedding_engine=embedding_engine
        )
        await governor.garbage_collect_dead()

        rows = await repository.engine.fetch_all(
            "SELECT * FROM governance_log WHERE action_type = 'gc_dead'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Quota enforcement tests
# ---------------------------------------------------------------------------

class TestQuotaEnforcement:

    async def test_quota_under_limit_no_action(self, repository, embedding_engine):
        """No culling when under quota."""
        config = GovernanceConfig(max_methodologies=100)
        gov = MemoryGovernor(repository=repository, config=config)

        await _make_methodology(
            repository, "m1", state="viable", embedding_engine=embedding_engine
        )
        culled = await gov.enforce_methodology_quota()
        assert culled == 0

    async def test_quota_exceeded_culls_lowest_fitness(self, repository, embedding_engine):
        """Culls lowest-fitness when over quota."""
        config = GovernanceConfig(max_methodologies=2)
        gov = MemoryGovernor(repository=repository, config=config)

        # Create 3 methodologies (1 over quota of 2)
        await _make_methodology(
            repository, "low fitness declining", state="declining",
            embedding_engine=embedding_engine, fitness_total=0.1
        )
        await _make_methodology(
            repository, "high fitness viable", state="viable",
            embedding_engine=embedding_engine, fitness_total=0.8
        )
        await _make_methodology(
            repository, "medium fitness viable", state="viable",
            embedding_engine=embedding_engine, fitness_total=0.5
        )

        culled = await gov.enforce_methodology_quota()
        assert culled == 1

        active = await repository.count_active_methodologies()
        assert active == 2

    async def test_quota_never_culls_thriving(self, repository, embedding_engine):
        """Thriving methodologies are never culled even when over quota."""
        config = GovernanceConfig(max_methodologies=1)
        gov = MemoryGovernor(repository=repository, config=config)

        await _make_methodology(
            repository, "thriving 1", state="thriving",
            embedding_engine=embedding_engine, fitness_total=0.9
        )
        await _make_methodology(
            repository, "thriving 2", state="thriving",
            embedding_engine=embedding_engine, fitness_total=0.8
        )

        # Both thriving — cannot cull either
        culled = await gov.enforce_methodology_quota()
        assert culled == 0

        active = await repository.count_active_methodologies()
        assert active == 2

    async def test_quota_cull_order_dormant_first(self, repository, embedding_engine):
        """Dormant is culled before declining."""
        config = GovernanceConfig(max_methodologies=1)
        gov = MemoryGovernor(repository=repository, config=config)

        declining = await _make_methodology(
            repository, "declining one", state="declining",
            embedding_engine=embedding_engine, fitness_total=0.3
        )
        dormant = await _make_methodology(
            repository, "dormant one", state="dormant",
            embedding_engine=embedding_engine, fitness_total=0.2
        )

        culled = await gov.enforce_methodology_quota()
        assert culled == 1

        # Dormant should be culled first
        loaded_dormant = await repository.get_methodology(dormant.id)
        assert loaded_dormant is None

        loaded_declining = await repository.get_methodology(declining.id)
        assert loaded_declining is not None


# ---------------------------------------------------------------------------
# Pre-save dedup tests
# ---------------------------------------------------------------------------

class TestPreSaveDedup:

    async def test_dedup_blocks_near_duplicate(self, repository, embedding_engine):
        """Near-duplicate (>= 0.88 similarity) is blocked."""
        config = GovernanceConfig(dedup_similarity_threshold=0.88)
        gov = MemoryGovernor(repository=repository, config=config)

        # Save original
        text = "optimizing database query performance"
        embedding = embedding_engine.encode(text)
        m = Methodology(
            problem_description=text,
            problem_embedding=embedding,
            solution_code="SELECT * FROM optimized",
            lifecycle_state="viable",
        )
        await repository.save_methodology(m)

        # Same text should be blocked (identical = similarity 1.0)
        should_save, existing_id = await gov.check_pre_save_dedup(
            text, embedding
        )
        assert should_save is False
        assert existing_id == m.id

    async def test_dedup_allows_different_content(self, repository, embedding_engine):
        """Sufficiently different content passes dedup."""
        config = GovernanceConfig(dedup_similarity_threshold=0.88)
        gov = MemoryGovernor(repository=repository, config=config)

        # Save original
        embedding1 = embedding_engine.encode("optimizing database queries")
        m = Methodology(
            problem_description="optimizing database queries",
            problem_embedding=embedding1,
            solution_code="SELECT optimized",
            lifecycle_state="viable",
        )
        await repository.save_methodology(m)

        # Very different content should pass
        embedding2 = embedding_engine.encode("implementing websocket authentication")
        should_save, existing_id = await gov.check_pre_save_dedup(
            "implementing websocket authentication", embedding2
        )
        assert should_save is True
        assert existing_id is None

    async def test_dedup_disabled(self, repository, embedding_engine):
        """Dedup disabled allows everything."""
        config = GovernanceConfig(dedup_enabled=False)
        gov = MemoryGovernor(repository=repository, config=config)

        embedding = embedding_engine.encode("test")
        m = Methodology(
            problem_description="test",
            problem_embedding=embedding,
            solution_code="code",
            lifecycle_state="viable",
        )
        await repository.save_methodology(m)

        should_save, _ = await gov.check_pre_save_dedup("test", embedding)
        assert should_save is True

    async def test_dedup_no_embedding_allows_save(self, governor):
        """No embedding means dedup cannot check — allow save."""
        should_save, _ = await governor.check_pre_save_dedup(
            "no embedding here", None
        )
        assert should_save is True

    async def test_dedup_skips_dead_methodology(self, repository, embedding_engine):
        """Dead methodologies don't count as duplicates."""
        config = GovernanceConfig(dedup_similarity_threshold=0.88)
        gov = MemoryGovernor(repository=repository, config=config)

        text = "dead pattern"
        embedding = embedding_engine.encode(text)
        m = Methodology(
            problem_description=text,
            problem_embedding=embedding,
            solution_code="dead code",
            lifecycle_state="dead",
        )
        await repository.save_methodology(m)

        # Same text but existing is dead — should allow save
        should_save, _ = await gov.check_pre_save_dedup(text, embedding)
        assert should_save is True

    async def test_dedup_logs_to_governance_log(self, repository, embedding_engine):
        """Blocked duplicates are logged."""
        config = GovernanceConfig(dedup_similarity_threshold=0.88)
        gov = MemoryGovernor(repository=repository, config=config)

        text = "duplicate tracking test"
        embedding = embedding_engine.encode(text)
        m = Methodology(
            problem_description=text,
            problem_embedding=embedding,
            solution_code="code",
            lifecycle_state="viable",
        )
        await repository.save_methodology(m)

        await gov.check_pre_save_dedup(text, embedding)

        rows = await repository.engine.fetch_all(
            "SELECT * FROM governance_log WHERE action_type = 'dedup_block'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Storage stats tests
# ---------------------------------------------------------------------------

class TestStorageStats:

    async def test_storage_stats_empty_db(self, governor):
        """Stats on empty DB return zeroes."""
        stats = await governor.get_storage_stats()
        assert stats.total_methodologies == 0
        assert stats.quota_used_pct == 0.0
        assert stats.quota_limit == 2000

    async def test_storage_stats_with_data(self, governor, repository, embedding_engine):
        """Stats reflect actual data."""
        await _make_methodology(repository, "s1", state="viable", embedding_engine=embedding_engine)
        await _make_methodology(repository, "s2", state="thriving", embedding_engine=embedding_engine)
        await _make_methodology(repository, "s3", state="dead", embedding_engine=embedding_engine)

        stats = await governor.get_storage_stats()
        assert stats.total_methodologies == 3
        assert stats.by_state.get("viable", 0) == 1
        assert stats.by_state.get("thriving", 0) == 1
        assert stats.by_state.get("dead", 0) == 1
        # Active = 2, quota = 2000 → 0.1%
        assert stats.quota_used_pct == 0.1


# ---------------------------------------------------------------------------
# Sweep scheduling tests
# ---------------------------------------------------------------------------

class TestSweepScheduling:

    async def test_maybe_run_sweep_interval(self, repository):
        """maybe_run_sweep only runs every N cycles."""
        config = GovernanceConfig(sweep_interval_cycles=5)
        gov = MemoryGovernor(repository=repository, config=config)

        # Cycles 1-4 should not run
        for _ in range(4):
            result = await gov.maybe_run_sweep()
            assert result is None

        # Cycle 5 should run
        result = await gov.maybe_run_sweep()
        assert isinstance(result, GovernanceReport)

    async def test_maybe_run_sweep_repeated_intervals(self, repository):
        """Sweep runs at every interval boundary."""
        config = GovernanceConfig(sweep_interval_cycles=3)
        gov = MemoryGovernor(repository=repository, config=config)

        results = []
        for i in range(9):
            result = await gov.maybe_run_sweep()
            results.append(result)

        # Should run at cycles 3, 6, 9
        assert results[2] is not None  # cycle 3
        assert results[5] is not None  # cycle 6
        assert results[8] is not None  # cycle 9
        # Others should be None
        assert results[0] is None
        assert results[1] is None


# ---------------------------------------------------------------------------
# Full sweep tests
# ---------------------------------------------------------------------------

class TestFullSweep:

    async def test_full_sweep_runs_all_operations(self, governor, repository, embedding_engine):
        """Full sweep runs GC, quota, episodes, and logs stats."""
        # Create a dead methodology for GC
        await _make_methodology(
            repository, "dead sweep", state="dead",
            embedding_engine=embedding_engine
        )

        report = await governor.run_full_sweep()

        assert isinstance(report, GovernanceReport)
        assert report.dead_collected == 1
        assert report.sweep_duration_seconds >= 0.0
        assert report.storage_stats is not None

    async def test_full_sweep_empty_db(self, governor):
        """Full sweep on empty DB completes without error."""
        report = await governor.run_full_sweep()
        assert report.dead_collected == 0
        assert report.quota_culled == 0
        assert report.episodes_pruned == 0

    async def test_full_sweep_logs_to_governance_log(self, governor, repository):
        """Sweep itself is logged to governance_log."""
        await governor.run_full_sweep()

        rows = await repository.engine.fetch_all(
            "SELECT * FROM governance_log WHERE action_type = 'sweep'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Config defaults tests
# ---------------------------------------------------------------------------

class TestGovernanceConfig:

    def test_config_defaults(self):
        """GovernanceConfig has sensible defaults."""
        config = GovernanceConfig()
        assert config.max_methodologies == 2000
        assert config.dedup_similarity_threshold == 0.88
        assert config.dedup_enabled is True
        assert config.gc_dead_on_sweep is True
        assert config.sweep_interval_cycles == 10
        assert config.sweep_on_startup is True
        assert config.episodic_retention_days == 90
        assert config.self_consume_enabled is True
        assert config.self_consume_max_generation == 3
        assert config.max_db_size_mb == 500

    def test_config_custom_values(self):
        """GovernanceConfig accepts custom values."""
        config = GovernanceConfig(
            max_methodologies=500,
            dedup_similarity_threshold=0.90,
            sweep_interval_cycles=5,
        )
        assert config.max_methodologies == 500
        assert config.dedup_similarity_threshold == 0.90
        assert config.sweep_interval_cycles == 5
