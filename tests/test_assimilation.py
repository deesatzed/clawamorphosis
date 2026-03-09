"""Tests for Capability Assimilation — repository methods, models, and engine.

All tests use REAL SQLite in-memory databases — no mocks, no placeholders.
"""

from __future__ import annotations

import json

import pytest

from claw.core.models import (
    CapabilityData,
    CapabilityIO,
    ComposabilityInterface,
    Methodology,
    Project,
    SynergyEdgeType,
    SynergyExploration,
    Task,
)
from claw.evolution.assimilation import (
    CapabilityComposer,
    SynergyDiscoverer,
    _canonical_pair,
    _parse_capability_json,
    _parse_synergy_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def sample_project(repository):
    """Create a test project."""
    project = Project(name="test-project", repo_path="/tmp/test")
    return await repository.create_project(project)


@pytest.fixture
async def sample_task(repository, sample_project):
    """Create a test task."""
    task = Task(
        project_id=sample_project.id,
        title="Test task",
        description="A test task for assimilation tests",
    )
    return await repository.create_task(task)


def _make_methodology(
    problem: str = "detect grokking events",
    solution: str = "def detect(): pass",
    capability_data: dict | None = None,
    **kwargs,
) -> Methodology:
    return Methodology(
        problem_description=problem,
        solution_code=solution,
        capability_data=capability_data,
        **kwargs,
    )


def _make_capability_data(**overrides) -> dict:
    """Create a CapabilityData dict with sensible defaults."""
    data = CapabilityData(
        inputs=[CapabilityIO(name="training_logs", type="metrics_data")],
        outputs=[CapabilityIO(name="grokking_events", type="event_list")],
        domain=["ml_training", "optimization"],
        composability=ComposabilityInterface(
            can_chain_after=["data_collection"],
            can_chain_before=["model_tuning"],
            standalone=True,
        ),
        capability_type="analysis",
    )
    d = data.model_dump()
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestCapabilityModels:
    def test_capability_io_defaults(self):
        io = CapabilityIO(name="input", type="text")
        assert io.required is True
        assert io.description == ""

    def test_composability_interface_defaults(self):
        ci = ComposabilityInterface()
        assert ci.can_chain_after == []
        assert ci.can_chain_before == []
        assert ci.standalone is True

    def test_capability_data_full(self):
        cd = CapabilityData(
            inputs=[CapabilityIO(name="x", type="code")],
            outputs=[CapabilityIO(name="y", type="patch")],
            domain=["refactoring"],
            composability=ComposabilityInterface(standalone=False),
            capability_type="transformation",
        )
        assert len(cd.inputs) == 1
        assert cd.domain == ["refactoring"]
        assert cd.composability.standalone is False

    def test_capability_data_serialization_roundtrip(self):
        cd = CapabilityData(
            inputs=[CapabilityIO(name="a", type="b")],
            outputs=[],
            domain=["test"],
        )
        d = cd.model_dump()
        cd2 = CapabilityData(**d)
        assert cd2.inputs[0].name == "a"
        assert cd2.domain == ["test"]

    def test_synergy_edge_types(self):
        assert SynergyEdgeType.FEEDS_INTO.value == "feeds_into"
        assert SynergyEdgeType.ENHANCES.value == "enhances"
        assert SynergyEdgeType.SYNERGY.value == "synergy"

    def test_synergy_exploration_defaults(self):
        se = SynergyExploration(cap_a_id="a", cap_b_id="b")
        assert se.result == "pending"
        assert se.synergy_score is None
        assert se.details == {}

    def test_methodology_with_capability_data(self):
        m = Methodology(
            problem_description="test",
            solution_code="code",
            capability_data=_make_capability_data(),
        )
        assert m.capability_data is not None
        assert m.capability_data["capability_type"] == "analysis"

    def test_methodology_without_capability_data(self):
        m = Methodology(
            problem_description="test",
            solution_code="code",
        )
        assert m.capability_data is None


# ---------------------------------------------------------------------------
# Repository: capability_data CRUD
# ---------------------------------------------------------------------------

class TestCapabilityDataRepository:
    async def test_save_methodology_with_capability_data(self, repository, sample_project, sample_task):
        cap_data = _make_capability_data()
        m = _make_methodology(capability_data=cap_data, source_task_id=sample_task.id)
        saved = await repository.save_methodology(m)

        fetched = await repository.get_methodology(saved.id)
        assert fetched is not None
        assert fetched.capability_data is not None
        assert fetched.capability_data["capability_type"] == "analysis"
        assert len(fetched.capability_data["inputs"]) == 1

    async def test_save_methodology_without_capability_data(self, repository, sample_project, sample_task):
        m = _make_methodology(source_task_id=sample_task.id)
        saved = await repository.save_methodology(m)

        fetched = await repository.get_methodology(saved.id)
        assert fetched is not None
        assert fetched.capability_data is None

    async def test_update_methodology_capability_data(self, repository, sample_project, sample_task):
        m = _make_methodology(source_task_id=sample_task.id)
        saved = await repository.save_methodology(m)

        cap_data = _make_capability_data(capability_type="detection")
        await repository.update_methodology_capability_data(saved.id, cap_data)

        fetched = await repository.get_methodology(saved.id)
        assert fetched.capability_data is not None
        assert fetched.capability_data["capability_type"] == "detection"

    async def test_get_methodologies_with_capabilities(self, repository, sample_project, sample_task):
        # One with, one without
        m1 = _make_methodology(
            problem="with cap",
            capability_data=_make_capability_data(),
            source_task_id=sample_task.id,
        )
        m2 = _make_methodology(problem="without cap", source_task_id=sample_task.id)
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)

        with_caps = await repository.get_methodologies_with_capabilities()
        assert len(with_caps) == 1
        assert with_caps[0].problem_description == "with cap"

    async def test_get_methodologies_without_capability_data(self, repository, sample_project, sample_task):
        m1 = _make_methodology(
            problem="enriched",
            capability_data=_make_capability_data(),
            source_task_id=sample_task.id,
        )
        m2 = _make_methodology(problem="unenriched", source_task_id=sample_task.id)
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)

        unenriched = await repository.get_methodologies_without_capability_data()
        assert len(unenriched) == 1
        assert unenriched[0].problem_description == "unenriched"

    async def test_dead_methodologies_excluded_from_capability_queries(
        self, repository, sample_project, sample_task
    ):
        m = _make_methodology(
            problem="dead cap",
            capability_data=_make_capability_data(),
            lifecycle_state="dead",
            source_task_id=sample_task.id,
        )
        await repository.save_methodology(m)

        with_caps = await repository.get_methodologies_with_capabilities()
        assert len(with_caps) == 0

        without_caps = await repository.get_methodologies_without_capability_data()
        assert len(without_caps) == 0


# ---------------------------------------------------------------------------
# Repository: synergy exploration log
# ---------------------------------------------------------------------------

class TestSynergyExplorationRepository:
    async def test_record_and_retrieve_exploration(self, repository):
        exp = SynergyExploration(
            cap_a_id="aaa",
            cap_b_id="bbb",
            result="synergy",
            synergy_score=0.85,
            synergy_type="feeds_into",
            exploration_method="4-signal",
            details={"io_score": 0.9, "domain_score": 0.7},
        )
        await repository.record_synergy_exploration(exp)

        fetched = await repository.get_synergy_exploration("aaa", "bbb")
        assert fetched is not None
        assert fetched.result == "synergy"
        assert fetched.synergy_score == 0.85
        assert fetched.details["io_score"] == 0.9

    async def test_canonical_ordering_on_get(self, repository):
        """get_synergy_exploration enforces canonical (a < b) ordering."""
        exp = SynergyExploration(
            cap_a_id="aaa",
            cap_b_id="zzz",
            result="no_match",
            synergy_score=0.3,
        )
        await repository.record_synergy_exploration(exp)

        # Query with reversed order — should still find it
        fetched = await repository.get_synergy_exploration("zzz", "aaa")
        assert fetched is not None
        assert fetched.result == "no_match"

    async def test_upsert_on_duplicate_pair(self, repository):
        """Recording same pair again updates instead of failing."""
        exp1 = SynergyExploration(
            cap_a_id="aaa",
            cap_b_id="bbb",
            result="pending",
        )
        await repository.record_synergy_exploration(exp1)

        exp2 = SynergyExploration(
            cap_a_id="aaa",
            cap_b_id="bbb",
            result="synergy",
            synergy_score=0.9,
        )
        await repository.record_synergy_exploration(exp2)

        fetched = await repository.get_synergy_exploration("aaa", "bbb")
        assert fetched.result == "synergy"
        assert fetched.synergy_score == 0.9

    async def test_get_unexplored_pairs(self, repository):
        # Record one explored pair
        exp = SynergyExploration(cap_a_id="aaa", cap_b_id="bbb", result="synergy")
        await repository.record_synergy_exploration(exp)

        # Check which of ["bbb", "ccc", "ddd"] are unexplored with "aaa"
        unexplored = await repository.get_unexplored_pairs("aaa", ["bbb", "ccc", "ddd"])
        assert "bbb" not in unexplored
        assert "ccc" in unexplored
        assert "ddd" in unexplored

    async def test_get_unexplored_pairs_empty_candidates(self, repository):
        result = await repository.get_unexplored_pairs("aaa", [])
        assert result == []

    async def test_get_synergy_stats(self, repository):
        for i, result in enumerate(["synergy", "synergy", "no_match", "error"]):
            exp = SynergyExploration(
                cap_a_id=f"a{i}",
                cap_b_id=f"b{i}",
                result=result,
                synergy_score=0.8 if result == "synergy" else 0.3,
            )
            await repository.record_synergy_exploration(exp)

        stats = await repository.get_synergy_stats()
        assert stats["total_explored"] == 4
        assert stats["by_result"]["synergy"] == 2
        assert stats["by_result"]["no_match"] == 1
        assert stats["avg_synergy_score"] > 0

    async def test_mark_stale_explorations(self, repository):
        # Use canonical ordering (cap_a_id < cap_b_id alphabetically)
        exp1 = SynergyExploration(cap_a_id="bbb", cap_b_id="del_me", result="synergy")
        exp2 = SynergyExploration(cap_a_id="aaa", cap_b_id="del_me", result="no_match")
        exp3 = SynergyExploration(cap_a_id="ccc", cap_b_id="ddd", result="synergy")
        await repository.record_synergy_exploration(exp1)
        await repository.record_synergy_exploration(exp2)
        await repository.record_synergy_exploration(exp3)

        count = await repository.mark_stale_explorations("del_me")
        assert count == 2

        # Verify they are stale
        f1 = await repository.get_synergy_exploration("bbb", "del_me")
        assert f1.result == "stale"
        f2 = await repository.get_synergy_exploration("aaa", "del_me")
        assert f2.result == "stale"
        # Unrelated one is untouched
        f3 = await repository.get_synergy_exploration("ccc", "ddd")
        assert f3.result == "synergy"

    async def test_nonexistent_exploration_returns_none(self, repository):
        result = await repository.get_synergy_exploration("nope", "nada")
        assert result is None


# ---------------------------------------------------------------------------
# Repository: methodology links by type
# ---------------------------------------------------------------------------

class TestMethodologyLinksByType:
    async def test_get_links_by_type(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="cap A", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="cap B", source_task_id=sample_task.id)
        m3 = _make_methodology(problem="cap C", source_task_id=sample_task.id)
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)
        await repository.save_methodology(m3)

        await repository.upsert_methodology_link(m1.id, m2.id, "feeds_into", 0.9)
        await repository.upsert_methodology_link(m1.id, m3.id, "co_retrieval", 1.0)

        feeds_into = await repository.get_methodology_links_by_type(m1.id, "feeds_into")
        assert len(feeds_into) == 1
        assert feeds_into[0]["target_id"] == m2.id

        co_ret = await repository.get_methodology_links_by_type(m1.id, "co_retrieval")
        assert len(co_ret) == 1


# ---------------------------------------------------------------------------
# Repository: synergy graph traversal
# ---------------------------------------------------------------------------

class TestSynergyGraph:
    async def test_graph_depth_1(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="root", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="neighbor", source_task_id=sample_task.id)
        m3 = _make_methodology(problem="far", source_task_id=sample_task.id)
        for m in [m1, m2, m3]:
            await repository.save_methodology(m)

        await repository.upsert_methodology_link(m1.id, m2.id, "synergy", 0.8)
        await repository.upsert_methodology_link(m2.id, m3.id, "feeds_into", 0.7)

        graph = await repository.get_synergy_graph(m1.id, depth=1)
        assert m1.id in graph["nodes"]
        assert m2.id in graph["nodes"]
        # m3 is at depth 2, should NOT be in depth-1 traversal
        assert m3.id not in graph["nodes"]

    async def test_graph_depth_2(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="root", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="middle", source_task_id=sample_task.id)
        m3 = _make_methodology(problem="leaf", source_task_id=sample_task.id)
        for m in [m1, m2, m3]:
            await repository.save_methodology(m)

        await repository.upsert_methodology_link(m1.id, m2.id, "synergy", 0.8)
        await repository.upsert_methodology_link(m2.id, m3.id, "feeds_into", 0.7)

        graph = await repository.get_synergy_graph(m1.id, depth=2)
        assert m1.id in graph["nodes"]
        assert m2.id in graph["nodes"]
        assert m3.id in graph["nodes"]
        assert len(graph["edges"]) == 2

    async def test_graph_no_edges(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="isolated", source_task_id=sample_task.id)
        await repository.save_methodology(m1)

        graph = await repository.get_synergy_graph(m1.id, depth=2)
        assert graph["nodes"] == {m1.id}
        assert graph["edges"] == []


# ---------------------------------------------------------------------------
# Repository: complementary capabilities
# ---------------------------------------------------------------------------

class TestComplementaryCapabilities:
    async def test_follows_synergy_edges(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="base", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="complement", source_task_id=sample_task.id)
        m3 = _make_methodology(problem="unrelated", source_task_id=sample_task.id)
        for m in [m1, m2, m3]:
            await repository.save_methodology(m)

        await repository.upsert_methodology_link(m1.id, m2.id, "synergy", 0.8)
        # co_retrieval should NOT be followed
        await repository.upsert_methodology_link(m1.id, m3.id, "co_retrieval", 1.0)

        complements = await repository.get_complementary_capabilities(m1.id)
        complement_ids = {c.id for c in complements}
        assert m2.id in complement_ids
        assert m3.id not in complement_ids

    async def test_follows_feeds_into_and_enhances(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="center", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="feeder", source_task_id=sample_task.id)
        m3 = _make_methodology(problem="enhancer", source_task_id=sample_task.id)
        for m in [m1, m2, m3]:
            await repository.save_methodology(m)

        await repository.upsert_methodology_link(m2.id, m1.id, "feeds_into", 0.9)
        await repository.upsert_methodology_link(m3.id, m1.id, "enhances", 0.7)

        complements = await repository.get_complementary_capabilities(m1.id)
        complement_ids = {c.id for c in complements}
        assert m2.id in complement_ids
        assert m3.id in complement_ids

    async def test_excludes_dead_methodologies(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="alive", source_task_id=sample_task.id)
        m2 = _make_methodology(
            problem="dead complement",
            lifecycle_state="dead",
            source_task_id=sample_task.id,
        )
        for m in [m1, m2]:
            await repository.save_methodology(m)

        await repository.upsert_methodology_link(m1.id, m2.id, "synergy", 0.8)

        complements = await repository.get_complementary_capabilities(m1.id)
        assert len(complements) == 0

    async def test_no_complements_returns_empty(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="lonely", source_task_id=sample_task.id)
        await repository.save_methodology(m1)

        complements = await repository.get_complementary_capabilities(m1.id)
        assert complements == []


# ---------------------------------------------------------------------------
# Integration: delete methodology marks synergy explorations stale
# ---------------------------------------------------------------------------

class TestDeleteMethodologyStalesSynergies:
    async def test_delete_marks_explorations_stale(self, repository, sample_project, sample_task):
        m1 = _make_methodology(problem="to delete", source_task_id=sample_task.id)
        m2 = _make_methodology(problem="stays", source_task_id=sample_task.id)
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)

        # Record exploration between them
        a, b = (m1.id, m2.id) if m1.id < m2.id else (m2.id, m1.id)
        exp = SynergyExploration(cap_a_id=a, cap_b_id=b, result="synergy", synergy_score=0.9)
        await repository.record_synergy_exploration(exp)

        # Delete m1
        deleted = await repository.delete_methodology(m1.id)
        assert deleted is True

        # Exploration should be stale
        fetched = await repository.get_synergy_exploration(a, b)
        assert fetched.result == "stale"


# ---------------------------------------------------------------------------
# Schema: synergy_exploration_log table exists
# ---------------------------------------------------------------------------

class TestSynergyExplorationSchema:
    async def test_table_exists(self, db_engine):
        row = await db_engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='synergy_exploration_log'"
        )
        assert row["cnt"] == 1

    async def test_unique_constraint(self, db_engine):
        """Inserting duplicate (cap_a_id, cap_b_id) violates UNIQUE."""
        await db_engine.execute(
            "INSERT INTO synergy_exploration_log (id, cap_a_id, cap_b_id) VALUES (?, ?, ?)",
            ["id1", "aaa", "bbb"],
        )
        with pytest.raises(Exception):
            await db_engine.execute(
                "INSERT INTO synergy_exploration_log (id, cap_a_id, cap_b_id) VALUES (?, ?, ?)",
                ["id2", "aaa", "bbb"],
            )

    async def test_capability_data_column_exists(self, db_engine):
        """The capability_data column exists on methodologies table."""
        row = await db_engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('methodologies') WHERE name = 'capability_data'"
        )
        assert row["cnt"] == 1


# ---------------------------------------------------------------------------
# Migration: apply_migrations adds columns idempotently
# ---------------------------------------------------------------------------

class TestMigrations:
    async def test_apply_migrations_idempotent(self, db_engine):
        """Running apply_migrations twice doesn't error."""
        await db_engine.apply_migrations()
        await db_engine.apply_migrations()
        # No exception means success

    async def test_migration_adds_capability_data_if_missing(self):
        """Migration 3 adds capability_data to existing schemas without it."""
        from claw.core.config import DatabaseConfig
        from claw.db.engine import DatabaseEngine

        config = DatabaseConfig(db_path=":memory:")
        engine = DatabaseEngine(config)
        await engine.connect()

        # Create methodologies table WITHOUT capability_data
        await engine.conn.executescript("""
            CREATE TABLE IF NOT EXISTS methodologies (
                id TEXT PRIMARY KEY,
                problem_description TEXT NOT NULL,
                solution_code TEXT NOT NULL,
                prism_data TEXT
            );
        """)
        await engine.conn.commit()

        # Apply migrations — should add the column
        await engine.apply_migrations()

        row = await engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM pragma_table_info('methodologies') WHERE name = 'capability_data'"
        )
        assert row["cnt"] == 1
        await engine.close()


# ---------------------------------------------------------------------------
# Assimilation Engine: Helpers
# ---------------------------------------------------------------------------

class TestCanonicalPair:
    def test_already_ordered(self):
        assert _canonical_pair("aaa", "bbb") == ("aaa", "bbb")

    def test_reversed(self):
        assert _canonical_pair("zzz", "aaa") == ("aaa", "zzz")

    def test_same_ids(self):
        assert _canonical_pair("x", "x") == ("x", "x")


class TestParseCapabilityJson:
    def test_valid_json(self):
        raw = json.dumps({
            "inputs": [{"name": "x", "type": "text", "required": True}],
            "outputs": [{"name": "y", "type": "code_patch"}],
            "domain": ["testing"],
            "composability": {"can_chain_after": [], "can_chain_before": [], "standalone": True},
            "capability_type": "analysis",
        })
        result = _parse_capability_json(raw)
        assert result is not None
        assert result["capability_type"] == "analysis"
        assert len(result["inputs"]) == 1

    def test_json_with_markdown_fencing(self):
        raw = '```json\n{"inputs": [], "outputs": [], "domain": ["test"], "capability_type": "detection"}\n```'
        result = _parse_capability_json(raw)
        assert result is not None
        assert result["capability_type"] == "detection"

    def test_invalid_json(self):
        result = _parse_capability_json("not json at all")
        assert result is None

    def test_missing_fields_get_defaults(self):
        raw = json.dumps({"domain": ["x"]})
        result = _parse_capability_json(raw)
        assert result is not None
        assert result["inputs"] == []
        assert result["outputs"] == []

    def test_non_dict_returns_none(self):
        result = _parse_capability_json('"just a string"')
        assert result is None


class TestParseSynergyJson:
    def test_valid_synergy_response(self):
        raw = json.dumps({
            "has_synergy": True,
            "synergy_type": "feeds_into",
            "synergy_score": 0.85,
            "direction": "a_to_b",
            "reasoning": "A outputs what B needs",
        })
        result = _parse_synergy_json(raw)
        assert result is not None
        assert result["synergy_score"] == 0.85

    def test_fenced_json(self):
        raw = '```\n{"has_synergy": false, "synergy_score": 0.1}\n```'
        result = _parse_synergy_json(raw)
        assert result is not None
        assert result["has_synergy"] is False

    def test_invalid_returns_none(self):
        assert _parse_synergy_json("garbage") is None


# ---------------------------------------------------------------------------
# SynergyDiscoverer: scoring methods (no LLM calls)
# ---------------------------------------------------------------------------

class TestIOCompatibility:
    def setup_method(self):
        self.discoverer = SynergyDiscoverer.__new__(SynergyDiscoverer)

    def test_perfect_forward_match(self):
        cap_a = {"outputs": [{"type": "metrics_data"}]}
        cap_b = {"inputs": [{"type": "metrics_data"}]}
        score = self.discoverer._check_io_compatibility(cap_a, cap_b)
        assert score == 1.0

    def test_no_match(self):
        cap_a = {"outputs": [{"type": "text"}]}
        cap_b = {"inputs": [{"type": "model_artifact"}]}
        score = self.discoverer._check_io_compatibility(cap_a, cap_b)
        assert score == 0.0

    def test_partial_match(self):
        cap_a = {"outputs": [{"type": "text"}, {"type": "metrics_data"}]}
        cap_b = {"inputs": [{"type": "metrics_data"}, {"type": "config"}]}
        score = self.discoverer._check_io_compatibility(cap_a, cap_b)
        # forward: intersection={metrics_data}, union={text, metrics_data, config} → 1/3
        assert 0.3 <= score <= 0.4

    def test_reverse_match_preferred(self):
        """If reverse match is stronger, use that."""
        cap_a = {"inputs": [{"type": "analysis"}], "outputs": []}
        cap_b = {"outputs": [{"type": "analysis"}], "inputs": []}
        score = self.discoverer._check_io_compatibility(cap_a, cap_b)
        assert score == 1.0

    def test_empty_outputs_returns_zero(self):
        cap_a = {"outputs": [], "inputs": []}
        cap_b = {"outputs": [], "inputs": []}
        score = self.discoverer._check_io_compatibility(cap_a, cap_b)
        assert score == 0.0


class TestDomainOverlap:
    def setup_method(self):
        self.discoverer = SynergyDiscoverer.__new__(SynergyDiscoverer)

    def test_identical_domains(self):
        cap_a = {"domain": ["ml_training", "optimization"]}
        cap_b = {"domain": ["ml_training", "optimization"]}
        score = self.discoverer._check_domain_overlap(cap_a, cap_b)
        assert score == 1.0

    def test_no_overlap(self):
        cap_a = {"domain": ["security"]}
        cap_b = {"domain": ["web_development"]}
        score = self.discoverer._check_domain_overlap(cap_a, cap_b)
        assert score == 0.0

    def test_partial_overlap(self):
        cap_a = {"domain": ["ml_training", "data_processing"]}
        cap_b = {"domain": ["ml_training", "optimization"]}
        score = self.discoverer._check_domain_overlap(cap_a, cap_b)
        # Jaccard: {ml_training} / {ml_training, data_processing, optimization} = 1/3
        assert abs(score - 1.0 / 3) < 0.01

    def test_both_empty(self):
        score = self.discoverer._check_domain_overlap({"domain": []}, {"domain": []})
        assert score == 0.0


# ---------------------------------------------------------------------------
# CapabilityComposer: merge logic (no LLM, no DB)
# ---------------------------------------------------------------------------

class TestCapabilityComposerMerge:
    def setup_method(self):
        self.composer = CapabilityComposer.__new__(CapabilityComposer)

    def test_merge_deduplicates_domains(self):
        cap_a = {"domain": ["ml_training", "optimization"], "inputs": [], "outputs": [], "composability": {}}
        cap_b = {"domain": ["optimization", "data_processing"], "inputs": [], "outputs": [], "composability": {}}
        merged = self.composer._merge_capability_data(cap_a, cap_b)
        assert set(merged["domain"]) == {"ml_training", "optimization", "data_processing"}

    def test_merge_removes_internal_io(self):
        """Inputs satisfied by the other capability's outputs become internal."""
        cap_a = {
            "inputs": [{"name": "raw_data", "type": "text", "required": True}],
            "outputs": [{"name": "analysis", "type": "analysis", "required": True}],
            "domain": [],
            "composability": {},
        }
        cap_b = {
            "inputs": [{"name": "input_analysis", "type": "analysis", "required": True}],
            "outputs": [{"name": "report", "type": "documentation", "required": True}],
            "domain": [],
            "composability": {},
        }
        merged = self.composer._merge_capability_data(cap_a, cap_b)
        # B's input (analysis) is satisfied by A's output (analysis) → internal
        input_types = {i["type"] for i in merged["inputs"]}
        assert "analysis" not in input_types
        assert "text" in input_types

    def test_merge_preserves_all_outputs(self):
        cap_a = {
            "inputs": [],
            "outputs": [{"name": "x", "type": "text"}],
            "domain": [],
            "composability": {},
        }
        cap_b = {
            "inputs": [],
            "outputs": [{"name": "y", "type": "code_patch"}],
            "domain": [],
            "composability": {},
        }
        merged = self.composer._merge_capability_data(cap_a, cap_b)
        output_types = {o["type"] for o in merged["outputs"]}
        assert "text" in output_types
        assert "code_patch" in output_types

    def test_merge_composability(self):
        cap_a = {
            "inputs": [], "outputs": [], "domain": [],
            "composability": {"can_chain_after": ["data_collection"], "can_chain_before": [], "standalone": True},
        }
        cap_b = {
            "inputs": [], "outputs": [], "domain": [],
            "composability": {"can_chain_after": [], "can_chain_before": ["reporting"], "standalone": False},
        }
        merged = self.composer._merge_capability_data(cap_a, cap_b)
        comp = merged["composability"]
        assert "data_collection" in comp["can_chain_after"]
        assert "reporting" in comp["can_chain_before"]
        assert comp["standalone"] is False  # AND of True and False

    def test_merged_capability_type_is_composite(self):
        cap_a = {"inputs": [], "outputs": [], "domain": [], "composability": {}}
        cap_b = {"inputs": [], "outputs": [], "domain": [], "composability": {}}
        merged = self.composer._merge_capability_data(cap_a, cap_b)
        assert merged["capability_type"] == "composite"


# ---------------------------------------------------------------------------
# CapabilityComposer: compose (real DB, no LLM)
# ---------------------------------------------------------------------------

class TestCapabilityComposerCompose:
    async def test_compose_creates_methodology(self, repository, sample_project, sample_task):
        from claw.core.config import load_config
        config = load_config()

        composer = CapabilityComposer(repository, config)

        m1 = _make_methodology(
            problem="detect grokking",
            solution="def detect(): pass",
            capability_data=_make_capability_data(
                domain=["ml_training"],
                capability_type="detection",
            ),
            source_task_id=sample_task.id,
        )
        m2 = _make_methodology(
            problem="fine-tune model",
            solution="def tune(): pass",
            capability_data=_make_capability_data(
                domain=["ml_training", "optimization"],
                capability_type="optimization",
            ),
            source_task_id=sample_task.id,
        )
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)

        composite_id = await composer.compose(m1, m2, synergy_type="feeds_into")
        assert composite_id is not None

        composite = await repository.get_methodology(composite_id)
        assert composite is not None
        assert composite.lifecycle_state == "embryonic"
        assert composite.generation == 1
        assert set(composite.parent_ids) == {m1.id, m2.id}
        assert "composite" in composite.tags
        assert "auto_composed" in composite.tags
        assert composite.capability_data is not None
        assert composite.capability_data["capability_type"] == "composite"

    async def test_compose_respects_generation_cap(self, repository, sample_project, sample_task):
        from claw.core.config import load_config
        config = load_config()

        composer = CapabilityComposer(repository, config)
        max_gen = config.governance.self_consume_max_generation

        m1 = _make_methodology(
            problem="high gen",
            capability_data=_make_capability_data(),
            generation=max_gen,
            source_task_id=sample_task.id,
        )
        m2 = _make_methodology(
            problem="also high gen",
            capability_data=_make_capability_data(),
            generation=max_gen,
            source_task_id=sample_task.id,
        )
        await repository.save_methodology(m1)
        await repository.save_methodology(m2)

        composite_id = await composer.compose(m1, m2)
        assert composite_id is None  # blocked by generation cap
