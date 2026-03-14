"""Tests for cam kb — Knowledge Browser feature.

Tests the 4 new repository methods and CLI subcommand --help smoke tests.
Uses REAL SQLite in-memory DB with seeded data — no mocks.
"""

import json

import pytest

from claw.core.models import Methodology, SynergyExploration


# ---------------------------------------------------------------------------
# Repository method tests
# ---------------------------------------------------------------------------


class TestKBRepositoryMethods:
    """Test the 4 new repository methods with real SQLite."""

    async def _seed_methodologies(self, repository, count=5):
        """Seed test methodologies with capability_data, novelty, and potential scores."""
        ids = []
        for i in range(count):
            m = Methodology(
                problem_description=f"Test capability #{i}: solving problem {i}",
                solution_code=f"def solve_{i}(): pass",
                methodology_notes=f"Notes for cap {i}",
                lifecycle_state="viable" if i < 3 else "embryonic",
                novelty_score=0.5 + i * 0.1,
                potential_score=0.3 + i * 0.12,
                capability_data={
                    "capability_type": "transformation" if i % 2 == 0 else "analysis",
                    "domain": [f"domain_{j}" for j in range(1 + i % 3)],
                    "io_types_in": ["code"],
                    "io_types_out": ["code"],
                    "composability_score": 0.5 + i * 0.05,
                    "standalone_viable": True,
                },
                tags=[f"tag{i}"],
                language="python",
            )
            await repository.save_methodology(m)
            ids.append(m.id)

            # Also populate FTS5 so search works
            await repository.engine.execute(
                "INSERT INTO methodology_fts (methodology_id, problem_description, methodology_notes, tags) VALUES (?, ?, ?, ?)",
                [m.id, m.problem_description, m.methodology_notes or "", json.dumps(m.tags)],
            )
        return ids

    async def _seed_synergies(self, repository, cap_ids):
        """Seed synergy exploration log entries between capabilities."""
        from datetime import UTC, datetime

        pairs = []
        for i in range(min(len(cap_ids) - 1, 3)):
            a, b = cap_ids[i], cap_ids[i + 1]
            if a > b:
                a, b = b, a
            exp = SynergyExploration(
                cap_a_id=a,
                cap_b_id=b,
                result="synergy",
                synergy_score=0.8 - i * 0.1,
                synergy_type="complementary",
                exploration_method="test",
                details={"reason": f"Test synergy {i}"},
            )
            await repository.record_synergy_exploration(exp)
            pairs.append((a, b))
        return pairs

    # --- get_top_synergy_edges ---

    async def test_get_top_synergy_edges_empty(self, repository):
        """Returns empty list when no synergy data."""
        result = await repository.get_top_synergy_edges()
        assert result == []

    async def test_get_top_synergy_edges_with_data(self, repository):
        """Returns synergy edges sorted by score descending."""
        ids = await self._seed_methodologies(repository)
        pairs = await self._seed_synergies(repository, ids)

        edges = await repository.get_top_synergy_edges(limit=10)
        assert len(edges) == len(pairs)
        # Check sorted descending
        scores = [e["synergy_score"] for e in edges]
        assert scores == sorted(scores, reverse=True)
        # Check fields present
        first = edges[0]
        assert "cap_a_summary" in first
        assert "cap_b_summary" in first
        assert "synergy_score" in first
        assert "synergy_type" in first
        assert "cap_a_domains" in first
        assert "cap_b_domains" in first

    async def test_get_top_synergy_edges_limit(self, repository):
        """Limit parameter works."""
        ids = await self._seed_methodologies(repository)
        await self._seed_synergies(repository, ids)

        edges = await repository.get_top_synergy_edges(limit=1)
        assert len(edges) == 1

    # --- get_novelty_potential_distribution ---

    async def test_novelty_potential_distribution_empty(self, repository):
        """Returns zeros when no methodologies have scores."""
        dist = await repository.get_novelty_potential_distribution()
        assert dist["total"] == 0
        assert dist["avg_novelty"] == 0.0

    async def test_novelty_potential_distribution_with_data(self, repository):
        """Returns correct aggregates with seeded data."""
        await self._seed_methodologies(repository, count=5)
        dist = await repository.get_novelty_potential_distribution()
        assert dist["total"] == 5
        assert dist["avg_novelty"] > 0
        assert dist["max_novelty"] >= dist["avg_novelty"]
        assert dist["min_novelty"] <= dist["avg_novelty"]
        assert dist["avg_potential"] > 0

    # --- get_cross_domain_capabilities ---

    async def test_cross_domain_empty(self, repository):
        """Returns empty list when no capabilities."""
        result = await repository.get_cross_domain_capabilities()
        assert result == []

    async def test_cross_domain_finds_bridges(self, repository):
        """Finds capabilities with multiple domains."""
        ids = await self._seed_methodologies(repository, count=5)
        # Cap at index 2 has domain: ["domain_0", "domain_1", "domain_2"]
        # Cap at index 4 has domain: ["domain_0", "domain_1", "domain_2"]
        bridges = await repository.get_cross_domain_capabilities(min_domains=2)
        assert len(bridges) > 0
        for b in bridges:
            domains = (b.capability_data or {}).get("domain", [])
            assert len(domains) >= 2

    async def test_cross_domain_min_3(self, repository):
        """min_domains=3 filters to only 3+ domain capabilities."""
        await self._seed_methodologies(repository, count=5)
        bridges = await repository.get_cross_domain_capabilities(min_domains=3)
        for b in bridges:
            domains = (b.capability_data or {}).get("domain", [])
            assert len(domains) >= 3

    # --- get_methodology_by_prefix ---

    async def test_prefix_lookup_not_found(self, repository):
        """Returns None for unknown prefix."""
        result = await repository.get_methodology_by_prefix("zzzzzz")
        assert result is None

    async def test_prefix_lookup_success(self, repository):
        """Finds methodology by ID prefix."""
        ids = await self._seed_methodologies(repository, count=1)
        prefix = ids[0][:8]
        result = await repository.get_methodology_by_prefix(prefix)
        assert result is not None
        assert result.id == ids[0]

    async def test_prefix_lookup_full_id(self, repository):
        """Full ID also works."""
        ids = await self._seed_methodologies(repository, count=1)
        result = await repository.get_methodology_by_prefix(ids[0])
        assert result is not None
        assert result.id == ids[0]


# ---------------------------------------------------------------------------
# CLI smoke tests (--help, not actual DB calls)
# ---------------------------------------------------------------------------


class TestKBCLISmoke:
    """Verify all kb subcommands register and have help text."""

    def test_app_name_is_cam(self):
        from claw.cli import app
        assert app.info.name == "cam"

    def test_kb_group_exists(self):
        from claw.cli import kb_app
        assert kb_app.info.name == "kb"

    def test_kb_insights_registered(self):
        from claw.cli import kb_app
        names = [cmd.name or cmd.callback.__name__ for cmd in kb_app.registered_commands]
        assert "insights" in names

    def test_kb_search_registered(self):
        from claw.cli import kb_app
        names = [cmd.name or cmd.callback.__name__ for cmd in kb_app.registered_commands]
        assert "search" in names

    def test_kb_capability_registered(self):
        from claw.cli import kb_app
        names = [cmd.name or cmd.callback.__name__ for cmd in kb_app.registered_commands]
        assert "capability" in names

    def test_kb_domains_registered(self):
        from claw.cli import kb_app
        names = [cmd.name or cmd.callback.__name__ for cmd in kb_app.registered_commands]
        assert "domains" in names

    def test_kb_synergies_registered(self):
        from claw.cli import kb_app
        names = [cmd.name or cmd.callback.__name__ for cmd in kb_app.registered_commands]
        assert "synergies" in names


# ---------------------------------------------------------------------------
# Integration tests with seeded data
# ---------------------------------------------------------------------------


class TestKBIntegration:
    """Integration tests exercising kb queries on a seeded DB."""

    async def _seed_full_kb(self, repository):
        """Seed a full knowledge base for integration testing."""
        ids = []
        for i in range(10):
            m = Methodology(
                problem_description=f"Integration cap {i}: error handling and retry logic #{i}",
                solution_code=f"def cap_{i}(): pass",
                methodology_notes=f"Detailed notes about capability {i}",
                lifecycle_state=["thriving", "viable", "embryonic", "declining", "dormant"][i % 5],
                novelty_score=0.1 + i * 0.08,
                potential_score=0.2 + i * 0.07,
                capability_data={
                    "capability_type": ["transformation", "analysis", "generation"][i % 3],
                    "domain": [f"domain_{j}" for j in range(1 + i % 4)],
                    "io_types_in": ["code", "text"],
                    "io_types_out": ["code"],
                    "composability_score": 0.6,
                    "standalone_viable": True,
                },
                tags=[f"error", f"retry"],
                language="python",
            )
            await repository.save_methodology(m)
            ids.append(m.id)
            # FTS5 entry
            await repository.engine.execute(
                "INSERT INTO methodology_fts (methodology_id, problem_description, methodology_notes, tags) VALUES (?, ?, ?, ?)",
                [m.id, m.problem_description, m.methodology_notes or "", json.dumps(m.tags)],
            )

        # Seed synergies
        for i in range(min(len(ids) - 1, 5)):
            a, b = ids[i], ids[i + 1]
            if a > b:
                a, b = b, a
            exp = SynergyExploration(
                cap_a_id=a,
                cap_b_id=b,
                result="synergy",
                synergy_score=0.9 - i * 0.1,
                synergy_type="complementary" if i % 2 == 0 else "pipeline",
                exploration_method="test",
                details={"reason": f"Integration synergy {i}"},
            )
            await repository.record_synergy_exploration(exp)

        return ids

    async def test_insights_all_queries_work(self, repository):
        """All queries used by kb insights return valid data."""
        ids = await self._seed_full_kb(repository)

        total = await repository.count_methodologies()
        assert total == 10

        states = await repository.count_methodologies_by_state()
        assert sum(states.values()) == 10

        dist = await repository.get_novelty_potential_distribution()
        assert dist["total"] == 10

        top_novel = await repository.get_most_novel_methodologies(limit=5)
        assert len(top_novel) == 5

        top_potential = await repository.get_high_potential_methodologies(limit=5)
        assert len(top_potential) == 5

        domains = await repository.get_domain_distribution()
        assert len(domains) > 0

        types = await repository.get_type_distribution()
        assert len(types) > 0

        edges = await repository.get_top_synergy_edges(limit=5)
        assert len(edges) == 5

        stats = await repository.get_synergy_stats()
        assert stats["total_explored"] > 0

    async def test_search_finds_capabilities(self, repository):
        """FTS5 search returns matching capabilities."""
        await self._seed_full_kb(repository)
        results = await repository.search_methodologies_text("error handling", limit=5)
        assert len(results) > 0
        # Every result should mention "error" in the problem description
        for m in results:
            assert "error" in m.problem_description.lower()

    async def test_capability_detail_by_prefix(self, repository):
        """Can retrieve full capability detail by prefix."""
        ids = await self._seed_full_kb(repository)
        prefix = ids[0][:8]
        m = await repository.get_methodology_by_prefix(prefix)
        assert m is not None
        assert m.id == ids[0]
        assert m.capability_data is not None
        assert m.novelty_score is not None
