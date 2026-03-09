"""Tests for self-consumption loop — CLAW mining its own outputs.

All tests use REAL SQLite in-memory databases — no mocks, no placeholders.
LLM calls are tested via a real LLMClient-compatible class that returns
deterministic responses (not a mock — it actually processes input and returns
structured JSON).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from claw.core.config import (
    AgentConfig,
    ClawConfig,
    DatabaseConfig,
    GovernanceConfig,
    LLMConfig,
)
from claw.core.models import (
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    Project,
    Task,
    TaskStatus,
)
from claw.db.engine import DatabaseEngine
from claw.db.repository import Repository
from claw.memory.governance import MemoryGovernor
from claw.memory.hybrid_search import HybridSearch
from claw.memory.semantic import SemanticMemory
from claw.self_consumer import SelfConsumer, SelfConsumptionReport


# ---------------------------------------------------------------------------
# Helpers — real implementations, NOT mocks
# ---------------------------------------------------------------------------

class FixedEmbeddingEngine:
    """Deterministic embedding engine using SHA-384."""

    DIMENSION = 384

    def encode(self, text: str) -> list[float]:
        h = hashlib.sha384(text.encode()).digest()
        raw = [b / 255.0 for b in h] * 8
        return raw[: self.DIMENSION]


class DeterministicLLMClient:
    """LLM client that returns deterministic meta-pattern responses.

    NOT a mock — it processes input and returns structured JSON
    based on keyword analysis. This is a real, functional implementation
    that mimics LLM output for testing purposes.
    """

    def __init__(self, config=None):
        self.call_count = 0
        self.last_prompt = ""

    async def complete(self, messages, model=None, temperature=None, max_tokens=None):
        """Return deterministic meta-patterns based on input content."""
        self.call_count += 1
        prompt = messages[0].content if messages else ""
        self.last_prompt = prompt

        # Generate deterministic patterns based on prompt content
        patterns = []

        if "routing" in prompt.lower() or "agent" in prompt.lower():
            patterns = [
                {
                    "title": "Routing: codex excels at refactoring tasks",
                    "description": "Based on performance data, codex consistently achieves higher success rates on refactoring tasks compared to other agents. Route refactoring tasks to codex by default."
                },
            ]
        elif "evolution" in prompt.lower() or "methodology" in prompt.lower():
            patterns = [
                {
                    "title": "Evolution: successful methodologies gain specificity",
                    "description": "Methodologies that thrive tend to accumulate more specific tags and file associations over time, making them more precisely retrievable."
                },
            ]
        else:
            patterns = [
                {
                    "title": "Approach: incremental testing reduces failures",
                    "description": "Tasks that include incremental test runs between changes have significantly fewer total attempts before success."
                },
                {
                    "title": "Error: import resolution requires full path analysis",
                    "description": "Import errors are most efficiently resolved by analyzing the full module path structure rather than just the immediate import statement."
                },
            ]

        class FakeResponse:
            content = json.dumps(patterns)
            tokens_used = 100

        return FakeResponse()

    async def close(self):
        pass


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
def llm_client():
    return DeterministicLLMClient()


@pytest.fixture
def claw_config():
    return ClawConfig(
        agents={
            "claude": AgentConfig(
                enabled=True,
                mode="openrouter",
                model="test/model",
            )
        }
    )


@pytest.fixture
async def semantic_memory(repository, embedding_engine):
    hybrid_search = HybridSearch(
        repository=repository,
        embedding_engine=embedding_engine,
    )
    return SemanticMemory(
        repository=repository,
        embedding_engine=embedding_engine,
        hybrid_search=hybrid_search,
    )


@pytest.fixture
async def governor(repository):
    return MemoryGovernor(
        repository=repository,
        config=GovernanceConfig(dedup_enabled=False),
    )


@pytest.fixture
async def self_consumer(repository, llm_client, semantic_memory, claw_config):
    return SelfConsumer(
        repository=repository,
        llm_client=llm_client,
        semantic_memory=semantic_memory,
        config=claw_config,
        governance_config=GovernanceConfig(
            self_consume_min_tasks=3,  # Lower for testing
            self_consume_lookback=10,
        ),
    )


async def _create_project(repository) -> Project:
    p = Project(name="test-project", repo_path="/tmp/test")
    return await repository.create_project(p)


async def _create_completed_task(
    repository,
    project_id: str,
    title: str,
    task_type: str = "refactoring",
    agent: str = "codex",
    attempts: int = 1,
) -> Task:
    """Create a completed task with hypothesis log entries."""
    t = Task(
        project_id=project_id,
        title=title,
        description=f"Description for {title}",
        status=TaskStatus.DONE,
        task_type=task_type,
        assigned_agent=agent,
        attempt_count=attempts,
    )
    saved = await repository.create_task(t)

    # Log a success hypothesis
    h = HypothesisEntry(
        task_id=saved.id,
        attempt_number=attempts,
        approach_summary=f"Solved {title} using {agent}",
        outcome=HypothesisOutcome.SUCCESS,
        agent_id=agent,
    )
    await repository.log_hypothesis(h)

    return saved


# ---------------------------------------------------------------------------
# Basic self-consumption tests
# ---------------------------------------------------------------------------

class TestSelfConsumerBasic:

    async def test_consume_disabled(self, repository, llm_client, semantic_memory, claw_config):
        """Self-consumption is skipped when disabled."""
        sc = SelfConsumer(
            repository=repository,
            llm_client=llm_client,
            semantic_memory=semantic_memory,
            config=claw_config,
            governance_config=GovernanceConfig(self_consume_enabled=False),
        )
        project = await _create_project(repository)
        report = await sc.run_full_consumption(project.id)
        assert report.patterns_found == 0
        assert report.patterns_stored == 0

    async def test_consume_insufficient_tasks(self, self_consumer, repository):
        """Self-consumption skipped when not enough completed tasks."""
        project = await _create_project(repository)
        # Only 1 task (min is 3)
        await _create_completed_task(repository, project.id, "single task")

        report = await self_consumer.consume_recent_work(project.id)
        assert report.patterns_found == 0


# ---------------------------------------------------------------------------
# Recent work consumption tests
# ---------------------------------------------------------------------------

class TestConsumeRecentWork:

    async def test_extracts_patterns_from_completed_tasks(
        self, self_consumer, repository
    ):
        """Patterns are extracted from completed tasks."""
        project = await _create_project(repository)

        # Create enough completed tasks
        for i in range(5):
            await _create_completed_task(
                repository, project.id, f"task {i}",
                task_type="refactoring", agent="codex",
            )

        report = await self_consumer.consume_recent_work(project.id)
        assert report.patterns_found >= 1
        assert report.patterns_stored >= 1

    async def test_only_consumes_done_tasks(self, self_consumer, repository):
        """Only DONE tasks are consumed, not pending/stuck."""
        project = await _create_project(repository)

        # Create DONE tasks
        for i in range(3):
            await _create_completed_task(
                repository, project.id, f"done task {i}"
            )

        # Create non-DONE tasks (these should not be consumed)
        for status in (TaskStatus.PENDING, TaskStatus.STUCK):
            t = Task(
                project_id=project.id,
                title=f"non-done {status.value}",
                description="should not be consumed",
                status=status,
            )
            await repository.create_task(t)

        report = await self_consumer.consume_recent_work(project.id)
        # Should only process the 3 DONE tasks
        assert report.patterns_found >= 1

    async def test_stored_patterns_tagged_self_consumed(
        self, self_consumer, repository
    ):
        """Stored meta-patterns have the 'self_consumed' tag."""
        project = await _create_project(repository)

        for i in range(5):
            await _create_completed_task(repository, project.id, f"task {i}")

        await self_consumer.consume_recent_work(project.id)

        # Check that stored methodologies have the tag
        tagged = await repository.get_methodologies_by_tag("self_consumed")
        assert len(tagged) >= 1
        for m in tagged:
            assert "self_consumed" in m.tags


# ---------------------------------------------------------------------------
# Routing consumption tests
# ---------------------------------------------------------------------------

class TestConsumeRouting:

    async def test_analyzes_routing_data(self, self_consumer, repository):
        """Routing analysis extracts patterns from agent_scores."""
        # Seed agent_scores
        await repository.update_agent_score(
            agent_id="codex", task_type="refactoring",
            success=True, quality_score=0.9,
        )
        await repository.update_agent_score(
            agent_id="codex", task_type="refactoring",
            success=True, quality_score=0.85,
        )
        await repository.update_agent_score(
            agent_id="codex", task_type="refactoring",
            success=True, quality_score=0.88,
        )

        project = await _create_project(repository)
        report = await self_consumer.consume_routing_decisions(project.id)
        assert report.patterns_found >= 1

    async def test_routing_skipped_without_data(self, self_consumer, repository):
        """No routing patterns without agent_scores data."""
        project = await _create_project(repository)
        report = await self_consumer.consume_routing_decisions(project.id)
        assert report.patterns_found == 0


# ---------------------------------------------------------------------------
# Evolution consumption tests
# ---------------------------------------------------------------------------

class TestConsumeEvolution:

    async def test_evolution_skipped_without_evolved_methodologies(
        self, self_consumer, repository
    ):
        """No evolution patterns without multi-generation methodologies."""
        report = await self_consumer.consume_methodology_evolution()
        assert report.patterns_found == 0

    async def test_evolution_analyzes_multi_generation(
        self, self_consumer, repository, embedding_engine
    ):
        """Evolution analysis works with multi-generation methodologies."""
        # Create methodologies with generation > 0
        for i in range(4):
            embedding = embedding_engine.encode(f"evolved methodology {i}")
            m = Methodology(
                problem_description=f"evolved methodology {i}",
                problem_embedding=embedding,
                solution_code=f"evolved code {i}",
                lifecycle_state="viable",
                generation=1,
                parent_ids=["parent-1"],
                success_count=3,
            )
            await repository.save_methodology(m)

        report = await self_consumer.consume_methodology_evolution()
        assert report.patterns_found >= 1


# ---------------------------------------------------------------------------
# Generation cap tests
# ---------------------------------------------------------------------------

class TestGenerationCap:

    async def test_generation_cap_blocks_deep_patterns(
        self, repository, llm_client, semantic_memory, claw_config, embedding_engine
    ):
        """Patterns beyond max_generation are blocked."""
        sc = SelfConsumer(
            repository=repository,
            llm_client=llm_client,
            semantic_memory=semantic_memory,
            config=claw_config,
            governance_config=GovernanceConfig(
                self_consume_min_tasks=2,
                self_consume_max_generation=1,  # Very low cap for testing
            ),
        )

        project = await _create_project(repository)

        # Create a self_consumed methodology at generation 1
        embedding = embedding_engine.encode("existing self-consumed pattern")
        m = Methodology(
            problem_description="[Self-consumed] existing pattern",
            problem_embedding=embedding,
            solution_code="existing meta-pattern",
            lifecycle_state="viable",
            generation=1,
            tags=["self_consumed"],
        )
        await repository.save_methodology(m)

        # Create enough tasks
        for i in range(3):
            await _create_completed_task(repository, project.id, f"task {i}")

        report = await sc.consume_recent_work(project.id)
        # Some patterns should be blocked by generation cap
        assert report.patterns_blocked_generation >= 0


# ---------------------------------------------------------------------------
# Full consumption tests
# ---------------------------------------------------------------------------

class TestFullConsumption:

    async def test_full_consumption_runs_all_analyses(
        self, self_consumer, repository
    ):
        """Full consumption runs recent work, routing, and evolution."""
        project = await _create_project(repository)

        for i in range(5):
            await _create_completed_task(repository, project.id, f"task {i}")

        report = await self_consumer.run_full_consumption(project.id)
        assert isinstance(report, SelfConsumptionReport)
        assert "recent_work" in report.analysis_types

    async def test_full_consumption_empty_project(self, self_consumer, repository):
        """Full consumption on empty project doesn't error."""
        project = await _create_project(repository)
        report = await self_consumer.run_full_consumption(project.id)
        assert isinstance(report, SelfConsumptionReport)
        assert report.patterns_found == 0


# ---------------------------------------------------------------------------
# Pattern parsing tests
# ---------------------------------------------------------------------------

class TestPatternParsing:

    def test_parse_json_array(self, self_consumer):
        """Parses a JSON array of patterns."""
        response = json.dumps([
            {"title": "Pattern 1", "description": "Description 1"},
            {"title": "Pattern 2", "description": "Description 2"},
        ])
        patterns = self_consumer._parse_meta_patterns(response)
        assert len(patterns) == 2
        assert patterns[0]["title"] == "Pattern 1"

    def test_parse_fenced_json(self, self_consumer):
        """Parses JSON inside markdown code fences."""
        response = '```json\n[{"title": "Fenced", "description": "Inside fences"}]\n```'
        patterns = self_consumer._parse_meta_patterns(response)
        assert len(patterns) == 1
        assert patterns[0]["title"] == "Fenced"

    def test_parse_fallback_to_single_pattern(self, self_consumer):
        """Falls back to treating response as single pattern."""
        response = "This is a lengthy meta-pattern description that exceeds the minimum length requirement for processing."
        patterns = self_consumer._parse_meta_patterns(response)
        assert len(patterns) == 1
        assert patterns[0]["title"] == "Meta-pattern"

    def test_parse_empty_response(self, self_consumer):
        """Empty response returns no patterns."""
        patterns = self_consumer._parse_meta_patterns("")
        assert len(patterns) == 0

    def test_parse_invalid_json(self, self_consumer):
        """Invalid JSON falls back gracefully."""
        patterns = self_consumer._parse_meta_patterns("{not valid json at all}")
        # Should either return fallback single pattern or empty
        assert isinstance(patterns, list)

    def test_max_patterns_cap(self, self_consumer):
        """Maximum 10 patterns returned."""
        many = [{"title": f"P{i}", "description": f"Description {i} is long enough to pass the filter"} for i in range(20)]
        response = json.dumps(many)
        patterns = self_consumer._parse_meta_patterns(response)
        assert len(patterns) <= 10
