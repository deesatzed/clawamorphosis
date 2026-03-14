"""Tests for CLAW database engine, schema, and repository."""

import struct

import pytest

from claw.core.models import (
    ActionTemplate,
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    PeerReview,
    Project,
    Task,
    TaskStatus,
    TokenCostRecord,
)


class TestDatabaseEngine:
    async def test_connect_and_schema(self, db_engine):
        """Engine connects and schema is created."""
        assert db_engine.conn is not None

    async def test_execute_and_fetch(self, db_engine):
        await db_engine.execute(
            "INSERT INTO projects (id, name, repo_path) VALUES (?, ?, ?)",
            ["p1", "test", "/tmp/test"],
        )
        row = await db_engine.fetch_one("SELECT * FROM projects WHERE id = ?", ["p1"])
        assert row is not None
        assert row["name"] == "test"

    async def test_fetch_all(self, db_engine):
        await db_engine.execute(
            "INSERT INTO projects (id, name, repo_path) VALUES (?, ?, ?)",
            ["p1", "test1", "/tmp/1"],
        )
        await db_engine.execute(
            "INSERT INTO projects (id, name, repo_path) VALUES (?, ?, ?)",
            ["p2", "test2", "/tmp/2"],
        )
        rows = await db_engine.fetch_all("SELECT * FROM projects ORDER BY name")
        assert len(rows) == 2

    async def test_transaction_commit(self, db_engine):
        async with db_engine.transaction():
            await db_engine.execute(
                "INSERT INTO projects (id, name, repo_path) VALUES (?, ?, ?)",
                ["p1", "tx-test", "/tmp/tx"],
            )
        row = await db_engine.fetch_one("SELECT * FROM projects WHERE id = ?", ["p1"])
        assert row is not None

    async def test_fts5_queryable(self, db_engine):
        await db_engine.execute(
            "INSERT INTO methodologies (id, problem_description, solution_code) VALUES (?, ?, ?)",
            ["m1", "Memory leak fix", "fix()"],
        )
        await db_engine.execute(
            "INSERT INTO methodology_fts (methodology_id, problem_description, methodology_notes, tags) VALUES (?, ?, ?, ?)",
            ["m1", "Memory leak fix", "", "[]"],
        )
        rows = await db_engine.fetch_all(
            "SELECT methodology_id FROM methodology_fts WHERE methodology_fts MATCH ?",
            ["memory"],
        )
        assert len(rows) == 1

    async def test_sqlite_vec_queryable(self, db_engine):
        vec = [0.5] * 384
        vec_bytes = struct.pack(f"<{len(vec)}f", *vec)
        await db_engine.execute(
            "INSERT INTO methodology_embeddings (methodology_id, embedding) VALUES (?, ?)",
            ["m1", vec_bytes],
        )
        query = struct.pack("<384f", *([0.5] * 384))
        rows = await db_engine.fetch_all(
            "SELECT methodology_id, distance FROM methodology_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT 5",
            [query],
        )
        assert len(rows) == 1
        assert rows[0]["distance"] < 0.001  # Identical vectors


class TestRepository:
    async def test_project_lifecycle(self, repository, sample_project):
        await repository.create_project(sample_project)
        got = await repository.get_project(sample_project.id)
        assert got is not None
        assert got.name == sample_project.name
        assert got.tech_stack == sample_project.tech_stack

    async def test_task_lifecycle(self, repository, sample_project, sample_task):
        await repository.create_project(sample_project)
        template = ActionTemplate(
            id="tmpl-seed",
            title="Seed runbook",
            problem_pattern="auth regression",
            execution_steps=["pytest -q tests/test_auth.py"],
            acceptance_checks=["pytest -q tests/test_auth.py"],
        )
        await repository.create_action_template(template)
        sample_task.action_template_id = template.id
        sample_task.execution_steps = ["pytest -q tests/test_auth.py"]
        sample_task.acceptance_checks = ["pytest -q tests/test_auth.py"]
        await repository.create_task(sample_task)

        # Get next task
        next_t = await repository.get_next_task(sample_project.id)
        assert next_t is not None
        assert next_t.title == sample_task.title

        # Update status
        await repository.update_task_status(sample_task.id, TaskStatus.CODING)
        got = await repository.get_task(sample_task.id)
        assert got.status == TaskStatus.CODING

        # Increment attempt
        await repository.increment_task_attempt(sample_task.id)
        got = await repository.get_task(sample_task.id)
        assert got.attempt_count == 1
        assert got.action_template_id == "tmpl-seed"
        assert got.execution_steps == ["pytest -q tests/test_auth.py"]
        assert got.acceptance_checks == ["pytest -q tests/test_auth.py"]

    async def test_action_template_lifecycle(self, repository):
        template = ActionTemplate(
            title="Node fix runbook",
            problem_pattern="failing API handler tests",
            execution_steps=["npm install", "npm run build"],
            acceptance_checks=["npm test -- --runInBand"],
            rollback_steps=["git restore src/api/handler.ts"],
            preconditions=["Node 20 installed"],
            source_repo="nanochat",
            confidence=0.6,
        )
        await repository.create_action_template(template)

        fetched = await repository.get_action_template(template.id)
        assert fetched is not None
        assert fetched.problem_pattern == "failing API handler tests"
        assert fetched.execution_steps == ["npm install", "npm run build"]
        assert fetched.acceptance_checks == ["npm test -- --runInBand"]

        listed = await repository.list_action_templates(source_repo="nanochat")
        assert any(t.id == template.id for t in listed)

        await repository.update_action_template_outcome(template.id, success=True)
        updated = await repository.get_action_template(template.id)
        assert updated is not None
        assert updated.success_count == 1
        assert updated.confidence > 0.6

        await repository.update_action_template_outcome(template.id, success=False)
        updated_2 = await repository.get_action_template(template.id)
        assert updated_2 is not None
        assert updated_2.failure_count == 1

    async def test_hypothesis_lifecycle(self, repository, sample_project, sample_task):
        await repository.create_project(sample_project)
        await repository.create_task(sample_task)

        h = HypothesisEntry(
            task_id=sample_task.id,
            attempt_number=1,
            approach_summary="Tried authentication fix",
            error_signature="AuthError",
            agent_id="claude",
        )
        await repository.log_hypothesis(h)

        failed = await repository.get_failed_approaches(sample_task.id)
        assert len(failed) == 1
        assert failed[0].agent_id == "claude"

        assert await repository.has_duplicate_error(sample_task.id, "AuthError") is True
        assert await repository.has_duplicate_error(sample_task.id, "OtherError") is False

    async def test_methodology_save_and_search(self, repository, sample_project, sample_task):
        await repository.create_project(sample_project)
        await repository.create_task(sample_task)

        m = Methodology(
            problem_description="Database connection pool leak",
            solution_code="pool.close()",
            problem_embedding=[0.2] * 384,
            tags=["database", "leak"],
            source_task_id=sample_task.id,
        )
        await repository.save_methodology(m)

        # FTS search
        results = await repository.search_methodologies_text("connection pool")
        assert len(results) >= 1

        # Vec search
        vec_results = await repository.find_similar_methodologies([0.2] * 384)
        assert len(vec_results) >= 1
        assert vec_results[0][1] > 0.99

    async def test_agent_scores(self, repository):
        await repository.update_agent_score("claude", "analysis", True, 5.0, 0.9, 0.02)
        await repository.update_agent_score("claude", "analysis", False, 10.0, 0.3, 0.05)

        scores = await repository.get_agent_scores("claude")
        assert len(scores) == 1
        assert scores[0]["successes"] == 1
        assert scores[0]["failures"] == 1
        assert scores[0]["total_attempts"] == 2

    async def test_token_cost_tracking(self, repository):
        tc = TokenCostRecord(
            agent_id="claude",
            model_used="test-model",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.05,
        )
        await repository.save_token_cost(tc)
        summary = await repository.get_token_cost_summary()
        assert summary["calls"] == 1
        assert summary["total_cost_usd"] == 0.05

    async def test_episodes(self, repository, sample_project):
        await repository.create_project(sample_project)
        ep_id = await repository.log_episode(
            session_id="test-session",
            event_type="test_event",
            event_data={"key": "value"},
            project_id=sample_project.id,
            agent_id="claude",
        )
        assert ep_id is not None
