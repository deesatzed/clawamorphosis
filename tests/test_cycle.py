"""Tests for CLAW cycle orchestration."""

from pathlib import Path

import pytest

from claw.core.models import (
    ActionTemplate,
    Project,
    Task,
    TaskContext,
    TaskOutcome,
    TaskStatus,
    VerificationResult,
)
from claw.cycle import MicroClaw


class TestMicroClaw:
    async def test_act_fails_when_agent_makes_no_workspace_changes(self, claw_context, sample_project, sample_task, tmp_path):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        workspace = tmp_path / "repo"
        workspace.mkdir()
        (workspace / "app.py").write_text("print('before')\n", encoding="utf-8")

        class NoChangeAgent:
            workspace_dir = str(workspace)

            async def run(self, task_ctx):
                return TaskOutcome(
                    approach_summary="Claimed update without touching files",
                    tests_passed=True,
                    files_changed=["fake.py"],
                    raw_output="updated fake.py",
                )

        ctx.agents["codex"] = NoChangeAgent()

        micro = MicroClaw(ctx, sample_project.id)
        task_ctx = TaskContext(task=sample_task)
        agent_id, _, outcome = await micro.act(("codex", task_ctx))

        assert agent_id == "codex"
        assert outcome.failure_reason == "no_workspace_changes"
        assert outcome.tests_passed is False
        assert outcome.files_changed == []
        assert outcome.diff == ""

    async def test_act_uses_real_workspace_changes(self, claw_context, sample_project, sample_task, tmp_path):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        workspace = tmp_path / "repo"
        workspace.mkdir()
        target = workspace / "app.py"
        target.write_text("print('before')\n", encoding="utf-8")

        class WriteAgent:
            workspace_dir = str(workspace)

            async def run(self, task_ctx):
                target.write_text("print('after')\n", encoding="utf-8")
                return TaskOutcome(
                    approach_summary="Updated app.py",
                    tests_passed=True,
                    files_changed=[],
                    raw_output="done",
                )

        ctx.agents["codex"] = WriteAgent()

        micro = MicroClaw(ctx, sample_project.id)
        task_ctx = TaskContext(task=sample_task)
        agent_id, _, outcome = await micro.act(("codex", task_ctx))

        assert agent_id == "codex"
        assert outcome.failure_reason is None
        assert outcome.files_changed == ["app.py"]
        assert "app.py" in outcome.diff

    async def test_grab_returns_task(self, claw_context, sample_project, sample_task):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        assert grabbed is not None
        assert grabbed.title == sample_task.title

    async def test_grab_returns_none_when_empty(self, claw_context, sample_project):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        assert grabbed is None

    async def test_grab_respects_priority(self, claw_context, sample_project):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)

        low = Task(project_id=sample_project.id, title="Low", description="low pri", priority=1)
        high = Task(project_id=sample_project.id, title="High", description="high pri", priority=10)
        await ctx.repository.create_task(low)
        await ctx.repository.create_task(high)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        assert grabbed.title == "High"

    async def test_evaluate_builds_context(self, claw_context, sample_project, sample_task):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        task_ctx = await micro.evaluate(grabbed)
        assert task_ctx.task.id == sample_task.id
        assert isinstance(task_ctx.forbidden_approaches, list)

    async def test_decide_routes_to_available_agent(self, claw_context, sample_project, sample_task):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        task_ctx = await micro.evaluate(grabbed)
        agent_id, decided_ctx = await micro.decide(task_ctx)

        # No agents in test context, but decide handles gracefully
        # The important thing is it doesn't crash
        assert isinstance(agent_id, str)

    async def test_full_cycle_status_tracking(self, claw_context, sample_project, sample_task):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        await ctx.repository.create_task(sample_task)

        micro = MicroClaw(ctx, sample_project.id)

        # Grab sets nothing yet
        grabbed = await micro.grab()
        assert grabbed is not None

        # Evaluate moves to EVALUATING
        task_ctx = await micro.evaluate(grabbed)
        got = await ctx.repository.get_task(sample_task.id)
        assert got.status == TaskStatus.EVALUATING

        # Decide moves to DISPATCHED
        agent_id, decided_ctx = await micro.decide(task_ctx)
        got = await ctx.repository.get_task(sample_task.id)
        assert got.status == TaskStatus.DISPATCHED

    async def test_evaluate_loads_action_template_into_context(
        self, claw_context, sample_project
    ):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        template = ActionTemplate(
            title="Auth patch template",
            problem_pattern="jwt auth regression",
            execution_steps=["pytest -q tests/test_auth.py"],
            acceptance_checks=["pytest -q tests/test_auth.py"],
            rollback_steps=["git restore src/auth.py"],
            preconditions=["pytest available in venv"],
        )
        await ctx.repository.create_action_template(template)
        task = Task(
            project_id=sample_project.id,
            title="Fix JWT bug",
            description="Repair login flow and verify",
            action_template_id=template.id,
        )
        await ctx.repository.create_task(task)

        micro = MicroClaw(ctx, sample_project.id)
        grabbed = await micro.grab()
        task_ctx = await micro.evaluate(grabbed)

        assert task_ctx.action_template is not None
        assert task_ctx.action_template.id == template.id
        assert any("Runbook execute:" in hint for hint in task_ctx.hints)
        assert any("Runbook verify:" in hint for hint in task_ctx.hints)

    async def test_learn_updates_action_template_feedback(
        self, claw_context, sample_project
    ):
        ctx = claw_context
        await ctx.repository.create_project(sample_project)
        template = ActionTemplate(
            title="Retry-safe template",
            problem_pattern="intermittent test failure",
            execution_steps=["pytest -q"],
            acceptance_checks=["pytest -q"],
            confidence=0.5,
        )
        await ctx.repository.create_action_template(template)
        task = Task(
            project_id=sample_project.id,
            title="Stabilize flaky tests",
            description="Stabilize and verify",
            action_template_id=template.id,
        )
        await ctx.repository.create_task(task)

        micro = MicroClaw(ctx, sample_project.id)
        verified = (
            "claude",
            TaskContext(task=task),
            TaskOutcome(
                approach_summary="Stabilized flaky test with deterministic fixture",
                tests_passed=True,
                files_changed=["tests/test_flaky.py"],
            ),
            VerificationResult(approved=True, quality_score=0.95),
        )
        await micro.learn(verified)

        updated_template = await ctx.repository.get_action_template(template.id)
        assert updated_template is not None
        assert updated_template.success_count == 1
        assert updated_template.confidence > 0.5
