"""Tests for CLAW Pydantic data models."""

import json

from claw.core.models import (
    ActionTemplate,
    AgentHealth,
    AgentMode,
    AgentResult,
    ComplexityTier,
    ContextBrief,
    CycleResult,
    EscalationDiagnosis,
    ExecutionState,
    FleetTask,
    HypothesisEntry,
    HypothesisOutcome,
    LifecycleState,
    Methodology,
    MethodologyLink,
    MethodologyType,
    OperationalMode,
    PeerReview,
    Project,
    Task,
    TaskContext,
    TaskOutcome,
    TaskStatus,
    TokenCostRecord,
    VerificationResult,
)


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING == "PENDING"
        assert TaskStatus.DISPATCHED == "DISPATCHED"
        assert TaskStatus.DONE == "DONE"

    def test_agent_mode(self):
        assert AgentMode.CLI == "cli"
        assert AgentMode.API == "api"
        assert AgentMode.CLOUD == "cloud"

    def test_operational_mode(self):
        assert OperationalMode.ATTENDED == "attended"
        assert OperationalMode.AUTONOMOUS == "autonomous"

    def test_lifecycle_state(self):
        assert LifecycleState.EMBRYONIC == "embryonic"
        assert LifecycleState.DEAD == "dead"


class TestProjectModel:
    def test_roundtrip(self):
        p = Project(name="test", repo_path="/tmp/test")
        j = p.model_dump_json()
        p2 = Project.model_validate_json(j)
        assert p2.name == "test"
        assert p2.repo_path == "/tmp/test"
        assert p2.id == p.id

    def test_defaults(self):
        p = Project(name="x", repo_path="/x")
        assert p.tech_stack == {}
        assert p.banned_dependencies == []
        assert p.id  # auto-generated


class TestTaskModel:
    def test_roundtrip(self):
        t = Task(project_id="p1", title="Fix bug", description="Fix it")
        j = t.model_dump_json()
        t2 = Task.model_validate_json(j)
        assert t2.title == "Fix bug"
        assert t2.status == TaskStatus.PENDING

    def test_claw_fields(self):
        t = Task(
            project_id="p1", title="X", description="Y",
            task_type="analysis", recommended_agent="claude",
        )
        assert t.task_type == "analysis"
        assert t.recommended_agent == "claude"
        assert t.escalation_count == 0

    def test_action_runbook_fields(self):
        t = Task(
            project_id="p1",
            title="Patch dependency",
            description="Upgrade package and verify behavior",
            action_template_id="tmpl-1",
            execution_steps=["npm install"],
            acceptance_checks=["npm test"],
        )
        assert t.action_template_id == "tmpl-1"
        assert t.execution_steps == ["npm install"]
        assert t.acceptance_checks == ["npm test"]


class TestTaskOutcome:
    def test_roundtrip(self):
        to = TaskOutcome(
            files_changed=["a.py", "b.py"],
            tests_passed=True,
            agent_id="claude",
            tokens_used=1500,
            cost_usd=0.05,
        )
        j = to.model_dump_json()
        to2 = TaskOutcome.model_validate_json(j)
        assert to2.files_changed == ["a.py", "b.py"]
        assert to2.agent_id == "claude"
        assert to2.cost_usd == 0.05


class TestVerificationResult:
    def test_roundtrip(self):
        vr = VerificationResult(
            approved=True,
            quality_score=0.95,
            violations=[],
        )
        j = vr.model_dump_json()
        vr2 = VerificationResult.model_validate_json(j)
        assert vr2.approved is True
        assert vr2.quality_score == 0.95


class TestAgentHealth:
    def test_roundtrip(self):
        ah = AgentHealth(
            agent_id="claude",
            available=True,
            mode=AgentMode.CLI,
            version="1.0.0",
        )
        j = ah.model_dump_json()
        ah2 = AgentHealth.model_validate_json(j)
        assert ah2.agent_id == "claude"
        assert ah2.available is True
        assert ah2.mode == AgentMode.CLI


class TestCycleResult:
    def test_roundtrip(self):
        cr = CycleResult(cycle_level="micro", success=True, tokens_used=500)
        j = cr.model_dump_json()
        cr2 = CycleResult.model_validate_json(j)
        assert cr2.cycle_level == "micro"
        assert cr2.success is True


class TestFleetTask:
    def test_roundtrip(self):
        ft = FleetTask(
            repo_path="/repos/test",
            repo_name="test",
            priority=5.0,
            budget_allocated_usd=2.0,
        )
        j = ft.model_dump_json()
        ft2 = FleetTask.model_validate_json(j)
        assert ft2.repo_name == "test"
        assert ft2.priority == 5.0


class TestMethodology:
    def test_roundtrip(self):
        m = Methodology(
            problem_description="Auth bug",
            solution_code="fix()",
            tags=["auth"],
            fitness_vector={"relevance": 0.8},
            parent_ids=["p1", "p2"],
        )
        j = m.model_dump_json()
        m2 = Methodology.model_validate_json(j)
        assert m2.tags == ["auth"]
        assert m2.fitness_vector["relevance"] == 0.8
        assert m2.parent_ids == ["p1", "p2"]


class TestActionTemplate:
    def test_roundtrip(self):
        template = ActionTemplate(
            title="Python bugfix runbook",
            problem_pattern="module import failure",
            execution_steps=["python -m pytest tests/test_imports.py -q"],
            acceptance_checks=["pytest -q tests/test_imports.py"],
            rollback_steps=["git checkout -- src/module.py"],
            preconditions=["Dependencies are installed"],
            source_repo="sample-repo",
            confidence=0.8,
            success_count=4,
            failure_count=1,
        )
        payload = template.model_dump_json()
        template_2 = ActionTemplate.model_validate_json(payload)
        assert template_2.problem_pattern == "module import failure"
        assert template_2.execution_steps[0].startswith("python -m")
        assert template_2.acceptance_checks == ["pytest -q tests/test_imports.py"]
        assert template_2.confidence == 0.8
