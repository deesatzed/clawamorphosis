"""CLAW Cycle — the core orchestration abstraction.

The Claw Cycle is a six-step loop: grab -> evaluate -> decide -> act -> verify -> learn
operating at four nested scales:

- MacroClaw (Fleet) — scans repo fleet, ranks by enhancement potential
- MesoClaw (Project) — runs evaluation battery on one repo, produces plan
- MicroClaw (Module) — takes one task, routes to agent, monitors/verifies
- NanoClaw (Self-improvement) — updates scores and routing after each task
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from claw.core.factory import ClawContext
from claw.core.models import (
    CycleResult,
    HypothesisEntry,
    HypothesisOutcome,
    Task,
    TaskContext,
    TaskOutcome,
    TaskStatus,
    VerificationResult,
)

logger = logging.getLogger("claw.cycle")


def _snapshot_workspace(workspace_dir: Optional[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not workspace_dir:
        return snapshot

    root = Path(workspace_dir)
    if not root.exists() or not root.is_dir():
        return snapshot

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        snapshot[str(rel)] = hashlib.sha1(data).hexdigest()
    return snapshot


def _compute_workspace_change(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], str]:
    changed_paths = sorted(set(before.keys()) | set(after.keys()))
    files_changed = [path for path in changed_paths if before.get(path) != after.get(path)]
    if not files_changed:
        return [], ""

    lines: list[str] = []
    for path in files_changed:
        if path not in before:
            lines.append(f"+++ {path}")
        elif path not in after:
            lines.append(f"--- {path}")
        else:
            lines.append(f"*** {path}")
    return files_changed, "\n".join(lines)


class ClawCycle(ABC):
    """Abstract base for all claw cycle levels."""

    def __init__(self, ctx: ClawContext, level: str):
        self.ctx = ctx
        self.level = level

    @abstractmethod
    async def grab(self) -> Any:
        """Select the next unit of work."""

    @abstractmethod
    async def evaluate(self, target: Any) -> Any:
        """Analyze the target for enhancement potential."""

    @abstractmethod
    async def decide(self, evaluation: Any) -> Any:
        """Choose the best approach/agent for the work."""

    @abstractmethod
    async def act(self, decision: Any) -> Any:
        """Execute the chosen approach."""

    @abstractmethod
    async def verify(self, result: Any) -> Any:
        """Validate the output (tests, quality gates)."""

    @abstractmethod
    async def learn(self, outcome: Any) -> None:
        """Update scores, memory, and routing from the outcome."""

    async def run_cycle(self, on_step=None) -> CycleResult:
        """Execute one complete grab->evaluate->decide->act->verify->learn cycle.

        Args:
            on_step: Optional callback ``(step_name: str, detail: str) -> None``
                     called at each phase transition for progress reporting.
        """
        def _step(name: str, detail: str = "") -> None:
            if on_step is not None:
                on_step(name, detail)

        start = time.monotonic()
        try:
            _step("grab", "Fetching next task...")
            target = await self.grab()
            if target is None:
                return CycleResult(cycle_level=self.level, success=False)

            _step("evaluate", f"Analyzing: {target.title[:60]}")
            evaluation = await self.evaluate(target)

            _step("decide", "Selecting best agent...")
            decision = await self.decide(evaluation)
            agent_id = decision[0] if isinstance(decision, tuple) else "unknown"
            _step("act", f"Agent '{agent_id}' working...")
            result = await self.act(decision)

            _step("verify", "Running verification checks...")
            verification = await self.verify(result)

            _step("learn", "Recording outcome...")
            await self.learn(verification)

            duration = time.monotonic() - start
            # Unpack the verification tuple for result fields
            v_agent_id = verification[0] if isinstance(verification, tuple) else None
            v_outcome = verification[2] if isinstance(verification, tuple) and len(verification) > 2 else TaskOutcome()
            v_result = verification[3] if isinstance(verification, tuple) and len(verification) > 3 else None
            _step("done", f"Cycle complete ({duration:.1f}s)")
            return CycleResult(
                cycle_level=self.level,
                task_id=getattr(target, "id", None),
                agent_id=v_agent_id,
                outcome=v_outcome,
                verification=v_result,
                success=True,
                tokens_used=v_outcome.tokens_used if v_outcome else 0,
                cost_usd=v_outcome.cost_usd if v_outcome else 0.0,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("Cycle %s failed: %s", self.level, e, exc_info=True)
            return CycleResult(
                cycle_level=self.level,
                success=False,
                duration_seconds=duration,
            )


class MicroClaw(ClawCycle):
    """Single-task cycle: grab one task -> route to agent -> verify -> learn.

    This is the Phase 1 implementation. It processes one task from the
    work queue through the full pipeline.
    """

    def __init__(
        self,
        ctx: ClawContext,
        project_id: str,
        session_id: Optional[str] = None,
    ):
        super().__init__(ctx, level="micro")
        self.project_id = project_id
        self.session_id = session_id or str(uuid.uuid4())
        self._current_task: Optional[Task] = None
        self._current_outcome: Optional[TaskOutcome] = None
        self._current_verification: Optional[VerificationResult] = None

    async def grab(self) -> Optional[Task]:
        """Get the next pending task for the project."""
        task = await self.ctx.repository.get_next_task(self.project_id)
        if task is None:
            logger.info("No pending tasks for project %s", self.project_id)
            return None

        self._current_task = task
        logger.info("Grabbed task: %s (priority=%d)", task.title, task.priority)

        # Log episode
        await self.ctx.repository.log_episode(
            session_id=self.session_id,
            event_type="task_grabbed",
            event_data={"task_id": task.id, "title": task.title},
            project_id=self.project_id,
            task_id=task.id,
            cycle_level="micro",
        )

        return task

    async def evaluate(self, task: Task) -> TaskContext:
        """Build enriched task context with forbidden approaches and hints."""
        await self.ctx.repository.update_task_status(task.id, TaskStatus.EVALUATING)

        # Get failed approaches for this task
        failed = await self.ctx.repository.get_failed_approaches(task.id)
        forbidden = [h.approach_summary for h in failed]

        # Enrich with project-wide error KB forbidden approaches
        if self.ctx.error_kb is not None:
            try:
                enriched = await self.ctx.error_kb.get_enriched_forbidden_approaches(
                    task.id, self.project_id
                )
                forbidden = enriched
            except Exception as e:
                logger.warning(
                    "Error KB enrichment failed for task %s: %s", task.id, e
                )

        # Query semantic memory for similar past solutions as hints
        hints: list[str] = []
        if self.ctx.semantic_memory is not None:
            try:
                similar = await self.ctx.semantic_memory.find_similar(
                    task.description, limit=3
                )
                if similar:
                    for s in similar:
                        if s.methodology and s.methodology.methodology_notes:
                            hints.append(
                                f"Similar past solution: {s.methodology.methodology_notes}"
                            )
                        # Graph-enhanced: follow synergy edges for complementary capabilities
                        if s.methodology and self.ctx.assimilation_engine is not None:
                            try:
                                complements = await self.ctx.repository.get_complementary_capabilities(
                                    s.methodology.id
                                )
                                for comp in complements[:2]:
                                    hints.append(
                                        f"Complementary capability: {comp.problem_description[:200]}"
                                    )
                            except Exception:
                                pass  # Non-critical enhancement
            except Exception as e:
                logger.warning(
                    "Semantic memory lookup failed for task %s: %s", task.id, e
                )

        # Surface top novel capabilities as hints (novelty >= 0.7)
        if self.ctx.repository is not None:
            try:
                novel = await self.ctx.repository.get_most_novel_methodologies(
                    limit=2, min_novelty=0.7
                )
                for nm in novel:
                    hints.append(
                        f"Novel capability (novelty={nm.novelty_score:.2f}): "
                        f"{nm.problem_description[:200]}"
                    )
            except Exception:
                pass  # Non-critical enhancement

        action_template = None
        if task.action_template_id:
            try:
                action_template = await self.ctx.repository.get_action_template(
                    task.action_template_id
                )
                if action_template is None:
                    logger.warning(
                        "Task %s references missing action template %s",
                        task.id,
                        task.action_template_id,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load action template %s for task %s: %s",
                    task.action_template_id,
                    task.id,
                    e,
                )

        # Add explicit runbook guidance for execution-oriented tasks.
        runbook_steps = list(task.execution_steps)
        runbook_checks = list(task.acceptance_checks)
        runbook_preconditions: list[str] = []
        runbook_rollback: list[str] = []
        if action_template is not None:
            if not runbook_steps:
                runbook_steps = list(action_template.execution_steps)
            if not runbook_checks:
                runbook_checks = list(action_template.acceptance_checks)
            runbook_preconditions = list(action_template.preconditions)
            runbook_rollback = list(action_template.rollback_steps)

        for precondition in runbook_preconditions[:3]:
            hints.append(f"Runbook precondition: {precondition}")
        for step in runbook_steps[:5]:
            hints.append(f"Runbook execute: {step}")
        for check in runbook_checks[:5]:
            hints.append(f"Runbook verify: {check}")
        for rollback in runbook_rollback[:2]:
            hints.append(f"Runbook rollback: {rollback}")

        task_ctx = TaskContext(
            task=task,
            forbidden_approaches=forbidden,
            hints=hints,
            action_template=action_template,
        )

        logger.info(
            "Evaluated task: %d forbidden approaches, %d hints",
            len(forbidden), len(hints),
        )
        return task_ctx

    async def decide(self, task_ctx: TaskContext) -> tuple[str, TaskContext]:
        """Decide which agent to use via Dispatcher + Degradation checks."""
        await self.ctx.repository.update_task_status(task_ctx.task.id, TaskStatus.DISPATCHED)

        # Check degradation: ensure at least one agent is healthy
        if self.ctx.degradation_manager is not None:
            if self.ctx.degradation_manager.is_all_down():
                logger.error("All agents down — escalating to human")
                return ("none", task_ctx)

        # Use Dispatcher for Bayesian routing (with 10% exploration)
        if self.ctx.dispatcher is not None:
            try:
                agent_id = await self.ctx.dispatcher.route_task(task_ctx.task, task_ctx)
            except Exception as e:
                logger.warning("Dispatcher routing failed: %s, falling back", e)
                agent_id = task_ctx.task.recommended_agent or "claude"
        else:
            agent_id = task_ctx.task.recommended_agent or "claude"

        # Check degradation for the chosen agent; get fallback if needed
        if self.ctx.degradation_manager is not None:
            healthy = self.ctx.degradation_manager.get_healthy_agents()
            if agent_id not in healthy:
                fallback = self.ctx.degradation_manager.get_fallback_agent(agent_id)
                if fallback is not None:
                    logger.info("Agent '%s' degraded, falling back to '%s'", agent_id, fallback)
                    agent_id = fallback

        if agent_id not in self.ctx.agents:
            available = list(self.ctx.agents.keys())
            if available:
                agent_id = available[0]
            else:
                logger.error("No agents available")
                return ("none", task_ctx)

        await self.ctx.repository.update_task_agent(task_ctx.task.id, agent_id)
        logger.info("Decided: routing to agent '%s'", agent_id)

        return (agent_id, task_ctx)

    async def act(self, decision: tuple[str, TaskContext]) -> tuple[str, TaskContext, TaskOutcome]:
        """Execute the task through the chosen agent, with budget check."""
        agent_id, task_ctx = decision

        if agent_id == "none" or agent_id not in self.ctx.agents:
            return (agent_id, task_ctx, TaskOutcome(
                agent_id=agent_id,
                failure_reason="no_agent",
                failure_detail="No agent available to execute task",
            ))

        # Budget check before dispatch
        if self.ctx.budget_enforcer is not None:
            budget_results = await self.ctx.budget_enforcer.check_all(
                task_id=task_ctx.task.id,
                project_id=self.project_id,
                agent_id=agent_id,
            )
            exceeded = [r for r in budget_results if r.exceeded]
            if exceeded:
                first = exceeded[0]
                logger.warning(
                    "Budget exceeded (%s): %s",
                    first.check_type, first.entity_id,
                )
                return (agent_id, task_ctx, TaskOutcome(
                    agent_id=agent_id,
                    failure_reason="budget_exceeded",
                    failure_detail=f"Budget cap hit: {first.check_type} ({first.entity_id})",
                ))

        await self.ctx.repository.update_task_status(task_ctx.task.id, TaskStatus.CODING)
        await self.ctx.repository.increment_task_attempt(task_ctx.task.id)

        agent = self.ctx.agents[agent_id]
        workspace_dir = getattr(agent, "workspace_dir", None)
        before_snapshot = _snapshot_workspace(workspace_dir)

        # Set token tracking context
        self.ctx.token_tracker.set_context(
            task_id=task_ctx.task.id,
            agent_id=agent_id,
            agent_role=agent_id,
        )

        outcome = await agent.run(task_ctx)
        after_snapshot = _snapshot_workspace(workspace_dir)
        actual_files_changed, actual_diff = _compute_workspace_change(before_snapshot, after_snapshot)

        # Trust the real workspace diff over model self-report.
        if actual_files_changed:
            outcome.files_changed = actual_files_changed
            outcome.diff = actual_diff
        elif not outcome.failure_reason:
            outcome.files_changed = []
            outcome.diff = ""
            outcome.failure_reason = "no_workspace_changes"
            outcome.failure_detail = (
                "Agent returned without modifying any workspace files."
            )
            outcome.tests_passed = False

        self._current_outcome = outcome

        logger.info(
            "Act complete: agent=%s, tests_passed=%s, files=%d",
            agent_id, outcome.tests_passed, len(outcome.files_changed),
        )

        return (agent_id, task_ctx, outcome)

    async def verify(self, result: tuple[str, TaskContext, TaskOutcome]) -> tuple[str, TaskContext, TaskOutcome, VerificationResult]:
        """Verify the agent's output using the full 7-check Verifier."""
        agent_id, task_ctx, outcome = result
        await self.ctx.repository.update_task_status(task_ctx.task.id, TaskStatus.REVIEWING)

        if self.ctx.verifier is not None and not outcome.failure_reason:
            # Use the full 7-check Verifier
            verification = await self.ctx.verifier.verify(
                outcome=outcome,
                task_context=task_ctx,
                workspace_dir=getattr(self.ctx.agents.get(agent_id), "workspace_dir", None),
            )
        else:
            # Fallback: basic checks if verifier unavailable or execution failed
            violations = []
            if outcome.failure_reason:
                violations.append({"check": "execution", "detail": outcome.failure_reason})
            if outcome.raw_output:
                for marker in ["TODO", "FIXME", "NotImplementedError", "placeholder", "mock"]:
                    if marker.lower() in outcome.raw_output.lower():
                        violations.append({"check": "placeholder_scan", "detail": f"Found '{marker}' in output"})

            verification = VerificationResult(
                approved=len(violations) == 0 and outcome.tests_passed,
                violations=violations,
                quality_score=1.0 if not violations else 0.5,
            )

        self._current_verification = verification

        logger.info(
            "Verify: approved=%s, violations=%d",
            verification.approved, len(verification.violations),
        )

        return (agent_id, task_ctx, outcome, verification)

    async def learn(self, verified: tuple[str, TaskContext, TaskOutcome, VerificationResult]) -> None:
        """Update memory, scores, error KB, and semantic memory from the outcome."""
        agent_id, task_ctx, outcome, verification = verified
        task = task_ctx.task

        if verification.approved:
            # Success path
            await self.ctx.repository.update_task_status(task.id, TaskStatus.DONE)

            # Log successful hypothesis
            attempt = await self.ctx.repository.get_next_hypothesis_attempt(task.id)
            await self.ctx.repository.log_hypothesis(HypothesisEntry(
                task_id=task.id,
                attempt_number=attempt,
                approach_summary=outcome.approach_summary[:500],
                outcome=HypothesisOutcome.SUCCESS,
                files_changed=outcome.files_changed,
                duration_seconds=outcome.duration_seconds,
                model_used=outcome.model_used,
                agent_id=agent_id,
            ))

            # Update agent score
            await self.ctx.repository.update_agent_score(
                agent_id=agent_id,
                task_type=task.task_type or "general",
                success=True,
                duration_seconds=outcome.duration_seconds,
                quality_score=verification.quality_score or 0.0,
                cost_usd=outcome.cost_usd,
            )

            # Save successful pattern to semantic memory + trigger assimilation
            if self.ctx.semantic_memory is not None and outcome.approach_summary:
                try:
                    saved_meth = await self.ctx.semantic_memory.save_solution(
                        problem_description=task.description,
                        solution_code=outcome.raw_output or outcome.approach_summary,
                        source_task_id=task.id,
                        methodology_notes=outcome.approach_summary,
                        tags=[task.task_type or "general"],
                    )
                    logger.info(
                        "Saved successful pattern to semantic memory for task %s",
                        task.title,
                    )
                    # Trigger capability assimilation on the newly saved methodology
                    if saved_meth and self.ctx.assimilation_engine is not None:
                        try:
                            await self.ctx.assimilation_engine.assimilate(saved_meth.id)
                        except Exception as ae:
                            logger.warning("Assimilation failed for %s: %s", saved_meth.id, ae)
                except Exception as e:
                    logger.warning(
                        "Failed to save pattern to semantic memory for task %s: %s",
                        task.id, e,
                    )

            # Extract cross-project patterns if enough completions
            if self.ctx.pattern_learner is not None:
                try:
                    patterns = await self.ctx.pattern_learner.extract_patterns(
                        self.project_id
                    )
                    if patterns:
                        logger.info(
                            "Extracted %d patterns from project %s",
                            len(patterns), self.project_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Pattern extraction failed for project %s: %s",
                        self.project_id, e,
                    )

            logger.info("Learned: task %s completed by %s", task.title, agent_id)

        else:
            # Failure path
            error_sig = outcome.failure_reason or "unknown"
            attempt = await self.ctx.repository.get_next_hypothesis_attempt(task.id)
            await self.ctx.repository.log_hypothesis(HypothesisEntry(
                task_id=task.id,
                attempt_number=attempt,
                approach_summary=outcome.approach_summary[:500] if outcome.approach_summary else "Failed attempt",
                outcome=HypothesisOutcome.FAILURE,
                error_signature=error_sig,
                error_full=outcome.failure_detail,
                files_changed=outcome.files_changed,
                duration_seconds=outcome.duration_seconds,
                model_used=outcome.model_used,
                agent_id=agent_id,
            ))

            # Update agent score (failure)
            await self.ctx.repository.update_agent_score(
                agent_id=agent_id,
                task_type=task.task_type or "general",
                success=False,
                duration_seconds=outcome.duration_seconds,
                quality_score=verification.quality_score or 0.0,
                cost_usd=outcome.cost_usd,
            )

            # Record failure in error KB for cross-task pattern detection
            if self.ctx.error_kb is not None:
                try:
                    await self.ctx.error_kb.record_attempt(
                        task_id=task.id,
                        attempt_number=attempt,
                        approach_summary=outcome.approach_summary or "Failed attempt",
                        outcome=HypothesisOutcome.FAILURE,
                        error_signature=error_sig,
                        error_full=outcome.failure_detail,
                        files_changed=outcome.files_changed,
                        duration_seconds=outcome.duration_seconds,
                        model_used=outcome.model_used,
                        agent_id=agent_id,
                    )
                    logger.info(
                        "Recorded failure in error KB for task %s (error: %s)",
                        task.id, error_sig,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to record error in KB for task %s: %s",
                        task.id, e,
                    )

            # Reset to PENDING for retry
            await self.ctx.repository.update_task_status(task.id, TaskStatus.PENDING)

            logger.info(
                "Learned: task %s failed by %s (error: %s)",
                task.title, agent_id, error_sig,
            )

        if task.action_template_id:
            try:
                await self.ctx.repository.update_action_template_outcome(
                    task.action_template_id,
                    verification.approved,
                )
            except Exception as e:
                logger.warning(
                    "Failed to update action template outcome for %s: %s",
                    task.action_template_id,
                    e,
                )

        # Governance sweep (periodic, amortized over cycles)
        if self.ctx.governance is not None:
            try:
                swept = await self.ctx.governance.maybe_run_sweep()
                if swept:
                    logger.info("Governance sweep completed during learn phase")
            except Exception as e:
                logger.warning("Governance sweep failed: %s", e)

        # Log episode
        await self.ctx.repository.log_episode(
            session_id=self.session_id,
            event_type="cycle_completed",
            event_data={
                "task_id": task.id,
                "agent_id": agent_id,
                "approved": verification.approved,
                "quality_score": verification.quality_score,
            },
            project_id=self.project_id,
            agent_id=agent_id,
            task_id=task.id,
            cycle_level="micro",
        )


class MesoClaw(ClawCycle):
    """Project-level cycle: evaluate repo -> plan tasks -> run MicroClaw for each.

    MesoClaw operates at the project level. It runs the evaluation battery
    against a repository to identify issues, plans tasks from the findings,
    stores the tasks, and then spawns MicroClaw cycles for each task.

    After all MicroClaw cycles complete, MesoClaw triggers prompt evolution
    if enough samples have been collected.
    """

    def __init__(
        self,
        ctx: ClawContext,
        project_id: str,
        repo_path: str,
        session_id: Optional[str] = None,
    ):
        super().__init__(ctx, level="meso")
        self.project_id = project_id
        self.repo_path = repo_path
        self.session_id = session_id or str(uuid.uuid4())

    async def grab(self) -> Any:
        """Return the repo path as the target."""
        return self.repo_path

    async def evaluate(self, target: Any) -> Any:
        """Run the evaluation battery against the repository.

        Uses the Evaluator to execute all 17 prompts (or as many as
        the dispatcher can handle) and collects the results into an
        EvaluationReport.
        """
        from claw.evaluator import Evaluator

        evaluator = Evaluator(
            repository=self.ctx.repository,
            dispatcher=self.ctx.dispatcher,
        )
        report = await evaluator.run_battery(self.project_id, str(target))

        logger.info(
            "MesoClaw evaluation complete: %d/%d prompts succeeded",
            report.successful_prompts, report.total_prompts,
        )
        return report

    async def decide(self, evaluation: Any) -> Any:
        """Plan tasks from evaluation results.

        Converts the EvaluationReport's phase/prompt results into
        EvaluationResult objects that the Planner can consume, then
        runs gap analysis to generate a prioritized task list.
        """
        from claw.planner import EvaluationResult, Planner

        planner = Planner(
            project_id=self.project_id,
            repository=self.ctx.repository,
        )

        # Convert EvaluationReport phases/prompts into EvaluationResult objects
        eval_results: list[EvaluationResult] = []
        for phase in evaluation.phases:
            for pr in phase.prompt_results:
                if pr.output:
                    eval_results.append(EvaluationResult(
                        prompt_name=pr.prompt_name,
                        findings=[pr.output],
                        severity="medium",
                        category=phase.phase_name,
                        raw_output=pr.output,
                    ))

        tasks = await planner.analyze_gaps(eval_results)
        logger.info(
            "MesoClaw planning complete: %d tasks generated from %d evaluation results",
            len(tasks), len(eval_results),
        )
        return tasks

    async def act(self, decision: Any) -> Any:
        """Store tasks and run MicroClaw for each.

        Creates each planned task in the database, then runs a MicroClaw
        cycle for each task in sequence. Collects all cycle results.
        """
        tasks = decision
        results: list[CycleResult] = []

        for task in tasks:
            await self.ctx.repository.create_task(task)

        micro = MicroClaw(
            ctx=self.ctx,
            project_id=self.project_id,
            session_id=self.session_id,
        )

        for _ in range(len(tasks)):
            result = await micro.run_cycle()
            results.append(result)

        logger.info(
            "MesoClaw executed %d MicroClaw cycles (%d successful)",
            len(results), sum(1 for r in results if r.success),
        )
        return results

    async def verify(self, result: Any) -> Any:
        """Aggregate MicroClaw results.

        Returns a tuple of (successes, total, results_list) for the
        learn phase to consume.
        """
        results: list[CycleResult] = result
        successes = sum(1 for r in results if r.success)
        total = len(results)

        logger.info(
            "MesoClaw verification: %d/%d MicroClaw cycles succeeded",
            successes, total,
        )
        return (successes, total, results)

    async def learn(self, outcome: Any) -> None:
        """Update routing and trigger prompt evolution after enough samples.

        After all MicroClaw cycles complete, evaluates A/B tests for
        all prompts that have both control and variant rows, and
        promotes winners. Also logs the meso-level cycle completion.
        """
        successes, total, results = outcome

        # Trigger prompt evolution after enough samples
        if self.ctx.prompt_evolver is not None and total >= 5:
            try:
                # Evaluate A/B tests for all prompts with active tests
                tests = await self.ctx.prompt_evolver.list_tests()
                for test_group in tests:
                    prompt_name = test_group["prompt_name"]
                    agent_id = test_group.get("agent_id")
                    eval_result = await self.ctx.prompt_evolver.evaluate_test(
                        prompt_name, agent_id
                    )
                    if eval_result.get("ready") and eval_result.get("winner"):
                        await self.ctx.prompt_evolver.promote_variant(
                            prompt_name,
                            eval_result["winner"],
                            agent_id,
                        )
                        logger.info(
                            "Promoted prompt variant '%s/%s' (agent=%s)",
                            prompt_name, eval_result["winner"], agent_id,
                        )
            except Exception as e:
                logger.warning("Prompt evolution failed: %s", e)

        await self.ctx.repository.log_episode(
            session_id=self.session_id,
            event_type="meso_cycle_completed",
            event_data={"successes": successes, "total": total},
            project_id=self.project_id,
            cycle_level="meso",
        )

        logger.info(
            "MesoClaw learn complete: %d/%d tasks succeeded", successes, total,
        )

    async def run_cycle(self, on_step=None) -> CycleResult:
        """Execute one complete MesoClaw cycle.

        Overrides the base run_cycle to handle MesoClaw-specific flow
        where grab() returns a string (repo_path) not a Task, and
        verify/learn return aggregated results.
        """
        def _step(name: str, detail: str = "") -> None:
            if on_step is not None:
                on_step(name, detail)

        start = time.monotonic()
        try:
            _step("grab", f"Targeting repo: {self.repo_path}")
            target = await self.grab()

            _step("evaluate", f"Running evaluation battery on {target}")
            evaluation = await self.evaluate(target)

            _step("decide", "Planning tasks from evaluation...")
            decision = await self.decide(evaluation)

            _step("act", f"Running {len(decision)} MicroClaw cycles...")
            result = await self.act(decision)

            _step("verify", "Aggregating results...")
            verification = await self.verify(result)

            _step("learn", "Updating routing and prompt evolution...")
            await self.learn(verification)

            duration = time.monotonic() - start
            successes, total, _results = verification
            _step("done", f"MesoClaw complete: {successes}/{total} ({duration:.1f}s)")

            return CycleResult(
                cycle_level=self.level,
                project_id=self.project_id,
                success=successes > 0,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("MesoClaw cycle failed: %s", e, exc_info=True)
            return CycleResult(
                cycle_level=self.level,
                project_id=self.project_id,
                success=False,
                duration_seconds=duration,
            )


class NanoClaw(ClawCycle):
    """Self-improvement cycle: update scores, routing, prompt variants.

    NanoClaw runs after task cycles to optimize the system itself.
    It evaluates current routing/prompt performance, extracts patterns
    from completed work, evaluates A/B tests, and promotes winning
    prompt variants.

    Unlike MicroClaw and MesoClaw, NanoClaw does not process external
    work -- it improves the internal machinery.
    """

    def __init__(self, ctx: ClawContext, project_id: str):
        super().__init__(ctx, level="nano")
        self.project_id = project_id

    async def grab(self) -> Any:
        """Get recent cycle results to learn from.

        Returns the project_id as the target for self-improvement.
        """
        return self.project_id

    async def evaluate(self, target: Any) -> Any:
        """Assess current routing and prompt performance.

        Gathers task status summary and pattern extraction summary
        to understand what can be improved.
        """
        summary = await self.ctx.repository.get_task_status_summary(target)

        pattern_summary = None
        if self.ctx.pattern_learner is not None:
            try:
                pattern_summary = await self.ctx.pattern_learner.get_pattern_summary(
                    target
                )
            except Exception as e:
                logger.warning("Pattern summary failed: %s", e)

        return {
            "task_summary": summary,
            "pattern_summary": pattern_summary,
            "project_id": target,
        }

    async def decide(self, evaluation: Any) -> Any:
        """Determine what self-improvement actions to take.

        Based on the evaluation, decides which optimization actions
        are available and should be executed.
        """
        actions: list[str] = []

        if self.ctx.prompt_evolver is not None:
            actions.append("evolve_prompts")
        if self.ctx.pattern_learner is not None:
            actions.append("extract_patterns")
        if self.ctx.self_consumer is not None:
            actions.append("self_consume")
        if self.ctx.assimilation_engine is not None:
            actions.append("enrich_capabilities")

        return actions

    async def act(self, decision: Any) -> Any:
        """Execute self-improvement actions.

        Runs prompt evolution (A/B test evaluation + promotion),
        pattern extraction, and self-consumption based on the
        decided actions.
        """
        actions = decision
        results: dict[str, Any] = {}

        if "evolve_prompts" in actions and self.ctx.prompt_evolver is not None:
            try:
                tests = await self.ctx.prompt_evolver.list_tests()
                promoted_count = 0
                for test_group in tests:
                    prompt_name = test_group["prompt_name"]
                    agent_id = test_group.get("agent_id")
                    eval_result = await self.ctx.prompt_evolver.evaluate_test(
                        prompt_name, agent_id
                    )
                    if eval_result.get("ready") and eval_result.get("winner"):
                        await self.ctx.prompt_evolver.promote_variant(
                            prompt_name,
                            eval_result["winner"],
                            agent_id,
                        )
                        promoted_count += 1
                results["prompt_evolution"] = f"evaluated {len(tests)} tests, promoted {promoted_count}"
            except Exception as e:
                results["prompt_evolution"] = f"failed: {e}"
                logger.warning("NanoClaw prompt evolution failed: %s", e)

        if "extract_patterns" in actions and self.ctx.pattern_learner is not None:
            try:
                patterns = await self.ctx.pattern_learner.extract_patterns(
                    self.project_id
                )
                results["patterns_extracted"] = len(patterns)
            except Exception as e:
                results["patterns_extracted"] = f"failed: {e}"
                logger.warning("NanoClaw pattern extraction failed: %s", e)

        if "self_consume" in actions and self.ctx.self_consumer is not None:
            try:
                sc_report = await self.ctx.self_consumer.run_full_consumption(
                    self.project_id
                )
                results["self_consumption"] = {
                    "patterns_found": sc_report.patterns_found,
                    "patterns_stored": sc_report.patterns_stored,
                    "blocked_dedup": sc_report.patterns_blocked_dedup,
                    "blocked_generation": sc_report.patterns_blocked_generation,
                    "analysis_types": sc_report.analysis_types,
                }
                logger.info(
                    "NanoClaw self-consumption: found=%d, stored=%d",
                    sc_report.patterns_found, sc_report.patterns_stored,
                )
            except Exception as e:
                results["self_consumption"] = f"failed: {e}"
                logger.warning("NanoClaw self-consumption failed: %s", e)

        # Capability enrichment sweep: find unenriched methodologies and assimilate
        if "enrich_capabilities" in actions and self.ctx.assimilation_engine is not None:
            try:
                self.ctx.assimilation_engine.reset_cycle_counter()
                unenriched = await self.ctx.repository.get_methodologies_without_capability_data(
                    limit=5
                )
                enriched_count = 0
                for meth in unenriched:
                    try:
                        result_info = await self.ctx.assimilation_engine.assimilate(meth.id)
                        if result_info.get("enriched"):
                            enriched_count += 1
                    except Exception as e:
                        logger.debug("Enrichment failed for %s: %s", meth.id, e)
                results["capability_enrichment"] = {
                    "unenriched_found": len(unenriched),
                    "enriched": enriched_count,
                }
                logger.info(
                    "NanoClaw capability enrichment: %d/%d enriched",
                    enriched_count, len(unenriched),
                )
            except Exception as e:
                results["capability_enrichment"] = f"failed: {e}"
                logger.warning("NanoClaw capability enrichment failed: %s", e)

        return results

    async def verify(self, result: Any) -> Any:
        """Verify self-improvement results.

        Self-improvement is self-verifying through A/B tests and
        pattern confidence scores. The act results are passed through.
        """
        return result

    async def learn(self, outcome: Any) -> None:
        """Log self-improvement cycle completion."""
        await self.ctx.repository.log_episode(
            session_id=str(uuid.uuid4()),
            event_type="nano_cycle_completed",
            event_data=outcome if isinstance(outcome, dict) else {"result": str(outcome)},
            project_id=self.project_id,
            cycle_level="nano",
        )

        logger.info(
            "NanoClaw cycle complete for project %s: %s",
            self.project_id, outcome,
        )

    async def run_cycle(self, on_step=None) -> CycleResult:
        """Execute one complete NanoClaw self-improvement cycle.

        Overrides the base run_cycle because NanoClaw's grab() returns
        a string (project_id) not a Task, and the verify/learn flow
        differs from the micro-level tuple unpacking.
        """
        def _step(name: str, detail: str = "") -> None:
            if on_step is not None:
                on_step(name, detail)

        start = time.monotonic()
        try:
            _step("grab", f"Self-improvement for project {self.project_id}")
            target = await self.grab()

            _step("evaluate", "Assessing routing and prompt performance...")
            evaluation = await self.evaluate(target)

            _step("decide", "Determining optimization actions...")
            decision = await self.decide(evaluation)

            _step("act", f"Executing: {', '.join(decision)}")
            result = await self.act(decision)

            _step("verify", "Verifying self-improvement...")
            verification = await self.verify(result)

            _step("learn", "Logging self-improvement outcome...")
            await self.learn(verification)

            duration = time.monotonic() - start
            _step("done", f"NanoClaw complete ({duration:.1f}s)")

            return CycleResult(
                cycle_level=self.level,
                project_id=self.project_id,
                success=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            logger.error("NanoClaw cycle failed: %s", e, exc_info=True)
            return CycleResult(
                cycle_level=self.level,
                project_id=self.project_id,
                success=False,
                duration_seconds=duration,
            )
