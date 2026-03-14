"""Abstract agent interface for CLAW.

All four agents (Claude, Codex, Gemini, Grok) implement this ABC.
Provides lifecycle timing, metrics, and structured TaskOutcome returns.

All agents can use OpenRouter mode (mode="openrouter") to route through
the OpenRouter API with any model. This is the recommended mode for
cost-controlled testing and multi-model comparison.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import httpx

from claw.core.models import AgentHealth, AgentMode, AgentResult, TaskContext, TaskOutcome


class AgentInterface(ABC):
    """Base class for all CLAW agents.

    Every agent follows the same lifecycle:
    1. Receive task context
    2. Execute (LLM calls, tool execution, etc.)
    3. Return a TaskOutcome with structured results
    4. Log metrics and errors throughout

    Subclasses must implement:
    - execute() — core task processing
    - health_check() — agent availability check
    - supported_modes — property listing modes (cli, api, cloud)
    - instruction_file — property with path to agent instruction file
    """

    def __init__(self, agent_id: str, name: str):
        """Initialize agent with id and name.

        Args:
            agent_id: Machine identifier (e.g., "claude", "codex", "gemini", "grok").
            name: Human-readable agent name (e.g., "Claude Code Agent").
        """
        self.agent_id = agent_id
        self.name = name
        self.logger = logging.getLogger(f"claw.agent.{agent_id}")
        self._metrics: dict[str, Any] = {
            "total_executed": 0,
            "total_errors": 0,
            "total_successes": 0,
            "last_duration_seconds": 0.0,
        }

    @abstractmethod
    async def execute(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute a task and return the outcome.

        Args:
            task: Enriched task context.
            context: Optional additional context (ContextBrief, etc.).

        Returns:
            TaskOutcome with files changed, test results, approach summary, etc.
        """

    @abstractmethod
    async def health_check(self) -> AgentHealth:
        """Check if this agent is available and operational.

        Returns:
            AgentHealth with availability status, mode, version, etc.
        """

    @property
    @abstractmethod
    def supported_modes(self) -> list[AgentMode]:
        """Return the modes this agent supports (cli, api, cloud)."""

    @property
    @abstractmethod
    def instruction_file(self) -> str:
        """Return the filename of this agent's instruction file (e.g., 'CLAUDE.md')."""

    async def run(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute the agent with lifecycle logging and metrics.

        This wraps execute() with start/complete/error tracking.
        Agents should override execute(), not run().
        """
        self._log_start(task)
        start = time.monotonic()

        try:
            result = await self.execute(task, context)
            duration = time.monotonic() - start
            result.duration_seconds = duration
            result.agent_id = self.agent_id
            self._metrics["total_executed"] += 1
            self._metrics["total_successes"] += 1
            self._metrics["last_duration_seconds"] = duration
            self._log_complete(duration, result)
            return result
        except Exception as e:
            duration = time.monotonic() - start
            self._metrics["total_executed"] += 1
            self._metrics["total_errors"] += 1
            self._metrics["last_duration_seconds"] = duration
            self._log_error(e)
            return TaskOutcome(
                agent_id=self.agent_id,
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _log_start(self, task: TaskContext) -> None:
        self.logger.info("[%s] Starting: task='%s'", self.name, task.task.title)

    def _log_complete(self, duration: float, result: TaskOutcome) -> None:
        status = "success" if result.tests_passed else "completed"
        self.logger.info(
            "[%s] Complete: status=%s (%.2fs)",
            self.name, status, duration,
        )

    def _log_error(self, error: Exception) -> None:
        self.logger.error(
            "[%s] Error: %s", self.name, error, exc_info=True,
        )

    def _resolve_workspace(self, task: TaskContext) -> Optional[str]:
        """Return a safe cwd for subprocess execution.

        Uses workspace_dir if set and valid. Never falls back to
        task.description to prevent path-traversal via untrusted input.
        """
        ws = getattr(self, "workspace_dir", None)
        if ws and Path(ws).is_dir():
            return ws
        return None

    async def execute_openrouter(
        self, task: TaskContext, context: Optional[Any] = None
    ) -> TaskOutcome:
        """Execute task via OpenRouter API (OpenAI-compatible).

        All agents share this method. It uses OPENROUTER_API_KEY and
        the model specified in the agent's config. No native SDK needed.
        """
        model = getattr(self, "model", None)
        if not model:
            return TaskOutcome(
                agent_id=self.agent_id,
                failure_reason="no_model",
                failure_detail="No model configured. Set model in claw.toml.",
            )

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return TaskOutcome(
                agent_id=self.agent_id,
                failure_reason="no_api_key",
                failure_detail="OPENROUTER_API_KEY not set in environment.",
            )

        prompt = self._build_openrouter_prompt(task, context)
        start = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/deesatzed/clawamorphosis",
                        "X-Title": "CLAW",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4096,
                    },
                )
                response.raise_for_status()
                data = response.json()

            duration = time.monotonic() - start

            # Parse response
            choices = data.get("choices", [])
            content = ""
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            usage = data.get("usage", {})
            tokens_used = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
            model_used = data.get("model", model)

            return TaskOutcome(
                approach_summary=content[:500],
                model_used=model_used,
                agent_id=self.agent_id,
                raw_output=content,
                tokens_used=tokens_used,
                tests_passed=True,
                duration_seconds=duration,
            )

        except httpx.HTTPStatusError as e:
            duration = time.monotonic() - start
            detail = str(e)
            try:
                err_body = e.response.json()
                detail = err_body.get("error", {}).get("message", detail)
            except Exception:
                pass
            return TaskOutcome(
                agent_id=self.agent_id,
                failure_reason=f"http_{e.response.status_code}",
                failure_detail=detail,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id=self.agent_id,
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _build_openrouter_prompt(
        self, task: TaskContext, context: Optional[Any] = None
    ) -> str:
        """Build prompt for OpenRouter execution. Agents can override."""
        parts = [f"# Task: {task.task.title}\n"]
        parts.append(task.task.description)

        execution_steps = list(task.task.execution_steps)
        acceptance_checks = list(task.task.acceptance_checks)

        if task.action_template is not None:
            if task.action_template.preconditions:
                parts.append("\n## Runbook Preconditions")
                for item in task.action_template.preconditions:
                    parts.append(f"- {item}")
            if not execution_steps:
                execution_steps = list(task.action_template.execution_steps)
            if not acceptance_checks:
                acceptance_checks = list(task.action_template.acceptance_checks)
            if task.action_template.rollback_steps:
                parts.append("\n## Rollback Steps")
                for step in task.action_template.rollback_steps:
                    parts.append(f"- {step}")

        if execution_steps:
            parts.append("\n## Execution Steps")
            for step in execution_steps:
                parts.append(f"- `{step}`")

        if acceptance_checks:
            parts.append("\n## Acceptance Checks")
            for check in acceptance_checks:
                parts.append(f"- `{check}`")

        if task.forbidden_approaches:
            parts.append("\n## Forbidden Approaches (already tried, failed)")
            for fa in task.forbidden_approaches:
                parts.append(f"- {fa}")

        if hasattr(task, "hints") and task.hints:
            parts.append("\n## Hints from Past Solutions")
            for hint in task.hints:
                parts.append(f"- {hint}")

        return "\n".join(parts)

    def get_metrics(self) -> dict[str, Any]:
        """Return a copy of the agent's runtime metrics."""
        return self._metrics.copy()
