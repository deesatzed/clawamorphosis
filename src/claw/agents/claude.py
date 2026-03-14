"""Claude Code Agent for CLAW.

Wraps Claude Code as a CLAW agent with two modes:
- CLI: Invokes `claude --print` subprocess (full CLI capabilities)
- API: Uses the Anthropic SDK directly for simpler queries

Model version comes from config — never hardcoded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from claw.agents.interface import AgentInterface
from claw.core.models import AgentHealth, AgentMode, TaskContext, TaskOutcome

logger = logging.getLogger("claw.agent.claude")


class ClaudeCodeAgent(AgentInterface):
    """Claude Code agent — analysis, docs, architecture, security."""

    def __init__(
        self,
        mode: AgentMode = AgentMode.CLI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        max_budget_usd: float = 1.0,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__(agent_id="claude", name="Claude Code Agent")
        self.mode = mode
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model  # User-set, never hardcoded
        self.timeout = timeout
        self.max_budget_usd = max_budget_usd
        self.workspace_dir = workspace_dir
        self._anthropic_client = None

    @property
    def supported_modes(self) -> list[AgentMode]:
        return [AgentMode.CLI, AgentMode.API]

    @property
    def instruction_file(self) -> str:
        return "CLAUDE.md"

    async def health_check(self) -> AgentHealth:
        """Check Claude Code availability."""
        if self.mode == AgentMode.OPENROUTER:
            return await self._openrouter_health_check()
        elif self.mode == AgentMode.CLI:
            return await self._cli_health_check()
        else:
            return await self._api_health_check()

    async def execute(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute a task using Claude Code."""
        if self.mode == AgentMode.OPENROUTER:
            return await self.execute_openrouter(task, context)
        elif self.mode == AgentMode.CLI:
            return await self._execute_cli(task, context)
        else:
            return await self._execute_api(task, context)

    async def _openrouter_health_check(self) -> AgentHealth:
        """Check OpenRouter availability for this agent."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return AgentHealth(
                agent_id="claude", available=False, mode=AgentMode.OPENROUTER,
                error="OPENROUTER_API_KEY not set",
            )
        if not self.model:
            return AgentHealth(
                agent_id="claude", available=False, mode=AgentMode.OPENROUTER,
                error="No model configured in claw.toml",
            )
        return AgentHealth(
            agent_id="claude", available=True, mode=AgentMode.OPENROUTER,
            version=f"openrouter:{self.model}",
        )

    # ------------------------------------------------------------------
    # CLI mode
    # ------------------------------------------------------------------

    async def _cli_health_check(self) -> AgentHealth:
        """Check if `claude` CLI is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                version = stdout.decode().strip()
                return AgentHealth(
                    agent_id="claude",
                    available=True,
                    mode=AgentMode.CLI,
                    version=version,
                )
            return AgentHealth(
                agent_id="claude",
                available=False,
                mode=AgentMode.CLI,
                error=stderr.decode().strip(),
            )
        except FileNotFoundError:
            return AgentHealth(
                agent_id="claude",
                available=False,
                mode=AgentMode.CLI,
                error="claude CLI not found in PATH",
            )
        except Exception as e:
            return AgentHealth(
                agent_id="claude",
                available=False,
                mode=AgentMode.CLI,
                error=str(e),
            )

    async def _execute_cli(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via `claude --print` subprocess."""
        prompt = self._build_prompt(task, context)

        cmd = [
            "claude", "--print",
            "--output-format", "json",
            "--dangerously-skip-permissions",
        ]

        if self.max_budget_usd > 0:
            cmd.extend(["--max-budget-usd", str(self.max_budget_usd)])

        cwd = self._resolve_workspace(task)

        proc = None
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=self.timeout,
            )

            duration = time.monotonic() - start
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")

            return self._parse_cli_result(stdout, stderr, proc.returncode or 0, duration)

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="claude",
                failure_reason="timeout",
                failure_detail=f"Claude CLI timed out after {self.timeout}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="claude",
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

    def _parse_cli_result(
        self, stdout: str, stderr: str, exit_code: int, duration: float
    ) -> TaskOutcome:
        """Parse Claude CLI JSON output into TaskOutcome."""
        raw_output = stdout
        tokens_used = 0
        cost_usd = 0.0
        model_used = self.model or ""

        # Try to parse JSON envelope from --output-format json
        try:
            data = json.loads(stdout)
            if isinstance(data, dict) and data.get("type") == "result":
                raw_output = data.get("result", stdout)
                usage = data.get("usage", {}) or data.get("modelUsage", {})
                if isinstance(usage, dict):
                    # Sum up tokens across all models if modelUsage
                    for model_info in (usage.values() if not usage.get("input_tokens") else [usage]):
                        if isinstance(model_info, dict):
                            tokens_used += model_info.get("input_tokens", 0)
                            tokens_used += model_info.get("output_tokens", 0)
                cost_info = data.get("costUsd") or data.get("cost_usd")
                if cost_info:
                    cost_usd = float(cost_info)
                model_used = data.get("model", model_used)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        # Detect test passage from output
        tests_passed = exit_code == 0 and "FAILED" not in raw_output.upper()

        # Extract changed files from git diff markers if present
        files_changed = _extract_files_from_output(raw_output)

        approach_summary = raw_output[:500] if raw_output else ""

        return TaskOutcome(
            files_changed=files_changed,
            test_output=stderr if stderr else "",
            tests_passed=tests_passed,
            approach_summary=approach_summary,
            model_used=model_used,
            agent_id="claude",
            raw_output=raw_output,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # API mode
    # ------------------------------------------------------------------

    async def _api_health_check(self) -> AgentHealth:
        """Check Anthropic API key and connectivity."""
        if not self.api_key:
            return AgentHealth(
                agent_id="claude",
                available=False,
                mode=AgentMode.API,
                error="ANTHROPIC_API_KEY not set",
            )
        try:
            client = self._get_anthropic_client()
            start = time.monotonic()
            # Minimal API call to verify connectivity
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            latency = (time.monotonic() - start) * 1000
            return AgentHealth(
                agent_id="claude",
                available=True,
                mode=AgentMode.API,
                version=response.model,
                latency_ms=latency,
            )
        except Exception as e:
            return AgentHealth(
                agent_id="claude",
                available=False,
                mode=AgentMode.API,
                error=str(e),
            )

    async def _execute_api(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via Anthropic API."""
        prompt = self._build_prompt(task, context)
        client = self._get_anthropic_client()

        start = time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            duration = time.monotonic() - start
            content = response.content[0].text if response.content else ""
            tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

            return TaskOutcome(
                approach_summary=content[:500],
                model_used=response.model,
                agent_id="claude",
                raw_output=content,
                tokens_used=tokens_used,
                tests_passed=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="claude",
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, task: TaskContext, context: Optional[Any] = None) -> str:
        """Build the prompt for Claude from task context."""
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

        if task.previous_escalation_diagnosis:
            parts.append(f"\n## Previous Diagnosis\n{task.previous_escalation_diagnosis}")

        if context and hasattr(context, "past_solutions") and context.past_solutions:
            parts.append("\n## Related Solutions from Memory")
            for sol in context.past_solutions[:3]:
                parts.append(f"- Problem: {sol.problem_description[:200]}")
                parts.append(f"  Solution: {sol.solution_code[:200]}")

        if context and hasattr(context, "project_rules") and context.project_rules:
            parts.append(f"\n## Project Rules\n{context.project_rules}")

        return "\n".join(parts)


def _extract_files_from_output(output: str) -> list[str]:
    """Extract file paths from Claude output."""
    files = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("--- FILE: ") and line.endswith(" ---"):
            fpath = line[len("--- FILE: "):-len(" ---")].strip()
            if fpath and fpath not in files:
                files.append(fpath)
    return files
