"""Grok Agent for CLAW.

Wraps xAI's Grok as a CLAW agent with two modes:
- CLI: Invokes `grok` subprocess (fast fixes, web lookup)
- API: Uses the OpenAI-compatible SDK via xAI base URL

Model version comes from config — never hardcoded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from claw.agents.interface import AgentInterface
from claw.core.models import AgentHealth, AgentMode, TaskContext, TaskOutcome

logger = logging.getLogger("claw.agent.grok")

# xAI API base URL for OpenAI-compatible requests
XAI_BASE_URL = "https://api.x.ai/v1"


class GrokAgent(AgentInterface):
    """Grok agent — fast fixes, web lookup, multi-agent reasoning."""

    def __init__(
        self,
        mode: AgentMode = AgentMode.CLI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        max_budget_usd: float = 1.0,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__(agent_id="grok", name="Grok Agent")
        self.mode = mode
        self.api_key = api_key or os.getenv("XAI_API_KEY", "")
        self.model = model  # User-set, never hardcoded
        self.timeout = timeout
        self.max_budget_usd = max_budget_usd
        self.workspace_dir = workspace_dir
        self._openai_client = None

    @property
    def supported_modes(self) -> list[AgentMode]:
        return [AgentMode.CLI, AgentMode.API]

    @property
    def instruction_file(self) -> str:
        return ".grok/GROK.md"

    async def health_check(self) -> AgentHealth:
        """Check Grok availability."""
        if self.mode == AgentMode.OPENROUTER:
            return await self._openrouter_health_check()
        elif self.mode == AgentMode.CLI:
            return await self._cli_health_check()
        else:
            return await self._api_health_check()

    async def execute(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute a task using Grok."""
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
                agent_id="grok", available=False, mode=AgentMode.OPENROUTER,
                error="OPENROUTER_API_KEY not set",
            )
        if not self.model:
            return AgentHealth(
                agent_id="grok", available=False, mode=AgentMode.OPENROUTER,
                error="No model configured in claw.toml",
            )
        return AgentHealth(
            agent_id="grok", available=True, mode=AgentMode.OPENROUTER,
            version=f"openrouter:{self.model}",
        )

    # ------------------------------------------------------------------
    # CLI mode
    # ------------------------------------------------------------------

    async def _cli_health_check(self) -> AgentHealth:
        """Check if `grok` CLI is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "grok", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                version = stdout.decode().strip()
                return AgentHealth(
                    agent_id="grok",
                    available=True,
                    mode=AgentMode.CLI,
                    version=version,
                )
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.CLI,
                error=stderr.decode().strip(),
            )
        except FileNotFoundError:
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.CLI,
                error="grok CLI not found in PATH",
            )
        except Exception as e:
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.CLI,
                error=str(e),
            )

    async def _execute_cli(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via `grok` subprocess with prompt on stdin."""
        prompt = self._build_prompt(task, context)

        cmd = ["grok"]

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
                agent_id="grok",
                failure_reason="timeout",
                failure_detail=f"Grok CLI timed out after {self.timeout}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="grok",
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
        """Parse Grok CLI output into TaskOutcome."""
        raw_output = stdout
        tokens_used = 0
        cost_usd = 0.0
        model_used = self.model or ""

        # Try to parse JSON envelope if the CLI outputs structured data
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                raw_output = data.get("result", data.get("output", stdout))
                usage = data.get("usage", {})
                if isinstance(usage, dict):
                    tokens_used += usage.get("input_tokens", 0)
                    tokens_used += usage.get("output_tokens", 0)
                    tokens_used += usage.get("prompt_tokens", 0)
                    tokens_used += usage.get("completion_tokens", 0)
                cost_info = data.get("cost_usd") or data.get("costUsd")
                if cost_info:
                    cost_usd = float(cost_info)
                model_used = data.get("model", model_used)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        # Detect test passage from output
        tests_passed = exit_code == 0 and "FAILED" not in raw_output.upper()

        # Extract changed files from output markers if present
        files_changed = _extract_files_from_output(raw_output)

        approach_summary = raw_output[:500] if raw_output else ""

        return TaskOutcome(
            files_changed=files_changed,
            test_output=stderr if stderr else "",
            tests_passed=tests_passed,
            approach_summary=approach_summary,
            model_used=model_used,
            agent_id="grok",
            raw_output=raw_output,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # API mode
    # ------------------------------------------------------------------

    async def _api_health_check(self) -> AgentHealth:
        """Check xAI API key and connectivity via OpenAI-compatible endpoint."""
        if not self.api_key:
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.API,
                error="XAI_API_KEY not set",
            )
        if not self.model:
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.API,
                error="No model configured for Grok agent (model must be user-set)",
            )
        try:
            client = self._get_openai_client()
            start = time.monotonic()
            # Minimal API call to verify connectivity
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            latency = (time.monotonic() - start) * 1000
            return AgentHealth(
                agent_id="grok",
                available=True,
                mode=AgentMode.API,
                version=response.model,
                latency_ms=latency,
            )
        except Exception as e:
            return AgentHealth(
                agent_id="grok",
                available=False,
                mode=AgentMode.API,
                error=str(e),
            )

    async def _execute_api(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via xAI API using OpenAI-compatible SDK."""
        if not self.model:
            return TaskOutcome(
                agent_id="grok",
                failure_reason="ConfigError",
                failure_detail="No model configured for Grok agent (model must be user-set)",
                duration_seconds=0.0,
            )

        prompt = self._build_prompt(task, context)
        client = self._get_openai_client()

        start = time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            duration = time.monotonic() - start
            content = ""
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content

            tokens_used = 0
            cost_usd = 0.0
            if response.usage:
                tokens_used = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

            return TaskOutcome(
                approach_summary=content[:500],
                model_used=response.model,
                agent_id="grok",
                raw_output=content,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                tests_passed=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="grok",
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _get_openai_client(self):
        """Get or create the OpenAI-compatible client for xAI."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=XAI_BASE_URL,
            )
        return self._openai_client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, task: TaskContext, context: Optional[Any] = None) -> str:
        """Build the prompt for Grok from task context."""
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
    """Extract file paths from Grok output."""
    files = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("--- FILE: ") and line.endswith(" ---"):
            fpath = line[len("--- FILE: "):-len(" ---")].strip()
            if fpath and fpath not in files:
                files.append(fpath)
    return files
