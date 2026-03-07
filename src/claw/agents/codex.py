"""Codex Agent for CLAW.

Wraps OpenAI's Codex CLI and API as a CLAW agent with three modes:
- CLI: Invokes `codex --quiet` subprocess with stdin prompt
- API: Uses the OpenAI SDK (Responses API) directly
- Cloud: Parallel worktrees (delegates to CLI for now, enhanced in Phase 4)

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

logger = logging.getLogger("claw.agent.codex")


class CodexAgent(AgentInterface):
    """Codex agent — parallel refactoring, bulk tests, CI/CD."""

    def __init__(
        self,
        mode: AgentMode = AgentMode.CLI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        max_tokens: int = 4096,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__(agent_id="codex", name="Codex Agent")
        self.mode = mode
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model  # User-set, never hardcoded
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.workspace_dir = workspace_dir
        self._openai_client = None

    @property
    def supported_modes(self) -> list[AgentMode]:
        return [AgentMode.CLI, AgentMode.API, AgentMode.CLOUD]

    @property
    def instruction_file(self) -> str:
        return "AGENTS.md"

    async def health_check(self) -> AgentHealth:
        """Check Codex availability based on current mode."""
        if self.mode == AgentMode.OPENROUTER:
            return await self._openrouter_health_check()
        elif self.mode == AgentMode.CLI:
            return await self._cli_health_check()
        elif self.mode == AgentMode.CLOUD:
            # Cloud mode uses CLI infrastructure for health checks
            return await self._cli_health_check()
        else:
            return await self._api_health_check()

    async def execute(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute a task using Codex."""
        if self.mode == AgentMode.OPENROUTER:
            return await self.execute_openrouter(task, context)
        elif self.mode == AgentMode.CLI:
            return await self._execute_cli(task, context)
        elif self.mode == AgentMode.CLOUD:
            # Cloud mode with parallel worktrees will be enhanced in Phase 4.
            # Currently delegates to CLI mode which provides the same Codex
            # subprocess execution without worktree parallelism.
            return await self._execute_cli(task, context)
        else:
            return await self._execute_api(task, context)

    async def _openrouter_health_check(self) -> AgentHealth:
        """Check OpenRouter availability for this agent."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return AgentHealth(
                agent_id="codex", available=False, mode=AgentMode.OPENROUTER,
                error="OPENROUTER_API_KEY not set",
            )
        if not self.model:
            return AgentHealth(
                agent_id="codex", available=False, mode=AgentMode.OPENROUTER,
                error="No model configured in claw.toml",
            )
        return AgentHealth(
            agent_id="codex", available=True, mode=AgentMode.OPENROUTER,
            version=f"openrouter:{self.model}",
        )

    # ------------------------------------------------------------------
    # CLI mode
    # ------------------------------------------------------------------

    async def _cli_health_check(self) -> AgentHealth:
        """Check if `codex` CLI is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "codex", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                version = stdout.decode().strip()
                return AgentHealth(
                    agent_id="codex",
                    available=True,
                    mode=self.mode,
                    version=version,
                )
            return AgentHealth(
                agent_id="codex",
                available=False,
                mode=self.mode,
                error=stderr.decode().strip(),
            )
        except FileNotFoundError:
            return AgentHealth(
                agent_id="codex",
                available=False,
                mode=self.mode,
                error="codex CLI not found in PATH",
            )
        except Exception as e:
            return AgentHealth(
                agent_id="codex",
                available=False,
                mode=self.mode,
                error=str(e),
            )

    async def _execute_cli(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via `codex --quiet` subprocess."""
        prompt = self._build_prompt(task, context)

        cmd = ["codex", "--quiet"]

        if self.model:
            cmd.extend(["--model", self.model])

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
                agent_id="codex",
                failure_reason="timeout",
                failure_detail=f"Codex CLI timed out after {self.timeout}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="codex",
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
        """Parse Codex CLI output into TaskOutcome.

        Codex CLI may output JSON or plain text. This method handles both,
        extracting file markers, token usage, and cost when available.
        """
        raw_output = stdout
        tokens_used = 0
        cost_usd = 0.0
        model_used = self.model or ""

        # Try to parse JSON output from Codex CLI
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                # Codex may return structured output with result/content fields
                raw_output = data.get("result", data.get("content", data.get("output", stdout)))
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

        # Extract changed files from output markers
        files_changed = _extract_files_from_output(raw_output)

        approach_summary = raw_output[:500] if raw_output else ""

        return TaskOutcome(
            files_changed=files_changed,
            test_output=stderr if stderr else "",
            tests_passed=tests_passed,
            approach_summary=approach_summary,
            model_used=model_used,
            agent_id="codex",
            raw_output=raw_output,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # API mode
    # ------------------------------------------------------------------

    async def _api_health_check(self) -> AgentHealth:
        """Check OpenAI API key and connectivity."""
        if not self.api_key:
            return AgentHealth(
                agent_id="codex",
                available=False,
                mode=AgentMode.API,
                error="OPENAI_API_KEY not set",
            )
        try:
            client = self._get_openai_client()
            start = time.monotonic()
            # Minimal API call to verify connectivity using the Responses API
            response = await asyncio.to_thread(
                client.responses.create,
                model=self.model,
                input="ping",
            )
            latency = (time.monotonic() - start) * 1000
            return AgentHealth(
                agent_id="codex",
                available=True,
                mode=AgentMode.API,
                version=response.model if hasattr(response, "model") else self.model,
                latency_ms=latency,
            )
        except Exception as e:
            return AgentHealth(
                agent_id="codex",
                available=False,
                mode=AgentMode.API,
                error=str(e),
            )

    async def _execute_api(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via OpenAI Responses API."""
        prompt = self._build_prompt(task, context)
        client = self._get_openai_client()

        start = time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.responses.create,
                model=self.model,
                input=prompt,
                max_output_tokens=self.max_tokens,
            )

            duration = time.monotonic() - start

            # Extract text content from response output items
            content = ""
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content_block in item.content:
                            if hasattr(content_block, "text"):
                                content += content_block.text
            elif hasattr(response, "output_text"):
                content = response.output_text or ""

            # Extract token usage from the response
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                tokens_used += getattr(usage, "input_tokens", 0)
                tokens_used += getattr(usage, "output_tokens", 0)

            model_used = response.model if hasattr(response, "model") else (self.model or "")

            # Extract changed files from the response content
            files_changed = _extract_files_from_output(content)

            return TaskOutcome(
                files_changed=files_changed,
                approach_summary=content[:500],
                model_used=model_used,
                agent_id="codex",
                raw_output=content,
                tokens_used=tokens_used,
                tests_passed=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="codex",
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _get_openai_client(self):
        """Lazily initialize and return the OpenAI client."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(api_key=self.api_key)
        return self._openai_client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, task: TaskContext, context: Optional[Any] = None) -> str:
        """Build the prompt for Codex from task context.

        Follows the same pattern as the Claude agent: structured prompt
        with task details, forbidden approaches, prior diagnosis, and
        related solutions from memory.
        """
        parts = [f"# Task: {task.task.title}\n"]
        parts.append(task.task.description)

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
    """Extract file paths from Codex output.

    Looks for multiple marker formats:
    - `--- FILE: path/to/file ---` (standard CLAW marker)
    - `+++ b/path/to/file` (git diff format)
    - `Modified: path/to/file` (Codex summary format)
    """
    files: list[str] = []
    for line in output.split("\n"):
        line = line.strip()

        # Standard CLAW file marker
        if line.startswith("--- FILE: ") and line.endswith(" ---"):
            fpath = line[len("--- FILE: "):-len(" ---")].strip()
            if fpath and fpath not in files:
                files.append(fpath)
            continue

        # Git diff new-file marker
        if line.startswith("+++ b/"):
            fpath = line[len("+++ b/"):].strip()
            if fpath and fpath not in files:
                files.append(fpath)
            continue

        # Codex "Modified:" summary lines
        if line.startswith("Modified: "):
            fpath = line[len("Modified: "):].strip()
            if fpath and fpath not in files:
                files.append(fpath)
            continue

    return files
