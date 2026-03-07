"""Gemini Agent for CLAW.

Wraps Google's Gemini as a CLAW agent with two modes:
- CLI: Invokes `gemini` subprocess
- API: Uses the google-genai SDK directly

Best for: Full-repo comprehension (1M context window), dependency analysis.

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

logger = logging.getLogger("claw.agent.gemini")

# Extensions to include when serializing a repo for the 1M context window.
_CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java",
    ".md", ".yaml", ".yml", ".toml", ".json", ".sql",
}

# Directories to skip during repo serialization.
_SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "dist", "build", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "egg-info",
}

# Maximum serialized repo size in bytes (900 KB leaves room for prompt).
_MAX_REPO_BYTES: int = 900 * 1024


class GeminiAgent(AgentInterface):
    """Gemini agent — full-repo comprehension, dependency analysis."""

    def __init__(
        self,
        mode: AgentMode = AgentMode.CLI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__(agent_id="gemini", name="Gemini Agent")
        self.mode = mode
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model  # User-set, never hardcoded
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self._genai_client = None

    @property
    def supported_modes(self) -> list[AgentMode]:
        return [AgentMode.CLI, AgentMode.API]

    @property
    def instruction_file(self) -> str:
        return "GEMINI.md"

    async def health_check(self) -> AgentHealth:
        """Check Gemini availability."""
        if self.mode == AgentMode.OPENROUTER:
            return await self._openrouter_health_check()
        elif self.mode == AgentMode.CLI:
            return await self._cli_health_check()
        else:
            return await self._api_health_check()

    async def execute(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute a task using Gemini."""
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
                agent_id="gemini", available=False, mode=AgentMode.OPENROUTER,
                error="OPENROUTER_API_KEY not set",
            )
        if not self.model:
            return AgentHealth(
                agent_id="gemini", available=False, mode=AgentMode.OPENROUTER,
                error="No model configured in claw.toml",
            )
        return AgentHealth(
            agent_id="gemini", available=True, mode=AgentMode.OPENROUTER,
            version=f"openrouter:{self.model}",
        )

    # ------------------------------------------------------------------
    # Repo serialization (1M context window leverage)
    # ------------------------------------------------------------------

    def _serialize_repo(self, repo_path: str) -> str:
        """Read all source files in a directory and concatenate with file headers.

        Filters by common code extensions, skips binary/build directories,
        and limits total size to 900 KB to leave room for the prompt within
        Gemini's 1M token context window.

        Args:
            repo_path: Absolute path to the repository root.

        Returns:
            A single string with all file contents, each prefixed by a header
            like ``--- FILE: relative/path/to/file.py ---``.
        """
        root = Path(repo_path)
        if not root.is_dir():
            logger.warning("Repo path is not a directory: %s", repo_path)
            return ""

        parts: list[str] = []
        total_bytes = 0

        for filepath in sorted(root.rglob("*")):
            # Skip directories themselves (we iterate files)
            if not filepath.is_file():
                continue

            # Skip files inside excluded directories
            rel = filepath.relative_to(root)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue

            # Filter by extension
            if filepath.suffix.lower() not in _CODE_EXTENSIONS:
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError) as exc:
                logger.debug("Skipping unreadable file %s: %s", filepath, exc)
                continue

            header = f"--- FILE: {rel} ---\n"
            chunk = header + content + "\n"
            chunk_bytes = len(chunk.encode("utf-8"))

            if total_bytes + chunk_bytes > _MAX_REPO_BYTES:
                parts.append(
                    f"\n--- TRUNCATED: repo serialization exceeded {_MAX_REPO_BYTES // 1024}KB limit ---\n"
                )
                break

            parts.append(chunk)
            total_bytes += chunk_bytes

        return "".join(parts)

    # ------------------------------------------------------------------
    # CLI mode
    # ------------------------------------------------------------------

    async def _cli_health_check(self) -> AgentHealth:
        """Check if `gemini` CLI is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "gemini", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                version = stdout.decode().strip()
                return AgentHealth(
                    agent_id="gemini",
                    available=True,
                    mode=AgentMode.CLI,
                    version=version,
                )
            return AgentHealth(
                agent_id="gemini",
                available=False,
                mode=AgentMode.CLI,
                error=stderr.decode().strip(),
            )
        except FileNotFoundError:
            return AgentHealth(
                agent_id="gemini",
                available=False,
                mode=AgentMode.CLI,
                error="gemini CLI not found in PATH",
            )
        except Exception as e:
            return AgentHealth(
                agent_id="gemini",
                available=False,
                mode=AgentMode.CLI,
                error=str(e),
            )

    async def _execute_cli(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via `gemini` subprocess."""
        prompt = self._build_prompt(task, context)

        cmd = ["gemini"]

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
                agent_id="gemini",
                failure_reason="timeout",
                failure_detail=f"Gemini CLI timed out after {self.timeout}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="gemini",
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
        """Parse Gemini CLI output into TaskOutcome."""
        raw_output = stdout
        tokens_used = 0
        cost_usd = 0.0
        model_used = self.model or ""

        # Attempt to parse structured JSON output
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                raw_output = data.get("text", data.get("result", stdout))
                usage = data.get("usage", {})
                if isinstance(usage, dict):
                    tokens_used += usage.get("prompt_tokens", 0)
                    tokens_used += usage.get("candidates_tokens", 0)
                    tokens_used += usage.get("total_tokens", 0)
                cost_info = data.get("cost_usd")
                if cost_info:
                    cost_usd = float(cost_info)
                model_used = data.get("model", model_used)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        tests_passed = exit_code == 0 and "FAILED" not in raw_output.upper()
        files_changed = _extract_files_from_output(raw_output)
        approach_summary = raw_output[:500] if raw_output else ""

        return TaskOutcome(
            files_changed=files_changed,
            test_output=stderr if stderr else "",
            tests_passed=tests_passed,
            approach_summary=approach_summary,
            model_used=model_used,
            agent_id="gemini",
            raw_output=raw_output,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # API mode
    # ------------------------------------------------------------------

    async def _api_health_check(self) -> AgentHealth:
        """Check Google AI API key and connectivity."""
        if not self.api_key:
            return AgentHealth(
                agent_id="gemini",
                available=False,
                mode=AgentMode.API,
                error="GOOGLE_API_KEY not set",
            )
        try:
            client = self._get_genai_client()
            start = time.monotonic()
            # Minimal API call to verify connectivity
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents="ping",
            )
            latency = (time.monotonic() - start) * 1000
            return AgentHealth(
                agent_id="gemini",
                available=True,
                mode=AgentMode.API,
                version=self.model or "",
                latency_ms=latency,
            )
        except Exception as e:
            return AgentHealth(
                agent_id="gemini",
                available=False,
                mode=AgentMode.API,
                error=str(e),
            )

    async def _execute_api(self, task: TaskContext, context: Optional[Any] = None) -> TaskOutcome:
        """Execute task via Google GenAI API."""
        prompt = self._build_prompt(task, context)
        client = self._get_genai_client()

        start = time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=prompt,
            )

            duration = time.monotonic() - start
            content = response.text or ""

            # Extract token usage from response metadata
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                tokens_used += getattr(meta, "prompt_token_count", 0) or 0
                tokens_used += getattr(meta, "candidates_token_count", 0) or 0

            return TaskOutcome(
                approach_summary=content[:500],
                model_used=self.model or "",
                agent_id="gemini",
                raw_output=content,
                tokens_used=tokens_used,
                tests_passed=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskOutcome(
                agent_id="gemini",
                failure_reason=type(e).__name__,
                failure_detail=str(e),
                duration_seconds=duration,
            )

    def _get_genai_client(self):
        """Lazily import and create the google-genai client."""
        if self._genai_client is None:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, task: TaskContext, context: Optional[Any] = None) -> str:
        """Build the prompt for Gemini from task context.

        Optionally includes the full serialized repo when workspace_dir is set,
        leveraging Gemini's 1M token context window for whole-repo comprehension.
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

        # Leverage Gemini's large context: include full repo serialization
        workspace = self.workspace_dir
        if not workspace and hasattr(task.task, "description"):
            # Fall back to task description if it looks like a path
            candidate = task.task.description.strip()
            if Path(candidate).is_dir():
                workspace = candidate

        if workspace and Path(workspace).is_dir():
            repo_context = self._serialize_repo(workspace)
            if repo_context:
                parts.append("\n## Full Repository Context")
                parts.append(repo_context)

        return "\n".join(parts)


def _extract_files_from_output(output: str) -> list[str]:
    """Extract file paths from Gemini output."""
    files = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("--- FILE: ") and line.endswith(" ---"):
            fpath = line[len("--- FILE: "):-len(" ---")].strip()
            if fpath and fpath not in files:
                files.append(fpath)
    return files
