"""Repo Mining for CLAW.

Scans local repositories, extracts patterns/features/ideas via LLM analysis,
stores findings in semantic memory, and generates enhancement tasks.

Usage:
    miner = RepoMiner(repository, llm_client, semantic_memory, config)
    report = await miner.mine_directory("/path/to/repos", project_id)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from claw.core.config import ClawConfig
from claw.core.models import Methodology, Project, Task, TaskStatus
from claw.db.repository import Repository
from claw.llm.client import LLMClient, LLMMessage, LLMResponse
from claw.memory.semantic import SemanticMemory

logger = logging.getLogger("claw.miner")

# Extensions to include when serializing a repo for mining.
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
    ".next", ".nuxt", "coverage", ".cache",
    "target",  # Rust/Java build output
}

# Maximum serialized repo size in bytes (900 KB).
_MAX_REPO_BYTES: int = 900 * 1024

# Valid categories for findings.
_VALID_CATEGORIES: set[str] = {
    "architecture", "ai_integration", "memory", "code_quality",
    "cli_ux", "testing", "data_processing", "security",
    "algorithm", "cross_cutting",
}

# Maximum findings per repo.
_MAX_FINDINGS_PER_REPO: int = 15


@dataclass
class MiningFinding:
    """A single extracted pattern/feature/idea from a mined repo."""
    title: str
    description: str
    category: str
    source_repo: str
    source_files: list[str] = field(default_factory=list)
    implementation_sketch: str = ""
    augmentation_notes: str = ""
    relevance_score: float = 0.5
    language: str = "python"


@dataclass
class RepoMiningResult:
    """Results from mining a single repo."""
    repo_name: str
    repo_path: str
    findings: list[MiningFinding] = field(default_factory=list)
    files_analyzed: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class MiningReport:
    """Aggregate results from mining a directory of repos."""
    repos_scanned: int = 0
    total_findings: int = 0
    tasks_generated: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    repo_results: list[RepoMiningResult] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)


def serialize_repo(repo_path: str | Path, max_bytes: int = _MAX_REPO_BYTES) -> tuple[str, int]:
    """Read all source files in a directory and concatenate with file headers.

    Filters by common code extensions, skips binary/build directories,
    and limits total size to max_bytes.

    Args:
        repo_path: Absolute path to the repository root.
        max_bytes: Maximum serialized size in bytes.

    Returns:
        Tuple of (serialized content, number of files read).
    """
    root = Path(repo_path)
    if not root.is_dir():
        logger.warning("Repo path is not a directory: %s", repo_path)
        return "", 0

    parts: list[str] = []
    total_bytes = 0
    file_count = 0

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue

        rel = filepath.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue

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

        if total_bytes + chunk_bytes > max_bytes:
            parts.append(
                f"\n--- TRUNCATED: repo serialization exceeded {max_bytes // 1024}KB limit ---\n"
            )
            break

        parts.append(chunk)
        total_bytes += chunk_bytes
        file_count += 1

    return "".join(parts), file_count


def parse_findings(llm_response: str, repo_name: str) -> list[MiningFinding]:
    """Extract MiningFinding objects from LLM JSON response.

    Handles ```json fences, validates required fields, filters by
    relevance score, and caps at _MAX_FINDINGS_PER_REPO.

    Args:
        llm_response: Raw text from the LLM containing a JSON array.
        repo_name: Name of the source repo (injected into each finding).

    Returns:
        List of validated MiningFinding objects.
    """
    cleaned = llm_response.strip()

    # Strip markdown code fences if present
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    # Try to find a JSON array in the response
    if not cleaned.startswith("["):
        # Look for array start in the response
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            cleaned = cleaned[arr_start:arr_end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse mining findings JSON: %s", e)
        return []

    if not isinstance(data, list):
        logger.warning("Mining findings response is not a JSON array")
        return []

    findings: list[MiningFinding] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        # Required fields
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        if not title or not description:
            continue

        # Category validation
        category = item.get("category", "").strip().lower()
        if category not in _VALID_CATEGORIES:
            category = "cross_cutting"

        # Relevance filter
        try:
            relevance = float(item.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        if relevance < 0.4:
            continue

        # Clamp relevance to [0.4, 1.0]
        relevance = min(max(relevance, 0.4), 1.0)

        # Source files
        source_files = item.get("source_files", [])
        if not isinstance(source_files, list):
            source_files = []
        source_files = [str(f) for f in source_files if f]

        finding = MiningFinding(
            title=title[:200],
            description=description[:2000],
            category=category,
            source_repo=repo_name,
            source_files=source_files[:20],
            implementation_sketch=str(item.get("implementation_sketch", ""))[:2000],
            augmentation_notes=str(item.get("augmentation_notes", ""))[:1000],
            relevance_score=relevance,
            language=str(item.get("language", "python"))[:20],
        )
        findings.append(finding)

        if len(findings) >= _MAX_FINDINGS_PER_REPO:
            break

    return findings


class RepoMiner:
    """Mines local repositories for patterns, features, and ideas.

    Uses LLMClient.complete() directly (not through agents/Dispatcher)
    since mining is analytical — a single large-context call per repo.

    Args:
        repository: Database access for creating tasks.
        llm_client: OpenRouter client for LLM calls.
        semantic_memory: For storing findings as methodologies.
        config: CLAW config for model selection.
    """

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        semantic_memory: SemanticMemory,
        config: ClawConfig,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.semantic_memory = semantic_memory
        self.config = config
        self._prompt_template: Optional[str] = None

    def _get_prompt_template(self) -> str:
        """Load the mining prompt template from prompts/repo-mine.md."""
        if self._prompt_template is None:
            prompt_path = Path(__file__).parent.parent.parent / "prompts" / "repo-mine.md"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Mining prompt not found: {prompt_path}")
            self._prompt_template = prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _get_mining_model(self) -> str:
        """Get the model to use for mining from config.

        Uses the claude agent's model since mining is analytical work.
        Falls back through other agents if claude is not configured.
        """
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = self.config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise ValueError("No model configured in any agent. Set a model in claw.toml.")

    async def mine_directory(
        self,
        base_path: str | Path,
        target_project_id: str,
        max_repos: int = 10,
        min_relevance: float = 0.6,
        generate_tasks: bool = True,
        on_repo_complete: Optional[Any] = None,
    ) -> MiningReport:
        """Discover repos in a directory and mine each.

        Args:
            base_path: Root directory to scan for git repos.
            target_project_id: Project ID to create tasks under.
            max_repos: Maximum repos to mine.
            min_relevance: Minimum relevance for task generation.
            generate_tasks: Whether to create enhancement tasks.
            on_repo_complete: Optional callback(repo_name, result) for progress.

        Returns:
            MiningReport with aggregate results.
        """
        base = Path(base_path).resolve()
        if not base.exists():
            raise FileNotFoundError(f"Directory not found: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        # Discover repos by looking for .git directories
        repos = _discover_repos(base)
        if not repos:
            logger.info("No git repos found in %s", base)
            return MiningReport()

        # Limit repos
        repos = repos[:max_repos]
        logger.info("Found %d repos to mine in %s", len(repos), base)

        report = MiningReport()
        start = time.monotonic()

        for repo_path, repo_name in repos:
            try:
                result = await self.mine_repo(repo_path, repo_name, target_project_id)
                report.repo_results.append(result)
                report.repos_scanned += 1
                report.total_findings += len(result.findings)
                report.total_cost_usd += result.cost_usd
                report.total_tokens += result.tokens_used

                if on_repo_complete:
                    on_repo_complete(repo_name, result)

            except Exception as e:
                logger.error("Failed to mine repo %s: %s", repo_name, e)
                report.repo_results.append(RepoMiningResult(
                    repo_name=repo_name,
                    repo_path=str(repo_path),
                    error=str(e),
                ))
                report.repos_scanned += 1

        # Generate tasks from all findings
        if generate_tasks:
            all_findings = []
            for result in report.repo_results:
                all_findings.extend(result.findings)

            tasks = await self._generate_tasks(
                all_findings, target_project_id, min_relevance
            )
            report.tasks = tasks
            report.tasks_generated = len(tasks)

        report.total_duration_seconds = time.monotonic() - start
        return report

    async def mine_repo(
        self,
        repo_path: str | Path,
        repo_name: str,
        target_project_id: str,
    ) -> RepoMiningResult:
        """Mine a single repository for patterns and features.

        Args:
            repo_path: Path to the repo root.
            repo_name: Human-readable repo name.
            target_project_id: Project ID for storing findings.

        Returns:
            RepoMiningResult with findings and metadata.
        """
        start = time.monotonic()
        repo_path = Path(repo_path)

        # Serialize repo content
        repo_content, file_count = serialize_repo(repo_path)
        if not repo_content:
            return RepoMiningResult(
                repo_name=repo_name,
                repo_path=str(repo_path),
                error="No source files found",
            )

        logger.info(
            "Serialized %s: %d files, %d bytes",
            repo_name, file_count, len(repo_content.encode()),
        )

        # Build prompt
        template = self._get_prompt_template()
        prompt = template.replace("{repo_content}", repo_content)

        # Call LLM
        model = self._get_mining_model()
        try:
            response: LLMResponse = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return RepoMiningResult(
                repo_name=repo_name,
                repo_path=str(repo_path),
                files_analyzed=file_count,
                duration_seconds=duration,
                error=f"LLM call failed: {e}",
            )

        # Parse findings
        findings = parse_findings(response.content, repo_name)
        logger.info("Extracted %d findings from %s", len(findings), repo_name)

        # Store each finding in semantic memory
        for finding in findings:
            try:
                await self.store_finding(finding, target_project_id)
            except Exception as e:
                logger.warning("Failed to store finding '%s': %s", finding.title, e)

        duration = time.monotonic() - start
        return RepoMiningResult(
            repo_name=repo_name,
            repo_path=str(repo_path),
            findings=findings,
            files_analyzed=file_count,
            tokens_used=response.tokens_used,
            cost_usd=0.0,  # Cost tracked by token_tracker separately
            duration_seconds=duration,
        )

    async def store_finding(
        self,
        finding: MiningFinding,
        target_project_id: str,
    ) -> str:
        """Store a mining finding in semantic memory as a Methodology.

        Args:
            finding: The extracted finding.
            target_project_id: Project to associate with (unused in methodology but tracked via tags).

        Returns:
            The methodology ID.
        """
        # Build a rich problem description for embedding
        problem_desc = (
            f"[Mined from {finding.source_repo}] {finding.title}: "
            f"{finding.description}"
        )

        # Build solution code from implementation sketch
        solution = (
            f"## {finding.title}\n\n"
            f"**Category:** {finding.category}\n"
            f"**Source:** {finding.source_repo}\n"
            f"**Relevance:** {finding.relevance_score:.2f}\n\n"
            f"### Description\n{finding.description}\n\n"
            f"### Implementation Sketch\n{finding.implementation_sketch}\n\n"
            f"### Augmentation Notes\n{finding.augmentation_notes}\n"
        )

        tags = [
            "mined",
            f"source:{finding.source_repo}",
            f"category:{finding.category}",
        ]

        methodology = await self.semantic_memory.save_solution(
            problem_description=problem_desc,
            solution_code=solution,
            methodology_notes=finding.augmentation_notes,
            tags=tags,
            language=finding.language,
            scope="global",
            methodology_type="PATTERN",
            files_affected=finding.source_files,
        )

        logger.debug("Stored finding '%s' as methodology %s", finding.title, methodology.id)
        return methodology.id

    async def _generate_tasks(
        self,
        findings: list[MiningFinding],
        target_project_id: str,
        min_relevance: float = 0.6,
    ) -> list[Task]:
        """Create enhancement tasks from high-relevance findings.

        Args:
            findings: All findings from mining.
            target_project_id: Project to create tasks under.
            min_relevance: Minimum relevance_score to generate a task.

        Returns:
            List of created Task objects.
        """
        tasks: list[Task] = []

        # Filter and sort by relevance
        eligible = [f for f in findings if f.relevance_score >= min_relevance]
        eligible.sort(key=lambda f: f.relevance_score, reverse=True)

        for finding in eligible:
            priority = _relevance_to_priority(finding.relevance_score)

            task = Task(
                project_id=target_project_id,
                title=f"[Mined:{finding.source_repo}] {finding.title}"[:200],
                description=(
                    f"## Enhancement from {finding.source_repo}\n\n"
                    f"**Category:** {finding.category}\n"
                    f"**Relevance:** {finding.relevance_score:.2f}\n"
                    f"**Language:** {finding.language}\n\n"
                    f"### What\n{finding.description}\n\n"
                    f"### How\n{finding.implementation_sketch}\n\n"
                    f"### Why\n{finding.augmentation_notes}\n\n"
                    f"### Source Files\n"
                    + "\n".join(f"- `{f}`" for f in finding.source_files)
                ),
                status=TaskStatus.PENDING,
                priority=priority,
                task_type=finding.category,
                recommended_agent=_category_to_agent(finding.category),
            )

            try:
                saved = await self.repository.create_task(task)
                tasks.append(saved)
                logger.info(
                    "Created task '%s' (priority=%d) from finding in %s",
                    saved.title[:60], priority, finding.source_repo,
                )
            except Exception as e:
                logger.warning("Failed to create task for '%s': %s", finding.title, e)

        return tasks


def _discover_repos(base: Path) -> list[tuple[Path, str]]:
    """Find git repositories under a base directory.

    Looks for .git directories up to 2 levels deep. Returns list of
    (repo_path, repo_name) tuples sorted by name.

    Args:
        base: Root directory to scan.

    Returns:
        List of (path, name) tuples for discovered repos.
    """
    repos: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    # Check if base itself is a repo
    if (base / ".git").exists():
        repos.append((base, base.name))
        seen.add(base.resolve())

    # Check immediate children
    try:
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            resolved = child.resolve()
            if resolved in seen:
                continue
            if child.name.startswith("."):
                continue
            if child.name in _SKIP_DIRS:
                continue
            if (child / ".git").exists():
                repos.append((child, child.name))
                seen.add(resolved)
                continue

            # Check one level deeper
            try:
                for grandchild in sorted(child.iterdir()):
                    if not grandchild.is_dir():
                        continue
                    gc_resolved = grandchild.resolve()
                    if gc_resolved in seen:
                        continue
                    if grandchild.name.startswith("."):
                        continue
                    if grandchild.name in _SKIP_DIRS:
                        continue
                    if (grandchild / ".git").exists():
                        repos.append((grandchild, grandchild.name))
                        seen.add(gc_resolved)
            except PermissionError:
                continue
    except PermissionError:
        pass

    return repos


def _relevance_to_priority(relevance: float) -> int:
    """Map relevance score to task priority (0-10 scale)."""
    if relevance >= 0.9:
        return 9
    if relevance >= 0.8:
        return 7
    if relevance >= 0.7:
        return 5
    if relevance >= 0.6:
        return 3
    return 1


def _category_to_agent(category: str) -> str:
    """Suggest an agent based on finding category."""
    mapping = {
        "architecture": "claude",
        "ai_integration": "claude",
        "memory": "claude",
        "code_quality": "codex",
        "cli_ux": "codex",
        "testing": "codex",
        "data_processing": "gemini",
        "security": "claude",
        "algorithm": "gemini",
        "cross_cutting": "grok",
    }
    return mapping.get(category, "claude")
