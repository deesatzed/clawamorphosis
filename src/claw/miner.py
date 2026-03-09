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
import os
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


@dataclass
class RepoCandidate:
    """A discovered git repo with metadata for dedup decisions."""
    path: Path
    name: str                # directory name (e.g., "ace-forecaster-v3")
    canonical_name: str      # stripped name (e.g., "ace-forecaster")
    depth: int               # nesting depth from scan root
    file_count: int = 0      # number of source files (proxy for completeness)
    last_commit_ts: float = 0.0  # timestamp of last git activity
    total_bytes: int = 0     # approximate source size


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
        governance: Any = None,
        assimilation_engine: Any = None,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.semantic_memory = semantic_memory
        self.config = config
        self.governance = governance
        self.assimilation_engine = assimilation_engine
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
        max_depth: int = 6,
        dedup_iterations: bool = True,
    ) -> MiningReport:
        """Discover repos in a directory and mine each.

        Args:
            base_path: Root directory to scan for git repos.
            target_project_id: Project ID to create tasks under.
            max_repos: Maximum repos to mine.
            min_relevance: Minimum relevance for task generation.
            generate_tasks: Whether to create enhancement tasks.
            on_repo_complete: Optional callback(repo_name, result) for progress.
            max_depth: Maximum directory depth for repo discovery.
            dedup_iterations: If True, dedup repo iterations by canonical name.

        Returns:
            MiningReport with aggregate results.
        """
        base = Path(base_path).resolve()
        if not base.exists():
            raise FileNotFoundError(f"Directory not found: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        # Discover repos by looking for .git directories
        candidates = _discover_repos(base, max_depth=max_depth)
        if not candidates:
            logger.info("No git repos found in %s", base)
            return MiningReport()

        # Dedup iterations if requested
        if dedup_iterations:
            candidates, skipped = _dedup_iterations(candidates)
            if skipped:
                logger.info(
                    "Dedup: %d selected, %d skipped",
                    len(candidates), len(skipped),
                )

        # Limit repos
        repos = [(c.path, c.name) for c in candidates[:max_repos]]
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

        # Check what CLAW already knows from this repo
        existing_knowledge = await self._check_already_mined(repo_name)

        # Build prompt
        template = self._get_prompt_template()
        prompt = template.replace("{repo_content}", repo_content)

        # Include existing knowledge context to avoid rediscovering patterns
        if existing_knowledge:
            context_block = (
                "\n\n# Context: CLAW already knows the following from this repo:\n"
                + "\n".join(f"- {title}" for title in existing_knowledge)
                + "\n\n# Instructions: DO NOT repeat the above. Focus on patterns not yet captured.\n"
            )
            prompt = context_block + prompt

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
    ) -> Optional[str]:
        """Store a mining finding in semantic memory as a Methodology.

        Applies enhanced quality gate and pre-save dedup before storing.

        Args:
            finding: The extracted finding.
            target_project_id: Project to associate with (unused in methodology but tracked via tags).

        Returns:
            The methodology ID, or None if blocked by quality gate or dedup.
        """
        # Enhanced quality gate
        passes, reason = self._enhanced_quality_gate(finding)
        if not passes:
            logger.info("Quality gate blocked finding '%s': %s", finding.title, reason)
            return None

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

        # Trigger capability assimilation
        if self.assimilation_engine is not None:
            try:
                await self.assimilation_engine.assimilate(methodology.id)
            except Exception as e:
                logger.warning("Assimilation failed for %s: %s", methodology.id, e)

        return methodology.id

    async def _check_already_mined(self, repo_name: str) -> list[str]:
        """Check what CLAW already knows from a repo.

        Searches semantic memory for methodologies tagged with source:{repo_name}.

        Returns:
            List of existing finding titles/descriptions.
        """
        try:
            existing = await self.repository.get_methodologies_by_tag(
                f"source:{repo_name}", limit=50
            )
            titles = [m.problem_description[:200] for m in existing]
            if titles:
                logger.info(
                    "Found %d existing findings from %s", len(titles), repo_name
                )
            return titles
        except Exception as e:
            logger.warning("Failed to check already-mined for %s: %s", repo_name, e)
            return []

    def _enhanced_quality_gate(
        self, finding: MiningFinding
    ) -> tuple[bool, str]:
        """Multi-dimensional quality gate beyond simple relevance.

        Checks:
        1. Relevance score >= 0.4 (existing minimum)
        2. Description length >= configured minimum
        3. Category is valid

        Returns:
            (passes, rejection_reason).
        """
        if finding.relevance_score < 0.4:
            return False, f"relevance too low ({finding.relevance_score:.2f} < 0.4)"

        min_desc = getattr(self.config, "governance", None)
        min_desc_len = 50
        if min_desc and hasattr(min_desc, "mining_min_description_length"):
            min_desc_len = min_desc.mining_min_description_length

        if len(finding.description) < min_desc_len:
            return False, f"description too short ({len(finding.description)} < {min_desc_len})"

        if finding.category not in _VALID_CATEGORIES:
            return False, f"invalid category: {finding.category}"

        return True, ""

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


def _canonicalize_name(name: str) -> str:
    """Strip version/variant suffixes from a repo directory name.

    Iteratively removes common suffixes like -v2, -final, -backup, _old,
    trailing digits after a dash, etc.

    Examples:
        "ace-forecaster-v3"  -> "ace-forecaster"
        "grokflow-cli-final" -> "grokflow-cli"
        "my-project-2"       -> "my-project"
        "tool-wip"           -> "tool"
        "tool-dev-v2"        -> "tool"
    """
    result = name.lower().strip()
    suffix_re = re.compile(
        r'[-_](v?\d+|final|latest|old|backup|copy|wip|dev|test|staging|prod|new|orig)$'
    )
    while True:
        new = suffix_re.sub('', result)
        if new == result:
            break
        result = new
    return result


def _collect_repo_metadata(repo_path: Path) -> tuple[int, float, int]:
    """Collect lightweight metadata for a repo (no subprocess calls).

    Returns:
        (file_count, last_commit_ts, total_bytes)
    """
    file_count = 0
    total_bytes = 0

    # Count source files in top-level directory only (fast scan)
    try:
        with os.scandir(repo_path) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    _, ext = os.path.splitext(entry.name)
                    if ext.lower() in _CODE_EXTENSIONS:
                        file_count += 1
                        try:
                            total_bytes += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            pass
    except (PermissionError, OSError):
        pass

    # Also count files in immediate subdirectories (one level deeper)
    try:
        with os.scandir(repo_path) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.') and entry.name not in _SKIP_DIRS:
                    try:
                        with os.scandir(entry.path) as sub_entries:
                            for sub in sub_entries:
                                if sub.is_file(follow_symlinks=False):
                                    _, ext = os.path.splitext(sub.name)
                                    if ext.lower() in _CODE_EXTENSIONS:
                                        file_count += 1
                                        try:
                                            total_bytes += sub.stat(follow_symlinks=False).st_size
                                        except OSError:
                                            pass
                    except (PermissionError, OSError):
                        pass
    except (PermissionError, OSError):
        pass

    # Use .git directory mtime as proxy for last commit timestamp
    last_commit_ts = 0.0
    git_dir = repo_path / ".git"
    for ref_name in ("refs/heads/main", "refs/heads/master", "HEAD"):
        ref_path = git_dir / ref_name
        try:
            last_commit_ts = max(last_commit_ts, ref_path.stat().st_mtime)
        except OSError:
            pass
    if last_commit_ts == 0.0:
        try:
            last_commit_ts = git_dir.stat().st_mtime
        except OSError:
            pass

    return file_count, last_commit_ts, total_bytes


def _discover_repos(
    base: Path,
    max_depth: int = 6,
) -> list[RepoCandidate]:
    """Find git repositories under a base directory using BFS.

    Scans up to max_depth levels deep using os.scandir() for performance.
    Stops descending into a directory once .git is found.
    Collects metadata for each repo to support iteration dedup.

    Args:
        base: Root directory to scan.
        max_depth: Maximum directory depth to search (default 6).

    Returns:
        List of RepoCandidate objects sorted by canonical_name then name.
    """
    candidates: list[RepoCandidate] = []
    seen: set[str] = set()  # resolved path strings for dedup

    # BFS queue: (directory_path, current_depth)
    frontier: list[tuple[Path, int]] = [(base, 0)]

    while frontier:
        next_frontier: list[tuple[Path, int]] = []

        for dir_path, depth in frontier:
            # Check if this directory is a git repo
            git_marker = dir_path / ".git"
            try:
                is_repo = git_marker.exists()
            except (PermissionError, OSError):
                is_repo = False

            if is_repo:
                try:
                    resolved = str(dir_path.resolve())
                except OSError:
                    resolved = str(dir_path)

                if resolved not in seen:
                    seen.add(resolved)
                    name = dir_path.name
                    file_count, last_commit_ts, total_bytes = _collect_repo_metadata(dir_path)
                    candidates.append(RepoCandidate(
                        path=dir_path,
                        name=name,
                        canonical_name=_canonicalize_name(name),
                        depth=depth,
                        file_count=file_count,
                        last_commit_ts=last_commit_ts,
                        total_bytes=total_bytes,
                    ))
                # Don't descend into repos — they're leaf nodes
                continue

            # Not a repo — descend if within depth limit
            if depth >= max_depth:
                continue

            try:
                with os.scandir(dir_path) as entries:
                    for entry in sorted(entries, key=lambda e: e.name):
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        if entry.name.startswith("."):
                            continue
                        if entry.name in _SKIP_DIRS:
                            continue
                        next_frontier.append((Path(entry.path), depth + 1))
            except (PermissionError, OSError):
                continue

        frontier = next_frontier

    # Sort by canonical_name, then by name for deterministic ordering
    candidates.sort(key=lambda c: (c.canonical_name, c.name))
    return candidates


def _dedup_iterations(
    candidates: list[RepoCandidate],
) -> tuple[list[RepoCandidate], list[tuple[RepoCandidate, str]]]:
    """Deduplicate repo iterations by canonical name.

    Groups candidates by canonical_name and picks the best version
    based on: last_commit_ts (primary), file_count (secondary),
    total_bytes (tertiary).

    Args:
        candidates: All discovered repo candidates.

    Returns:
        (selected, skipped) where skipped includes (candidate, reason) tuples.
    """
    from collections import defaultdict

    groups: dict[str, list[RepoCandidate]] = defaultdict(list)
    for c in candidates:
        groups[c.canonical_name].append(c)

    selected: list[RepoCandidate] = []
    skipped: list[tuple[RepoCandidate, str]] = []

    for canonical, group in sorted(groups.items()):
        if len(group) == 1:
            selected.append(group[0])
            continue

        # Score: sort by (last_commit_ts, file_count, total_bytes) descending
        group.sort(
            key=lambda c: (c.last_commit_ts, c.file_count, c.total_bytes),
            reverse=True,
        )

        winner = group[0]
        selected.append(winner)

        for loser in group[1:]:
            skipped.append((
                loser,
                f"superseded by {winner.name} ({winner.path})",
            ))

        if len(group) > 1:
            logger.info(
                "Dedup '%s': selected '%s' (%d files, ts=%.0f), skipped %d iterations",
                canonical, winner.name, winner.file_count, winner.last_commit_ts,
                len(group) - 1,
            )

    selected.sort(key=lambda c: (c.canonical_name, c.name))
    return selected, skipped


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
