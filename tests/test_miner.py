"""Tests for CLAW repo mining module (claw.miner).

Covers:
    1. serialize_repo() — file serialization with extension/dir filtering
    2. parse_findings() — LLM JSON response parsing and validation
    3. _discover_repos() — git repository discovery in directory trees
    4. _relevance_to_priority() — relevance-to-priority mapping
    5. _category_to_agent() — category-to-agent mapping
    6. Data classes — MiningFinding, RepoMiningResult, MiningReport
    7. RepoMiner.store_finding() — async methodology storage
    8. RepoMiner._generate_tasks() — async task generation from findings
    9. CLI registration — mine command exists

All tests use REAL dependencies — no mocks, no placeholders, no cached responses.
Database tests use the real SQLite in-memory engine from conftest.py.
Filesystem tests use pytest's tmp_path fixture for isolation.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from claw.core.config import AgentConfig, ClawConfig, load_config
from claw.core.models import Methodology, Project, Task, TaskStatus
from claw.db.embeddings import EmbeddingEngine
from claw.memory.hybrid_search import HybridSearch
from claw.memory.semantic import SemanticMemory
from claw.miner import (
    MiningFinding,
    MiningReport,
    RepoCandidate,
    RepoMiner,
    RepoMiningResult,
    _MAX_FINDINGS_PER_REPO,
    _SKIP_DIRS,
    _VALID_CATEGORIES,
    _canonicalize_name,
    _category_to_agent,
    _collect_repo_metadata,
    _dedup_iterations,
    _discover_repos,
    _relevance_to_priority,
    parse_findings,
    serialize_repo,
)


# ---------------------------------------------------------------------------
# Deterministic embedding engine (same pattern as test_memory.py)
# ---------------------------------------------------------------------------

class FixedEmbeddingEngine:
    """Deterministic embedding engine using SHA-384 for reproducible tests.

    Hashes the input text with SHA-384 to produce 48 bytes, then
    normalizes each byte to [0, 1] and repeats 8x to fill 384 floats.
    """

    DIMENSION = 384

    def encode(self, text: str) -> list[float]:
        h = hashlib.sha384(text.encode()).digest()
        raw = [b / 255.0 for b in h] * 8
        return raw[: self.DIMENSION]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding_engine() -> FixedEmbeddingEngine:
    return FixedEmbeddingEngine()


@pytest.fixture
async def hybrid_search(repository, embedding_engine):
    return HybridSearch(
        repository=repository,
        embedding_engine=embedding_engine,
    )


@pytest.fixture
async def semantic_memory(repository, embedding_engine, hybrid_search):
    return SemanticMemory(
        repository=repository,
        embedding_engine=embedding_engine,
        hybrid_search=hybrid_search,
    )


@pytest.fixture
def miner_config() -> ClawConfig:
    """ClawConfig with at least one agent enabled and model set."""
    config = load_config()
    # Ensure claude agent is configured for _get_mining_model()
    config.agents["claude"] = AgentConfig(
        enabled=True,
        mode="api",
        model="test-model/claude-test",
    )
    return config


@pytest.fixture
async def repo_miner(repository, semantic_memory, miner_config):
    """RepoMiner with real database, real SemanticMemory, but no LLM client.

    Tests that exercise mine_repo() or mine_directory() are skipped because
    they require a real LLM call. Tests for store_finding() and
    _generate_tasks() work without the LLM client.
    """
    from claw.llm.client import LLMClient
    # LLMClient is constructed but not called in the tests below
    llm_client = LLMClient(config=miner_config.llm)
    return RepoMiner(
        repository=repository,
        llm_client=llm_client,
        semantic_memory=semantic_memory,
        config=miner_config,
    )


# ===========================================================================
# 1. serialize_repo()
# ===========================================================================

class TestSerializeRepo:
    """Tests for serialize_repo() — pure filesystem operations."""

    def test_includes_python_files(self, tmp_path):
        """Python files (.py) are included in serialization."""
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1
        assert "--- FILE: main.py ---" in content
        assert "print('hello')" in content

    def test_includes_js_and_ts_files(self, tmp_path):
        """JavaScript and TypeScript files are included."""
        (tmp_path / "app.js").write_text("const x = 1;", encoding="utf-8")
        (tmp_path / "utils.ts").write_text("export const y = 2;", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 2
        assert "--- FILE: app.js ---" in content
        assert "--- FILE: utils.ts ---" in content

    def test_includes_config_files(self, tmp_path):
        """Config files (.toml, .yaml, .yml, .json) are included."""
        (tmp_path / "config.toml").write_text("[section]\nkey = 'value'", encoding="utf-8")
        (tmp_path / "data.json").write_text('{"key": "value"}', encoding="utf-8")
        (tmp_path / "spec.yaml").write_text("name: test", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 3

    def test_skips_binary_extensions(self, tmp_path):
        """Binary files (.exe, .png, .jpg, .dll) are excluded."""
        (tmp_path / "app.exe").write_bytes(b"\x00\x01\x02")
        (tmp_path / "logo.png").write_bytes(b"\x89PNG")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "real.py").write_text("x = 1", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1
        assert "app.exe" not in content
        assert "logo.png" not in content

    def test_skips_git_directory(self, tmp_path):
        """Files inside .git/ are excluded."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]", encoding="utf-8")
        (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1
        assert ".git" not in content.replace("--- FILE:", "")

    def test_skips_node_modules(self, tmp_path):
        """Files inside node_modules/ are excluded."""
        nm = tmp_path / "node_modules" / "lodash"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {};", encoding="utf-8")
        (tmp_path / "app.js").write_text("require('lodash')", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1
        assert "node_modules" not in content

    def test_skips_pycache(self, tmp_path):
        """Files inside __pycache__/ are excluded."""
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "main.cpython-312.pyc").write_bytes(b"\x00")
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1
        assert "__pycache__" not in content

    def test_respects_max_bytes(self, tmp_path):
        """Serialization stops when max_bytes is exceeded."""
        # Create files that exceed a small limit
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}\n" * 50, encoding="utf-8")
        content, count = serialize_repo(tmp_path, max_bytes=500)
        # Should have stopped before processing all files
        assert count < 10
        assert "TRUNCATED" in content

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty string and 0 file count."""
        content, count = serialize_repo(tmp_path)
        assert content == ""
        assert count == 0

    def test_correct_file_count(self, tmp_path):
        """File count matches actual files serialized."""
        (tmp_path / "a.py").write_text("a = 1", encoding="utf-8")
        (tmp_path / "b.py").write_text("b = 2", encoding="utf-8")
        (tmp_path / "c.js").write_text("const c = 3;", encoding="utf-8")
        (tmp_path / "skip.exe").write_bytes(b"\x00")
        content, count = serialize_repo(tmp_path)
        assert count == 3

    def test_skips_unreadable_files(self, tmp_path):
        """Unreadable files are gracefully skipped."""
        good = tmp_path / "good.py"
        good.write_text("x = 1", encoding="utf-8")
        bad = tmp_path / "bad.py"
        bad.write_text("secret", encoding="utf-8")
        # Remove read permission
        bad.chmod(0o000)
        try:
            content, count = serialize_repo(tmp_path)
            # On some systems (root), file may still be readable
            assert count >= 1
            assert "good.py" in content
        finally:
            # Restore permissions for cleanup
            bad.chmod(0o644)

    def test_file_header_format(self, tmp_path):
        """Each file is prefixed with --- FILE: relative/path --- header."""
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "app.py").write_text("pass", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert "--- FILE: src/app.py ---" in content

    def test_nonexistent_path_returns_empty(self, tmp_path):
        """Non-existent path returns empty string and 0 count."""
        content, count = serialize_repo(tmp_path / "nonexistent")
        assert content == ""
        assert count == 0

    def test_skips_all_skip_dirs(self, tmp_path):
        """All directories in _SKIP_DIRS are excluded."""
        for skip_dir in list(_SKIP_DIRS)[:5]:  # Test a subset for speed
            d = tmp_path / skip_dir
            d.mkdir(exist_ok=True)
            (d / "file.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        content, count = serialize_repo(tmp_path)
        assert count == 1


# ===========================================================================
# 2. parse_findings()
# ===========================================================================

class TestParseFindings:
    """Tests for parse_findings() — JSON parsing and validation."""

    def _make_finding_dict(self, **overrides) -> dict:
        """Helper to create a valid finding dict."""
        base = {
            "title": "Pattern Found",
            "description": "A useful pattern discovered in the repo",
            "category": "architecture",
            "relevance_score": 0.8,
            "source_files": ["src/main.py"],
            "implementation_sketch": "def impl(): pass",
            "augmentation_notes": "Could improve CLAW's architecture",
            "language": "python",
        }
        base.update(overrides)
        return base

    def test_parses_valid_json_array(self):
        """Valid JSON array of findings is parsed correctly."""
        findings_data = [self._make_finding_dict()]
        response = json.dumps(findings_data)
        results = parse_findings(response, "test-repo")
        assert len(results) == 1
        assert results[0].title == "Pattern Found"
        assert results[0].source_repo == "test-repo"

    def test_handles_json_fences(self):
        """JSON wrapped in ```json fences is parsed correctly."""
        findings_data = [self._make_finding_dict()]
        response = f"```json\n{json.dumps(findings_data)}\n```"
        results = parse_findings(response, "test-repo")
        assert len(results) == 1
        assert results[0].title == "Pattern Found"

    def test_handles_malformed_json(self):
        """Malformed JSON returns empty list."""
        results = parse_findings("{not valid json[", "test-repo")
        assert results == []

    def test_handles_completely_invalid_text(self):
        """Non-JSON text returns empty list."""
        results = parse_findings("This is just plain text with no JSON.", "test-repo")
        assert results == []

    def test_filters_low_relevance(self):
        """Findings with relevance_score < 0.4 are filtered out."""
        findings_data = [
            self._make_finding_dict(title="High", relevance_score=0.8),
            self._make_finding_dict(title="Low", relevance_score=0.2),
            self._make_finding_dict(title="Zero", relevance_score=0.0),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 1
        assert results[0].title == "High"

    def test_caps_at_max_findings(self):
        """Results are capped at _MAX_FINDINGS_PER_REPO (15)."""
        findings_data = [
            self._make_finding_dict(title=f"Finding {i}", relevance_score=0.8)
            for i in range(25)
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == _MAX_FINDINGS_PER_REPO

    def test_missing_title_excluded(self):
        """Findings without a title are excluded."""
        findings_data = [
            self._make_finding_dict(title=""),
            self._make_finding_dict(title="Valid Title"),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 1
        assert results[0].title == "Valid Title"

    def test_missing_description_excluded(self):
        """Findings without a description are excluded."""
        findings_data = [
            self._make_finding_dict(description=""),
            self._make_finding_dict(description="Valid desc"),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 1

    def test_validates_and_defaults_category(self):
        """Invalid category defaults to 'cross_cutting'."""
        findings_data = [
            self._make_finding_dict(category="unknown_category"),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 1
        assert results[0].category == "cross_cutting"

    def test_valid_categories_pass_through(self):
        """All valid categories pass through without defaulting."""
        for category in _VALID_CATEGORIES:
            findings_data = [self._make_finding_dict(category=category)]
            results = parse_findings(json.dumps(findings_data), "test-repo")
            assert len(results) == 1
            assert results[0].category == category

    def test_clamps_relevance_score(self):
        """Relevance scores are clamped to [0.4, 1.0]."""
        findings_data = [
            self._make_finding_dict(title="Clamped High", relevance_score=1.5),
            self._make_finding_dict(title="Just Right", relevance_score=0.7),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 2
        high = next(f for f in results if f.title == "Clamped High")
        assert high.relevance_score == 1.0
        right = next(f for f in results if f.title == "Just Right")
        assert right.relevance_score == 0.7

    def test_non_array_json_returns_empty(self):
        """Non-array JSON (e.g., a dict) returns empty list."""
        results = parse_findings('{"title": "not an array"}', "test-repo")
        assert results == []

    def test_finds_json_array_not_at_start(self):
        """JSON array embedded in text is extracted and parsed."""
        prefix = "Here are the findings I discovered:\n\n"
        findings_data = [self._make_finding_dict()]
        response = prefix + json.dumps(findings_data) + "\n\nThat's all."
        results = parse_findings(response, "test-repo")
        assert len(results) == 1

    def test_source_repo_injected(self):
        """source_repo field is set to the provided repo_name."""
        findings_data = [self._make_finding_dict()]
        results = parse_findings(json.dumps(findings_data), "my-cool-repo")
        assert results[0].source_repo == "my-cool-repo"

    def test_invalid_relevance_type_filtered(self):
        """Non-numeric relevance_score causes finding to be filtered (score=0.0 < 0.4)."""
        findings_data = [
            self._make_finding_dict(relevance_score="not a number"),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 0

    def test_source_files_non_list_defaults_to_empty(self):
        """Non-list source_files defaults to empty list."""
        findings_data = [
            self._make_finding_dict(source_files="not-a-list"),
        ]
        results = parse_findings(json.dumps(findings_data), "test-repo")
        assert len(results) == 1
        assert results[0].source_files == []


# ===========================================================================
# 3. _discover_repos()
# ===========================================================================

class TestDiscoverRepos:
    """Tests for _discover_repos() — git repository discovery."""

    def test_finds_base_level_repo(self, tmp_path):
        """If base itself has .git, it is discovered."""
        (tmp_path / ".git").mkdir()
        repos = _discover_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].path == tmp_path
        assert repos[0].name == tmp_path.name

    def test_finds_repos_one_level_deep(self, tmp_path):
        """Repos at immediate children level are discovered."""
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        (repo_a / ".git").mkdir()

        repo_b = tmp_path / "repo-b"
        repo_b.mkdir()
        (repo_b / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "repo-a" in names
        assert "repo-b" in names

    def test_finds_repos_two_levels_deep(self, tmp_path):
        """Repos at grandchild level are discovered."""
        org = tmp_path / "github"
        org.mkdir()
        project = org / "my-project"
        project.mkdir()
        (project / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "my-project" in names

    def test_skips_hidden_directories(self, tmp_path):
        """Directories starting with . (except .git itself) are skipped."""
        hidden = tmp_path / ".hidden-repo"
        hidden.mkdir()
        (hidden / ".git").mkdir()

        visible = tmp_path / "visible-repo"
        visible.mkdir()
        (visible / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "visible-repo" in names
        assert ".hidden-repo" not in names

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        repos = _discover_repos(tmp_path)
        assert repos == []

    def test_skips_skip_dirs_at_child_level(self, tmp_path):
        """Directories in _SKIP_DIRS are not scanned for repos."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / ".git").mkdir()

        real = tmp_path / "real-repo"
        real.mkdir()
        (real / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "real-repo" in names
        assert "node_modules" not in names

    def test_finds_repos_at_depth_4(self, tmp_path):
        """Repos nested 4 levels deep are discovered with default depth."""
        deep = tmp_path / "org" / "team" / "category" / "my-deep-repo"
        deep.mkdir(parents=True)
        (deep / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "my-deep-repo" in names

    def test_respects_max_depth(self, tmp_path):
        """Repos beyond max_depth are not discovered.

        BFS depth counting: base=0, a=1, b=2, deep-repo=3.
        max_depth controls how deep we descend into non-repo dirs.
        To discover depth=3, we need max_depth >= 3 so we descend
        into b (depth=2, which is < 3) and find deep-repo at depth 3.
        """
        # a/b/deep-repo = depth 3 from base
        deep = tmp_path / "a" / "b" / "deep-repo"
        deep.mkdir(parents=True)
        (deep / ".git").mkdir()

        # depth=1 should miss repos at depth 3
        repos = _discover_repos(tmp_path, max_depth=1)
        names = {r.name for r in repos}
        assert "deep-repo" not in names

        # depth=3 should find it
        repos = _discover_repos(tmp_path, max_depth=3)
        names = {r.name for r in repos}
        assert "deep-repo" in names

    def test_collects_file_count_metadata(self, tmp_path):
        """Discovered repos have file_count metadata from source files."""
        repo = tmp_path / "my-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "main.py").write_text("x = 1", encoding="utf-8")
        (repo / "utils.py").write_text("y = 2", encoding="utf-8")
        (repo / "data.bin").write_bytes(b"\x00")  # not a code file

        repos = _discover_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].file_count == 2  # only .py files counted

    def test_sets_canonical_name(self, tmp_path):
        """Discovered repos have canonical_name set."""
        repo = tmp_path / "my-project-v3"
        repo.mkdir()
        (repo / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].canonical_name == "my-project"

    def test_does_not_descend_into_repos(self, tmp_path):
        """Once .git is found, don't look for nested repos inside."""
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / ".git").mkdir()

        inner = outer / "vendor" / "inner"
        inner.mkdir(parents=True)
        (inner / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r.name for r in repos}
        assert "outer" in names
        assert "inner" not in names


# ===========================================================================
# 4. _relevance_to_priority()
# ===========================================================================

class TestRelevanceToPriority:
    """Tests for _relevance_to_priority() — score-to-priority mapping."""

    def test_very_high_relevance(self):
        """0.9+ maps to priority 9."""
        assert _relevance_to_priority(0.9) == 9
        assert _relevance_to_priority(0.95) == 9
        assert _relevance_to_priority(1.0) == 9

    def test_high_relevance(self):
        """0.8-0.89 maps to priority 7."""
        assert _relevance_to_priority(0.8) == 7
        assert _relevance_to_priority(0.85) == 7
        assert _relevance_to_priority(0.89) == 7

    def test_medium_relevance(self):
        """0.7-0.79 maps to priority 5."""
        assert _relevance_to_priority(0.7) == 5
        assert _relevance_to_priority(0.75) == 5
        assert _relevance_to_priority(0.79) == 5

    def test_low_relevance(self):
        """0.6-0.69 maps to priority 3."""
        assert _relevance_to_priority(0.6) == 3
        assert _relevance_to_priority(0.65) == 3
        assert _relevance_to_priority(0.69) == 3

    def test_very_low_relevance(self):
        """Below 0.6 maps to priority 1."""
        assert _relevance_to_priority(0.5) == 1
        assert _relevance_to_priority(0.3) == 1
        assert _relevance_to_priority(0.0) == 1


# ===========================================================================
# 5. _category_to_agent()
# ===========================================================================

class TestCategoryToAgent:
    """Tests for _category_to_agent() — category-to-agent mapping."""

    def test_architecture_maps_to_claude(self):
        assert _category_to_agent("architecture") == "claude"

    def test_ai_integration_maps_to_claude(self):
        assert _category_to_agent("ai_integration") == "claude"

    def test_memory_maps_to_claude(self):
        assert _category_to_agent("memory") == "claude"

    def test_security_maps_to_claude(self):
        assert _category_to_agent("security") == "claude"

    def test_code_quality_maps_to_codex(self):
        assert _category_to_agent("code_quality") == "codex"

    def test_cli_ux_maps_to_codex(self):
        assert _category_to_agent("cli_ux") == "codex"

    def test_testing_maps_to_codex(self):
        assert _category_to_agent("testing") == "codex"

    def test_data_processing_maps_to_gemini(self):
        assert _category_to_agent("data_processing") == "gemini"

    def test_algorithm_maps_to_gemini(self):
        assert _category_to_agent("algorithm") == "gemini"

    def test_cross_cutting_maps_to_grok(self):
        assert _category_to_agent("cross_cutting") == "grok"

    def test_unknown_category_defaults_to_claude(self):
        assert _category_to_agent("totally_unknown") == "claude"
        assert _category_to_agent("") == "claude"


# ===========================================================================
# 6. Data classes
# ===========================================================================

class TestDataClasses:
    """Tests for MiningFinding, RepoMiningResult, MiningReport dataclasses."""

    def test_mining_finding_defaults(self):
        """MiningFinding has correct defaults."""
        f = MiningFinding(
            title="Test",
            description="Desc",
            category="testing",
            source_repo="repo",
        )
        assert f.source_files == []
        assert f.implementation_sketch == ""
        assert f.augmentation_notes == ""
        assert f.relevance_score == 0.5
        assert f.language == "python"

    def test_mining_finding_custom_fields(self):
        """MiningFinding accepts custom values."""
        f = MiningFinding(
            title="Custom Pattern",
            description="A custom pattern",
            category="algorithm",
            source_repo="my-repo",
            source_files=["file1.py", "file2.py"],
            implementation_sketch="def algo(): pass",
            augmentation_notes="Improves speed",
            relevance_score=0.95,
            language="rust",
        )
        assert f.title == "Custom Pattern"
        assert f.language == "rust"
        assert f.relevance_score == 0.95
        assert len(f.source_files) == 2

    def test_repo_mining_result_defaults(self):
        """RepoMiningResult has correct defaults."""
        r = RepoMiningResult(repo_name="test", repo_path="/tmp/test")
        assert r.findings == []
        assert r.files_analyzed == 0
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0
        assert r.duration_seconds == 0.0
        assert r.error is None

    def test_repo_mining_result_with_error(self):
        """RepoMiningResult can carry an error message."""
        r = RepoMiningResult(
            repo_name="broken",
            repo_path="/tmp/broken",
            error="LLM call failed",
        )
        assert r.error == "LLM call failed"

    def test_mining_report_defaults(self):
        """MiningReport has correct defaults."""
        report = MiningReport()
        assert report.repos_scanned == 0
        assert report.total_findings == 0
        assert report.tasks_generated == 0
        assert report.total_cost_usd == 0.0
        assert report.total_tokens == 0
        assert report.total_duration_seconds == 0.0
        assert report.repo_results == []
        assert report.tasks == []

    def test_mining_report_accumulation(self):
        """MiningReport fields can be accumulated."""
        report = MiningReport()
        report.repos_scanned = 3
        report.total_findings = 10
        report.tasks_generated = 5
        report.total_cost_usd = 0.15
        assert report.repos_scanned == 3
        assert report.total_findings == 10
        assert report.tasks_generated == 5


# ===========================================================================
# 7. RepoMiner.store_finding() — async, real database
# ===========================================================================

class TestStoreFinding:
    """Tests for RepoMiner.store_finding() — stores in SemanticMemory."""

    async def test_store_finding_returns_methodology_id(self, repo_miner, repository, sample_project):
        """store_finding returns a methodology ID string."""
        await repository.create_project(sample_project)
        finding = MiningFinding(
            title="Useful Pattern",
            description="A pattern that could improve CLAW's memory system",
            category="memory",
            source_repo="external-repo",
            source_files=["src/memory.py"],
            implementation_sketch="class Memory: ...",
            augmentation_notes="Could be adapted for semantic search",
            relevance_score=0.85,
        )
        method_id = await repo_miner.store_finding(finding, sample_project.id)
        assert isinstance(method_id, str)
        assert len(method_id) > 0

    async def test_store_finding_creates_methodology_with_global_scope(self, repo_miner, repository, sample_project):
        """Stored methodology has scope='global' and type='PATTERN'."""
        await repository.create_project(sample_project)
        finding = MiningFinding(
            title="Architecture Pattern",
            description="Event sourcing pattern for state management",
            category="architecture",
            source_repo="event-store-lib",
            relevance_score=0.9,
        )
        method_id = await repo_miner.store_finding(finding, sample_project.id)

        # Verify via direct repository lookup
        methodology = await repository.get_methodology(method_id)
        assert methodology is not None
        assert methodology.scope == "global"
        assert methodology.methodology_type == "PATTERN"

    async def test_store_finding_tags_include_mined_and_source(self, repo_miner, repository, sample_project):
        """Methodology tags include 'mined' and 'source:{repo_name}'."""
        await repository.create_project(sample_project)
        finding = MiningFinding(
            title="Security Pattern",
            description="Input sanitization pattern",
            category="security",
            source_repo="secure-lib",
            relevance_score=0.75,
        )
        method_id = await repo_miner.store_finding(finding, sample_project.id)
        methodology = await repository.get_methodology(method_id)
        assert "mined" in methodology.tags
        assert "source:secure-lib" in methodology.tags
        assert "category:security" in methodology.tags

    async def test_store_finding_problem_description_contains_repo_name(self, repo_miner, repository, sample_project):
        """Problem description includes [Mined from repo_name] prefix."""
        await repository.create_project(sample_project)
        finding = MiningFinding(
            title="CLI Pattern",
            description="Rich console progress bars",
            category="cli_ux",
            source_repo="rich-cli",
            relevance_score=0.7,
        )
        method_id = await repo_miner.store_finding(finding, sample_project.id)
        methodology = await repository.get_methodology(method_id)
        assert "[Mined from rich-cli]" in methodology.problem_description


# ===========================================================================
# 8. RepoMiner._generate_tasks() — async, real database
# ===========================================================================

class TestGenerateTasks:
    """Tests for RepoMiner._generate_tasks() — task creation from findings."""

    async def test_filters_by_min_relevance(self, repo_miner, repository, sample_project):
        """Findings below min_relevance are not turned into tasks."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="High Relevance",
                description="Important pattern",
                category="architecture",
                source_repo="repo-a",
                relevance_score=0.8,
            ),
            MiningFinding(
                title="Low Relevance",
                description="Less important pattern",
                category="testing",
                source_repo="repo-b",
                relevance_score=0.3,  # Below any reasonable min_relevance
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert len(tasks) == 1
        assert "High Relevance" in tasks[0].title

    async def test_creates_tasks_with_correct_priority(self, repo_miner, repository, sample_project):
        """Task priority matches _relevance_to_priority mapping."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="Critical Pattern",
                description="Very important",
                category="security",
                source_repo="secure-repo",
                relevance_score=0.95,
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert len(tasks) == 1
        assert tasks[0].priority == 9  # 0.95 >= 0.9 -> priority 9

    async def test_sets_recommended_agent(self, repo_miner, repository, sample_project):
        """Task recommended_agent is set based on finding category."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="Data Processing Enhancement",
                description="Better CSV parsing",
                category="data_processing",
                source_repo="csv-lib",
                relevance_score=0.75,
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert len(tasks) == 1
        assert tasks[0].recommended_agent == "gemini"

    async def test_task_title_includes_mined_prefix(self, repo_miner, repository, sample_project):
        """Task title includes [Mined:{repo}] prefix."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="Test Pattern",
                description="Helpful testing approach",
                category="testing",
                source_repo="test-lib",
                relevance_score=0.8,
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert len(tasks) == 1
        assert tasks[0].title.startswith("[Mined:test-lib]")

    async def test_generates_multiple_tasks_sorted_by_relevance(self, repo_miner, repository, sample_project):
        """Multiple tasks are generated, sorted by relevance descending."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="Medium",
                description="Medium importance",
                category="code_quality",
                source_repo="repo-a",
                relevance_score=0.7,
            ),
            MiningFinding(
                title="High",
                description="High importance",
                category="architecture",
                source_repo="repo-b",
                relevance_score=0.9,
            ),
            MiningFinding(
                title="Low",
                description="Low importance",
                category="testing",
                source_repo="repo-c",
                relevance_score=0.6,
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert len(tasks) == 3
        # Tasks should be ordered by relevance descending
        assert "High" in tasks[0].title
        assert "Medium" in tasks[1].title
        assert "Low" in tasks[2].title

    async def test_empty_findings_returns_empty_list(self, repo_miner, repository, sample_project):
        """No findings produces no tasks."""
        await repository.create_project(sample_project)
        tasks = await repo_miner._generate_tasks([], sample_project.id, min_relevance=0.6)
        assert tasks == []

    async def test_task_status_is_pending(self, repo_miner, repository, sample_project):
        """Generated tasks have PENDING status."""
        await repository.create_project(sample_project)
        findings = [
            MiningFinding(
                title="Pending Test",
                description="Should be pending",
                category="testing",
                source_repo="repo",
                relevance_score=0.8,
            ),
        ]
        tasks = await repo_miner._generate_tasks(findings, sample_project.id, min_relevance=0.6)
        assert tasks[0].status == TaskStatus.PENDING


# ===========================================================================
# 9. CLI registration
# ===========================================================================

class TestCLIRegistration:
    """Tests for mine command registration in CLI app."""

    def test_mine_command_exists(self):
        """The 'mine' command is registered in the typer app."""
        from claw.cli import app
        # Typer stores name=None for commands using the function name as command name.
        # Resolve effective name by falling back to the callback function name.
        command_names = [
            cmd.name or (cmd.callback.__name__ if cmd.callback else None)
            for cmd in app.registered_commands
        ]
        assert "mine" in command_names


# ===========================================================================
# 10. _canonicalize_name()
# ===========================================================================

class TestCanonicalizeName:
    """Tests for _canonicalize_name() — version/variant suffix stripping."""

    def test_strips_version_suffix(self):
        """Strips -v2, -v3, _v10 etc."""
        assert _canonicalize_name("ace-forecaster-v3") == "ace-forecaster"
        assert _canonicalize_name("project_v2") == "project"
        assert _canonicalize_name("tool-v10") == "tool"

    def test_strips_common_suffixes(self):
        """Strips -final, -latest, -backup, -copy, -wip, -old, -new, -orig."""
        assert _canonicalize_name("project-final") == "project"
        assert _canonicalize_name("project-latest") == "project"
        assert _canonicalize_name("project-backup") == "project"
        assert _canonicalize_name("project-copy") == "project"
        assert _canonicalize_name("project-wip") == "project"
        assert _canonicalize_name("project-old") == "project"
        assert _canonicalize_name("project-new") == "project"
        assert _canonicalize_name("project-orig") == "project"

    def test_strips_env_suffixes(self):
        """Strips -dev, -test, -staging, -prod."""
        assert _canonicalize_name("api-dev") == "api"
        assert _canonicalize_name("api-test") == "api"
        assert _canonicalize_name("api-staging") == "api"
        assert _canonicalize_name("api-prod") == "api"

    def test_strips_trailing_digits(self):
        """Strips bare trailing digits after dash: -2, -3, etc."""
        assert _canonicalize_name("project-2") == "project"
        assert _canonicalize_name("tool-42") == "tool"

    def test_iterative_stripping(self):
        """Strips multiple suffixes iteratively."""
        assert _canonicalize_name("tool-dev-v2") == "tool"
        assert _canonicalize_name("project-old-backup") == "project"

    def test_preserves_meaningful_names(self):
        """Doesn't strip parts that are part of the project name."""
        assert _canonicalize_name("grokflow-cli") == "grokflow-cli"
        assert _canonicalize_name("ace-forecaster") == "ace-forecaster"
        assert _canonicalize_name("my-awesome-project") == "my-awesome-project"

    def test_lowercases(self):
        """Names are lowercased."""
        assert _canonicalize_name("MyProject-V2") == "myproject"

    def test_underscore_separator(self):
        """Works with underscore separator too."""
        assert _canonicalize_name("project_final") == "project"
        assert _canonicalize_name("tool_backup") == "tool"

    def test_empty_and_simple(self):
        """Handles edge cases."""
        assert _canonicalize_name("x") == "x"
        assert _canonicalize_name("project") == "project"


# ===========================================================================
# 11. _dedup_iterations()
# ===========================================================================

class TestDedupIterations:
    """Tests for _dedup_iterations() — picking best version per canonical name."""

    def _make_candidate(self, name: str, **kwargs) -> RepoCandidate:
        """Helper to create a RepoCandidate with defaults."""
        from pathlib import Path
        return RepoCandidate(
            path=Path(f"/repos/{name}"),
            name=name,
            canonical_name=_canonicalize_name(name),
            depth=kwargs.get("depth", 1),
            file_count=kwargs.get("file_count", 5),
            last_commit_ts=kwargs.get("last_commit_ts", 1000.0),
            total_bytes=kwargs.get("total_bytes", 5000),
        )

    def test_single_repo_passes_through(self):
        """Single repo with unique canonical name is always selected."""
        candidates = [self._make_candidate("my-project")]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 1
        assert len(skipped) == 0
        assert selected[0].name == "my-project"

    def test_dedup_picks_newest(self):
        """When multiple iterations exist, picks the one with latest commit."""
        candidates = [
            self._make_candidate("project-v1", last_commit_ts=1000.0),
            self._make_candidate("project-v2", last_commit_ts=2000.0),
            self._make_candidate("project-v3", last_commit_ts=3000.0),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 1
        assert selected[0].name == "project-v3"
        assert len(skipped) == 2

    def test_dedup_uses_file_count_tiebreaker(self):
        """When timestamps are equal, picks the one with most files."""
        candidates = [
            self._make_candidate("project-v1", last_commit_ts=1000.0, file_count=5),
            self._make_candidate("project-v2", last_commit_ts=1000.0, file_count=20),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 1
        assert selected[0].name == "project-v2"

    def test_dedup_uses_total_bytes_tiebreaker(self):
        """When timestamp and file_count are equal, picks largest."""
        candidates = [
            self._make_candidate("project-v1", last_commit_ts=1000.0, file_count=5, total_bytes=1000),
            self._make_candidate("project-v2", last_commit_ts=1000.0, file_count=5, total_bytes=5000),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 1
        assert selected[0].name == "project-v2"

    def test_different_canonical_names_all_selected(self):
        """Repos with different canonical names are all selected."""
        candidates = [
            self._make_candidate("project-a"),
            self._make_candidate("project-b"),
            self._make_candidate("tool-x"),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 3
        assert len(skipped) == 0

    def test_skipped_includes_reason(self):
        """Skipped entries include the superseding repo in the reason."""
        candidates = [
            self._make_candidate("project-v1", last_commit_ts=1000.0),
            self._make_candidate("project-v2", last_commit_ts=2000.0),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(skipped) == 1
        reason = skipped[0][1]
        assert "superseded by" in reason
        assert "project-v2" in reason

    def test_mixed_groups(self):
        """Mix of unique repos and iteration groups."""
        candidates = [
            self._make_candidate("alpha"),               # unique
            self._make_candidate("beta-v1", last_commit_ts=1000.0),
            self._make_candidate("beta-v2", last_commit_ts=2000.0),
            self._make_candidate("gamma"),               # unique
            self._make_candidate("delta-old", last_commit_ts=500.0),
            self._make_candidate("delta-new", last_commit_ts=3000.0),
        ]
        selected, skipped = _dedup_iterations(candidates)
        assert len(selected) == 4  # alpha, beta-v2, gamma, delta-new
        assert len(skipped) == 2   # beta-v1, delta-old
        selected_names = {c.name for c in selected}
        assert "alpha" in selected_names
        assert "beta-v2" in selected_names
        assert "gamma" in selected_names
        assert "delta-new" in selected_names


# ===========================================================================
# 12. _collect_repo_metadata()
# ===========================================================================

class TestCollectRepoMetadata:
    """Tests for _collect_repo_metadata() — lightweight metadata collection."""

    def test_counts_source_files(self, tmp_path):
        """Counts files matching _CODE_EXTENSIONS in top level."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "utils.py").write_text("y = 2", encoding="utf-8")
        (tmp_path / "data.bin").write_bytes(b"\x00")

        file_count, _, total_bytes = _collect_repo_metadata(tmp_path)
        assert file_count >= 2  # at least top-level .py files

    def test_includes_subdirectory_files(self, tmp_path):
        """Counts files in immediate subdirectories too."""
        (tmp_path / ".git").mkdir()
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("pass", encoding="utf-8")
        (src / "lib.py").write_text("pass", encoding="utf-8")

        file_count, _, _ = _collect_repo_metadata(tmp_path)
        assert file_count >= 2

    def test_uses_git_mtime_for_timestamp(self, tmp_path):
        """Uses .git directory mtime as last_commit_ts."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

        _, last_commit_ts, _ = _collect_repo_metadata(tmp_path)
        assert last_commit_ts > 0

    def test_handles_no_git_dir(self, tmp_path):
        """Returns 0 timestamp when no .git directory."""
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        _, last_commit_ts, _ = _collect_repo_metadata(tmp_path)
        assert last_commit_ts == 0.0


# ===========================================================================
# 13. RepoCandidate dataclass
# ===========================================================================

class TestRepoCandidate:
    """Tests for RepoCandidate dataclass."""

    def test_defaults(self):
        """RepoCandidate has correct defaults."""
        from pathlib import Path
        c = RepoCandidate(
            path=Path("/repos/test"),
            name="test",
            canonical_name="test",
            depth=1,
        )
        assert c.file_count == 0
        assert c.last_commit_ts == 0.0
        assert c.total_bytes == 0

    def test_custom_fields(self):
        """RepoCandidate accepts custom values."""
        from pathlib import Path
        c = RepoCandidate(
            path=Path("/repos/project-v2"),
            name="project-v2",
            canonical_name="project",
            depth=3,
            file_count=42,
            last_commit_ts=1709900000.0,
            total_bytes=150000,
        )
        assert c.canonical_name == "project"
        assert c.file_count == 42
        assert c.depth == 3
