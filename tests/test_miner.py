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
    RepoMiner,
    RepoMiningResult,
    _MAX_FINDINGS_PER_REPO,
    _SKIP_DIRS,
    _VALID_CATEGORIES,
    _category_to_agent,
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
        assert repos[0][0] == tmp_path
        assert repos[0][1] == tmp_path.name

    def test_finds_repos_one_level_deep(self, tmp_path):
        """Repos at immediate children level are discovered."""
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        (repo_a / ".git").mkdir()

        repo_b = tmp_path / "repo-b"
        repo_b.mkdir()
        (repo_b / ".git").mkdir()

        repos = _discover_repos(tmp_path)
        names = {r[1] for r in repos}
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
        names = {r[1] for r in repos}
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
        names = {r[1] for r in repos}
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
        names = {r[1] for r in repos}
        assert "real-repo" in names
        assert "node_modules" not in names


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
