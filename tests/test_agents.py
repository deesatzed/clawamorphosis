"""Tests for CLAW agent interface and all four agents.

Covers: AgentInterface (ABC), ClaudeCodeAgent, CodexAgent, GeminiAgent, GrokAgent.

Every test uses REAL objects -- no mocks, no placeholders, no cached responses.
"""

import json
import os

import pytest

from claw.agents.interface import AgentInterface
from claw.agents.claude import ClaudeCodeAgent
from claw.agents.claude import _extract_files_from_output as claude_extract_files
from claw.agents.codex import CodexAgent
from claw.agents.codex import _extract_files_from_output as codex_extract_files
from claw.agents.gemini import GeminiAgent, _MAX_REPO_BYTES
from claw.agents.gemini import _extract_files_from_output as gemini_extract_files
from claw.agents.grok import GrokAgent
from claw.agents.grok import _extract_files_from_output as grok_extract_files
from claw.core.models import AgentHealth, AgentMode, TaskContext, TaskOutcome, Task


# ===========================================================================
# AgentInterface ABC
# ===========================================================================

class TestAgentInterface:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AgentInterface("test", "Test Agent")


# ===========================================================================
# ClaudeCodeAgent
# ===========================================================================

class TestClaudeCodeAgent:
    def test_instantiation(self):
        agent = ClaudeCodeAgent(mode=AgentMode.CLI)
        assert agent.agent_id == "claude"
        assert agent.name == "Claude Code Agent"
        assert AgentMode.CLI in agent.supported_modes
        assert AgentMode.API in agent.supported_modes
        assert agent.instruction_file == "CLAUDE.md"

    def test_metrics_initialized(self):
        agent = ClaudeCodeAgent()
        metrics = agent.get_metrics()
        assert metrics["total_executed"] == 0
        assert metrics["total_errors"] == 0

    def test_parse_cli_result_success(self):
        agent = ClaudeCodeAgent()
        result = agent._parse_cli_result(
            stdout="--- FILE: src/main.py ---\nfixed the bug\n--- FILE: tests/test.py ---\nadded tests",
            stderr="",
            exit_code=0,
            duration=5.0,
        )
        assert result.agent_id == "claude"
        assert result.tests_passed is True
        assert "src/main.py" in result.files_changed
        assert "tests/test.py" in result.files_changed
        assert result.duration_seconds == 5.0

    def test_parse_cli_result_failure(self):
        agent = ClaudeCodeAgent()
        result = agent._parse_cli_result(
            stdout="Error: FAILED test",
            stderr="tests failed",
            exit_code=1,
            duration=3.0,
        )
        assert result.tests_passed is False

    def test_parse_cli_json_envelope(self):
        envelope = json.dumps({
            "type": "result",
            "result": "Fixed the issue",
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "costUsd": 0.05,
            "model": "claude-opus",
        })
        agent = ClaudeCodeAgent()
        result = agent._parse_cli_result(envelope, "", 0, 10.0)
        assert result.tokens_used > 0
        assert result.cost_usd == 0.05

    def test_build_prompt_basic(self):
        agent = ClaudeCodeAgent()
        task = Task(project_id="p1", title="Fix login", description="Fix the login page bug")
        ctx = TaskContext(task=task)
        prompt = agent._build_prompt(ctx)
        assert "Fix login" in prompt
        assert "Fix the login page bug" in prompt

    def test_build_prompt_with_forbidden(self):
        agent = ClaudeCodeAgent()
        task = Task(project_id="p1", title="Fix login", description="Fix it")
        ctx = TaskContext(
            task=task,
            forbidden_approaches=["Used regex parsing", "Modified the database directly"],
        )
        prompt = agent._build_prompt(ctx)
        assert "Forbidden Approaches" in prompt
        assert "regex parsing" in prompt

    def test_extract_files_from_output(self):
        output = (
            "--- FILE: src/main.py ---\ncode here\n"
            "--- FILE: tests/test_main.py ---\ntest code\n"
            "Some other text\n"
        )
        files = claude_extract_files(output)
        assert files == ["src/main.py", "tests/test_main.py"]

    def test_extract_files_no_duplicates(self):
        output = "--- FILE: a.py ---\nfoo\n--- FILE: a.py ---\nbar\n"
        files = claude_extract_files(output)
        assert files == ["a.py"]

    def test_extract_files_empty_output(self):
        assert claude_extract_files("") == []
        assert claude_extract_files("no file markers here") == []


# ===========================================================================
# CodexAgent
# ===========================================================================

class TestCodexAgent:
    def test_instantiation(self):
        agent = CodexAgent(mode=AgentMode.CLI)
        assert agent.agent_id == "codex"
        assert agent.name == "Codex Agent"
        assert AgentMode.CLI in agent.supported_modes
        assert AgentMode.API in agent.supported_modes
        assert AgentMode.CLOUD in agent.supported_modes
        assert agent.instruction_file == "AGENTS.md"

    def test_instantiation_defaults(self):
        agent = CodexAgent()
        assert agent.mode == AgentMode.CLI
        assert agent.timeout == 600
        assert agent.max_tokens == 4096
        assert agent.model is None

    def test_metrics_initialized(self):
        agent = CodexAgent()
        metrics = agent.get_metrics()
        assert metrics["total_executed"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["last_duration_seconds"] == 0.0

    def test_parse_cli_result_success(self):
        agent = CodexAgent()
        result = agent._parse_cli_result(
            stdout=(
                "--- FILE: src/utils.py ---\nrefactored utilities\n"
                "--- FILE: tests/test_utils.py ---\nadded new tests"
            ),
            stderr="",
            exit_code=0,
            duration=7.5,
        )
        assert result.agent_id == "codex"
        assert result.tests_passed is True
        assert "src/utils.py" in result.files_changed
        assert "tests/test_utils.py" in result.files_changed
        assert result.duration_seconds == 7.5
        assert result.approach_summary != ""

    def test_parse_cli_result_failure(self):
        agent = CodexAgent()
        result = agent._parse_cli_result(
            stdout="FAILED: compilation error in module X",
            stderr="build error details",
            exit_code=1,
            duration=2.0,
        )
        assert result.tests_passed is False
        assert result.agent_id == "codex"
        assert result.test_output == "build error details"
        assert result.duration_seconds == 2.0

    def test_parse_cli_result_failure_exit_code_only(self):
        """Exit code 1 alone should mean tests_passed is False, even without FAILED in output."""
        agent = CodexAgent()
        result = agent._parse_cli_result(
            stdout="Some generic error output",
            stderr="",
            exit_code=1,
            duration=1.0,
        )
        assert result.tests_passed is False

    def test_parse_cli_result_failure_keyword_only(self):
        """FAILED in output with exit_code=0 should still mean tests_passed is False."""
        agent = CodexAgent()
        result = agent._parse_cli_result(
            stdout="Test suite FAILED with 3 errors",
            stderr="",
            exit_code=0,
            duration=4.0,
        )
        assert result.tests_passed is False

    def test_parse_cli_json_envelope(self):
        envelope = json.dumps({
            "result": "Refactored the module successfully",
            "usage": {"prompt_tokens": 800, "completion_tokens": 400},
            "cost_usd": 0.03,
            "model": "codex-model-v1",
        })
        agent = CodexAgent()
        result = agent._parse_cli_result(envelope, "", 0, 12.0)
        assert result.tokens_used == 800 + 400
        assert result.cost_usd == 0.03
        assert result.model_used == "codex-model-v1"
        assert result.duration_seconds == 12.0

    def test_parse_cli_json_envelope_input_output_tokens(self):
        """Codex JSON can use input_tokens/output_tokens format too."""
        envelope = json.dumps({
            "content": "Done",
            "usage": {"input_tokens": 500, "output_tokens": 250},
            "costUsd": 0.02,
            "model": "o1-mini",
        })
        agent = CodexAgent()
        result = agent._parse_cli_result(envelope, "", 0, 5.0)
        assert result.tokens_used == 500 + 250
        assert result.cost_usd == 0.02
        assert result.model_used == "o1-mini"

    def test_parse_cli_json_envelope_mixed_token_keys(self):
        """Codex sums across all four possible token keys."""
        envelope = json.dumps({
            "output": "All done",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "prompt_tokens": 200,
                "completion_tokens": 75,
            },
        })
        agent = CodexAgent()
        result = agent._parse_cli_result(envelope, "", 0, 1.0)
        assert result.tokens_used == 100 + 50 + 200 + 75

    def test_build_prompt_basic(self):
        agent = CodexAgent()
        task = Task(project_id="p1", title="Add tests", description="Write unit tests for auth module")
        ctx = TaskContext(task=task)
        prompt = agent._build_prompt(ctx)
        assert "Add tests" in prompt
        assert "Write unit tests for auth module" in prompt

    def test_build_prompt_with_forbidden(self):
        agent = CodexAgent()
        task = Task(project_id="p1", title="Refactor DB", description="Improve DB layer")
        ctx = TaskContext(
            task=task,
            forbidden_approaches=["Used raw SQL strings", "Dropped the table"],
        )
        prompt = agent._build_prompt(ctx)
        assert "Forbidden Approaches" in prompt
        assert "raw SQL strings" in prompt
        assert "Dropped the table" in prompt

    def test_build_prompt_with_previous_diagnosis(self):
        agent = CodexAgent()
        task = Task(project_id="p1", title="Fix race", description="Fix the race condition")
        ctx = TaskContext(
            task=task,
            previous_escalation_diagnosis="The issue is a deadlock in the connection pool",
        )
        prompt = agent._build_prompt(ctx)
        assert "Previous Diagnosis" in prompt
        assert "deadlock in the connection pool" in prompt

    def test_extract_files_standard_marker(self):
        output = "--- FILE: src/api.py ---\ncode\n--- FILE: src/models.py ---\nmore code"
        files = codex_extract_files(output)
        assert "src/api.py" in files
        assert "src/models.py" in files

    def test_extract_files_git_diff_marker(self):
        output = "+++ b/src/handler.py\n@@ -1,3 +1,5 @@\n+new line"
        files = codex_extract_files(output)
        assert "src/handler.py" in files

    def test_extract_files_modified_marker(self):
        output = "Modified: src/config.py\nModified: src/utils.py"
        files = codex_extract_files(output)
        assert "src/config.py" in files
        assert "src/utils.py" in files

    def test_extract_files_mixed_markers(self):
        output = (
            "--- FILE: a.py ---\ncode\n"
            "+++ b/b.py\ndiff\n"
            "Modified: c.py\n"
        )
        files = codex_extract_files(output)
        assert files == ["a.py", "b.py", "c.py"]

    def test_extract_files_no_duplicates(self):
        output = (
            "--- FILE: x.py ---\nfoo\n"
            "+++ b/x.py\nbar\n"
            "Modified: x.py\n"
        )
        files = codex_extract_files(output)
        assert files == ["x.py"]

    def test_extract_files_empty(self):
        assert codex_extract_files("") == []
        assert codex_extract_files("nothing special") == []


# ===========================================================================
# GeminiAgent
# ===========================================================================

class TestGeminiAgent:
    def test_instantiation(self):
        agent = GeminiAgent(mode=AgentMode.CLI)
        assert agent.agent_id == "gemini"
        assert agent.name == "Gemini Agent"
        assert AgentMode.CLI in agent.supported_modes
        assert AgentMode.API in agent.supported_modes
        assert AgentMode.CLOUD not in agent.supported_modes
        assert agent.instruction_file == "GEMINI.md"

    def test_instantiation_defaults(self):
        agent = GeminiAgent()
        assert agent.mode == AgentMode.CLI
        assert agent.timeout == 600
        assert agent.model is None
        assert agent.workspace_dir is None

    def test_metrics_initialized(self):
        agent = GeminiAgent()
        metrics = agent.get_metrics()
        assert metrics["total_executed"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["last_duration_seconds"] == 0.0

    def test_parse_cli_result_success(self):
        agent = GeminiAgent()
        result = agent._parse_cli_result(
            stdout=(
                "--- FILE: lib/parser.py ---\nimproved parser\n"
                "--- FILE: lib/lexer.py ---\nupdated lexer"
            ),
            stderr="",
            exit_code=0,
            duration=8.2,
        )
        assert result.agent_id == "gemini"
        assert result.tests_passed is True
        assert "lib/parser.py" in result.files_changed
        assert "lib/lexer.py" in result.files_changed
        assert result.duration_seconds == 8.2

    def test_parse_cli_result_failure(self):
        agent = GeminiAgent()
        result = agent._parse_cli_result(
            stdout="Analysis FAILED: could not parse repository",
            stderr="error details here",
            exit_code=1,
            duration=4.5,
        )
        assert result.tests_passed is False
        assert result.agent_id == "gemini"
        assert result.test_output == "error details here"

    def test_parse_cli_result_failure_exit_code_only(self):
        agent = GeminiAgent()
        result = agent._parse_cli_result(
            stdout="Some output without failure keyword",
            stderr="",
            exit_code=1,
            duration=1.0,
        )
        assert result.tests_passed is False

    def test_parse_cli_json_envelope(self):
        envelope = json.dumps({
            "text": "Dependency analysis complete",
            "usage": {
                "prompt_tokens": 2000,
                "candidates_tokens": 1000,
                "total_tokens": 3000,
            },
            "cost_usd": 0.01,
            "model": "gemini-2.5-pro",
        })
        agent = GeminiAgent()
        result = agent._parse_cli_result(envelope, "", 0, 15.0)
        # Gemini sums prompt_tokens + candidates_tokens + total_tokens
        assert result.tokens_used == 2000 + 1000 + 3000
        assert result.cost_usd == 0.01
        assert result.model_used == "gemini-2.5-pro"
        assert result.duration_seconds == 15.0

    def test_parse_cli_json_envelope_partial_usage(self):
        """JSON with only some usage fields should still parse correctly."""
        envelope = json.dumps({
            "result": "Done",
            "usage": {"total_tokens": 5000},
        })
        agent = GeminiAgent()
        result = agent._parse_cli_result(envelope, "", 0, 3.0)
        assert result.tokens_used == 5000

    def test_build_prompt_basic(self):
        agent = GeminiAgent()
        task = Task(
            project_id="p1",
            title="Analyze dependencies",
            description="Map all dependencies and find circular imports",
        )
        ctx = TaskContext(task=task)
        prompt = agent._build_prompt(ctx)
        assert "Analyze dependencies" in prompt
        assert "Map all dependencies" in prompt

    def test_build_prompt_with_forbidden(self):
        agent = GeminiAgent()
        task = Task(project_id="p1", title="Fix imports", description="Clean up imports")
        ctx = TaskContext(
            task=task,
            forbidden_approaches=["Removed all imports", "Used wildcard imports"],
        )
        prompt = agent._build_prompt(ctx)
        assert "Forbidden Approaches" in prompt
        assert "Removed all imports" in prompt
        assert "wildcard imports" in prompt

    def test_build_prompt_with_previous_diagnosis(self):
        agent = GeminiAgent()
        task = Task(project_id="p1", title="Fix perf", description="Improve performance")
        ctx = TaskContext(
            task=task,
            previous_escalation_diagnosis="N+1 query pattern detected in ORM layer",
        )
        prompt = agent._build_prompt(ctx)
        assert "Previous Diagnosis" in prompt
        assert "N+1 query pattern" in prompt

    def test_extract_files_from_output(self):
        output = (
            "--- FILE: src/app.py ---\napp code\n"
            "--- FILE: src/config.py ---\nconfig code\n"
            "plain text here\n"
        )
        files = gemini_extract_files(output)
        assert files == ["src/app.py", "src/config.py"]

    def test_extract_files_no_duplicates(self):
        output = "--- FILE: x.py ---\na\n--- FILE: x.py ---\nb\n"
        files = gemini_extract_files(output)
        assert files == ["x.py"]

    def test_extract_files_empty(self):
        assert gemini_extract_files("") == []

    # -----------------------------------------------------------------------
    # _serialize_repo tests
    # -----------------------------------------------------------------------

    def test_serialize_repo_basic(self, tmp_path):
        """Serialize a directory with a few .py files and verify output."""
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        (tmp_path / "README.txt").write_text("This is a readme", encoding="utf-8")

        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))

        assert "--- FILE: main.py ---" in result
        assert "print('hello')" in result
        assert "--- FILE: utils.py ---" in result
        assert "def add(a, b)" in result
        # README.txt should NOT be included (.txt not in _CODE_EXTENSIONS)
        assert "README.txt" not in result

    def test_serialize_repo_includes_supported_extensions(self, tmp_path):
        """Verify that multiple supported code extensions are picked up."""
        (tmp_path / "app.py").write_text("# python", encoding="utf-8")
        (tmp_path / "index.js").write_text("// javascript", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")
        (tmp_path / "schema.sql").write_text("CREATE TABLE t(id INT);", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")  # not supported

        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))

        assert "--- FILE: app.py ---" in result
        assert "--- FILE: index.js ---" in result
        assert "--- FILE: config.yaml ---" in result
        assert "--- FILE: schema.sql ---" in result
        assert "data.csv" not in result

    def test_serialize_repo_skips_excluded_dirs(self, tmp_path):
        """__pycache__, .git, node_modules, etc. should be skipped."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("cached", encoding="utf-8")

        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("git config", encoding="utf-8")

        node_dir = tmp_path / "node_modules"
        node_dir.mkdir()
        (node_dir / "package.json").write_text("{}", encoding="utf-8")

        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "activate.py").write_text("activate", encoding="utf-8")

        # This file SHOULD be included
        (tmp_path / "real.py").write_text("real code", encoding="utf-8")

        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))

        assert "--- FILE: real.py ---" in result
        assert "cached.py" not in result
        assert "__pycache__" not in result
        assert ".git" not in result or "--- FILE: .git" not in result
        assert "node_modules" not in result or "--- FILE: node_modules" not in result

    def test_serialize_repo_skips_nested_excluded_dirs(self, tmp_path):
        """Excluded dirs nested deeper in the tree should also be skipped."""
        nested = tmp_path / "src" / "__pycache__"
        nested.mkdir(parents=True)
        (nested / "module.py").write_text("cached module", encoding="utf-8")

        (tmp_path / "src" / "app.py").write_text("app", encoding="utf-8")

        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))

        assert "--- FILE: src/app.py ---" in result or "--- FILE: src\\app.py ---" in result
        assert "module.py" not in result

    def test_serialize_repo_size_limit(self, tmp_path):
        """When total content exceeds 900KB, a truncation notice should appear."""
        # Create files that collectively exceed _MAX_REPO_BYTES (900KB)
        # Each file is ~100KB, so 10 files = ~1MB which exceeds 900KB
        for i in range(12):
            content = f"# File {i}\n" + ("x" * 100_000) + "\n"
            (tmp_path / f"big_{i:02d}.py").write_text(content, encoding="utf-8")

        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))

        assert "TRUNCATED" in result
        assert "900KB" in result
        # The total output should not dramatically exceed the limit
        # (the truncation message is appended after exceeding, so allow some margin)
        result_bytes = len(result.encode("utf-8"))
        # Should be roughly at the limit -- allow up to limit + 200KB for the
        # last accepted chunk plus the truncation notice
        assert result_bytes < _MAX_REPO_BYTES + 200_000

    def test_serialize_repo_nonexistent_dir(self):
        """Non-existent directory should return empty string."""
        agent = GeminiAgent()
        result = agent._serialize_repo("/nonexistent/path/that/does/not/exist")
        assert result == ""

    def test_serialize_repo_empty_dir(self, tmp_path):
        """Empty directory should return empty string."""
        agent = GeminiAgent()
        result = agent._serialize_repo(str(tmp_path))
        assert result == ""

    def test_serialize_repo_file_not_dir(self, tmp_path):
        """Passing a file path instead of directory should return empty string."""
        fpath = tmp_path / "file.py"
        fpath.write_text("content", encoding="utf-8")
        agent = GeminiAgent()
        result = agent._serialize_repo(str(fpath))
        assert result == ""


# ===========================================================================
# GrokAgent
# ===========================================================================

class TestGrokAgent:
    def test_instantiation(self):
        agent = GrokAgent(mode=AgentMode.CLI)
        assert agent.agent_id == "grok"
        assert agent.name == "Grok Agent"
        assert AgentMode.CLI in agent.supported_modes
        assert AgentMode.API in agent.supported_modes
        assert AgentMode.CLOUD not in agent.supported_modes
        assert agent.instruction_file == ".grok/GROK.md"

    def test_instantiation_defaults(self):
        agent = GrokAgent()
        assert agent.mode == AgentMode.CLI
        assert agent.timeout == 600
        assert agent.max_budget_usd == 1.0
        assert agent.model is None

    def test_metrics_initialized(self):
        agent = GrokAgent()
        metrics = agent.get_metrics()
        assert metrics["total_executed"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["last_duration_seconds"] == 0.0

    def test_parse_cli_result_success(self):
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout=(
                "--- FILE: src/fix.py ---\nquick fix applied\n"
                "--- FILE: tests/test_fix.py ---\ntest added"
            ),
            stderr="",
            exit_code=0,
            duration=1.2,
        )
        assert result.agent_id == "grok"
        assert result.tests_passed is True
        assert "src/fix.py" in result.files_changed
        assert "tests/test_fix.py" in result.files_changed
        assert result.duration_seconds == 1.2

    def test_parse_cli_result_success_no_files(self):
        """Successful result with no file markers should still pass."""
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout="Analysis complete. No changes needed.",
            stderr="",
            exit_code=0,
            duration=0.5,
        )
        assert result.tests_passed is True
        assert result.files_changed == []

    def test_parse_cli_result_failure(self):
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout="Error: FAILED to apply fix",
            stderr="syntax error in output",
            exit_code=1,
            duration=0.8,
        )
        assert result.tests_passed is False
        assert result.agent_id == "grok"
        assert result.test_output == "syntax error in output"

    def test_parse_cli_result_failure_exit_code_only(self):
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout="Output without failure keyword",
            stderr="",
            exit_code=1,
            duration=0.3,
        )
        assert result.tests_passed is False

    def test_parse_cli_result_failure_keyword_only(self):
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout="Some test FAILED unexpectedly",
            stderr="",
            exit_code=0,
            duration=2.0,
        )
        assert result.tests_passed is False

    def test_parse_cli_json_envelope(self):
        envelope = json.dumps({
            "result": "Fixed the fast route",
            "usage": {
                "input_tokens": 600,
                "output_tokens": 300,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            "cost_usd": 0.008,
            "model": "grok-beta",
        })
        agent = GrokAgent()
        result = agent._parse_cli_result(envelope, "", 0, 2.5)
        assert result.tokens_used == 600 + 300
        assert result.cost_usd == 0.008
        assert result.model_used == "grok-beta"

    def test_parse_cli_json_envelope_costUsd_key(self):
        """Grok supports both cost_usd and costUsd keys."""
        envelope = json.dumps({
            "output": "Done",
            "usage": {"prompt_tokens": 400, "completion_tokens": 200},
            "costUsd": 0.015,
        })
        agent = GrokAgent()
        result = agent._parse_cli_result(envelope, "", 0, 1.0)
        assert result.tokens_used == 400 + 200
        assert result.cost_usd == 0.015

    def test_parse_cli_json_envelope_all_token_keys(self):
        """Grok sums all four possible token keys."""
        envelope = json.dumps({
            "result": "All done",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "prompt_tokens": 200,
                "completion_tokens": 75,
            },
        })
        agent = GrokAgent()
        result = agent._parse_cli_result(envelope, "", 0, 1.0)
        assert result.tokens_used == 100 + 50 + 200 + 75

    def test_parse_cli_plain_text_no_json(self):
        """Plain text output (not JSON) should still parse correctly."""
        agent = GrokAgent()
        result = agent._parse_cli_result(
            stdout="Here is the analysis result with no JSON structure",
            stderr="",
            exit_code=0,
            duration=3.0,
        )
        assert result.tests_passed is True
        assert result.tokens_used == 0
        assert result.cost_usd == 0.0
        assert result.approach_summary.startswith("Here is the analysis")

    def test_parse_cli_model_from_constructor(self):
        """When model is set on the agent, it should appear in the result if not in JSON."""
        agent = GrokAgent(model="grok-3")
        result = agent._parse_cli_result(
            stdout="plain text output",
            stderr="",
            exit_code=0,
            duration=1.0,
        )
        assert result.model_used == "grok-3"

    def test_build_prompt_basic(self):
        agent = GrokAgent()
        task = Task(project_id="p1", title="Quick fix", description="Fix the typo in README")
        ctx = TaskContext(task=task)
        prompt = agent._build_prompt(ctx)
        assert "Quick fix" in prompt
        assert "Fix the typo in README" in prompt

    def test_build_prompt_with_forbidden(self):
        agent = GrokAgent()
        task = Task(project_id="p1", title="Fix API", description="Fix the API endpoint")
        ctx = TaskContext(
            task=task,
            forbidden_approaches=["Disabled authentication", "Hardcoded credentials"],
        )
        prompt = agent._build_prompt(ctx)
        assert "Forbidden Approaches" in prompt
        assert "Disabled authentication" in prompt
        assert "Hardcoded credentials" in prompt

    def test_build_prompt_with_previous_diagnosis(self):
        agent = GrokAgent()
        task = Task(project_id="p1", title="Debug crash", description="Fix the crash")
        ctx = TaskContext(
            task=task,
            previous_escalation_diagnosis="Null pointer in session handler",
        )
        prompt = agent._build_prompt(ctx)
        assert "Previous Diagnosis" in prompt
        assert "Null pointer in session handler" in prompt

    def test_build_prompt_no_forbidden_section_when_empty(self):
        """When forbidden_approaches is empty, the section should not appear."""
        agent = GrokAgent()
        task = Task(project_id="p1", title="Task", description="Do something")
        ctx = TaskContext(task=task, forbidden_approaches=[])
        prompt = agent._build_prompt(ctx)
        assert "Forbidden Approaches" not in prompt

    def test_build_prompt_no_diagnosis_section_when_none(self):
        """When previous_escalation_diagnosis is None, the section should not appear."""
        agent = GrokAgent()
        task = Task(project_id="p1", title="Task", description="Do something")
        ctx = TaskContext(task=task)
        prompt = agent._build_prompt(ctx)
        assert "Previous Diagnosis" not in prompt

    def test_extract_files_from_output(self):
        output = (
            "--- FILE: hotfix.py ---\nfix code\n"
            "--- FILE: tests/test_hotfix.py ---\ntest code\n"
        )
        files = grok_extract_files(output)
        assert files == ["hotfix.py", "tests/test_hotfix.py"]

    def test_extract_files_no_duplicates(self):
        output = "--- FILE: a.py ---\nx\n--- FILE: a.py ---\ny\n"
        files = grok_extract_files(output)
        assert files == ["a.py"]

    def test_extract_files_empty(self):
        assert grok_extract_files("") == []
        assert grok_extract_files("no markers here") == []


# ===========================================================================
# Cross-agent consistency tests
# ===========================================================================

class TestCrossAgentConsistency:
    """Verify uniform behavior across all four agents."""

    def _make_agents(self):
        return [
            ClaudeCodeAgent(),
            CodexAgent(),
            GeminiAgent(),
            GrokAgent(),
        ]

    def test_all_agents_have_unique_ids(self):
        agents = self._make_agents()
        ids = [a.agent_id for a in agents]
        assert len(ids) == len(set(ids))

    def test_all_agents_have_names(self):
        for agent in self._make_agents():
            assert agent.name
            assert isinstance(agent.name, str)

    def test_all_agents_have_instruction_files(self):
        for agent in self._make_agents():
            assert agent.instruction_file
            assert isinstance(agent.instruction_file, str)

    def test_all_agents_support_cli(self):
        for agent in self._make_agents():
            assert AgentMode.CLI in agent.supported_modes

    def test_all_agents_support_api(self):
        for agent in self._make_agents():
            assert AgentMode.API in agent.supported_modes

    def test_all_agents_metrics_start_at_zero(self):
        for agent in self._make_agents():
            metrics = agent.get_metrics()
            assert metrics["total_executed"] == 0
            assert metrics["total_errors"] == 0
            assert metrics["total_successes"] == 0

    def test_all_agents_build_prompt_includes_title_and_description(self):
        task = Task(project_id="p1", title="Universal Task", description="Universal description")
        ctx = TaskContext(task=task)
        for agent in self._make_agents():
            prompt = agent._build_prompt(ctx)
            assert "Universal Task" in prompt, f"{agent.agent_id} missing title in prompt"
            assert "Universal description" in prompt, f"{agent.agent_id} missing description in prompt"

    def test_all_agents_build_prompt_includes_forbidden(self):
        task = Task(project_id="p1", title="T", description="D")
        ctx = TaskContext(
            task=task,
            forbidden_approaches=["Bad approach"],
        )
        for agent in self._make_agents():
            prompt = agent._build_prompt(ctx)
            assert "Forbidden Approaches" in prompt, f"{agent.agent_id} missing forbidden section"
            assert "Bad approach" in prompt, f"{agent.agent_id} missing forbidden item"

    def test_all_agents_build_prompt_includes_runbook_sections(self):
        task = Task(
            project_id="p1",
            title="T",
            description="D",
            execution_steps=["pytest -q tests/test_api.py"],
            acceptance_checks=["pytest -q tests/test_api.py"],
        )
        ctx = TaskContext(task=task)
        for agent in self._make_agents():
            prompt = agent._build_prompt(ctx)
            assert "Execution Steps" in prompt, f"{agent.agent_id} missing execution steps section"
            assert "Acceptance Checks" in prompt, f"{agent.agent_id} missing acceptance checks section"
            assert "pytest -q tests/test_api.py" in prompt

    def test_all_agents_parse_success(self):
        stdout = "--- FILE: f.py ---\ncode"
        for agent in self._make_agents():
            result = agent._parse_cli_result(stdout, "", 0, 1.0)
            assert result.tests_passed is True
            assert result.agent_id == agent.agent_id

    def test_all_agents_parse_failure_on_exit_code(self):
        for agent in self._make_agents():
            result = agent._parse_cli_result("some output", "", 1, 1.0)
            assert result.tests_passed is False
            assert result.agent_id == agent.agent_id

    def test_all_agents_parse_failure_on_failed_keyword(self):
        for agent in self._make_agents():
            result = agent._parse_cli_result("test FAILED", "", 0, 1.0)
            assert result.tests_passed is False
