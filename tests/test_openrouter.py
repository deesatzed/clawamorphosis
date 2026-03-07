"""Tests for OpenRouter mode across all four CLAW agents.

Covers:
1. AgentMode.OPENROUTER enum value
2. OpenRouter health check for each agent (with/without API key, with/without model)
3. OpenRouter execute routing (OPENROUTER mode delegates to execute_openrouter)
4. _build_openrouter_prompt() method on base AgentInterface
5. execute_openrouter() success, error, and edge-case paths

Justification for unittest.mock usage:
We are testing OUR code's control flow, parsing, and error handling in
execute_openrouter(). The actual httpx.AsyncClient.post call is the
external boundary. We patch ONLY that network call to avoid real API
charges during CI. This is the accepted pattern per the user's explicit
instruction for this test file.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claw.agents.claude import ClaudeCodeAgent
from claw.agents.codex import CodexAgent
from claw.agents.gemini import GeminiAgent
from claw.agents.grok import GrokAgent
from claw.agents.interface import AgentInterface
from claw.core.models import AgentHealth, AgentMode, Task, TaskContext, TaskOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task_context(
    title: str = "Test Task",
    description: str = "Do the thing",
    forbidden: list[str] | None = None,
    hints: list[str] | None = None,
) -> TaskContext:
    """Build a real TaskContext for tests."""
    task = Task(project_id="proj-1", title=title, description=description)
    return TaskContext(
        task=task,
        forbidden_approaches=forbidden or [],
        hints=hints or [],
    )


def _make_openrouter_response_json(
    content: str = "Here is the answer.",
    model: str = "openai/gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> dict:
    """Build a real OpenRouter chat/completions response dict."""
    return {
        "id": "gen-abc123",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_httpx_response(
    status_code: int = 200,
    json_body: dict | None = None,
) -> httpx.Response:
    """Build a real httpx.Response object with the given status and body."""
    body = json.dumps(json_body or {}).encode("utf-8")
    return httpx.Response(
        status_code=status_code,
        content=body,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def _all_agents_openrouter(model: str = "openai/gpt-4o") -> list[AgentInterface]:
    """Create all 4 agents in OPENROUTER mode with a model set."""
    return [
        ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model=model),
        CodexAgent(mode=AgentMode.OPENROUTER, model=model),
        GeminiAgent(mode=AgentMode.OPENROUTER, model=model),
        GrokAgent(mode=AgentMode.OPENROUTER, model=model),
    ]


# ===========================================================================
# 1. AgentMode.OPENROUTER enum
# ===========================================================================

class TestAgentModeOpenRouter:
    def test_openrouter_enum_exists(self):
        assert hasattr(AgentMode, "OPENROUTER")
        assert AgentMode.OPENROUTER.value == "openrouter"

    def test_openrouter_is_string_enum(self):
        assert isinstance(AgentMode.OPENROUTER, str)
        assert AgentMode.OPENROUTER == "openrouter"

    def test_openrouter_in_all_values(self):
        values = [m.value for m in AgentMode]
        assert "openrouter" in values


# ===========================================================================
# 2. OpenRouter health check per agent
# ===========================================================================

class TestOpenRouterHealthCheck:
    """Health check tests for all 4 agents in OPENROUTER mode."""

    # -- No API key --

    async def test_claude_health_no_api_key(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="test-model")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            health = await agent.health_check()
        assert isinstance(health, AgentHealth)
        assert health.available is False
        assert health.agent_id == "claude"
        assert health.mode == AgentMode.OPENROUTER
        assert "OPENROUTER_API_KEY" in (health.error or "")

    async def test_codex_health_no_api_key(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="test-model")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            health = await agent.health_check()
        assert health.available is False
        assert health.agent_id == "codex"
        assert health.mode == AgentMode.OPENROUTER

    async def test_gemini_health_no_api_key(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="test-model")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            health = await agent.health_check()
        assert health.available is False
        assert health.agent_id == "gemini"
        assert health.mode == AgentMode.OPENROUTER

    async def test_grok_health_no_api_key(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="test-model")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            health = await agent.health_check()
        assert health.available is False
        assert health.agent_id == "grok"
        assert health.mode == AgentMode.OPENROUTER

    # -- No model --

    async def test_claude_health_no_model(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model=None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is False
        assert "model" in (health.error or "").lower()

    async def test_codex_health_no_model(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model=None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is False
        assert "model" in (health.error or "").lower()

    async def test_gemini_health_no_model(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model=None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is False
        assert "model" in (health.error or "").lower()

    async def test_grok_health_no_model(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model=None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is False
        assert "model" in (health.error or "").lower()

    # -- Healthy (key + model present) --

    async def test_claude_health_ok(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="anthropic/claude-3.5-sonnet")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is True
        assert health.agent_id == "claude"
        assert health.mode == AgentMode.OPENROUTER
        assert "openrouter:" in (health.version or "")
        assert health.error is None

    async def test_codex_health_ok(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="openai/gpt-4o")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is True
        assert health.agent_id == "codex"
        assert health.version == "openrouter:openai/gpt-4o"

    async def test_gemini_health_ok(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="google/gemini-2.5-pro")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is True
        assert health.agent_id == "gemini"
        assert health.version == "openrouter:google/gemini-2.5-pro"

    async def test_grok_health_ok(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="xai/grok-3")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            health = await agent.health_check()
        assert health.available is True
        assert health.agent_id == "grok"
        assert health.version == "openrouter:xai/grok-3"


# ===========================================================================
# 3. OpenRouter execute routing
# ===========================================================================

class TestOpenRouterRouting:
    """Verify that mode=OPENROUTER routes execute() to execute_openrouter()."""

    async def test_claude_routes_to_openrouter(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()
        with patch.object(agent, "execute_openrouter", new_callable=AsyncMock) as patched:
            patched.return_value = TaskOutcome(agent_id="claude", tests_passed=True)
            result = await agent.execute(ctx)
            patched.assert_awaited_once_with(ctx, None)
        assert result.agent_id == "claude"

    async def test_codex_routes_to_openrouter(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()
        with patch.object(agent, "execute_openrouter", new_callable=AsyncMock) as patched:
            patched.return_value = TaskOutcome(agent_id="codex", tests_passed=True)
            result = await agent.execute(ctx)
            patched.assert_awaited_once_with(ctx, None)
        assert result.agent_id == "codex"

    async def test_gemini_routes_to_openrouter(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()
        with patch.object(agent, "execute_openrouter", new_callable=AsyncMock) as patched:
            patched.return_value = TaskOutcome(agent_id="gemini", tests_passed=True)
            result = await agent.execute(ctx)
            patched.assert_awaited_once_with(ctx, None)
        assert result.agent_id == "gemini"

    async def test_grok_routes_to_openrouter(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()
        with patch.object(agent, "execute_openrouter", new_callable=AsyncMock) as patched:
            patched.return_value = TaskOutcome(agent_id="grok", tests_passed=True)
            result = await agent.execute(ctx)
            patched.assert_awaited_once_with(ctx, None)
        assert result.agent_id == "grok"


# ===========================================================================
# 4. _build_openrouter_prompt()
# ===========================================================================

class TestBuildOpenRouterPrompt:
    """Test the base-class _build_openrouter_prompt() method."""

    def test_basic_prompt_includes_title_and_description(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="m")
        ctx = _make_task_context(title="Fix login bug", description="The login page crashes")
        prompt = agent._build_openrouter_prompt(ctx)
        assert "Fix login bug" in prompt
        assert "The login page crashes" in prompt

    def test_prompt_includes_forbidden_approaches(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="m")
        ctx = _make_task_context(
            forbidden=["Used regex", "Hardcoded values"],
        )
        prompt = agent._build_openrouter_prompt(ctx)
        assert "Forbidden Approaches" in prompt
        assert "Used regex" in prompt
        assert "Hardcoded values" in prompt

    def test_prompt_includes_hints(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="m")
        ctx = _make_task_context(hints=["Try async approach", "Check the cache layer"])
        prompt = agent._build_openrouter_prompt(ctx)
        assert "Hints from Past Solutions" in prompt
        assert "Try async approach" in prompt
        assert "Check the cache layer" in prompt

    def test_prompt_no_forbidden_section_when_empty(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="m")
        ctx = _make_task_context(forbidden=[])
        prompt = agent._build_openrouter_prompt(ctx)
        assert "Forbidden Approaches" not in prompt

    def test_prompt_no_hints_section_when_empty(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="m")
        ctx = _make_task_context(hints=[])
        prompt = agent._build_openrouter_prompt(ctx)
        assert "Hints from Past Solutions" not in prompt

    def test_prompt_all_agents_produce_consistent_output(self):
        ctx = _make_task_context(
            title="Universal",
            description="Universal desc",
            forbidden=["Bad approach"],
            hints=["Good hint"],
        )
        for agent in _all_agents_openrouter():
            prompt = agent._build_openrouter_prompt(ctx)
            assert "Universal" in prompt, f"{agent.agent_id} missing title"
            assert "Universal desc" in prompt, f"{agent.agent_id} missing description"
            assert "Bad approach" in prompt, f"{agent.agent_id} missing forbidden"
            assert "Good hint" in prompt, f"{agent.agent_id} missing hint"


# ===========================================================================
# 5. execute_openrouter() — full method tests
# ===========================================================================

class TestExecuteOpenRouter:
    """Test the base-class execute_openrouter() method with patched httpx."""

    # -- Precondition failures (no API call needed) --

    async def test_no_model_returns_failure(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model=None)
        ctx = _make_task_context()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            result = await agent.execute_openrouter(ctx)
        assert result.failure_reason == "no_model"
        assert "model" in (result.failure_detail or "").lower()
        assert result.tests_passed is False

    async def test_no_api_key_returns_failure(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="openai/gpt-4o")
        ctx = _make_task_context()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            result = await agent.execute_openrouter(ctx)
        assert result.failure_reason == "no_api_key"
        assert "OPENROUTER_API_KEY" in (result.failure_detail or "")

    # -- Successful API call --

    async def test_successful_execution(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="google/gemini-2.5-pro")
        ctx = _make_task_context(title="Analyze deps", description="Find circular imports")

        response_data = _make_openrouter_response_json(
            content="Found 3 circular imports in modules A, B, C.",
            model="google/gemini-2.5-pro",
            prompt_tokens=200,
            completion_tokens=80,
        )
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-real-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.tests_passed is True
        assert result.agent_id == "gemini"
        assert result.model_used == "google/gemini-2.5-pro"
        assert result.tokens_used == 280
        assert "circular imports" in (result.approach_summary or "")
        assert "circular imports" in (result.raw_output or "")
        assert result.failure_reason is None
        assert result.duration_seconds > 0

    async def test_successful_execution_all_agents(self):
        """All 4 agents produce valid TaskOutcome from a successful OpenRouter call."""
        response_data = _make_openrouter_response_json(
            content="Task completed successfully.",
            model="openai/gpt-4o",
            prompt_tokens=50,
            completion_tokens=25,
        )
        fake_response = _make_httpx_response(200, response_data)

        ctx = _make_task_context()

        for agent in _all_agents_openrouter():
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
                with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                    result = await agent.execute_openrouter(ctx)

            assert result.tests_passed is True, f"{agent.agent_id} should pass"
            assert result.agent_id == agent.agent_id, f"{agent.agent_id} wrong agent_id"
            assert result.tokens_used == 75, f"{agent.agent_id} wrong token count"
            assert result.failure_reason is None, f"{agent.agent_id} should have no failure"

    # -- Empty choices --

    async def test_empty_choices_returns_empty_content(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="xai/grok-3")
        ctx = _make_task_context()

        response_data = {
            "id": "gen-xyz",
            "model": "xai/grok-3",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.tests_passed is True
        assert result.approach_summary == ""
        assert result.raw_output == ""
        assert result.tokens_used == 10

    # -- HTTP errors --

    async def test_http_401_unauthorized(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        error_body = {"error": {"message": "Invalid API key", "code": 401}}
        fake_response = _make_httpx_response(401, error_body)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-bad"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "http_401"
        assert "Invalid API key" in (result.failure_detail or "")
        assert result.tests_passed is False

    async def test_http_429_rate_limit(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        error_body = {"error": {"message": "Rate limit exceeded"}}
        fake_response = _make_httpx_response(429, error_body)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "http_429"
        assert "Rate limit" in (result.failure_detail or "")

    async def test_http_500_server_error(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        error_body = {"error": {"message": "Internal server error"}}
        fake_response = _make_httpx_response(500, error_body)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "http_500"
        assert result.tests_passed is False

    async def test_http_error_without_json_body(self):
        """HTTP error where the response body is not valid JSON."""
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        # Build a response with non-JSON body
        fake_response = httpx.Response(
            status_code=502,
            content=b"Bad Gateway",
            headers={"content-type": "text/plain"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "http_502"
        assert result.tests_passed is False
        assert result.duration_seconds >= 0

    # -- Network/connection errors --

    async def test_connection_error(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("Connection refused"),
            ):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "ConnectError"
        assert "Connection refused" in (result.failure_detail or "")
        assert result.tests_passed is False

    async def test_timeout_error(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout("Read timed out"),
            ):
                result = await agent.execute_openrouter(ctx)

        assert result.failure_reason == "ReadTimeout"
        assert result.tests_passed is False

    # -- Response parsing edge cases --

    async def test_response_with_no_usage_field(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        response_data = {
            "id": "gen-no-usage",
            "model": "test/model",
            "choices": [{"message": {"content": "Answer"}, "index": 0}],
        }
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.tests_passed is True
        assert result.tokens_used == 0
        assert result.raw_output == "Answer"

    async def test_response_model_field_overrides_agent_model(self):
        """The model in the response should be used, not the agent's config model."""
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="requested/model")
        ctx = _make_task_context()

        response_data = _make_openrouter_response_json(
            content="Done",
            model="actual/model-served",
        )
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.model_used == "actual/model-served"

    async def test_long_content_truncated_in_approach_summary(self):
        """approach_summary should be truncated to 500 chars."""
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        long_content = "A" * 1000
        response_data = _make_openrouter_response_json(content=long_content)
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert len(result.approach_summary) == 500
        assert len(result.raw_output) == 1000

    async def test_duration_tracked(self):
        """duration_seconds should be positive for a successful call."""
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        response_data = _make_openrouter_response_json()
        fake_response = _make_httpx_response(200, response_data)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_response):
                result = await agent.execute_openrouter(ctx)

        assert result.duration_seconds >= 0

    async def test_duration_tracked_on_error(self):
        """duration_seconds should be set even when an error occurs."""
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="test/model")
        ctx = _make_task_context()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-key"}):
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("fail"),
            ):
                result = await agent.execute_openrouter(ctx)

        assert result.duration_seconds >= 0


# ===========================================================================
# 6. Instantiation with OPENROUTER mode
# ===========================================================================

class TestOpenRouterInstantiation:
    """Verify agents can be instantiated with OPENROUTER mode."""

    def test_claude_openrouter_mode(self):
        agent = ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model="anthropic/claude-3.5-sonnet")
        assert agent.mode == AgentMode.OPENROUTER
        assert agent.model == "anthropic/claude-3.5-sonnet"
        assert agent.agent_id == "claude"

    def test_codex_openrouter_mode(self):
        agent = CodexAgent(mode=AgentMode.OPENROUTER, model="openai/gpt-4o")
        assert agent.mode == AgentMode.OPENROUTER
        assert agent.model == "openai/gpt-4o"
        assert agent.agent_id == "codex"

    def test_gemini_openrouter_mode(self):
        agent = GeminiAgent(mode=AgentMode.OPENROUTER, model="google/gemini-2.5-pro")
        assert agent.mode == AgentMode.OPENROUTER
        assert agent.model == "google/gemini-2.5-pro"
        assert agent.agent_id == "gemini"

    def test_grok_openrouter_mode(self):
        agent = GrokAgent(mode=AgentMode.OPENROUTER, model="xai/grok-3")
        assert agent.mode == AgentMode.OPENROUTER
        assert agent.model == "xai/grok-3"
        assert agent.agent_id == "grok"


# ===========================================================================
# 7. Cross-agent consistency for OpenRouter
# ===========================================================================

class TestOpenRouterCrossAgent:
    """Verify uniform OpenRouter behavior across all four agents."""

    async def test_all_agents_health_check_structure(self):
        """All agents produce AgentHealth with correct fields in OPENROUTER mode."""
        for agent in _all_agents_openrouter("test/model"):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
                health = await agent.health_check()
            assert isinstance(health, AgentHealth)
            assert health.available is True
            assert health.mode == AgentMode.OPENROUTER
            assert health.agent_id == agent.agent_id
            assert health.version == f"openrouter:test/model"

    async def test_all_agents_no_key_produce_same_error_shape(self):
        for agent in _all_agents_openrouter("test/model"):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENROUTER_API_KEY", None)
                health = await agent.health_check()
            assert health.available is False
            assert "OPENROUTER_API_KEY" in (health.error or "")

    async def test_all_agents_no_model_produce_same_error_shape(self):
        agents = [
            ClaudeCodeAgent(mode=AgentMode.OPENROUTER, model=None),
            CodexAgent(mode=AgentMode.OPENROUTER, model=None),
            GeminiAgent(mode=AgentMode.OPENROUTER, model=None),
            GrokAgent(mode=AgentMode.OPENROUTER, model=None),
        ]
        for agent in agents:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
                health = await agent.health_check()
            assert health.available is False
            assert "model" in (health.error or "").lower(), f"{agent.agent_id} error missing 'model'"

    def test_all_agents_build_openrouter_prompt_same_output(self):
        """_build_openrouter_prompt is inherited from base class, should be identical."""
        ctx = _make_task_context(
            title="Same Task",
            description="Same description",
            forbidden=["A"],
            hints=["B"],
        )
        prompts = []
        for agent in _all_agents_openrouter():
            prompts.append(agent._build_openrouter_prompt(ctx))

        # All should produce the same prompt (shared base-class method)
        assert len(set(prompts)) == 1, "All agents should produce identical OpenRouter prompts"
