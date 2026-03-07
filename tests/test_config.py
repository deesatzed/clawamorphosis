"""Tests for CLAW configuration system."""

from pathlib import Path

from claw.core.config import (
    AgentConfig,
    ClawConfig,
    DatabaseConfig,
    FleetConfig,
    RoutingConfig,
    load_config,
)


class TestLoadConfig:
    def test_loads_claw_toml(self):
        config = load_config()
        assert config.database.db_path == "data/claw.db"
        assert config.llm.base_url == "https://openrouter.ai/api/v1"

    def test_agents_section(self):
        config = load_config()
        assert "claude" in config.agents
        assert config.agents["claude"].enabled is True
        assert config.agents["claude"].mode == "openrouter"

    def test_routing_config(self):
        config = load_config()
        assert config.routing.exploration_rate == 0.10
        assert "analysis" in config.routing.static_priors
        assert config.routing.static_priors["analysis"] == "claude"

    def test_fleet_config(self):
        config = load_config()
        assert config.fleet.max_concurrent_repos == 4
        assert config.fleet.enhancement_branch_prefix == "claw/enhancement"

    def test_evolution_config(self):
        config = load_config()
        assert config.evolution.ab_test_sample_size == 20

    def test_security_config(self):
        config = load_config()
        assert "ANTHROPIC_API_KEY" in config.security.safe_env_vars
        assert "/System" in config.security.forbidden_paths


class TestDefaults:
    def test_database_default(self):
        dc = DatabaseConfig()
        assert dc.db_path == "data/claw.db"

    def test_routing_default(self):
        rc = RoutingConfig()
        assert rc.exploration_rate == 0.10

    def test_agent_config_default(self):
        ac = AgentConfig()
        assert ac.enabled is False
        assert ac.mode == "cli"
