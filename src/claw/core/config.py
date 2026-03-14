"""Configuration loader for CLAW.

Loads config from claw.toml (TOML format), with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import toml
from pydantic import BaseModel, Field

from claw.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    db_path: str = "data/claw.db"


class LLMConfig(BaseModel):
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    default_temperature: float = 0.3
    default_max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_cap: float = 60.0
    fallback_models: list[str] = Field(default_factory=list)
    model_failure_threshold: int = 2
    model_cooldown_seconds: int = 90


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    api_key_env: str = "GOOGLE_API_KEY"
    task_type: Optional[str] = "RETRIEVAL_DOCUMENT"
    required_model: Optional[str] = None


class MemoryConfig(BaseModel):
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    vector_weight: float = 0.6
    text_weight: float = 0.4


class OrchestratorConfig(BaseModel):
    max_retries: int = 5
    council_trigger: int = 2
    max_council: int = 3
    max_tokens_per_task: int = 100_000
    exploration_rate: float = 0.10
    loop_guard_max_repeats: int = 2
    pipeline_adaptation_enabled: bool = True


class SentinelConfig(BaseModel):
    llm_deep_check: bool = True
    drift_threshold: float = 0.40
    quality_score_threshold: float = 0.60


class SecurityConfig(BaseModel):
    autonomy_level: str = "SUPERVISED"
    rate_limit_per_hour: int = 200
    allowed_commands: list[str] = Field(
        default_factory=lambda: [
            "git", "pytest", "python3", "pip", "npm", "npx",
            "cargo", "rustc", "go", "make", "ls", "find", "grep",
            "cat", "head", "tail", "wc", "diff", "ruff", "mypy",
        ]
    )
    forbidden_paths: list[str] = Field(
        default_factory=lambda: [
            "/etc", "/root", "/var", "/tmp",
            "/System", "/Library", "/Applications",
            "/usr/bin", "/usr/sbin",
        ]
    )
    safe_env_vars: list[str] = Field(
        default_factory=lambda: [
            "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL",
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "GOOGLE_API_KEY", "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
    )


class TokenTrackingConfig(BaseModel):
    enabled: bool = True
    jsonl_path: str = "data/token_costs.jsonl"
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015


class AgentConfig(BaseModel):
    """Per-agent configuration."""
    enabled: bool = False
    mode: str = "cli"  # cli, api, cloud
    api_key_env: str = ""
    max_concurrent: int = 2
    timeout: int = 300
    model: Optional[str] = None  # User-set; never hardcoded
    max_budget_usd: float = 1.0


class RoutingConfig(BaseModel):
    exploration_rate: float = 0.10
    score_decay_factor: float = 0.95
    min_samples_for_routing: int = 5
    static_priors: dict[str, str] = Field(default_factory=lambda: {
        "analysis": "claude",
        "documentation": "claude",
        "refactoring": "codex",
        "bulk_tests": "codex",
        "dependency_analysis": "gemini",
        "full_repo_comprehension": "gemini",
        "quick_fixes": "grok",
        "web_lookup": "grok",
    })


class EvolutionConfig(BaseModel):
    ab_test_sample_size: int = 20
    mutation_rate: float = 0.1
    promotion_threshold: float = 0.6


class GovernanceConfig(BaseModel):
    """Memory governance configuration."""
    max_methodologies: int = 2000
    quota_warning_pct: float = 0.80
    gc_dead_on_sweep: bool = True
    dedup_similarity_threshold: float = 0.88
    dedup_enabled: bool = True
    episodic_retention_days: int = 90
    sweep_interval_cycles: int = 10
    sweep_on_startup: bool = True
    self_consume_enabled: bool = True
    self_consume_min_tasks: int = 10
    self_consume_max_generation: int = 3
    self_consume_lookback: int = 20
    max_db_size_mb: int = 500
    mining_min_description_length: int = 20


class AssimilationConfig(BaseModel):
    """Capability assimilation configuration."""
    enabled: bool = True
    synergy_candidate_limit: int = 20
    synergy_score_threshold: float = 0.6
    auto_compose_threshold: float = 0.8
    max_compositions_per_cycle: int = 3
    io_compatibility_weight: float = 0.3
    domain_overlap_weight: float = 0.2
    embedding_similarity_weight: float = 0.3
    llm_analysis_weight: float = 0.2
    # Novelty scoring
    novelty_enabled: bool = True
    novelty_nearest_neighbor_k: int = 5
    novelty_nn_weight: float = 0.35
    novelty_domain_uniqueness_weight: float = 0.25
    novelty_type_rarity_weight: float = 0.15
    novelty_centroid_distance_weight: float = 0.25
    # Potential scoring
    potential_io_generality_weight: float = 0.30
    potential_composability_weight: float = 0.25
    potential_domain_breadth_weight: float = 0.20
    potential_standalone_weight: float = 0.10
    potential_llm_weight: float = 0.15
    potential_llm_threshold: float = 0.4
    # Lifecycle + retrieval
    novelty_lifecycle_protection_days: int = 90
    novelty_protection_threshold: float = 0.7
    novelty_retrieval_boost: float = 0.15
    potential_retrieval_boost: float = 0.10


class FleetConfig(BaseModel):
    max_concurrent_repos: int = 4
    enhancement_branch_prefix: str = "claw/enhancement"
    max_cost_per_repo_usd: float = 5.0
    max_cost_per_day_usd: float = 50.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ClawConfig(BaseModel):
    """Top-level CLAW configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    sentinel: SentinelConfig = Field(default_factory=SentinelConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    token_tracking: TokenTrackingConfig = Field(default_factory=TokenTrackingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    fleet: FleetConfig = Field(default_factory=FleetConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    assimilation: AssimilationConfig = Field(default_factory=AssimilationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

class PromptLoader:
    """Loads prompt templates from prompts/ directory."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"
        self.prompts_dir = prompts_dir

    def load(self, name: str, default: str = "") -> str:
        path = self.prompts_dir / name
        if path.exists():
            return path.read_text().strip()
        return default


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[Path] = None) -> ClawConfig:
    """Load CLAW config from TOML file.

    Args:
        config_path: Path to claw.toml. Defaults to ./claw.toml relative to project root.

    Returns:
        Validated ClawConfig instance.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "claw.toml"

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = toml.load(f)

    # Convert agents section: TOML nested tables → dict[str, AgentConfig]
    agents_raw = raw.pop("agents", {})
    agents = {}
    for agent_name, agent_data in agents_raw.items():
        if isinstance(agent_data, dict):
            agents[agent_name] = AgentConfig(**agent_data)

    # Environment variable overrides
    db_path_env = os.getenv("CLAW_DB_PATH")
    if db_path_env:
        raw.setdefault("database", {})["db_path"] = db_path_env

    config = ClawConfig(**raw, agents=agents)
    return config
