"""Component factory and dependency injection for CLAW.

ClawFactory.create() builds the full dependency graph and returns
a ClawContext dataclass with all wired components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from claw.core.config import ClawConfig, load_config
from claw.core.models import AgentMode
from claw.db.engine import DatabaseEngine
from claw.db.embeddings import EmbeddingEngine
from claw.db.repository import Repository
from claw.llm.client import LLMClient
from claw.llm.token_tracker import TokenTracker
from claw.security.policy import AutonomyLevel, SecurityPolicy
from claw.agents.interface import AgentInterface

logger = logging.getLogger("claw.factory")


@dataclass
class ClawContext:
    """All wired components for a CLAW session."""
    config: ClawConfig
    engine: DatabaseEngine
    repository: Repository
    embeddings: EmbeddingEngine
    llm_client: LLMClient
    token_tracker: TokenTracker
    security: SecurityPolicy
    agents: dict[str, AgentInterface] = field(default_factory=dict)
    dispatcher: Any = None
    verifier: Any = None
    budget_enforcer: Any = None
    degradation_manager: Any = None
    health_monitor: Any = None
    error_kb: Any = None
    semantic_memory: Any = None
    prompt_evolver: Any = None
    pattern_learner: Any = None
    miner: Any = None

    async def close(self) -> None:
        """Cleanly shut down all components."""
        await self.llm_client.close()
        await self.engine.close()
        logger.info("ClawContext closed")


class ClawFactory:
    """Builds the complete CLAW dependency graph."""

    @staticmethod
    async def create(
        config_path: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
    ) -> ClawContext:
        """Create a fully wired ClawContext.

        Args:
            config_path: Path to claw.toml. Defaults to ./claw.toml.
            workspace_dir: Working directory for agent operations.
        """
        config = load_config(config_path)

        # Database
        engine = DatabaseEngine(config.database)
        await engine.connect()
        await engine.initialize_schema()
        repository = Repository(engine)

        # Embeddings
        embeddings = EmbeddingEngine(config.embeddings)

        # LLM client
        llm_client = LLMClient(config.llm)

        # Token tracker
        token_tracker = TokenTracker(
            repository=repository,
            jsonl_path=config.token_tracking.jsonl_path if config.token_tracking.enabled else None,
            cost_per_1k_input=config.token_tracking.cost_per_1k_input,
            cost_per_1k_output=config.token_tracking.cost_per_1k_output,
        )

        # Security
        ws = workspace_dir or Path(".").resolve()
        autonomy = AutonomyLevel.SUPERVISED
        sec_cfg = config.security
        if sec_cfg.autonomy_level.upper() == "FULL":
            autonomy = AutonomyLevel.FULL
        elif sec_cfg.autonomy_level.upper() == "READ_ONLY":
            autonomy = AutonomyLevel.READ_ONLY

        security = SecurityPolicy(
            autonomy=autonomy,
            workspace_dir=ws,
            allowed_commands=sec_cfg.allowed_commands,
            forbidden_paths=sec_cfg.forbidden_paths,
            max_actions_per_hour=sec_cfg.rate_limit_per_hour,
            safe_env_vars=sec_cfg.safe_env_vars,
        )

        # Agents
        agents: dict[str, AgentInterface] = {}
        for agent_name, agent_cfg in config.agents.items():
            if not agent_cfg.enabled:
                continue
            agent = _create_agent(agent_name, agent_cfg, workspace_dir=str(ws))
            if agent:
                agents[agent_name] = agent

        # Dispatcher
        from claw.dispatcher import Dispatcher
        dispatcher = Dispatcher(
            agents=agents,
            exploration_rate=config.orchestrator.exploration_rate,
            repository=repository,
        )

        # Verifier
        from claw.verifier import Verifier
        verifier = Verifier(
            embedding_engine=embeddings,
            banned_dependencies=getattr(config.sentinel, "banned_dependencies", []) if hasattr(config, "sentinel") else [],
            drift_threshold=getattr(config.sentinel, "drift_threshold", 0.40) if hasattr(config, "sentinel") else 0.40,
            llm_client=llm_client,
        )

        # Health Monitor
        from claw.orchestrator.health_monitor import HealthMonitor
        health_monitor = HealthMonitor(
            repository=repository,
            config=config.orchestrator,
        )

        # Budget Enforcer
        from claw.budget import BudgetEnforcer
        budget_enforcer = BudgetEnforcer(
            repository=repository,
            config=config,
        )

        # Degradation Manager
        from claw.degradation import DegradationManager
        degradation_manager = DegradationManager(
            health_monitor=health_monitor,
            dispatcher=dispatcher,
            all_agent_ids=list(agents.keys()) if agents else None,
        )

        # Error KB
        from claw.memory.error_kb import ErrorKB
        error_kb = ErrorKB(repository=repository)

        # Semantic Memory
        from claw.memory.semantic import SemanticMemory
        from claw.memory.hybrid_search import HybridSearch
        hybrid_search = HybridSearch(
            repository=repository,
            embedding_engine=embeddings,
        )
        semantic_memory = SemanticMemory(
            repository=repository,
            embedding_engine=embeddings,
            hybrid_search=hybrid_search,
        )

        # Prompt Evolver
        from claw.evolution.prompt_evolver import PromptEvolver
        prompt_evolver = PromptEvolver(
            repository=repository,
            semantic_memory=semantic_memory,
            error_kb=error_kb,
        )

        # Pattern Learner
        from claw.evolution.pattern_learner import PatternLearner
        pattern_learner = PatternLearner(
            repository=repository,
            semantic_memory=semantic_memory,
        )

        # Repo Miner
        from claw.miner import RepoMiner
        miner = RepoMiner(
            repository=repository,
            llm_client=llm_client,
            semantic_memory=semantic_memory,
            config=config,
        )

        ctx = ClawContext(
            config=config,
            engine=engine,
            repository=repository,
            embeddings=embeddings,
            llm_client=llm_client,
            token_tracker=token_tracker,
            security=security,
            agents=agents,
            dispatcher=dispatcher,
            verifier=verifier,
            budget_enforcer=budget_enforcer,
            degradation_manager=degradation_manager,
            health_monitor=health_monitor,
            error_kb=error_kb,
            semantic_memory=semantic_memory,
            prompt_evolver=prompt_evolver,
            pattern_learner=pattern_learner,
            miner=miner,
        )

        agent_names = list(agents.keys()) if agents else ["none"]
        logger.info(
            "ClawContext created: db=%s, agents=[%s], evolution=[error_kb, semantic_memory, prompt_evolver, pattern_learner]",
            config.database.db_path,
            ", ".join(agent_names),
        )
        return ctx


def _create_agent(
    name: str,
    agent_cfg: Any,
    workspace_dir: Optional[str] = None,
) -> Optional[AgentInterface]:
    """Create a single agent by name."""
    import os

    mode = AgentMode(agent_cfg.mode)
    api_key = os.getenv(agent_cfg.api_key_env, "") if agent_cfg.api_key_env else ""

    if name == "claude":
        from claw.agents.claude import ClaudeCodeAgent
        return ClaudeCodeAgent(
            mode=mode,
            api_key=api_key,
            model=agent_cfg.model,
            timeout=agent_cfg.timeout,
            max_budget_usd=agent_cfg.max_budget_usd,
            workspace_dir=workspace_dir,
        )

    if name == "codex":
        from claw.agents.codex import CodexAgent
        return CodexAgent(
            mode=mode,
            api_key=api_key,
            model=agent_cfg.model,
            timeout=agent_cfg.timeout,
            max_tokens=getattr(agent_cfg, "max_tokens", 4096),
            workspace_dir=workspace_dir,
        )

    if name == "gemini":
        from claw.agents.gemini import GeminiAgent
        return GeminiAgent(
            mode=mode,
            api_key=api_key,
            model=agent_cfg.model,
            timeout=agent_cfg.timeout,
            workspace_dir=workspace_dir,
        )

    if name == "grok":
        from claw.agents.grok import GrokAgent
        return GrokAgent(
            mode=mode,
            api_key=api_key,
            model=agent_cfg.model,
            timeout=agent_cfg.timeout,
            max_budget_usd=agent_cfg.max_budget_usd,
            workspace_dir=workspace_dir,
        )

    logger.warning("Unknown agent name: '%s'", name)
    return None
