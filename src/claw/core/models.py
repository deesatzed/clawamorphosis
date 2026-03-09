"""All Pydantic data models for CLAW.

Defines the data contracts used across all agents, database operations,
and orchestration. Every table row and inter-agent message has a model here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    """Generate a new string UUID for SQLite TEXT PRIMARY KEY."""
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    EVALUATING = "EVALUATING"
    PLANNING = "PLANNING"
    DISPATCHED = "DISPATCHED"
    CODING = "CODING"
    REVIEWING = "REVIEWING"
    STUCK = "STUCK"
    DONE = "DONE"


class HypothesisOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class LifecycleState(str, enum.Enum):
    EMBRYONIC = "embryonic"
    VIABLE = "viable"
    THRIVING = "thriving"
    DECLINING = "declining"
    DORMANT = "dormant"
    DEAD = "dead"


class MethodologyType(str, enum.Enum):
    BUG_FIX = "BUG_FIX"
    PATTERN = "PATTERN"
    DECISION = "DECISION"
    GOTCHA = "GOTCHA"


class ComplexityTier(str, enum.Enum):
    TRIVIAL = "TRIVIAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class AgentMode(str, enum.Enum):
    """How an agent is invoked."""
    CLI = "cli"
    API = "api"
    CLOUD = "cloud"
    OPENROUTER = "openrouter"


class OperationalMode(str, enum.Enum):
    """CLAW operational modes."""
    ATTENDED = "attended"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


# ---------------------------------------------------------------------------
# Database row models
# ---------------------------------------------------------------------------

class Project(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    repo_path: str
    tech_stack: dict[str, Any] = Field(default_factory=dict)
    project_rules: Optional[str] = None
    banned_dependencies: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Task(BaseModel):
    id: str = Field(default_factory=_new_id)
    project_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    task_type: Optional[str] = None
    recommended_agent: Optional[str] = None
    assigned_agent: Optional[str] = None
    context_snapshot_id: Optional[str] = None
    attempt_count: int = 0
    escalation_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None


class HypothesisEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    task_id: str
    attempt_number: int
    approach_summary: str
    outcome: HypothesisOutcome = HypothesisOutcome.FAILURE
    error_signature: Optional[str] = None
    error_full: Optional[str] = None
    files_changed: list[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    model_used: Optional[str] = None
    agent_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class CapabilityIO(BaseModel):
    """Single input or output port of a capability."""
    name: str
    type: str  # "text", "code_patch", "metrics_data", "event_list", etc.
    required: bool = True
    description: str = ""


class ComposabilityInterface(BaseModel):
    """Describes how a capability can chain with others."""
    can_chain_after: list[str] = Field(default_factory=list)  # domain tags
    can_chain_before: list[str] = Field(default_factory=list)
    standalone: bool = True


class CapabilityData(BaseModel):
    """Structured capability metadata stored as JSON in methodologies.capability_data."""
    inputs: list[CapabilityIO] = Field(default_factory=list)
    outputs: list[CapabilityIO] = Field(default_factory=list)
    domain: list[str] = Field(default_factory=list)
    composability: ComposabilityInterface = Field(default_factory=ComposabilityInterface)
    capability_type: str = "transformation"  # transformation, analysis, generation, validation


class SynergyEdgeType(str, enum.Enum):
    DEPENDS_ON = "depends_on"
    ENHANCES = "enhances"
    COMPETES_WITH = "competes_with"
    FEEDS_INTO = "feeds_into"
    SYNERGY = "synergy"
    CO_RETRIEVAL = "co_retrieval"


class SynergyExploration(BaseModel):
    """A record of an explored capability pair."""
    id: str = Field(default_factory=_new_id)
    cap_a_id: str
    cap_b_id: str
    explored_at: datetime = Field(default_factory=_now)
    result: str = "pending"  # pending, synergy, no_match, error, stale
    synergy_score: Optional[float] = None
    synergy_type: Optional[str] = None
    edge_id: Optional[str] = None
    exploration_method: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class Methodology(BaseModel):
    id: str = Field(default_factory=_new_id)
    problem_description: str
    problem_embedding: Optional[list[float]] = None
    solution_code: str
    methodology_notes: Optional[str] = None
    source_task_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    language: Optional[str] = None
    scope: str = "project"
    methodology_type: Optional[str] = None
    files_affected: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    lifecycle_state: str = "viable"
    retrieval_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_retrieved_at: Optional[datetime] = None
    generation: int = 0
    fitness_vector: dict[str, float] = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    prism_data: Optional[dict] = None
    capability_data: Optional[dict] = None
    novelty_score: Optional[float] = None
    potential_score: Optional[float] = None


class PeerReview(BaseModel):
    id: str = Field(default_factory=_new_id)
    task_id: str
    model_used: str
    diagnosis: str
    recommended_approach: Optional[str] = None
    reasoning: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class ContextSnapshot(BaseModel):
    id: str = Field(default_factory=_new_id)
    task_id: str
    attempt_number: int
    git_ref: str
    file_manifest: Optional[dict[str, str]] = None
    created_at: datetime = Field(default_factory=_now)


class MethodologyLink(BaseModel):
    id: str = Field(default_factory=_new_id)
    source_id: str
    target_id: str
    link_type: str = "co_retrieval"
    strength: float = 1.0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Token cost tracking
# ---------------------------------------------------------------------------

class TokenCostRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_role: str = ""
    agent_id: Optional[str] = None
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Inter-agent message models (CLAW-specific)
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Standardized output from any agent."""
    agent_name: str
    status: str  # "success", "failure", "blocked"
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class TaskOutcome(BaseModel):
    """Output of any agent executing a task (replaces BuildResult)."""
    files_changed: list[str] = Field(default_factory=list)
    test_output: str = ""
    tests_passed: bool = False
    diff: str = ""
    approach_summary: str = ""
    model_used: Optional[str] = None
    agent_id: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None
    self_audit: str = ""
    raw_output: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class VerificationResult(BaseModel):
    """Output of Verifier — audit gate decision (replaces SentinelVerdict)."""
    approved: bool = False
    violations: list[dict[str, str]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    quality_score: Optional[float] = None
    tests_before: Optional[int] = None
    tests_after: Optional[int] = None


class EscalationDiagnosis(BaseModel):
    """Output of escalation — peer review strategy (replaces CouncilDiagnosis)."""
    strategy_shift: str
    new_approach: str
    reasoning: str
    model_used: str


class AgentHealth(BaseModel):
    """Health check result for an agent."""
    agent_id: str
    available: bool = False
    mode: Optional[AgentMode] = None
    version: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class CycleResult(BaseModel):
    """Result of one claw cycle iteration."""
    cycle_level: str  # "micro", "meso", "macro", "nano"
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    outcome: TaskOutcome = Field(default_factory=TaskOutcome)
    verification: Optional[VerificationResult] = None
    success: bool = False
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class FleetTask(BaseModel):
    """A repo in the fleet queue."""
    id: str = Field(default_factory=_new_id)
    repo_path: str
    repo_name: str
    priority: float = 0.0
    status: str = "pending"
    enhancement_branch: Optional[str] = None
    last_evaluated: Optional[datetime] = None
    evaluation_score: Optional[float] = None
    budget_allocated_usd: float = 0.0
    budget_used_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Context models (pipeline flow)
# ---------------------------------------------------------------------------

class TaskContext(BaseModel):
    """Enriched task context for the pipeline."""
    task: Task
    forbidden_approaches: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    checkpoint_ref: Optional[str] = None
    previous_escalation_diagnosis: Optional[str] = None


class ContextBrief(BaseModel):
    """Full context assembled for agent execution."""
    task: Task
    past_solutions: list[Methodology] = Field(default_factory=list)
    forbidden_approaches: list[str] = Field(default_factory=list)
    project_rules: Optional[str] = None
    escalation_diagnosis: Optional[str] = None
    retrieval_confidence: float = 0.0
    retrieval_conflicts: list[str] = Field(default_factory=list)
    retrieval_strategy_hint: Optional[str] = None
    complexity_tier: Optional[str] = None
    sentinel_feedback: list[dict[str, str]] = Field(default_factory=list)
    retrieved_methodology_ids: list[str] = Field(default_factory=list)


class ExecutionState(BaseModel):
    """Shared typed execution state passed through the agent pipeline."""
    task_id: str
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    current_phase: str = "init"
    attempt_number: int = 0
    token_budget_remaining: int = 100_000
    tokens_used: int = 0
    complexity_tier: Optional[ComplexityTier] = None
    quality_score: Optional[float] = None
    agent_id: Optional[str] = None
