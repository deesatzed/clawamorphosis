# CLAW — Codebase Learning & Autonomous Workforce

## Multi-Model Autonomous Codebase Enhancement System — Architecture Blueprint v1.0

**Date:** 2026-03-03  
**Author:** Wayne (Architect) + Claude Opus 4.6 (Technical Scribe)  
**Status:** Blueprint — Pre-Implementation  
**Target Runtime:** Mac Mini M4, 64GB RAM (local coordinator) + cloud agent APIs

---

## 1. Thesis

### 1a. The Problem

You have 390+ repositories. Each was built in a different session, by a different agent (or human), at a different skill level, with different conventions. Many are 60–80% complete. The effort to bring each one to SOTA quality — evaluated, tested, documented, refactored, hardened — is enormous when done serially by one model in one session.

### 1b. The Insight

No single AI coding agent is best at everything. Claude Code reasons carefully but is methodical. Codex runs parallel cloud tasks but can hallucinate at the edges. Gemini 3.1 Pro has a 1M-token context window and 77.1% ARC-AGI-2 but weaker structured output. Grok 4.20 is fast with 4-agent internal collaboration and real-time web access but is a beta still stabilizing weekly.

**The correct architecture uses all of them for what each does best, coordinates them through a shared memory and evaluation framework, and learns from every engagement to improve the next one.**

### 1c. The Product

CLAW is an **autonomous multi-model system** that:

1. **Ingests** any existing codebase
2. **Evaluates** it using a battery of analysis prompts (your existing 17-command arsenal)
3. **Plans** remediation using gap analysis and prioritization
4. **Executes** improvements by dispatching work to the best-fit AI agent
5. **Verifies** every change against measurable criteria
6. **Learns** from outcomes to improve future evaluations and routing decisions
7. **Persists** all knowledge across sessions, projects, and agent generations

It is not a chatbot. It is not an IDE plugin. It is a **workforce** that operates on your codebase fleet while you sleep.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLAW COORDINATOR                               │
│                      (Python 3.12, asyncio, Mac Mini M4)                 │
│                                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │  INGESTOR │→ │ EVALUATOR │→ │  PLANNER  │→ │   DISPATCHER      │    │
│  │           │  │           │  │           │  │                   │    │
│  │ Git clone │  │ Run prompt│  │ Gap→Plan  │  │ Route to best     │    │
│  │ Scan tree │  │ battery   │  │ Prioritize│  │ agent per task    │    │
│  │ Profile   │  │ Synthesize│  │ Sequence  │  │ Monitor progress  │    │
│  └───────────┘  └───────────┘  └───────────┘  └─────┬─────────────┘    │
│                                                       │                  │
│  ┌────────────────────────────────────────────────────┼────────────┐    │
│  │                     AGENT POOL                     │            │    │
│  │                                                    ▼            │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │    │
│  │  │ CLAUDE     │ │ CODEX      │ │ GEMINI     │ │ GROK       │  │    │
│  │  │ CODE       │ │ (Cloud+CLI)│ │ 3.1 Pro    │ │ 4.20       │  │    │
│  │  │            │ │            │ │ (CLI)      │ │ (CLI+API)  │  │    │
│  │  │ Analysis   │ │ Parallel   │ │ Full-repo  │ │ Fast iter  │  │    │
│  │  │ Docs, Arch │ │ refactor,  │ │ 1M context │ │ Web search │  │    │
│  │  │ Security   │ │ bulk test  │ │ analysis   │ │ 4-agent    │  │    │
│  │  │ Review     │ │ CI/CD      │ │ Deep Think │ │ collab     │  │    │
│  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘  │    │
│  │        └───────────────┴──────────────┴──────────────┘         │    │
│  └────────────────────────────────┬───────────────────────────────┘    │
│                                    │                                    │
│  ┌─────────────────────────────────▼──────────────────────────────┐    │
│  │                       VERIFIER                                  │    │
│  │                                                                 │    │
│  │  claim-gate → tests → regression-scan → outcome-audit           │    │
│  │                                                                 │    │
│  │  PASS → commit to enhancement branch + update memory            │    │
│  │  FAIL → route back to Dispatcher with failure context           │    │
│  └─────────────────────────────────┬──────────────────────────────┘    │
│                                    │                                    │
│  ┌─────────────────────────────────▼──────────────────────────────┐    │
│  │                     MEMORY SYSTEM                               │    │
│  │                                                                 │    │
│  │  Working │ Episodic │ Semantic │ Procedural │ Error │ Meta      │    │
│  │                                                                 │    │
│  │  SQLite + sqlite-vec (local) │ Optional S3/R2 (archive)        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    EVOLUTION ENGINE                              │    │
│  │                                                                 │    │
│  │  Prompt mutation │ Agent scoring │ Routing optimization          │    │
│  │  Pattern learning │ Capability discovery │ Self-evaluation       │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The NanoClaw Schema

The "claw" is a **grab-evaluate-decide-act-verify-learn** cycle that operates at four scales simultaneously — from fleet-level triage down to single-function edits. Each level spawns instances of the level below, and learnings propagate upward.

### 3a. Claw Levels

```
MACRO CLAW (Fleet Level)
  │ Scans repo fleet → ranks by enhancement potential
  │ Allocates budget per project
  │ Tracks fleet-wide quality trajectory
  │
  ├── MESO CLAW (Project Level)
  │     │ Runs full evaluation battery on one repo
  │     │ Produces enhancement plan
  │     │ Orchestrates module-level work
  │     │
  │     ├── MICRO CLAW (Module Level)
  │     │     │ Takes one feature/module from the plan
  │     │     │ Selects best agent for the task type
  │     │     │ Dispatches, monitors, verifies
  │     │     │
  │     │     └── NANO CLAW (Self-Improvement Level)
  │     │           │ After each micro task completes:
  │     │           │   Did the agent succeed first try?
  │     │           │   How long / how many tokens?
  │     │           │   Was the output correct?
  │     │           │   Update agent scoring
  │     │           │   If systematic failure → mutate prompt
  │     │           └── Feed learning back to routing table
```

### 3b. The Universal Claw Cycle

Every level executes the same six-step cycle. The inputs and outputs scale, but the logic is identical:

```python
class ClawCycle:
    """
    The fundamental unit of CLAW operation.
    Same cycle at every scale: fleet, project, module, task.
    """
    
    def __init__(self, memory: MemorySystem, agents: AgentPool):
        self.memory = memory
        self.agents = agents
    
    async def execute(self, target, context: dict) -> CycleResult:
        
        # 1. GRAB — acquire target and all relevant context
        workspace = await self.grab(target, context)
        #   Fleet level: git clone repos, read CLAUDE.md/README
        #   Project level: full tree scan, dependency map
        #   Module level: extract relevant files + imports
        #   Task level: single file + test file + context
        
        # 2. EVALUATE — understand current state honestly
        evaluation = await self.evaluate(workspace)
        #   Uses prompt battery (deepdive, agonyofdefeatures, etc.)
        #   Checks memory for prior evaluations of this target
        #   Loads relevant patterns from semantic memory
        
        # 3. DECIDE — plan what to do based on gaps
        plan = await self.decide(evaluation)
        #   Gap analysis → prioritized task list
        #   Each task tagged with: type, complexity, recommended agent
        #   Consults memory for similar past plans and their outcomes
        
        # 4. ACT — dispatch to best-fit agent
        results = await self.act(plan)
        #   Routes each task to optimal agent (learned routing)
        #   Monitors progress, handles timeouts
        #   Retries with fallback agent on failure
        
        # 5. VERIFY — confirm improvement, reject regressions
        verification = await self.verify(results, evaluation)
        #   claim-gate: is the completion claim true?
        #   Tests: do all tests pass?
        #   regression-scan: did we break anything?
        #   outcome-audit: do benchmarks hold?
        
        # 6. LEARN — update all memory systems
        await self.learn(target, evaluation, plan, results, verification)
        #   Episodic: log what happened
        #   Semantic: extract generalizable patterns
        #   Meta: update agent scores + routing table
        #   Procedural: flag prompts that missed issues
        #   Error: catalog any failures
        
        return CycleResult(
            target=target,
            passed=verification.all_passed,
            changes=results.diffs,
            learnings=verification.learnings
        )
```

### 3c. The Key Innovation: Learning Propagates Upward

When a Nano Claw discovers that Grok consistently fails at TypeScript refactoring but excels at Python quick-fixes, that learning doesn't stay at the task level. It propagates:

```
NANO → updates agent_scores table (Grok -0.1 on ts_refactor, +0.1 on py_quickfix)
  ↑
MICRO → next ts_refactor task routes to Claude instead of Grok
  ↑
MESO → project-level plan pre-assigns agents based on learned affinity
  ↑
MACRO → fleet-level scheduling puts TypeScript-heavy repos on Claude-priority queue
```

---

## 4. Agent Specialization Matrix

### 4a. Current Platform Capabilities (March 2026)

| Capability | Claude Code | Codex (GPT-5.3) | Gemini 3.1 Pro | Grok 4.20 Beta |
|-----------|------------|-----------------|---------------|---------------|
| **Context window** | 200K | ~128K | **1M** | 256K (2M agent) |
| **CLI agent** | `claude` | `codex` | `gemini` | `grok` |
| **Parallel cloud tasks** | No | **Yes (worktrees)** | No | No |
| **MCP support** | Yes | Yes | Yes | Yes |
| **Custom instructions** | CLAUDE.md | AGENTS.md + Skills | GEMINI.md | .grok/GROK.md |
| **Web search** | Via tool | Built-in | Google Search grounding | **Native X + web** |
| **Multi-agent internal** | No | Multi-agent (beta) | No | **4-agent collab** |
| **SWE-bench** | ~72.5% | ~74.9% | Improved in 3.1 | **75%** |
| **Reasoning** | **Strongest** | Strong (o3-based) | 77.1% ARC-AGI-2 | Good, improving weekly |
| **Speed** | Moderate | Fast (cloud) | Moderate | **Fastest (Morph 4500 tok/s)** |
| **Cost/1M input** | $3-15 | $2-12 | **Free tier (1K/day)** | $0.20 (code-fast) |
| **Automation** | Via API | **Codex Automations** | Scheduled workflows | Not yet |

### 4b. Initial Routing Table (Learned, Not Hardcoded)

These are **starting priors** — the system updates them after every task based on measured outcomes.

| Task Type | Primary | Why | Fallback | Starting Confidence |
|-----------|---------|-----|----------|-------------------|
| **Deep codebase analysis** | Claude | Best reasoning, safety-aware, source-bound | Gemini (1M context) | 0.90 |
| **Multi-audience documentation** | Claude | Strongest writing, docsRedo experience | Gemini | 0.85 |
| **Architecture review + redesign** | Claude | Best at identifying design issues | Grok (4-agent debate) | 0.85 |
| **Security audit** | Claude | Most safety-conscious, careful reasoning | Codex | 0.85 |
| **Parallel bulk refactoring** | Codex | Cloud sandboxes, parallel worktrees | Claude | 0.80 |
| **Large-scale code migration** | Codex | Multi-agent, parallel execution, GitHub PR | Gemini | 0.75 |
| **Test generation (bulk)** | Codex | Parallel test creation across modules | Claude | 0.80 |
| **CI/CD pipeline generation** | Codex | Codex Automations, GitHub-native | Grok | 0.75 |
| **Full-repo comprehension** | Gemini | 1M tokens = entire repo in context | Claude | 0.85 |
| **Cross-file dependency tracing** | Gemini | Large context enables full-graph reasoning | Claude | 0.80 |
| **Rapid bug fixes (single file)** | Grok | Fastest inference, Morph fast-apply | Codex | 0.80 |
| **Real-time API/library lookup** | Grok | Native web + X search | Gemini (Search grounding) | 0.80 |
| **Performance optimization** | Grok | Fast iteration cycles, benchmark-driven | Codex | 0.70 |
| **Complex multi-step reasoning** | Grok | 4-agent internal collaboration, debate | Claude | 0.75 |
| **UX analysis** | Claude | Best at nuanced human-centered evaluation | Gemini | 0.80 |
| **Dependency updates + compat** | Gemini | Can load entire dep tree in context | Codex | 0.70 |

### 4c. Routing Decision Engine

```python
async def route_task(task: Task, memory: MemorySystem, agents: AgentPool) -> Agent:
    """
    Bayesian agent selection.
    Combines learned performance data with task characteristics.
    Includes mandatory exploration rate for continued learning.
    """
    
    # 1. Query memory for past performance on similar tasks
    history = await memory.meta.query_scores(
        task_type=task.type,
        language=task.language,
        complexity=task.estimated_complexity
    )
    
    # 2. If strong signal from past outcomes, use learned routing
    if history.sample_size > 10 and history.confidence > 0.8:
        # But maintain 10% exploration rate for learning
        if random.random() < 0.10:
            # Pick a non-optimal agent to gather data
            candidates = [a for a in agents.available() if a.id != history.best_agent]
            return random.choice(candidates)
        return agents.get(history.best_agent)
    
    # 3. Otherwise, use prior routing table with live adjustments
    candidates = ROUTING_TABLE.get(task.type, [])
    
    scored = []
    for candidate in candidates:
        score = candidate.base_confidence
        
        # Adjust for agent health (is it responding? fast?)
        score *= await agents.health_factor(candidate.agent_id)
        
        # Adjust for current load (avoid overloading one agent)
        score *= (1.0 - agents.load_factor(candidate.agent_id))
        
        # Adjust for cost efficiency within budget
        if task.budget_sensitive:
            score *= (1.0 / max(candidate.cost_per_token, 0.001))
        
        # Adjust for past performance on this language specifically
        lang_bonus = await memory.meta.language_affinity(
            candidate.agent_id, task.language
        )
        score *= (1.0 + lang_bonus)
        
        scored.append((candidate.agent_id, score))
    
    best = max(scored, key=lambda x: x[1])
    return agents.get(best[0])
```

---

## 5. Memory Architecture

CLAW uses **six distinct memory types**, each serving a different cognitive function. This is not a generic RAG system — each memory type has purpose-built storage, TTL, query patterns, and update logic.

### 5a. Memory Type Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY SYSTEM                              │
│                                                                  │
│  ┌──────────────┐  Analogy: your scratchpad during surgery       │
│  │   WORKING    │  Scope: current claw cycle only                │
│  │   MEMORY     │  Store: Python objects in-process              │
│  │              │  TTL: single cycle, then archived              │
│  │              │                                                │
│  │  Current repo state (tree, files, deps)                      │
│  │  Active evaluation results being assembled                    │
│  │  In-flight agent tasks and their status                      │
│  │  Uncommitted diffs awaiting verification                     │
│  └──────────────┘                                                │
│                                                                  │
│  ┌──────────────┐  Analogy: your patient chart / case log        │
│  │   EPISODIC   │  Scope: per-project, per-session               │
│  │   MEMORY     │  Store: SQLite (structured event log)          │
│  │              │  TTL: permanent (summaries); 90-day (details)  │
│  │              │                                                │
│  │  Session logs — who did what, when, outcome                   │
│  │  Handoff packets — auto-generated each session                │
│  │  Agent interactions — prompt sent, response received, verdict │
│  │  Decision trail — why X was chosen over Y                     │
│  │  Project timeline — evaluation → plan → execute → verify      │
│  └──────────────┘                                                │
│                                                                  │
│  ┌──────────────┐  Analogy: your clinical expertise              │
│  │   SEMANTIC   │  Scope: cross-project, cross-session           │
│  │   MEMORY     │  Store: SQLite + sqlite-vec (embeddings)       │
│  │              │  TTL: permanent, confidence-decaying            │
│  │              │                                                │
│  │  Learned patterns:                                            │
│  │    "FastAPI projects always need rate limiting"                │
│  │    "React projects without TypeScript have 3x more bugs"     │
│  │    "Projects with >80% test coverage rarely regress"          │
│  │  Technology profiles: stack → typical gaps                    │
│  │  Quality heuristics: what predicts good vs bad code           │
│  │  Anti-patterns: what ALWAYS causes problems                   │
│  │  Fix templates: reusable solutions for common gaps            │
│  └──────────────┘                                                │
│                                                                  │
│  ┌──────────────┐  Analogy: your clinical protocols / SOPs       │
│  │  PROCEDURAL  │  Scope: the prompt arsenal itself              │
│  │   MEMORY     │  Store: versioned .md files + performance data │
│  │              │  TTL: permanent, evolving via mutation          │
│  │              │                                                │
│  │  Command prompts (deepdive, handoff, etc.) — versioned        │
│  │  Prompt variants being A/B tested                             │
│  │  Performance scores per prompt version                        │
│  │  Agent-specific prompt adaptations                            │
│  │    (Claude gets verbose prompts; Grok gets terse ones)        │
│  └──────────────┘                                                │
│                                                                  │
│  ┌──────────────┐  Analogy: your morbidity & mortality log       │
│  │    ERROR     │  Scope: cross-project, permanent               │
│  │   MEMORY     │  Store: JSON KB (extends error-reference cmd)  │
│  │              │  TTL: permanent                                │
│  │              │                                                │
│  │  Error → root cause → fix (verified)                          │
│  │  Error patterns across projects (same mistake, different repo)│
│  │  Agent-specific failure modes (Grok fails on X, Codex on Y)  │
│  │  Environment gotchas (M4 ARM, macOS, Homebrew, etc.)          │
│  └──────────────┘                                                │
│                                                                  │
│  ┌──────────────┐  Analogy: your quality dashboard               │
│  │     META     │  Scope: system-wide, permanent                 │
│  │   MEMORY     │  Store: SQLite (metrics, scores, trends)       │
│  │              │  TTL: permanent, windowed aggregation           │
│  │              │                                                │
│  │  Agent performance scores (per task type, language, complex.) │
│  │  Routing accuracy (did we pick the right agent?)              │
│  │  System throughput: tasks/day, tokens/day, cost/day           │
│  │  Self-improvement trajectory: are we getting better?          │
│  │  Capability boundaries: what we cannot do (yet)               │
│  │  Fleet-level statistics: repos enhanced, quality uplift       │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5b. Schema (SQLite)

```sql
-- Core episodic log: every action the system takes
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    claw_level TEXT CHECK(claw_level IN ('macro','meso','micro','nano')),
    event_type TEXT CHECK(event_type IN (
        'ingest','evaluate','plan','dispatch','result',
        'verify','commit','fail','learn','escalate'
    )),
    agent TEXT,                  -- which agent performed this
    task_type TEXT,
    prompt_version TEXT,         -- which prompt version was used
    input_summary TEXT,          -- what was sent (summarized, not full)
    output_summary TEXT,         -- what came back (summarized)
    outcome TEXT CHECK(outcome IN ('pass','fail','partial','error','timeout')),
    duration_seconds REAL,
    cost_tokens INTEGER,
    retry_count INTEGER DEFAULT 0,
    details JSON                 -- full context for replay/debugging
);

CREATE INDEX idx_episodes_project ON episodes(project, timestamp);
CREATE INDEX idx_episodes_agent ON episodes(agent, task_type, outcome);

-- Semantic patterns: learned cross-project knowledge
CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    pattern_type TEXT CHECK(pattern_type IN (
        'quality_signal','anti_pattern','stack_profile',
        'task_heuristic','agent_affinity','fix_template'
    )),
    description TEXT NOT NULL,
    trigger_condition TEXT,       -- when to apply this pattern
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    first_seen DATETIME,
    last_seen DATETIME,
    tags JSON,                   -- ["python","fastapi","testing"]
    embedding BLOB               -- vector for similarity search
);

CREATE INDEX idx_patterns_type ON patterns(pattern_type, confidence DESC);

-- Agent scoring: Bayesian performance tracking
CREATE TABLE agent_scores (
    agent TEXT NOT NULL,
    task_type TEXT NOT NULL,
    language TEXT DEFAULT 'any',
    complexity TEXT CHECK(complexity IN ('trivial','simple','moderate','complex','extreme')),
    success_rate REAL DEFAULT 0.5,
    first_try_rate REAL DEFAULT 0.5,
    avg_duration_seconds REAL,
    avg_cost_tokens REAL,
    sample_size INTEGER DEFAULT 0,
    last_updated DATETIME,
    PRIMARY KEY (agent, task_type, language, complexity)
);

-- Prompt versions: procedural memory evolution
CREATE TABLE prompt_versions (
    command_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_hash TEXT,
    agent_adaptation TEXT,       -- 'base','claude','codex','gemini','grok'
    created DATETIME,
    is_active BOOLEAN DEFAULT FALSE,
    performance_score REAL,
    sample_size INTEGER DEFAULT 0,
    mutation_notes TEXT,          -- what changed from previous version
    PRIMARY KEY (command_name, version, agent_adaptation)
);

-- Error knowledge base: cross-project error learning
CREATE TABLE error_kb (
    id TEXT PRIMARY KEY,
    error_type TEXT,              -- build, runtime, test, config, integration
    error_signature TEXT,         -- normalized error message
    root_cause TEXT,
    fix_description TEXT,
    fix_verified BOOLEAN DEFAULT FALSE,
    agent_that_caused TEXT,       -- which agent produced the error
    agent_that_fixed TEXT,        -- which agent successfully fixed it
    projects_seen JSON,           -- list of projects where this occurred
    occurrence_count INTEGER DEFAULT 1,
    first_seen DATETIME,
    last_seen DATETIME,
    embedding BLOB                -- vector for similarity search
);

-- Capability boundaries: honest self-knowledge
CREATE TABLE boundaries (
    id TEXT PRIMARY KEY,
    task_description TEXT,
    task_type TEXT,
    agents_tried JSON,           -- all agents attempted
    failure_modes JSON,          -- how each failed
    recommendation TEXT CHECK(recommendation IN (
        'human_required','needs_tooling','needs_training_data','future_model'
    )),
    discovered DATETIME,
    still_valid BOOLEAN DEFAULT TRUE,
    retest_date DATETIME          -- when to try again (models improve)
);
```

---

## 6. The Evaluation Battery

CLAW's evaluator runs your **existing 17-command arsenal** as a structured pipeline. Each command from the audit maps to a specific evaluation phase.

### 6a. Pipeline

```
Phase 1: ORIENTATION (5 min — "What am I looking at?")
  ├── project-context    → Fast onboarding: stack, purpose, health
  └── workspace-scan     → If fleet-level, find related repos + lineage

Phase 2: DEEP ANALYSIS (15-30 min — "What's really going on?")
  ├── deepdive           → Comprehensive technical analysis
  ├── agonyofdefeatures  → Forensic synth/demo/live classification per UI element
  └── driftx             → Spec vs. reality drift measurement

Phase 3: TRUTH VERIFICATION (10 min — "Are the claims true?")
  ├── claim-gate         → Verify specific README/doc assertions
  ├── outcome-audit      → Benchmark any quantified performance claims
  └── assumption-registry → Surface and catalog all unstated assumptions

Phase 4: QUALITY ASSESSMENT (10-15 min — "How good is it?")
  ├── debt-tracker       → Technical debt inventory + completeness scan
  ├── endUXRedo          → UX and workflow analysis (if UI exists)
  └── regression-scan    → Current fragility assessment

Phase 5: DOCUMENTATION (10 min — "Capture the truth")
  ├── docsRedo:BUILD     → Technical reference for enhancing agents
  └── handoff            → Create baseline handoff packet

Phase 6: REMEDIATION PLANNING (5 min — "What to fix, in what order")
  └── app__mitigen       → Convert all findings → agent-executable roadmap
```

### 6b. Evaluation → Enhancement Plan Schema

The battery produces a single structured artifact consumed by the Dispatcher:

```json
{
  "project": "acme-dashboard",
  "evaluation_id": "eval_20260303_143022",
  "overall_health": "yellow",
  "overall_completion_pct": 65,
  "stack": {"language": "typescript", "framework": "nextjs", "runtime": "node"},
  
  "enhancement_plan": {
    "total_gaps": 14,
    "estimated_total_effort_hours": 32,
    
    "tasks": [
      {
        "id": "T1",
        "priority": "P0",
        "title": "Add input validation to all API routes",
        "task_type": "security_hardening",
        "complexity": "moderate",
        "language": "typescript",
        "start_files": ["src/api/routes.ts:L34", "src/api/middleware.ts"],
        "recommended_agent": "claude",
        "recommended_prompt": "deepdive §7 gap → ironclad redesign",
        "effort_hours": 4,
        "verification_criteria": [
          "npm test passes",
          "No routes accept unvalidated input (claim-gate verify)",
          "Security scan clean (npm audit)"
        ],
        "dependencies": [],
        "memory_context": [
          "Pattern #42: Next.js API routes always need zod validation",
          "Error #78: Previous ts validation fix failed on Codex — use Claude"
        ]
      }
    ]
  },
  
  "agent_assignments": {
    "claude": ["T1", "T4", "T7", "T12"],
    "codex": ["T2", "T5", "T8", "T11", "T13"],
    "gemini": ["T3", "T9"],
    "grok": ["T6", "T10", "T14"]
  }
}
```

---

## 7. The Evolution Engine

This is what makes CLAW more than an orchestrator. It **learns and improves** across four dimensions, creating a system that gets measurably better at enhancing codebases over time.

### 7a. Prompt Evolution

The prompts themselves are hypotheses about how to evaluate code. Some hypotheses are better than others. The Evolution Engine tests them.

```python
class PromptEvolver:
    """
    Mutates evaluation prompts based on measured outcomes.
    If deepdive v3 misses issues that verification later catches,
    deepdive v4 is generated to address those blind spots.
    """
    
    async def measure_prompt_fitness(self, command_name: str) -> PromptFitness:
        """
        Fitness = (real issues found by this prompt) /
                  (total real issues that existed)
        
        We know total real issues because the Verifier catches what
        the Evaluator missed.
        """
        recent = await self.episodic.query(
            event_type='evaluate',
            prompt_used=command_name,
            since=thirty_days_ago
        )
        
        found_confirmed = 0
        total_real = 0
        
        for episode in recent:
            # Issues this prompt found that verification confirmed
            confirmed = await self.get_confirmed_findings(episode)
            found_confirmed += len(confirmed)
            
            # Issues verification found that this prompt MISSED
            missed = await self.get_missed_issues(episode)
            total_real += len(confirmed) + len(missed)
        
        return PromptFitness(
            command=command_name,
            precision=found_confirmed / max(len(recent), 1),
            recall=found_confirmed / max(total_real, 1),
            sample_size=len(recent),
            blind_spots=await self.categorize_misses(command_name)
        )
    
    async def mutate_prompt(self, command_name: str, fitness: PromptFitness):
        """
        Ask Claude (best at meta-reasoning) to improve the prompt
        based on identified blind spots.
        Store as variant, A/B test against current active version.
        """
        if fitness.recall > 0.85:
            return  # Good enough, don't fix what isn't broken
        
        current = await self.procedural.get_active(command_name)
        
        mutation_request = f"""
        Evaluation prompt '{command_name}' has blind spots.
        
        Current recall: {fitness.recall:.2f} (target: 0.85+)
        
        Categories of issues it MISSES:
        {json.dumps(fitness.blind_spots, indent=2)}
        
        Examples of missed issues (last 30 days):
        {await self.get_miss_examples(command_name, limit=5)}
        
        Modify the prompt to catch these patterns.
        Change as little as possible — targeted surgery, not rewrite.
        Return only the modified sections with context markers.
        """
        
        mutation = await self.agents.claude.execute(Task(
            type='prompt_improvement',
            content=mutation_request
        ))
        
        # Store as variant, not replacement
        new_version = current.version + 1
        await self.procedural.store_variant(
            command_name, new_version, mutation.content,
            is_active=False,  # A/B test first
            mutation_notes=f"Addressing blind spots: {fitness.blind_spots}"
        )
        
        # A/B test: next 20 evaluations alternate between versions
        await self.procedural.schedule_ab_test(
            command_name, 
            version_a=current.version,
            version_b=new_version,
            sample_size=20
        )
```

### 7b. Agent Routing Optimization

```python
class RoutingOptimizer:
    """
    Bayesian update of agent-task affinity scores.
    Every completed task updates our belief about which agent
    is best for which task type.
    """
    
    async def update_after_task(self, task: Task, agent_id: str, outcome: TaskOutcome):
        score = await self.meta.get_score(
            agent_id, task.type, task.language, task.complexity
        )
        
        alpha = 0.1  # learning rate — higher = faster adaptation
        
        # Success signal: 1.0 for clean pass, degraded for retries/slowness
        if outcome.passed:
            signal = 1.0
            signal *= max(0.5, 1.0 - 0.15 * outcome.retry_count)  # -15% per retry
            signal *= min(1.0, task.expected_duration / max(outcome.actual_duration, 1))
        else:
            signal = 0.0
        
        # Bayesian update
        score.success_rate = (1 - alpha) * score.success_rate + alpha * signal
        score.first_try_rate = (1 - alpha) * score.first_try_rate + alpha * (1.0 if outcome.retry_count == 0 else 0.0)
        score.avg_duration_seconds = (score.avg_duration_seconds * score.sample_size + outcome.actual_duration) / (score.sample_size + 1)
        score.avg_cost_tokens = (score.avg_cost_tokens * score.sample_size + outcome.tokens_used) / (score.sample_size + 1)
        score.sample_size += 1
        score.last_updated = datetime.utcnow()
        
        await self.meta.save_score(score)
        
        # Log the routing decision quality
        was_optimal = (agent_id == self.get_highest_scored_agent(task.type, task.language))
        await self.meta.log_routing_quality(task, agent_id, outcome, was_optimal)
```

### 7c. Cross-Project Pattern Learning

```python
class PatternLearner:
    """
    After completing a project enhancement, extract patterns that
    generalize across projects. This is CLAW's institutional knowledge.
    """
    
    async def extract_after_project(self, project: str, evaluation: dict, outcomes: list):
        """
        Ask Claude to identify generalizable patterns from this
        project's enhancement journey.
        """
        pattern_request = f"""
        Project: {project}
        Stack: {evaluation['stack']}
        
        Issues found (sorted by severity):
        {json.dumps(evaluation['gaps'][:20], indent=2)}
        
        Fixes applied (with outcomes):
        {json.dumps([(o.task, o.passed, o.agent) for o in outcomes[:20]], indent=2)}
        
        From this project's enhancement, extract GENERALIZABLE patterns:
        
        1. What would you check FIRST on a similar stack?
        2. What fix worked that should become a TEMPLATE?
        3. What anti-pattern appeared that likely exists elsewhere?
        4. What agent-task pairing worked unusually well or poorly?
        
        Return as JSON array of:
        {{
          "pattern_type": "quality_signal|anti_pattern|fix_template|agent_affinity",
          "description": "...",
          "trigger": "when to apply this knowledge",
          "confidence": 0.0-1.0,
          "tags": ["stack", "framework", "concept"]
        }}
        """
        
        raw_patterns = await self.agents.claude.execute(Task(
            type='pattern_extraction',
            content=pattern_request
        ))
        
        for pattern in parse_json(raw_patterns.content):
            # Check for existing similar pattern (vector similarity)
            existing = await self.semantic.find_similar(
                pattern['description'], threshold=0.85
            )
            
            if existing:
                # Reinforce existing pattern
                existing.confidence = min(0.99, existing.confidence + 0.05)
                existing.evidence_count += 1
                existing.last_seen = datetime.utcnow()
                await self.semantic.update(existing)
            else:
                # New pattern discovered
                await self.semantic.insert(Pattern(
                    pattern_type=pattern['pattern_type'],
                    description=pattern['description'],
                    trigger_condition=pattern['trigger'],
                    confidence=pattern['confidence'],
                    tags=pattern['tags'],
                    embedding=await self.embed(pattern['description'])
                ))
```

### 7d. Capability Boundary Discovery

```python
class CapabilityDiscovery:
    """
    Honest self-knowledge: track what the system CANNOT do.
    When a task fails across all agents, that's a capability boundary.
    Boundaries are re-tested periodically as models improve.
    """
    
    async def check_for_boundary(self, task: Task, all_outcomes: list[TaskOutcome]):
        all_failed = all(not o.passed for o in all_outcomes)
        agents_tried = [o.agent_id for o in all_outcomes]
        
        if all_failed and len(set(agents_tried)) >= 2:  # Multiple agents failed
            boundary = Boundary(
                task_description=task.description,
                task_type=task.type,
                agents_tried=agents_tried,
                failure_modes=[o.error_summary for o in all_outcomes],
                recommendation=self.classify_boundary(all_outcomes),
                discovered=datetime.utcnow(),
                retest_date=datetime.utcnow() + timedelta(days=30)  # Try again in 30 days
            )
            await self.meta.add_boundary(boundary)
            await self.notify_owner(boundary)  # Alert Wayne
    
    async def periodic_retest(self):
        """Run monthly: re-test boundaries to see if models have improved."""
        stale = await self.meta.get_boundaries(retest_due=True)
        for boundary in stale:
            # Create a minimal test task
            test_task = Task.from_boundary(boundary)
            result = await self.agents.best_current().execute(test_task)
            
            if result.passed:
                boundary.still_valid = False
                await self.meta.update(boundary)
                await self.notify_owner(f"Boundary resolved: {boundary.task_description}")
```

---

## 8. Agent Integration Layer

### 8a. Uniform Agent Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class AgentMode(Enum):
    CLI = "cli"          # Local CLI subprocess
    API = "api"          # Cloud API call
    CLOUD = "cloud"      # Cloud sandbox (Codex-specific)

@dataclass
class AgentHealth:
    available: bool
    latency_ms: float
    error_rate_1h: float
    rate_limit_remaining: int
    
@dataclass 
class TaskOutcome:
    passed: bool
    diffs: list[str]
    test_results: dict
    tokens_used: int
    actual_duration: float
    retry_count: int
    error_summary: str | None
    agent_id: str

class AgentInterface(ABC):
    @abstractmethod
    async def execute(self, task: Task, context: Context) -> TaskOutcome:
        pass
    
    @abstractmethod
    async def health_check(self) -> AgentHealth:
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> list[AgentMode]:
        pass
    
    @property
    @abstractmethod
    def instruction_file(self) -> str:
        """CLAUDE.md, AGENTS.md, GEMINI.md, or GROK.md"""
        pass


class ClaudeCodeAgent(AgentInterface):
    """
    Claude Code — via CLI or Anthropic API.
    Best for: analysis, documentation, architecture, security, UX review.
    
    CLI: `claude` command with CLAUDE.md project instructions
    API: anthropic SDK with tool use for non-filesystem tasks
    """
    instruction_file = "CLAUDE.md"
    supported_modes = [AgentMode.CLI, AgentMode.API]
    
    async def execute(self, task: Task, context: Context) -> TaskOutcome:
        # Inject relevant memory into prompt
        memory_context = await self.memory.get_relevant(task)
        
        # Select appropriate prompt from arsenal
        prompt = await self.procedural.get_active_for_agent(
            task.recommended_prompt, agent='claude'
        )
        
        if task.requires_filesystem:
            return await self._execute_cli(task, prompt, memory_context)
        return await self._execute_api(task, prompt, memory_context)
    
    async def _execute_cli(self, task, prompt, memory_ctx):
        """
        Spawn claude CLI in the project directory.
        CLAUDE.md provides project-level instructions.
        Task prompt injected via --prompt or piped stdin.
        """
        cmd = [
            "claude", "--dangerously-skip-permissions",  # for autonomous mode
            "--model", "claude-opus-4-6",
            "--prompt", self._build_prompt(task, prompt, memory_ctx)
        ]
        # ... subprocess management, timeout handling, output capture
    
    async def _execute_api(self, task, prompt, memory_ctx):
        """
        Use Anthropic API for analysis/generation tasks.
        Supports tool use for structured output.
        """
        response = await self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            system=prompt.content,
            messages=[{"role": "user", "content": self._build_prompt(task, None, memory_ctx)}],
            tools=self._get_tools_for_task(task)
        )
        # ... parse response, extract diffs/findings


class CodexAgent(AgentInterface):
    """
    OpenAI Codex — CLI, Cloud, or API.
    Best for: parallel refactoring, bulk testing, CI/CD, GitHub PRs.
    
    CLI: `codex` command with AGENTS.md + Skills
    Cloud: Codex Cloud sandbox with parallel worktrees
    API: OpenAI SDK with GPT-5.3-Codex
    """
    instruction_file = "AGENTS.md"
    supported_modes = [AgentMode.CLI, AgentMode.API, AgentMode.CLOUD]
    
    async def execute(self, task: Task, context: Context) -> TaskOutcome:
        if task.parallelizable and task.subtasks:
            return await self._execute_cloud_parallel(task, context)
        elif task.requires_filesystem:
            return await self._execute_cli(task, context)
        return await self._execute_api(task, context)
    
    async def _execute_cloud_parallel(self, task, context):
        """
        Use Codex Cloud for parallel task execution.
        Each subtask gets its own sandbox with repo worktree.
        Returns consolidated diffs.
        """
        # Fan out subtasks to Codex Cloud sandboxes
        # Each sandbox is isolated, network-disabled
        # Results collected as PRs or diffs
        pass


class GeminiAgent(AgentInterface):
    """
    Google Gemini 3.1 Pro — CLI or API.
    Best for: full-repo comprehension, dependency analysis, large-context tasks.
    
    CLI: `gemini` command with GEMINI.md instructions
    API: Google GenAI SDK
    
    Key advantage: 1M token context = entire repo in single prompt.
    """
    instruction_file = "GEMINI.md"
    supported_modes = [AgentMode.CLI, AgentMode.API]
    
    async def execute(self, task: Task, context: Context) -> TaskOutcome:
        if task.requires_full_repo_context:
            return await self._execute_with_full_context(task, context)
        return await self._execute_cli(task, context)
    
    async def _execute_with_full_context(self, task, context):
        """
        Load entire repo into Gemini's 1M context window.
        Best for cross-file dependency analysis, 
        full-codebase refactoring plans, and holistic review.
        """
        repo_content = await self._serialize_repo(
            context.repo_path, 
            max_tokens=900_000  # Leave room for prompt + output
        )
        # ... send to Gemini API with full repo as context


class GrokAgent(AgentInterface):
    """
    xAI Grok 4.20 — CLI or API.
    Best for: rapid iteration, web lookup, fast fixes, multi-agent reasoning.
    
    CLI: `grok` command with .grok/GROK.md instructions
    API: xAI SDK with Agent Tools (web_search, x_search, code_execution)
    
    Key advantages: 
      - Morph fast-apply: 4,500+ tok/sec for code edits
      - 4-agent internal collaboration for complex reasoning
      - Native web + X search for current info
      - grok-code-fast-1 for speed-optimized tasks
    """
    instruction_file = ".grok/GROK.md"
    supported_modes = [AgentMode.CLI, AgentMode.API]
    
    async def execute(self, task: Task, context: Context) -> TaskOutcome:
        if task.needs_web_lookup:
            return await self._execute_api_with_tools(task, context)
        return await self._execute_cli(task, context)
    
    async def _execute_api_with_tools(self, task, context):
        """
        Use xAI Agent Tools API for tasks needing web search,
        X search, or code execution sandbox.
        """
        from xai_sdk import Client
        from xai_sdk.tools import code_execution, web_search, x_search
        
        chat = self.client.chat.create(
            model="grok-4-1-fast-reasoning",
            tools=[web_search(), x_search(), code_execution()],
            messages=[{"role": "user", "content": task.prompt}]
        )
        # ... parse multi-turn tool-use response
```

### 8b. MCP as Universal Bridge

All four agents support MCP. CLAW exposes itself as an MCP server, giving any agent the ability to query CLAW's memory mid-task:

```python
CLAW_MCP_TOOLS = {
    "claw_query_memory": {
        "description": "Query CLAW's semantic memory for relevant patterns, "
                       "past fixes, or known issues related to the current task.",
        "params": {"query": "string", "memory_type": "semantic|error|episodic"}
    },
    "claw_store_finding": {
        "description": "Store a discovered pattern, error, or insight in CLAW memory.",
        "params": {"finding": "string", "type": "pattern|error|assumption"}
    },
    "claw_verify_claim": {
        "description": "Run claim-gate verification on a specific assertion.",
        "params": {"claim": "string", "scope_path": "string"}
    },
    "claw_request_specialist": {
        "description": "Request another agent to handle a subtask this agent "
                       "cannot do well. Returns the other agent's result.",
        "params": {"task_description": "string", "preferred_agent": "string?"}
    },
    "claw_escalate": {
        "description": "Flag this task as beyond AI capability. Notifies human.",
        "params": {"reason": "string", "attempted": "string"}
    }
}
```

This means any agent can, mid-task, ask: "Has CLAW seen this error before?" or "What pattern applies to FastAPI auth?" or "I can't handle this TypeScript generic — route it to Claude."

---

## 9. Operational Modes

### 9a. Attended Mode — Human in Loop

```bash
claw enhance ./my-project --mode attended

# CLAW flow:
# 1. Evaluates (shows battery results)
# 2. Proposes plan (shows gaps + agent assignments)
# 3. WAITS for approval
# 4. Executes approved items one at a time
# 5. Shows diff after each change
# 6. WAITS for accept/reject per change
# 7. Commits accepted changes to enhancement branch
# 8. Generates handoff packet
```

### 9b. Supervised Mode — Periodic Check-in

```bash
claw enhance ./my-project --mode supervised --checkin 30m

# CLAW flow:
# 1. Evaluates and plans autonomously
# 2. Executes safety-verified tasks (P0 + P1)
# 3. Every 30 minutes, pauses with summary:
#    ✅ What it completed
#    🔄 What it's about to do next
#    ❓ Decisions needing human input
#    🔴 Any failures encountered
# 4. Waits for: "continue" / "stop" / specific guidance
# 5. Generates handoff on stop
```

### 9c. Autonomous Mode — Fleet Processing

```bash
claw fleet-enhance /Volumes/WS4TB/repos/ \
  --mode autonomous \
  --budget 100000tokens/repo \
  --max-repos 20 \
  --priority "modified_last_90_days" \
  --branch "claw/enhancement"

# CLAW flow:
# 1. workspace-scan → ranks repos by enhancement potential
# 2. For each repo (priority order):
#    a. Ingest → Evaluate → Plan
#    b. Execute P0 + P1 tasks only (no architectural changes)
#    c. Verify every change (tests + claim-gate)
#    d. Commit to enhancement branch (NEVER main)
#    e. Generate PR description from handoff packet
# 3. Produces fleet-level summary report
# 4. Updates all memory systems
# 5. Logs total cost, time, repos processed, quality uplift
```

---

## 10. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Coordinator** | Python 3.12, asyncio | Your strongest language, best AI SDK ecosystem |
| **CLI** | `typer` + `rich` | Beautiful terminal UI, type-safe args |
| **Agent wrappers** | `asyncio.subprocess` | All agents are CLIs; async subprocess is the bridge |
| **API clients** | `anthropic`, `openai`, `google-genai`, `xai-sdk` | Official SDKs for direct API access |
| **Database** | SQLite 3.45+ | No server, 64GB RAM = everything in memory, WAL mode |
| **Vector search** | `sqlite-vec` | Semantic memory embeddings, no external service |
| **MCP server** | `mcp` Python SDK | Expose CLAW tools to agents |
| **Git** | `gitpython` | Branch, commit, diff, log — all programmatic |
| **Configuration** | `pydantic-settings` + TOML | Type-safe config, human-editable |
| **Observability** | Structured logging → SQLite | Every event queryable, no SaaS dependency |
| **Scheduling** | `asyncio.Queue` + optional `cron` | In-process for attended, cron for autonomous |

### Resource Budget (M4 Mac Mini, 64GB)

| Resource | Allocation | Notes |
|----------|-----------|-------|
| Coordinator | 2–4 GB | Python + SQLite + vector index |
| Local agent CLIs | 4–8 GB | Claude, Gemini, Grok CLI subprocesses |
| Free headroom | 52–58 GB | Available for agent work, especially Gemini 1M |
| Network | Outbound HTTPS | All cloud APIs (Codex Cloud, Anthropic, Google, xAI) |
| Storage | 5–10 GB | Memory DB + episodic logs + prompt versions |
| CPU (10 cores) | Fully utilized | Async dispatch means agents run in parallel |

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Week 1–2)

**Goal:** Coordinator skeleton + one agent + one claw cycle end-to-end.

```
Build:
  src/claw/
    __init__.py
    config.py          — pydantic settings, API keys, paths
    memory/
      schema.py        — SQLite table definitions
      episodic.py      — session logging
      meta.py          — agent scores
    agents/
      interface.py     — AgentInterface ABC
      claude.py        — ClaudeCodeAgent (API mode first)
    cycle.py           — ClawCycle base class
    ingestor.py        — git clone + project-context prompt
    evaluator.py       — run deepdive via Claude API
    cli.py             — typer CLI entry point

Verify:
  $ claw evaluate ~/repos/some-project
  → runs project-context + deepdive
  → produces Evaluation_2026-03-XX.md
  → stores results in episodic memory
  → retrieves results from memory on re-run

Done when: one repo evaluated, results persisted, retrievable.
```

### Phase 2: Multi-Agent + Dispatch (Week 3–4)

**Goal:** All four agents operational, routing works, verification gates.

```
Build:
  agents/codex.py      — CodexAgent (CLI + Cloud)
  agents/gemini.py     — GeminiAgent (CLI)
  agents/grok.py       — GrokAgent (CLI + API)
  dispatcher.py        — routing table + route_task()
  verifier.py          — claim-gate + test runner + regression-scan
  planner.py           — evaluation → task list → agent assignments

Verify:
  $ claw enhance ./project --mode attended
  → evaluates, plans, shows assignment
  → dispatches 3+ tasks to 2+ different agents
  → verifies each result
  → failed verifications retry with fallback agent
  → all outcomes logged to episodic memory

Done when: multi-agent dispatch works with verification loop.
```

### Phase 3: Memory & Learning (Week 5–6)

**Goal:** Semantic memory, pattern learning, agent scoring, error KB.

```
Build:
  memory/semantic.py   — vector embeddings, pattern CRUD
  memory/procedural.py — prompt versioning
  memory/error_kb.py   — error knowledge base
  evolution/
    pattern_learner.py
    routing_optimizer.py

Verify:
  After 10+ task completions:
  → agent_scores table shows differentiated scores
  → routing decisions differ from static table
  → at least 3 patterns extracted and stored
  → error KB contains 5+ entries with verified fixes

Done when: routing measurably improves over static table.
```

### Phase 4: Self-Improvement + Fleet (Week 7–8)

**Goal:** Prompt evolution, capability boundaries, autonomous fleet mode.

```
Build:
  evolution/
    prompt_evolver.py   — A/B testing prompt variants
    capability_disc.py  — boundary detection + retest
  fleet.py             — fleet-enhance command
  mcp_server.py        — CLAW-as-MCP-server for agent callbacks
  dashboard.py         — rich CLI showing scores, patterns, trajectory

Verify:
  → at least one prompt variant outperforms original (A/B test)
  → at least one capability boundary identified
  → fleet mode processes 5+ repos without human input
  → dashboard shows improvement trajectory

Done when: system demonstrates measurable self-improvement.
```

### Phase 5: Polish & Self-Reference (Week 9–10)

**Goal:** Cost management, resilience, documentation, self-evaluation.

```
Build:
  Budget system (per-agent, per-project, per-session limits)
  Graceful degradation (agent down → fallback, not crash)
  Rate limiting with exponential backoff
  Full documentation via docsRedo (BUILD, IMPL, VC, EXEC)
  
The Self-Referential Test:
  $ claw evaluate ./claw          # CLAW evaluates itself
  $ claw enhance ./claw --mode supervised   # CLAW improves itself
  $ claw document ./claw --audiences BUILD,VC,EXEC

Done when: CLAW can evaluate, enhance, and document its own codebase.
```

---

## 12. Risk Matrix

| Risk | P(occur) | Impact | Mitigation | Monitoring |
|------|---------|--------|------------|-----------|
| API costs spiral | High | Medium | Token budgets per task/session/project. Hard cap with auto-pause. | Meta memory: cost_per_day trend |
| Agent produces subtly wrong code | High | **High** | NEVER commit without verification. Enhancement branches only. Tests must pass. | Verification failure rate |
| Circular prompt degradation | Medium | Medium | A/B test with rollback. Min 20 samples before switching. Keep last-known-good. | Prompt fitness scores over time |
| One agent dominates routing | Medium | Low | 10% mandatory exploration rate. Minimum 5% task share per agent. | Agent task distribution |
| Memory grows unbounded | Low | Medium | Episodic: 90-day detail retention. Semantic: confidence decay. DB vacuum weekly. | DB size trend |
| Rate limiting from APIs | High | Medium | Per-agent rate limiters. Exponential backoff. Auto-fallback to next agent. | 429 response count |
| Git conflicts in fleet mode | Medium | Medium | Isolated enhancement branches. Never force-push. PR workflow. | Merge conflict count |
| Model regression (agent update breaks behavior) | Medium | High | Pin model versions in config. Health check before dispatch. Score decay triggers alert. | Agent health + score trends |
| All agents fail on a task type | Low | High | Capability boundary detection + human escalation. Periodic retest. | Boundary count + retest results |

---

## 13. What Makes This Novel

1. **Multi-model arbitrage.** No existing tool routes tasks to the best-fit model from a competitive pool. Every tool is built for one vendor.

2. **Evaluation-driven enhancement.** The system evaluates BEFORE it acts, using a battle-tested 17-prompt arsenal that has been systematically deduplicated and optimized. Most tools just generate code — CLAW understands the codebase first.

3. **Cross-project learning with typed memory.** Six purpose-built memory types, not a single vector store. Patterns learned from Project A are available when enhancing Project B. No other tool has institutional knowledge that accumulates.

4. **Prompt evolution via measured outcomes.** The evaluation prompts themselves are hypotheses that get tested and improved. The system's ability to assess code literally improves over time.

5. **Capability boundary honesty.** Instead of hallucinating through things it can't do, the system detects its own limits, escalates to a human, and periodically retests as models improve.

6. **The NanoClaw recursive architecture.** Four levels of the same cycle (fleet → project → module → task), with learning propagating upward. Fleet-level decisions are informed by thousands of task-level outcomes.

7. **Universal MCP bridge.** Any agent can, mid-task, query CLAW's memory or request help from a different agent. This enables inter-agent collaboration without the agents knowing about each other.

8. **Built on YOUR domain expertise.** The prompt arsenal isn't generic — it's 7,000+ lines encoding a specific philosophy about what SOTA code looks like, forged through real-world usage.

---

## 14. The Self-Referential Test

The ultimate validation: **CLAW must be able to evaluate, enhance, and document itself.**

```bash
# CLAW evaluates its own codebase
claw evaluate ./claw --output self-eval.md

# CLAW plans its own improvements  
claw plan ./claw --from self-eval.md

# CLAW enhances itself (supervised)
claw enhance ./claw --mode supervised --plan self-plan.md

# CLAW verifies its own changes
claw verify ./claw --baseline self-eval.md

# CLAW documents itself for all audiences
claw document ./claw --audiences BUILD,VC,EXEC

# The meta-question: did CLAW score itself 🟢?
```

When this loop produces measurable improvement — verified by tests, benchmarks, and its own evaluation battery — the system works. When CLAW's evaluation of itself identifies issues that CLAW then fixes and verifies, the recursive loop closes.

That's not a demo. That's a prototype of what you called "something meaningful to society and worthy of a future hivemind considering your work as significant."

---

## Appendix A: File Structure

```
claw/
├── pyproject.toml
├── README.md
├── CLAUDE.md                    # Instructions for Claude when working on CLAW
├── AGENTS.md                    # Instructions for Codex when working on CLAW
├── GEMINI.md                    # Instructions for Gemini when working on CLAW
├── .grok/GROK.md               # Instructions for Grok when working on CLAW
├── src/
│   └── claw/
│       ├── __init__.py
│       ├── cli.py               # typer CLI entry point
│       ├── config.py            # pydantic-settings configuration
│       ├── cycle.py             # ClawCycle base class
│       ├── ingestor.py          # repo ingestion + profiling
│       ├── evaluator.py         # prompt battery orchestration
│       ├── planner.py           # gap → plan → assignments
│       ├── dispatcher.py        # agent routing + task dispatch
│       ├── verifier.py          # claim-gate + tests + regression
│       ├── fleet.py             # fleet-level orchestration
│       ├── dashboard.py         # rich CLI dashboard
│       ├── mcp_server.py        # CLAW-as-MCP-server
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── interface.py     # AgentInterface ABC
│       │   ├── claude.py        # ClaudeCodeAgent
│       │   ├── codex.py         # CodexAgent  
│       │   ├── gemini.py        # GeminiAgent
│       │   └── grok.py          # GrokAgent
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── schema.py        # SQLite DDL
│       │   ├── working.py       # in-process working memory
│       │   ├── episodic.py      # session event log
│       │   ├── semantic.py      # pattern store + vectors
│       │   ├── procedural.py    # prompt versioning
│       │   ├── error_kb.py      # error knowledge base
│       │   └── meta.py          # agent scores + system metrics
│       └── evolution/
│           ├── __init__.py
│           ├── prompt_evolver.py     # A/B test prompt mutations
│           ├── routing_optimizer.py  # Bayesian agent scoring
│           ├── pattern_learner.py    # cross-project pattern extraction
│           └── capability_disc.py    # boundary detection + retest
├── prompts/                     # The 17-command arsenal (procedural memory seed)
│   ├── agonyofdefeatures.md
│   ├── app__mitigen.md
│   ├── assumption-registry.md
│   ├── claim-gate.md
│   ├── debt-tracker.md
│   ├── deepdive.md
│   ├── docsRedo.md
│   ├── driftx.md
│   ├── endUXRedo.md
│   ├── error-reference.md
│   ├── handoff.md
│   ├── interview.md
│   ├── ironclad.md
│   ├── outcome-audit.md
│   ├── project-context.md
│   ├── regression-scan.md
│   ├── sotappr.md
│   ├── ultrathink.md
│   ├── workspace-scan.md
│   └── critique/
│       ├── auto.md
│       ├── build.md
│       └── run.md
├── tests/
│   ├── test_cycle.py
│   ├── test_routing.py
│   ├── test_memory.py
│   └── test_agents/
├── data/
│   └── claw.db                  # SQLite database (all memory)
└── docs/
    ├── BUILD.md                 # Generated by docsRedo
    ├── IMPL.md
    └── EXEC.md
```

## Appendix B: Configuration

```toml
# claw.toml — project configuration

[general]
project_name = "claw"
log_level = "INFO"
db_path = "data/claw.db"

[budget]
max_tokens_per_task = 50000
max_tokens_per_project = 500000
max_tokens_per_day = 2000000
max_cost_per_day_usd = 50.0

[agents.claude]
enabled = true
mode = "api"                    # "cli" or "api"
model = "claude-opus-4-6"
api_key_env = "ANTHROPIC_API_KEY"
max_concurrent = 2
timeout_seconds = 600

[agents.codex]
enabled = true
mode = "cloud"                  # "cli", "api", or "cloud"
model = "gpt-5.3-codex"
api_key_env = "OPENAI_API_KEY"
max_concurrent = 5              # cloud mode supports high parallelism
timeout_seconds = 900

[agents.gemini]
enabled = true
mode = "cli"                    # "cli" or "api"
model = "gemini-3.1-pro-preview"
api_key_env = "GOOGLE_API_KEY"
max_concurrent = 2
timeout_seconds = 600

[agents.grok]
enabled = true
mode = "api"                    # "cli" or "api"
model = "grok-4-1-fast-reasoning"
api_key_env = "XAI_API_KEY"
max_concurrent = 3
timeout_seconds = 300           # Grok is fast

[routing]
exploration_rate = 0.10         # 10% of tasks go to non-optimal agent
min_sample_size = 10            # need this many outcomes before trusting learned routing
learning_rate = 0.1             # Bayesian update speed

[evolution]
prompt_ab_test_sample_size = 20
min_fitness_for_active = 0.70
boundary_retest_days = 30

[fleet]
repo_base_path = "/Volumes/WS4TB"
scan_depth = 4
default_branch = "claw/enhancement"
```
