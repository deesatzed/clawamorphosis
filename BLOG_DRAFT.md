# We Built a System That Improves Its Own Code — Here's What Happened

*A multi-model AI orchestrator that evaluates, plans, executes, and verifies codebase improvements. Then we pointed it at itself.*

---

## The Problem with Single-Agent Coding

AI coding agents are impressive. Claude Code, Codex, Gemini, Grok — each can write functions, fix bugs, generate tests. But hand any one of them a complex, real-world codebase and ask "make this better," and you hit the same walls:

- **No verification.** The agent claims it fixed the bug. Did it? You're back to reading diffs.
- **No memory.** It solves the same problem three different ways across three sessions, learning nothing.
- **No routing intelligence.** Every task goes to the same agent, whether it's a security audit or a CSS tweak.
- **No budget control.** One runaway task burns through your API credits before breakfast.

We built CLAW to solve all four.

## What CLAW Actually Is

CLAW (Codebase Learning & Autonomous Workforce) is a Python system that coordinates four AI coding agents through a pipeline:

```
Evaluate → Plan → Dispatch → Act → Verify → Learn
```

The "Evaluate" step runs an 18-prompt analysis battery against the target repository — architecture review, drift detection, claim verification, debt tracking, security scanning. The battery supports multiple modes (full, quick, structural, auto) so you can tune depth vs. speed depending on the context.

The "Dispatch" step routes each task to the best-fit agent using Bayesian scoring. Claude gets the analysis work. Codex gets bulk refactoring. Gemini gets full-repo comprehension (1M context). Grok gets quick fixes. But these aren't hardcoded — the system starts with priors and updates them based on actual outcomes.

The "Verify" step is a 7-check audit gate. No agent output passes without clearing: dependency jail, placeholder scan, drift alignment, style match, chaos resistance, claim validation, and optional LLM deep review. Failed verification triggers a retry with a *different* agent and the failed approach added to a forbidden list.

The "Learn" step updates agent scores, stores successful patterns in semantic memory, and records failed approaches in an error KB — so the same mistake is never repeated.

## The Architecture: NanoClaw Hierarchy

The core insight is fractal: the same 6-step cycle operates at four scales.

**MacroClaw** (fleet level) scans hundreds of repositories, ranks them by enhancement potential, and allocates budgets. **MesoClaw** (project level) runs the evaluation battery against one repo and produces a plan. **MicroClaw** (task level) takes one task, routes it to an agent, and monitors execution. **NanoClaw** (self-improvement) updates scores and routing after every task.

MesoClaw and NanoClaw are fully wired into the execution cycle — not just architectural concepts, but active participants in every enhancement run. The fleet-level MacroClaw orchestration is available via `claw fleet-enhance`, which scans a directory of repositories, ranks them, and processes them autonomously.

Learning propagates upward. A task-level failure informs project-level routing, which informs fleet-level scheduling. The system gets smarter at every scale with every execution.

## Seven Memory Systems

Most AI tools are stateless — every session starts from scratch. CLAW maintains seven distinct memory types:

1. **Working memory** — current cycle state, discarded after execution
2. **Episodic memory** — session event log, 90-day retention
3. **Semantic memory** — cross-project patterns stored as 384-dimensional embeddings with confidence decay
4. **Procedural memory** — versioned prompt arsenal with A/B testing
5. **Error KB** — cross-project error database mapping errors to root causes to verified fixes
6. **Meta memory** — agent performance scores via Bayesian Beta distributions
7. **Hybrid search** — vector similarity + full-text search with MMR re-ranking

When CLAW encounters a problem it's seen before — even in a different project — it retrieves the successful approach and avoids known-failed approaches. The fitness of each stored pattern decays over time unless reinforced by successful reuse.

## Prompt Evolution: Self-Improving Instructions

Every agent receives instructions through prompt templates. The prompt evolution engine:

1. **Mutates** existing prompts (adds constraints, changes structure, sharpens focus)
2. **A/B tests** mutations against the original (20-sample minimum)
3. **Promotes** winners based on Bayesian comparison of success rates
4. **Retires** losers

This means the instructions agents receive get better over time — without any manual tuning. The system discovers what phrasing works best for each agent on each task type.

## OpenRouter: Unified Multi-Model Access

A key operational decision: all four agents now support **OpenRouter mode**. Instead of managing four separate API keys and four different SDKs, any agent can route through OpenRouter with a single API key. This enables cost-controlled multi-model comparison — you can test whether a cheaper model handles certain task types just as well as a premium one.

Round-robin model testing confirmed all four agent slots working through OpenRouter, with models including gemini-flash-lite, qwen, minimax, and glm-5. The practical benefit: you can swap underlying models weekly (as new ones release) without touching agent code. The Bayesian routing layer measures outcomes regardless of which model is behind each agent.

## The Showpiece: CLAW Enhances Itself

The most honest test of an autonomous coding system is self-reference: can it improve its own code?

CLAW is not a toy. It's ~18,000 lines of Python across 58 files — async throughout, SQLite with vector search, four agent integrations (all OpenRouter-capable), seven memory systems, an evolution engine, budget enforcement, graceful degradation, and an 18-prompt evaluation battery. It has 1,153 passing tests.

When we run `claw enhance . --mode attended`, CLAW:

1. **Evaluates itself** using the 18-prompt battery — deepdive analysis, drift detection, claim verification, technical debt tracking, security scanning
2. **Plans improvements** — the MesoClaw planner converts evaluation findings into prioritized tasks with dependency ordering (security before features, infrastructure before implementation)
3. **Routes each task** to the best-fit agent — Claude for the analysis-heavy work, with Bayesian scoring updating after each outcome
4. **Executes** — the agent reads CLAW's own source code and produces changes
5. **Verifies** — the 7-check gate runs against CLAW's own test suite, rejecting placeholder code, scanning for drift, validating claims
6. **Learns** — the NanoClaw loop records what worked in semantic memory, updates agent scores and routing, and stores failed approaches in the error KB

The recursion is real. CLAW's Verifier runs CLAW's own `pytest` suite. CLAW's drift detector compares its own documentation against its own implementation. CLAW's claim-gate validates assertions in its own README.

## What We Learned Building This

**Verification is the bottleneck, not generation.** AI agents can produce code quickly. The hard part is knowing whether that code is correct, consistent, and complete. Our 7-check verification gate catches what humans would miss in code review — and what agents would miss checking their own work.

**Routing matters more than model quality.** A mediocre agent doing what it's good at outperforms a great agent doing what it's bad at. The static routing priors (Claude for analysis, Codex for refactoring) are a starting point. After 20+ tasks, the Bayesian routing diverges from the initial table based on measured outcomes.

**Memory across sessions is transformative.** The first time CLAW encounters a new error pattern, it fumbles. The second time, it retrieves the solution from semantic memory and applies it immediately. Cross-project memory means a lesson learned in one codebase benefits every future codebase.

**Budget enforcement prevents catastrophe.** Without caps, a single stuck task can burn through $50 of API credits looping on a problem. Four-level budget enforcement (per-task, per-project, per-day, per-agent) with auto-pause ensures costs stay predictable.

**Self-improvement is not magic — it's bookkeeping.** Prompt evolution, routing optimization, and pattern learning are all just careful tracking of what worked and what didn't, combined with statistical methods (Thompson sampling, Bayesian Beta distributions, fitness-weighted retrieval) to make better decisions next time.

**OpenRouter changes the economics.** Being able to swap models per agent without code changes means you can run cost experiments. Does a $0.10/M-token model handle boilerplate refactoring as well as a $15/M-token model? The Bayesian routing will tell you, with data.

## Technical Details

- **Language:** Python 3.12, asyncio throughout
- **Database:** SQLite with WAL mode, sqlite-vec for vector search, FTS5 for full-text
- **Agents:** Claude Code, Codex, Gemini, Grok — all supporting native SDK and OpenRouter modes
- **Memory:** 384-dimensional embeddings via sentence-transformers, hybrid vector + text retrieval with MMR
- **Budget:** 4-level USD-denominated caps with auto-pause and fallback routing
- **Evolution:** Bayesian A/B testing for prompt variants, Thompson sampling for routing
- **Fleet:** MacroClaw orchestration via `claw fleet-enhance` for multi-repo processing
- **Evaluation:** 18-prompt battery with selectable modes (full/quick/structural/auto)

## Try It

```bash
git clone https://github.com/deesatzed/clawamorphosis.git
cd clawamorphosis
pip install -e ".[dev]"
claw setup                            # Configure API keys and models
claw evaluate .                       # CLAW analyzes itself
claw evaluate . --battery-mode quick  # Fast evaluation with core prompts only
claw enhance . --mode attended        # CLAW improves itself
claw fleet-enhance /path/to/repos/    # Process a fleet of repositories
claw results                          # See what happened
```

The code is MIT licensed. The prompts are included. The self-referential test is the default demo.

---

*CLAW is ~18,000 lines of Python, 1,153 tests, 18 evaluation prompts, 7 memory systems, 4 agent integrations (all OpenRouter-capable), 30+ database tables, and 7 CLI commands. It was built by harvesting 42 battle-tested components from a production SWE agent orchestrator and writing 19 new files for multi-model fleet architecture. The total codebase — source, tests, config, prompts — was built across a single implementation sprint.*
