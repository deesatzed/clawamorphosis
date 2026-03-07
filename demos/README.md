# CLAW Showpiece Demos

Three self-contained demonstrations that prove CLAW works on real code with real AI agents.

Each demo is a single shell script. One prerequisite: an OpenRouter API key.

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Demo 1: The Mirror

**CLAW evaluates itself.**

```bash
./demos/01-the-mirror.sh
```

Four AI agents simultaneously analyze CLAW's 18,000 LOC codebase for documentation drift, technical debt, architecture issues, and feature completeness. Each finding includes file:line evidence.

- **Runtime:** ~2-4 minutes
- **LLM calls:** 5 (orientation + deep analysis)
- **Shows:** Multi-agent dispatch, Bayesian routing, structured analysis

## Demo 2: The Heist

**Cross-repo knowledge transfer.**

```bash
./demos/02-the-heist.sh
```

CLAW mines the `ralfed/` reference implementation, extracts transferable patterns (circuit breakers, state machines, audit trails), stores each as a 384-dimensional vector embedding in semantic memory, then generates prioritized enhancement tasks for CLAW itself.

- **Runtime:** ~1-3 minutes
- **LLM calls:** 1 (mining analysis)
- **Shows:** Repo mining, semantic memory with real embeddings, task generation with agent routing

## Demo 3: The Gauntlet

**Full autonomous cycle with verification gate.**

```bash
./demos/03-the-gauntlet.sh
```

A goal is injected, then CLAW runs the complete pipeline: grab the task, query memory for hints, route to the best agent via Bayesian scoring, execute via OpenRouter, run the 7-check verification gate (dependency jail, chaos check, placeholder scan, drift alignment via cosine similarity, claim validation), then update agent scores and save patterns.

- **Runtime:** ~2-4 minutes
- **LLM calls:** 2-3 (dispatch + verification)
- **Shows:** Full grab-evaluate-decide-act-verify-learn cycle, 7-check gate, learning feedback

## What makes these different

| Feature | ChatGPT wrapper | CLAW |
|---------|----------------|------|
| Models | 1 | 4 agents, Bayesian routing |
| Verification | None | 7-check gate with semantic drift detection |
| Memory | None | 384-dim embeddings, cross-repo knowledge transfer |
| Learning | None | Agent scores update after every task |
| Cross-repo | No | Mines patterns from repo A, applies to repo B |
| Self-referential | No | Evaluates and improves its own code |

## Run all three

```bash
./demos/01-the-mirror.sh && ./demos/02-the-heist.sh && ./demos/03-the-gauntlet.sh
```

Each demo resets the database for a clean run. Total time: ~8-12 minutes.
