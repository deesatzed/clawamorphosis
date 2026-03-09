# CLAW Coverage Action Plan

**Date:** 2026-03-03
**Current Coverage:** 79% (1,079 tests, 5,910 statements, 1,241 missed)
**Previous:** 74% (825 tests) → 79% after Tier 2 round (+254 tests)
**Target:** 100% (per project policy, gaps need action plan unless waived)

## Modules at 100% (No Action Needed)

| Module | Stmts |
|--------|-------|
| `__init__.py` (multiple) | 1 |
| `core/exceptions.py` | 47 |
| `core/models.py` | 234 |
| `budget.py` | 101 |
| `degradation.py` | 111 |
| `memory/meta.py` | 118 |
| `memory/fitness.py` | 45 |
| `orchestrator/loop_guard.py` | 20 |

## Modules at 90%+ (Minimal Gaps — Defensive Code)

| Module | Coverage | Missing |
|--------|----------|---------|
| `cycle.py` | 97% | Exception handler in `run_cycle()` |
| `db/repository.py` | 98% | 5 defensive None guards |
| `db/embeddings.py` | 96% | Model download failure |
| `orchestrator/adaptation.py` | 97% | Exception handler in `from_task()` |
| `orchestrator/health_monitor.py` | 94% | Timezone-aware branch, circuit reset logging |
| `memory/hybrid_search.py` | 95% | Text-only fallback edge case |
| `memory/episodic.py` | 95% | Retention policy edge case |
| `memory/semantic.py` | 91% | Exception handlers in record_* methods |
| `orchestrator/budget_hints.py` | 95% | Single edge case line |
| `orchestrator/complexity.py` | 94% | Two keyword edge cases |
| `orchestrator/diagnostics.py` | 96% | 4 minor lines |
| `orchestrator/metrics.py` | 99% | 1 line |
| `evolution/capability_disc.py` | 97% | JSON parse fallback |
| `evolution/routing_optimizer.py` | 95% | Exploration fallback |
| `core/config.py` | 90% | Config parse error paths |
| `evolution/pattern_learner.py` | 90% | Pattern extraction edge cases |

## Tier 1: Requires External Services (API Keys / CLI Tools)

These modules contain code paths that execute real API calls or CLI subprocesses. Testing requires live external services.

| Module | Coverage | Missing | Blocker |
|--------|----------|---------|---------|
| `cli.py` | 0% | 108 lines | `ClawFactory.create()` + real DB path. **Self-referential test confirmed working manually.** |
| `llm/client.py` | 49% | 90 lines | Real HTTP calls to OpenRouter API. Requires `OPENROUTER_API_KEY`. |
| `agents/claude.py` | 53% | 70 lines | CLI/API execution paths. Requires `ANTHROPIC_API_KEY` or `claude` CLI. |
| `agents/codex.py` | 51% | 87 lines | CLI/API/Cloud execution. Requires `OPENAI_API_KEY` or `codex` CLI. |
| `agents/gemini.py` | 59% | 79 lines | CLI/API execution. Requires `GOOGLE_API_KEY` or `gemini` CLI. |
| `agents/grok.py` | 52% | 76 lines | CLI/API execution. Requires `XAI_API_KEY` or `grok` CLI. |
| `core/factory.py` | 49% | 35 lines | `ClawFactory.create()` integration wiring (agent creation). |
| `evaluator.py` | 84% | 28 lines | Dispatcher execution path with live agents. |

**Total Tier 1:** 573 lines (46% of all uncovered lines)

**Action:** Create `tests/test_integration.py` with `@requires_*` skip markers. Tests run automatically when API keys are available, skipped otherwise. Each agent gets a health_check + simple execute integration test.

## Tier 2: Remaining Testable Gaps

| Module | Coverage | Gap Description | Action |
|--------|----------|-----------------|--------|
| `dashboard.py` | 66% | Plain-text fallback paths (Rich always installed), fleet status detail | Test with Rich disabled or fleet repo data |
| `fleet.py` | 79% | Git operations (branch creation, scanning edge cases) | More tmp_path git repo tests |
| `mcp_server.py` | 57% | Deeper handler paths (semantic_memory integration, workspace scanning) | Wire SemanticMemory into MCP tests |
| `verifier.py` | 71% | Full LLM deep review, complex verification paths | Requires agent for deep review |
| `security/policy.py` | 84% | Rate limit overflow, workspace boundary edge cases | Additional policy tests |
| `memory/lifecycle.py` | 77% | `check_niche_collision()` body (requires sqlite-vec vectors) | Depends on sqlite-vec in test env |
| `planner.py` | 84% | Gap analysis with real evaluation data | Wire evaluator output into planner |
| `memory/error_kb.py` | 85% | Cross-agent failure patterns, enriched forbidden | More multi-agent scenario tests |
| `agents/interface.py` | 85% | `run()` method execution | Covered by cycle tests indirectly |
| `llm/token_tracker.py` | 82% | JSONL persistence, DB persistence | Persist integration test |
| `db/engine.py` | 76% | Error paths, vec extension fallback | Engine error handling tests |
| `dispatcher.py` | 87% | Bayesian routing with real scores | Integration with meta memory |
| `orchestrator/arbitration.py` | 88% | Multi-candidate scoring | More arbitration scenarios |
| `evolution/prompt_evolver.py` | 86% | `evolve_prompt()` full pipeline | Full evolution pipeline test |

**Total Tier 2:** 525 lines (42% of all uncovered lines)

## Tier 3: Defensive / Unreachable Code

| Module | Lines | Description |
|--------|-------|-------------|
| Various `except Exception` handlers | ~50 | Standard error handling, infrastructure failure paths |
| `cycle.py:88-91` | 4 | Exception in run_cycle (all paths tested individually) |
| `repository.py:794,797-798` | 3 | `_parse_dt` edge cases for malformed datetimes |
| `adaptation.py:55-56` | 2 | Exception handler in `from_task()` |

**Total Tier 3:** ~143 lines (12% of all uncovered lines)

**Action:** Request waiver for Tier 3 defensive code paths. These require deliberately breaking system resources to trigger.

## Coverage Summary

| Category | Lines Missed | % of Total Missed |
|----------|-------------|-------------------|
| Tier 1 (External Services) | 573 | 46% |
| Tier 2 (Testable Gaps) | 525 | 42% |
| Tier 3 (Defensive Code) | 143 | 12% |
| **Total** | **1,241** | **100%** |

## Test Suite Progress

| Phase | Tests Added | Running Total |
|-------|------------|---------------|
| Phase 1 | 97 | 97 |
| Phase 2 | 296 | 393 |
| Phase 3 | 159 | 552 |
| Phase 4 | 137 | 689 |
| Phase 5 | 136 | 825 |
| Tier 2 Round 1 | 254 | 1,079 |

## Self-Referential Test (Step 5.4)

**Status: PASSED**

```
$ python -m claw.cli evaluate .
CLAW Evaluation: multiclaw
  Repository: /Users/o2satz/multiclaw
  Project ID: 060ff7e1-cd06-4745-af7a-9a898b701af9
  Database: data/claw.db
  Total Files: 376
  Language: Python
  Evaluation stored in data/claw.db
```

CLAW successfully evaluated its own codebase, stored results in SQLite, and closed cleanly.
