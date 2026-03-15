# CAM Proven Capabilities

This document is the evidence-oriented companion to the README.

Its job is to answer four questions:

1. what CAM is trying to do
2. what CAM has actually been shown to do
3. what command lines were used
4. where the current limits still are

## What CAM Is Trying To Be

CAM is a repo operator with memory.

That means:
- it can inspect a repo
- it can learn transferable patterns from other repos
- it can propose new app ideas from that learning
- it can create a spec-backed task against a target repo
- it can validate outcomes instead of trusting agent self-report

The important distinction is that CAM is not supposed to be a passive knowledge notebook. It is supposed to help build, fix, and create.

## Proven Areas

## 1. Fresh-clone bootstrap works

Verified flow:

```bash
git clone https://github.com/deesatzed/clawamorphosis.git
cd clawamorphosis
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/cam --help
.venv/bin/cam govern stats
```

What this proves:
- the package installs from a clean clone
- the CLI entrypoint works
- the database can initialize on a fresh clone

## 2. CAM can discover extracted source trees, not just real git clones

Verified command:

```bash
.venv/bin/cam mine tests/fixtures/embedding_forge --scan-only --depth 3 --max-repos 5
```

Observed result:

```text
CLAW Repo Scanner (scan-only)
  Directory: /Users/o2satz/multiclaw/tests/fixtures/embedding_forge
  Depth: 3
  Dedup: True

Scanning for repos...
Discovered Repos (1 total, 1 selected)
...
Summary
  Total discovered: 1
  Selected: 1
  Skipped (dedup): 0
  Will mine: 1
```

Why it matters:
- a lot of candidate repos arrive as zip downloads
- CAM no longer requires `.git` metadata just to study them

## 3. CAM can benchmark the standalone Forge path locally

Verified command:

```bash
.venv/bin/cam forge-benchmark --max-minutes 1
```

Observed result:

```text
CAM Forge Benchmark
  Repo: tests/fixtures/embedding_forge/repo
  Note: tests/fixtures/embedding_forge/note.md
  Knowledge pack: tests/fixtures/embedding_forge/knowledge_pack.jsonl
  Out: data/forge_benchmark_fixture
  Time guardrail: 1 minute(s)

Benchmark complete.
  Status: pass
  Docs: 7
  Best lift: 0.00%
  Best config: anchor_dim=8 residual_dim=8 anchor_weight=1.2 residual_weight=0.8
  Summary: data/forge_benchmark_fixture/benchmark_summary.json
```

Why it matters:
- benchmark execution is real and reproducible
- CAM reports the current quality honestly
- current fixture result is non-catastrophic, but not a clear uplift over baseline

## 4. CAM rejects fake execution success when no files changed

Backed by tests in:
- [tests/test_cycle.py](../tests/test_cycle.py)
- [tests/test_create_benchmark_spec.py](../tests/test_create_benchmark_spec.py)

What is enforced:
- if an agent claims success but the target workspace is unchanged, CAM marks the run as failed
- validation also fails if the repo remains unchanged since spec creation

Why it matters:
- this is one of the most important anti-vaporware checks in the repo

## 5. CAM `ideate` is more robust against imperfect model JSON output

Backed by tests in:
- [tests/test_llm.py](../tests/test_llm.py)

What is enforced:
- JSON parsing now recovers from raw control characters inside string fields, such as literal newlines or tabs emitted by a model

Why it matters:
- ideation commands fail less often on otherwise-usable model output

## 6. Create-spec validation is a first-class step, not an afterthought

Backed by tests in:
- [tests/test_create_benchmark_spec.py](../tests/test_create_benchmark_spec.py)

What is enforced:
- spec files are created deterministically
- executable acceptance checks run during validation
- plain-English checks are treated as manual checks instead of being executed as shell commands
- unchanged repos fail validation

## Current Verified Test Slice

Command run on March 15, 2026:

```bash
pytest -q tests/test_llm.py tests/test_miner.py tests/test_db.py tests/test_create_benchmark_spec.py tests/test_embedding_forge_benchmark.py tests/test_cli_ux.py
```

Observed result:

```text
159 passed in 0.43s
```

Coverage represented by that slice:
- CLI command coverage
- miner behavior
- database bootstrap and schema behavior
- create/validate helper behavior
- Forge benchmark harness
- ideate parser hardening

## What CAM Can Accomplish Today

With correct model/API configuration, CAM is currently positioned to do these classes of work:

- repo evaluation and triage
- bounded repo enhancement workflows
- repo-fleet mining and knowledge extraction
- app ideation from cross-repo synthesis
- spec-backed repo creation orchestration
- validation of created repos against explicit checks
- knowledge export for standalone downstream apps

## What CAM Does Not Yet Prove

These are the important non-claims.

- CAM does not yet prove that `create --execute` will autonomously build any requested app end-to-end without supervision.
- CAM does not yet prove positive retrieval lift for standalone Forge on the fixture corpus.
- CAM does not yet prove that every ideated app concept is implementable or worthwhile.
- CAM does not yet prove product-market fit for any generated app concept.

Those gaps are not hidden. They are the next engineering targets.

## Suggested Reading Order

1. [README.md](../README.md) for the public overview
2. [CAM_COMMAND_GUIDE.md](CAM_COMMAND_GUIDE.md) for command-by-command usage
3. [CAM_BEGINNER_ASSIMILATION_GUIDE.md](CAM_BEGINNER_ASSIMILATION_GUIDE.md) for learning/build workflows
