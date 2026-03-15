# CAM Example Workflows

This document shows real command examples and representative outputs.

The goal is to make CAM concrete.

## Workflow 1: Fresh Clone Smoke Test

Command:

```bash
.venv/bin/cam --help
.venv/bin/cam govern stats
```

Observed output excerpt:

```text
Memory Governance Stats
  Total methodologies:  1469
  Active (non-dead):    1469
  Quota: 1469/2000 (73.5%)
  DB size: 50.25 MB
  Episodes: 3
```

What this shows:
- CLI installed
- DB initialized
- runtime healthy enough to proceed

## Workflow 2: Preview Repo Mining Without Spending Model Calls

Command:

```bash
.venv/bin/cam mine tests/fixtures/embedding_forge --scan-only --depth 3 --max-repos 5
```

Observed output excerpt:

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

What this shows:
- CAM can discover a source-tree style repo
- you can preview repo selection before making model calls

## Workflow 3: Export CAM Knowledge For Standalone Use

Command:

```bash
.venv/bin/cam forge-export \
  --out data/cam_knowledge_pack_docs_example.jsonl \
  --max-methodologies 5 \
  --max-tasks 5 \
  --max-minutes 1
```

Observed output:

```text
CAM Forge Export
  Database: data/claw.db
  Out: data/cam_knowledge_pack_docs_example.jsonl
  Max methodologies: 5
  Max tasks: 5
  Time guardrail: 1 minute(s)

Knowledge pack exported.
  Total: 10
  Methodologies: 5
  Tasks: 5
  File: data/cam_knowledge_pack_docs_example.jsonl
```

What this shows:
- CAM can package learned knowledge into a neutral JSONL file
- the pack can include both methodologies and tasks

Observed file facts:

```text
10 data/cam_knowledge_pack_docs_example.jsonl
```

Example knowledge-pack entries:

```json
{"id": "meth:261b6e4d-cc93-4e65-8db2-7bae4bbc4044", "title": "[Mined from autoresearch-macos] Soft-Capped Logit Normalization...", "modality": "memory_methodology", ...}
{"id": "meth:b7923a94-98c0-41e8-b1d7-a185d8e7ff60", "title": "[Mined from autoresearch-macos] BOS-Aligned Best-Fit Data Packing...", "modality": "memory_methodology", ...}
```

What this means in practice:
- CAM is not only storing knowledge inside its own DB
- it can hand that knowledge to a separate application

## Workflow 4: Create A Spec-Backed Task

Command:

```bash
mkdir -p tmp/doc-example-app
.venv/bin/cam create tmp/doc-example-app \
  --repo-mode augment \
  --request "Create a minimal example repo skeleton for docs validation." \
  --spec "Must contain a README.md file" \
  --check "python -V" \
  --no-preview \
  --max-minutes 1
```

Observed output excerpt:

```text
CAM Create
  Repo: /Users/o2satz/multiclaw/tmp/doc-example-app
  Mode: augment
  Spec file: /Users/o2satz/multiclaw/data/create_specs/20260315-074936-doc-example-app-create-spec.json
  Purpose: convert CAM memory + your request into an executable creation task

Quickstart goal created.
  Task ID: 36f91f65-1323-455c-b1ab-cc4d09b30ea5
  Project: doc-example-app
  Agent: claude
  Priority: high (8)
```

What this shows:
- `create` writes a real spec file
- `create` also creates a real task in CAM’s database
- CAM has now recorded the requested outcome in a form that can be validated later

## Workflow 5: Validate Against The Saved Spec

After the `create` step above, a `README.md` file was added to the repo and validation was run.

Command:

```bash
printf '# Doc Example App\n\nThis repo exists to demonstrate CAM validation.\n' > tmp/doc-example-app/README.md
.venv/bin/cam validate --spec-file data/create_specs/20260315-074936-doc-example-app-create-spec.json --max-minutes 1
```

Observed output:

```text
CAM Validate
  Spec file: /Users/o2satz/multiclaw/data/create_specs/20260315-074936-doc-example-app-create-spec.json
  Repo: /Users/o2satz/multiclaw/tmp/doc-example-app
  Checks run: 1

Validation passed.
  OK python -V
```

What this shows:
- validation reads the saved spec file
- validation checks the actual target repo
- validation runs acceptance checks
- a repo can move from requested state to validated state with a concrete check result

## Workflow 6: Standalone Forge Benchmark

Command:

```bash
.venv/bin/cam forge-benchmark --max-minutes 1
```

Observed output:

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

What this shows:
- benchmark execution is real and local
- results are written to a real summary artifact
- CAM is reporting the benchmark honestly, even when the best result only matches baseline

## What These Workflows Add Up To

Taken together, these examples show that CAM can already do a meaningful subset of the promised operator loop:

- initialize from a fresh clone
- inspect and enumerate repos
- export learned memory for external use
- turn a requested outcome into a spec-backed task
- validate the target repo against that spec
- benchmark a standalone downstream path

What they do not yet prove is universal autonomous app creation. That remains the harder open problem.
