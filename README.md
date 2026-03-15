# CAM

CAM is a multi-agent codebase worker built around a simple idea:

1. inspect a repository
2. learn from other repositories
3. turn that learning into concrete repo work
4. validate the result instead of trusting agent narration

This repository is documented for a new operator, not as an internal lab notebook.

## What Is Novel About CAM

CAM is not just a wrapper around a chat model. The distinctive parts are the workflow and the safety checks around it.

- `mine` turns outside repos into reusable CAM memory instead of one-off notes.
- `ideate` combines stored CAM knowledge with candidate repos to propose new standalone app concepts.
- `create` writes a real creation spec, not just a prompt, so the requested outcome is explicit and reviewable.
- `validate` checks the created repo against the saved spec and acceptance rules.
- `create --execute` no longer trusts an agent saying "I changed files". CAM now checks the actual workspace diff and marks the run as failed if nothing changed.
- `forge-export` lets CAM hand off what it knows as a neutral JSONL knowledge pack, so a standalone app can consume CAM’s knowledge without importing CAM itself.
- `mine` can detect extracted source trees even when they are not full `.git` clones, which matters when you are evaluating zip-downloaded repos.

The practical result is that CAM is designed to help with real repo work, not just repo discussion.

## What CAM Can Do Right Now

Today CAM can:

- evaluate one repo and decide what looks worth improving
- mine a folder of repos and store transferable patterns in CAM memory
- search and inspect what CAM has already learned
- ideate novel app concepts using both stored CAM knowledge and candidate repos
- create a spec-backed task for a fixed repo, augmented repo, or new repo
- validate whether a created repo actually changed and whether executable checks passed
- export CAM knowledge into a standalone knowledge pack
- run a deterministic standalone Forge benchmark on fixture data

## What Has Been Proven, Not Just Claimed

The items below are backed either by direct command runs in this repo or by targeted automated tests.

### Proven by direct command execution

As of March 15, 2026, these commands were run successfully in this repo:

- fresh-clone smoke test path:
  - `.venv/bin/cam --help`
  - `.venv/bin/cam govern stats`
- source-tree scan path:
  - `.venv/bin/cam mine tests/fixtures/embedding_forge --scan-only --depth 3 --max-repos 5`
- standalone benchmark path:
  - `.venv/bin/cam forge-benchmark --max-minutes 1`

### Proven by targeted automated tests

A targeted verification run passed on March 15, 2026:

```bash
pytest -q tests/test_llm.py tests/test_miner.py tests/test_db.py tests/test_create_benchmark_spec.py tests/test_embedding_forge_benchmark.py tests/test_cli_ux.py
```

Result:

```text
159 passed in 0.43s
```

That test slice covers:

- CLI command surface and UX
- fresh-database bootstrap and migrations
- source-tree mining behavior
- create-spec generation and validation logic
- rejection of unchanged repo executions
- standalone Forge regression benchmark
- resilient JSON parsing for `cam ideate`

## What CAM Has Explicitly Been Hardened Against

These are important because they are easy places for agent systems to become fake.

- False success reports from agents with no real file changes
  - CAM now detects this and marks the execution as failed.
- Fresh-clone DB bootstrap failures
  - fixed and retested from a clean clone.
- `ideate` crashing on imperfect JSON-like model output
  - parser hardened to recover from raw control characters inside JSON strings.
- Zip-downloaded repo folders being invisible to `mine`
  - CAM can now discover source-tree style repos without `.git` metadata.

## Honest Limits

CAM is strong as an operator and orchestrator, but it is not magic.

- `cam create --execute` is safer than before, but it is not yet a guaranteed autonomous app-builder.
- `validate` proves basic correctness against the saved spec and checks; it does not prove product quality by itself.
- `benchmark` is only as strong as the benchmark corpus and metrics you feed it.
- standalone Forge currently has a real benchmark harness, but not yet a proven positive retrieval lift on the fixture corpus. The current best fixture run matches baseline rather than beating it.

That last point matters. CAM is built to fail honestly instead of pretending the benchmark improved when it did not.

## Requirements

- Python 3.12+
- `git`
- API access for the models you plan to use
- enough local disk for `data/claw.db`

## Install

```bash
git clone https://github.com/deesatzed/clawamorphosis.git
cd clawamorphosis
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Notes:

- install is heavy because CAM currently pulls ML dependencies such as `torch`, `transformers`, and `sentence-transformers`
- if shell activation is unreliable in your environment, you can run CAM directly as `.venv/bin/cam`

## Smoke Test

Use these exact commands first:

```bash
.venv/bin/cam --help
.venv/bin/cam govern stats
```

What this proves:

- the CLI is installed
- the database can initialize on a fresh clone
- the local runtime is basically healthy

Example output from `cam govern stats`:

```text
Memory Governance Stats
  Total methodologies:  1469
  Active (non-dead):    1469
  Quota: 1469/2000 (73.5%)
  DB size: 50.25 MB
  Episodes: 3
```

Your numbers will differ. The point is that the command should complete cleanly and print memory/database stats instead of crashing.

## Configure

Use the interactive setup first:

```bash
.venv/bin/cam setup
```

Or export the keys you need before running CAM:

```bash
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

## First Practical Workflows

### 1. Review one repo before touching it

```bash
.venv/bin/cam evaluate /path/to/repo --mode quick
.venv/bin/cam enhance /path/to/repo --dry-run
```

Use this when you want CAM to tell you what it would change before it tries to change anything.

### 2. Learn from outside repos

```bash
.venv/bin/cam mine /path/to/source-repos \
  --target /path/to/target-repo \
  --max-repos 2 \
  --depth 2 \
  --max-minutes 15
```

Use this when you want CAM to extract reusable patterns from outside repos and make that knowledge available to a target project.

### 3. Invent new app ideas from CAM memory plus repo inputs

```bash
.venv/bin/cam ideate /path/to/source-repos \
  --focus "Invent useful standalone apps that build, troubleshoot, or create" \
  --ideas 3 \
  --max-repos 4 \
  --max-minutes 10
```

Use this when you want CAM to propose new product directions, not just summarize repos.

### 4. Create or modify a target repo from that context

```bash
.venv/bin/cam create /path/to/target-repo \
  --repo-mode new \
  --request "Build the app I described" \
  --spec "Must be standalone" \
  --check "pytest -q" \
  --max-minutes 20
```

If you want CAM to attempt execution immediately:

```bash
.venv/bin/cam create /path/to/target-repo \
  --repo-mode new \
  --request "Build the app I described" \
  --check "pytest -q" \
  --execute \
  --max-minutes 20
```

### 5. Validate the result before trusting it

```bash
.venv/bin/cam validate --spec-file data/create_specs/<spec-file>.json --max-minutes 5
```

This is the line between "the agent said it did it" and "the repo actually matches the requested spec closely enough to pass checks".

### 6. Benchmark only after validation passes

```bash
.venv/bin/cam benchmark --max-minutes 5
```

### 7. Export learned knowledge for a standalone app

```bash
.venv/bin/cam forge-export \
  --out data/cam_knowledge_pack.jsonl \
  --max-methodologies 200 \
  --max-tasks 200 \
  --max-minutes 5
```

## Reproducible Example Outputs

### Example: scan-only repo discovery without spending model calls

Command:

```bash
.venv/bin/cam mine tests/fixtures/embedding_forge --scan-only --depth 3 --max-repos 5
```

Observed output:

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

Remove --scan-only to mine these repos.
```

What this proves:
- CAM can discover a source-tree style repo from fixture data
- you can preview repo discovery before spending tokens

### Example: standalone Forge benchmark

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

What this proves:
- the benchmark harness runs locally on repo-contained fixture data
- the current best fixture configuration passes the catastrophic regression floor
- CAM is not overstating results: on this fixture run, best lift matched baseline rather than beating it

## Why This Matters

A lot of agent systems can talk about repos. CAM is trying to be useful in a harder way:

- learn from repo fleets
- turn that learning into explicit tasks and specs
- create or modify codebases against those specs
- fail honestly when execution did not really happen
- export what it learned so a separate app can consume it

That is the core build.

## Core Commands

| Command | Purpose |
| --- | --- |
| `cam setup` | Configure agent keys, models, budgets, and defaults |
| `cam evaluate <repo>` | Inspect one repository and produce findings |
| `cam enhance <repo>` | Run the full improve-and-verify loop on one repository |
| `cam mine <dir>` | Learn from repositories in a directory |
| `cam ideate <dir>` | Generate novel standalone app concepts from CAM memory plus repo inputs |
| `cam create <repo>` | Create, augment, or fix a repository from a task request |
| `cam validate` | Check the created result against its saved spec |
| `cam benchmark` | Measure output quality after validation |
| `cam forge-export` | Export CAM knowledge into a neutral knowledge pack |
| `cam forge-benchmark` | Run the standalone Forge regression benchmark |
| `cam results` | Inspect past task outcomes |
| `cam status` | Inspect system and budget status |
| `cam kb ...` | Search and inspect what CAM has learned |

## Documentation Map

- Full command-by-command reference: [docs/CAM_COMMAND_GUIDE.md](docs/CAM_COMMAND_GUIDE.md)
- Beginner assimilation walkthrough: [docs/CAM_BEGINNER_ASSIMILATION_GUIDE.md](docs/CAM_BEGINNER_ASSIMILATION_GUIDE.md)
- Proven capabilities and example transcripts: [docs/CAM_PROVEN_CAPABILITIES.md](docs/CAM_PROVEN_CAPABILITIES.md)
- Short operator quick-reference: [docs/CAM_OPERATOR_CHEATSHEET.md](docs/CAM_OPERATOR_CHEATSHEET.md)
- End-to-end example workflows and outputs: [docs/CAM_EXAMPLE_WORKFLOWS.md](docs/CAM_EXAMPLE_WORKFLOWS.md)

## Development

Run the targeted verification slice used for the claims above:

```bash
.venv/bin/pytest -q tests/test_llm.py tests/test_miner.py tests/test_db.py tests/test_create_benchmark_spec.py tests/test_embedding_forge_benchmark.py tests/test_cli_ux.py
```

Or run the full suite:

```bash
.venv/bin/pytest
```

Show CLI help:

```bash
.venv/bin/cam --help
```
