# CAM

CAM is a multi-agent codebase worker for three jobs:

1. inspect a repository
2. mine patterns from other repositories
3. create or modify a target repository, then validate the result

This repository is documented for a new operator, not as an internal lab notebook.

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

Note:

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

## First Run

Evaluate a repository:

```bash
.venv/bin/cam evaluate /path/to/repo
```

Mine two source repos to learn from them:

```bash
.venv/bin/cam mine /path/to/source-repos \
  --target /path/to/target-repo \
  --max-repos 2 \
  --depth 2 \
  --max-minutes 15
```

Create or modify a target repo from that learned context:

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

Validate the result:

```bash
.venv/bin/cam validate --spec-file data/create_specs/<spec-file>.json --max-minutes 5
```

Benchmark only after validation passes:

```bash
.venv/bin/cam benchmark --max-minutes 5
```

Export learned knowledge for a standalone app:

```bash
.venv/bin/cam forge-export \
  --out data/cam_knowledge_pack.jsonl \
  --max-methodologies 200 \
  --max-tasks 200 \
  --max-minutes 5
```

## Core Commands

| Command | Purpose |
| --- | --- |
| `cam setup` | Configure agent keys, models, budgets, and defaults |
| `cam evaluate <repo>` | Inspect one repository and produce findings |
| `cam enhance <repo>` | Run the full improve-and-verify loop on one repository |
| `cam mine <dir>` | Learn from repositories in a directory |
| `cam create <repo>` | Create, augment, or fix a repository from a task request |
| `cam validate` | Check the created result against its saved spec |
| `cam benchmark` | Measure output quality after validation |
| `cam forge-export` | Export CAM knowledge into a neutral knowledge pack |
| `cam results` | Inspect past task outcomes |
| `cam status` | Inspect system and budget status |

## Command Reference

For the full command-by-command guide with syntax, behavior, and example use cases, see [docs/CAM_COMMAND_GUIDE.md](/Users/o2satz/multiclaw/docs/CAM_COMMAND_GUIDE.md).

## Notes

- CAM uses minute-based wall-clock guardrails on mining, create, validate, and benchmark flows.
- `cam create --execute` rejects runs that report success without real workspace file changes.
- Runtime data is written under `data/` and is not committed.

## Development

Run tests:

```bash
.venv/bin/pytest
```

Show CLI help:

```bash
.venv/bin/cam --help
```
