# CAM Command Guide

This guide documents the current `cam` CLI as implemented in the repository. For each command, it explains:
- what the command is for
- what it actually does
- the basic syntax
- one concrete example use case

For evidence-backed examples, tested claims, and real command transcripts, also see [CAM_PROVEN_CAPABILITIES.md](CAM_PROVEN_CAPABILITIES.md).

## Before You Start

From the repo root:

```bash
cd /Users/o2satz/multiclaw
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Basic smoke test:

```bash
.venv/bin/cam --help
.venv/bin/cam govern stats
```

If you prefer not to activate the shell environment, use `.venv/bin/cam ...` directly.

## Mental Model

CAM has four main jobs:

1. Evaluate codebases
CAM studies a repo and decides what is worth changing.

2. Create or improve code
CAM can add goals, create a new task/spec, and execute work against a repo.

3. Learn from other repos
CAM can mine folders of repos, store reusable patterns, and let you search that knowledge later.

4. Validate and benchmark
CAM can check whether a created repo matches its requested spec and whether a Forge-style output performs acceptably.

## Quick Workflow Map

### If you want CAM to study a repo and improve it

```bash
cam evaluate /path/to/repo
cam enhance /path/to/repo --dry-run
cam enhance /path/to/repo
```

### If you want CAM to study outside repos and create a new app

```bash
cam mine /path/to/repo-folder --target /path/to/new-app --max-repos 4
cam ideate /path/to/repo-folder --ideas 3
cam create /path/to/new-app --repo-mode new --request "Build the chosen app"
cam validate --spec-file data/create_specs/<spec-file>.json
```

### If you want CAM to improve itself from outside repos

```bash
cam mine /path/to/repo-folder --target /Users/o2satz/multiclaw --max-repos 4
cam kb insights
cam kb search "test generation"
```

## Top-Level Commands

## `cam evaluate`

Purpose:
Analyze a single repository and score its enhancement potential.

What it does:
- runs structural repo analysis
- can run a full or partial evaluation battery through agents
- stores evaluation results in the SQLite database
- helps CAM decide what is worth working on next

Syntax:

```bash
cam evaluate <repo> [--mode auto|full|quick|structural] [--config claw.toml]
```

Modes:
- `auto`: use full evaluation if agents are configured, otherwise structural only
- `full`: structural analysis plus the full evaluation battery
- `quick`: structural analysis plus a smaller prompt set
- `structural`: no agent calls; just inspect the repo structure

Example use case:
You found a repo and want to know if CAM thinks it is worth improving.

```bash
cam evaluate /Users/o2satz/projects/my-app --mode quick
```

## `cam enhance`

Purpose:
Run CAM’s main improvement loop on one repository.

What it does:
- evaluates the repo
- plans tasks
- dispatches tasks to agents
- verifies outcomes
- records results back into CAM memory

Syntax:

```bash
cam enhance <repo> [--mode attended|supervised|autonomous] [--max-tasks N] [--battery] [--dry-run]
```

Key options:
- `--mode`: autonomy level
- `--max-tasks`: cap the number of tasks processed
- `--battery`: use the full evaluation battery first
- `--dry-run`: preview tasks without executing changes

Example use case:
You want CAM to propose and then execute a bounded improvement pass on an app.

```bash
cam enhance /Users/o2satz/projects/my-app --mode attended --max-tasks 3
```

## `cam fleet-enhance`

Purpose:
Run enhancement across many repos in one folder.

What it does:
- scans a directory for repos
- ranks them by enhancement potential
- allocates budget across the fleet
- runs enhancement in ranked order
- works on branches, not directly on `main`

Syntax:

```bash
cam fleet-enhance <repos_dir> [--mode supervised] [--max-repos N] [--max-tasks N] [--budget USD] [--strategy proportional|equal]
```

Example use case:
You have 20 internal repos and want CAM to spend effort only on the highest-value ones.

```bash
cam fleet-enhance /Users/o2satz/workspace/repos --max-repos 5 --budget 30
```

## `cam results`

Purpose:
Show prior task execution results stored in the database.

What it does:
- lists recent task outcomes
- lets you filter by project
- helps you review what CAM already tried

Syntax:

```bash
cam results [--limit N] [--project PROJECT_ID]
```

Example use case:
You want to inspect recent CAM executions after a session.

```bash
cam results --limit 10
```

## `cam status`

Purpose:
Show CAM system status.

What it does:
- prints overall CAM runtime/database status
- useful as a quick health check

Syntax:

```bash
cam status
```

Example use case:
You want to confirm CAM is configured and the database is reachable.

```bash
cam status
```

## `cam runbook`

Purpose:
Inspect the planned execution steps for a task.

What it does:
- shows the task’s execution steps
- shows acceptance checks
- helps you inspect what CAM intends to do before running it

Syntax:

```bash
cam runbook <task_id>
```

Example use case:
You created a task and want to inspect its plan before execution.

```bash
cam runbook task_abc123
```

## `cam quickstart`

Purpose:
Create a goal quickly, preview the runbook, and optionally execute immediately.

What it does:
- creates a task for a repo
- lets you attach steps and checks
- previews the runbook
- can execute the task immediately

Syntax:

```bash
cam quickstart <repo> --title "..." --description "..." [--type bug_fix] [--step "..."] [--check "..."] [--execute]
```

Example use case:
You want a fast path to tell CAM, “fix this thing,” without manually building the task structure.

```bash
cam quickstart /Users/o2satz/projects/my-app \
  --title "Repair failing tests" \
  --description "Fix the broken auth tests and restore green CI" \
  --check "pytest -q" \
  --preview
```

## `cam create`

Purpose:
Create a fixed repo, augmented repo, or brand-new repo from a requested outcome.

What it does:
- writes a creation spec JSON under `data/create_specs/`
- creates a real CAM task tied to the target repo
- can preview the runbook
- can execute the task immediately
- can use prior mined CAM knowledge when relevant

Syntax:

```bash
cam create <repo> --request "..." [--repo-mode fixed|augment|new] [--spec "..."] [--step "..."] [--check "..."] [--execute] [--max-minutes N]
```

Repo modes:
- `fixed`: repair an existing repo
- `augment`: add capabilities to an existing repo
- `new`: create a new repo/project outcome

Example use case:
You want CAM to build a new standalone app using prior mined knowledge.

```bash
cam create /Users/o2satz/projects/embedding-worker \
  --repo-mode new \
  --request "Create a standalone CLI that reads a CAM knowledge pack and proposes finetuning jobs for small models" \
  --spec "Must be standalone" \
  --spec "Must not import CAM runtime code" \
  --check "pytest -q" \
  --max-minutes 20
```

## `cam add-goal`

Purpose:
Add a custom goal/task to a repository without going through full creation flow.

What it does:
- creates a task in CAM’s database
- records title, description, type, priority, steps, and checks
- is intended to be picked up by later enhancement runs

Syntax:

```bash
cam add-goal <repo> --title "..." --description "..." [--type analysis|testing|documentation|security|refactoring|bug_fix|architecture|dependency_analysis] [--step "..."] [--check "..."]
```

Example use case:
You know exactly what CAM should investigate next and want that stored as a task.

```bash
cam add-goal /Users/o2satz/multiclaw \
  --title "Evaluate finetuning path" \
  --description "Design a practical small-model finetuning path for CAM where it is clearly worth the cost" \
  --type architecture
```

## `cam ideate`

Purpose:
Generate novel app ideas using CAM’s stored knowledge plus candidate repos in a folder.

What it does:
- discovers repos or source trees in a directory
- pulls repo-specific findings CAM already mined
- pulls high-potential and high-novelty CAM methodologies
- asks an LLM for new standalone app concepts
- writes JSON and Markdown ideation artifacts to `data/ideation/`
- can optionally promote one idea into a real `cam create` task/spec

Syntax:

```bash
cam ideate <directory> [--focus "..."] [--ideas 3] [--max-repos 4] [--depth 3] [--agent claude|codex|gemini|grok] [--promote N --target-repo /path/to/repo --repo-mode new] [--max-minutes N]
```

Example use case:
You want CAM to propose three new app concepts from a folder of candidate repos.

```bash
cam ideate /Users/o2satz/multiclaw/Repo2Eval \
  --focus "Invent useful standalone apps that combine CAM knowledge with these repos" \
  --ideas 3 \
  --max-repos 4
```

Example with promotion:

```bash
cam ideate /Users/o2satz/multiclaw/Repo2Eval \
  --focus "Invent a standout standalone app around finetuning or build automation" \
  --ideas 3 \
  --promote 1 \
  --target-repo /Users/o2satz/projects/new-app-from-cam \
  --repo-mode new
```

## `cam mine`

Purpose:
Study a folder of repos and extract reusable patterns, features, and ideas.

What it does:
- scans a directory for git repos and source-tree style repos
- analyzes each repo via LLM
- stores transferable findings in CAM semantic memory
- can generate enhancement tasks for a target project
- supports scan-only preview mode with no model calls
- keeps a persistent mining ledger so unchanged repos are skipped by default

Syntax:

```bash
cam mine <directory> [--target /path/to/project] [--max-repos N] [--min-relevance 0.6] [--tasks/--no-tasks] [--depth N] [--dedup/--no-dedup] [--skip-known/--no-skip-known] [--force-rescan] [--scan-only] [--max-minutes N]
```

Key options:
- `--target`: where the mined findings should be considered relevant
- `--min-relevance`: threshold for task generation
- `--scan-only`: preview discovered repos without spending model calls
- `--skip-known`: skip repos already mined when unchanged
- `--force-rescan`: ignore the mining ledger and rescan selected repos
- `--tasks`: whether to generate tasks from findings

Example use case: improve CAM itself

```bash
cam mine /Users/o2satz/multiclaw/Repo2Eval \
  --target /Users/o2satz/multiclaw \
  --max-repos 4 \
  --max-minutes 20
```

Example use case: check a folder again later without wasting tokens

```bash
cam mine /Users/o2satz/multiclaw/Repo2Eval \
  --scan-only \
  --max-repos 10
```

If CAM says `Will mine: 0`, the selected repos are unchanged and would be skipped.

If you know you want to rerun them anyway:

```bash
cam mine /Users/o2satz/multiclaw/Repo2Eval \
  --max-repos 10 \
  --force-rescan
```

## `cam mine-report`

Purpose:
Inspect a repo folder against CAM’s persistent mining ledger before spending tokens.

What it does:
- discovers repos/source trees in a directory
- compares each one against the mining ledger
- shows whether each repo is `new`, `changed`, or `unchanged`
- shows when a repo was last mined and how many findings/tokens were recorded

Syntax:

```bash
cam mine-report <directory> [--depth N] [--dedup/--no-dedup] [--changed-only]
```

Example use case:
You added more repos to `Repo2Eval` and want to know what actually needs scanning.

```bash
cam mine-report /Users/o2satz/multiclaw/Repo2Eval --depth 3
```

Example use case: only show the repos that would justify fresh work

```bash
cam mine-report /Users/o2satz/multiclaw/Repo2Eval --depth 3 --changed-only
```

## `cam assimilation-report`

Purpose:
Show whether CAM’s assimilated knowledge is merely stored, actively reused, operationalized, or proven useful.

What it does:
- classifies methodologies into continuum stages:
  - `stored`
  - `enriched`
  - `retrieved`
  - `operationalized`
  - `proven`
- separately flags high-potential methodologies that may become useful later
- uses existing metadata like:
  - retrieval counts
  - success counts
  - capability metadata
  - potential score
  - linked action templates

Syntax:

```bash
cam assimilation-report [--limit N] [--future-threshold 0.65]
```

Example use case:
You want to know whether CAM’s mined knowledge is real operational fuel or just archived memory.

```bash
cam assimilation-report --limit 10
```

Example use case:
You want to raise the bar for what counts as a future candidate.

```bash
cam assimilation-report --limit 15 --future-threshold 0.75
```

## `cam reassess`

Purpose:
Actively re-score old methodologies against a new task so CAM can decide what prior knowledge should be revived now.

What it does:
- takes a task description and optional repo context
- derives activation triggers from prior methodologies
- scores methodologies against the new task using:
  - task/repo keyword overlap
  - potential score
  - novelty
  - retrieval evidence
  - success evidence
  - action-template presence
- separates “recommended now” from “future watchlist”
- explains why each recommendation was surfaced

Syntax:

```bash
cam reassess [repo] --task "..." [--limit N] [--min-score 0.2] [--future-threshold 0.65]
```

Example use case:
You want CAM to reactivate prior knowledge for a repo repair task instead of just showing stored memory.

```bash
cam reassess --task "repair broken tests with ast-based refactoring" --limit 10
```

Example use case:
You want repo context to influence what CAM revives.

```bash
cam reassess /path/to/repo --task "add evaluation and rollback for finetuning pipeline" --limit 10
```

Example use case: support a new app build

```bash
cam mine /Users/o2satz/multiclaw/Repo2Eval \
  --target /Users/o2satz/projects/new-app \
  --max-repos 4 \
  --max-minutes 20
```

## `cam forge-export`

Purpose:
Export CAM memory into a standalone Forge knowledge pack.

What it does:
- reads CAM methodologies and tasks from the database
- writes a neutral JSONL knowledge pack
- gives outside tools/apps a way to use CAM knowledge without importing CAM runtime

Syntax:

```bash
cam forge-export [--out data/cam_knowledge_pack.jsonl] [--db path/to/claw.db] [--max-methodologies N] [--max-tasks N] [--max-minutes N]
```

Example use case:
You want a standalone app to consume CAM’s learned knowledge.

```bash
cam forge-export \
  --out data/cam_knowledge_pack.jsonl \
  --max-methodologies 200 \
  --max-tasks 200
```

## `cam forge-benchmark`

Purpose:
Run the standalone Forge regression benchmark with a wall-clock limit.

What it does:
- executes the standalone benchmark harness
- compares Forge-style output against the baseline retrieval path
- writes benchmark summary artifacts to an output directory

Syntax:

```bash
cam forge-benchmark [--repo PATH] [--note PATH] [--knowledge-pack PATH] [--out PATH] [--max-minutes N]
```

Example use case:
You changed the standalone Forge flow and want a fixed regression check.

```bash
cam forge-benchmark --max-minutes 5
```

## `cam validate`

Purpose:
Check whether a created repo actually matches the saved creation spec.

What it does:
- loads a `cam create` spec JSON
- checks repo existence and baseline state
- runs executable acceptance checks
- distinguishes between shell checks and plain-English manual checks
- fails if the repo never materially changed

Syntax:

```bash
cam validate --spec-file data/create_specs/<spec-file>.json [--max-minutes N]
```

Example use case:
You asked CAM to build something and want to know whether it actually delivered.

```bash
cam validate --spec-file data/create_specs/20260314-my-app-create-spec.json
```

## `cam benchmark`

Purpose:
Benchmark Forge output after validation.

What it does:
- runs the benchmark harness on a repo, note, and knowledge pack
- writes output metrics
- is intended to be a performance/quality step, not the first validation step

Syntax:

```bash
cam benchmark [--repo PATH] [--note PATH] [--knowledge-pack PATH] [--out PATH] [--max-minutes N]
```

Example use case:
You already validated a created app and now want a quality score comparison.

```bash
cam benchmark --out data/forge_benchmark_after_validation
```

## `cam govern`

Purpose:
Manage and inspect CAM’s memory governance layer.

What it does:
Depending on the action, it can:
- show memory counts and DB usage
- run a governance sweep
- garbage-collect dead methodologies
- enforce quota
- prune old episodes

Syntax:

```bash
cam govern [stats|sweep|gc|quota|prune]
```

Actions:
- `stats`: show current governance stats
- `sweep`: run a full governance sweep
- `gc`: garbage collect dead methodologies
- `quota`: enforce methodology quota
- `prune`: prune old episodes

Example use case:
You want to see whether CAM’s memory database is healthy.

```bash
cam govern stats
```

## `cam setup`

Purpose:
Configure API keys, models, and agent settings interactively.

What it does:
- walks through agent/provider configuration
- writes settings into `claw.toml`

Syntax:

```bash
cam setup
```

Example use case:
You cloned CAM on a new machine and need to configure models and keys.

```bash
cam setup
```

## `cam synergies`

Purpose:
Show CAM’s capability synergy graph summary.

What it does:
- reports synergy relationships between learned capabilities
- can show detailed edge lists with `--verbose`
- helps identify which ideas combine well across repos/domains

Syntax:

```bash
cam synergies [--verbose]
```

Example use case:
You want to see which learned capabilities are reinforcing each other.

```bash
cam synergies --verbose
```

## `cam prism-demo`

Purpose:
Demonstrate CAM’s PRISM multi-scale embedding concept.

What it does:
- runs the PRISM demonstration path
- is mainly a demo/inspection command rather than a normal production workflow

Syntax:

```bash
cam prism-demo [--verbose]
```

Example use case:
You want to inspect the PRISM embedding demo behavior.

```bash
cam prism-demo
```

## `cam kb` Knowledge Browser

The `kb` group lets you inspect what CAM has already learned.

### `cam kb insights`

Purpose:
Show the high-level knowledge summary.

What it does:
- top capabilities
- domain map
- synergy highlights
- score distributions

Syntax:

```bash
cam kb insights
```

Example:

```bash
cam kb insights
```

### `cam kb search`

Purpose:
Search learned capabilities with natural language.

What it does:
- uses hybrid vector plus full-text search
- returns relevant learned capabilities from CAM memory

Syntax:

```bash
cam kb search "<query>" [--limit N]
```

Example:

```bash
cam kb search "repo repair and test generation" --limit 5
```

### `cam kb capability`

Purpose:
Inspect one specific capability in detail.

What it does:
- shows the full capability record
- shows related items and synergies
- accepts a full ID or an ID prefix

Syntax:

```bash
cam kb capability <capability_id_or_prefix>
```

Example:

```bash
cam kb capability 4f1d2a
```

### `cam kb domains`

Purpose:
Show the domain landscape of CAM’s learned knowledge.

What it does:
- groups capabilities into domains
- highlights bridge areas between domains

Syntax:

```bash
cam kb domains
```

Example:

```bash
cam kb domains
```

### `cam kb synergies`

Purpose:
Show the strongest synergy edges in CAM memory.

What it does:
- surfaces cross-repo and cross-domain combinations
- helps identify promising syntheses

Syntax:

```bash
cam kb synergies [--limit N]
```

Example:

```bash
cam kb synergies --limit 15
```

## Recommended Workflows

## Workflow A: Review a repo before asking CAM to change it

```bash
cam evaluate /path/to/repo --mode quick
cam enhance /path/to/repo --dry-run
cam runbook <task_id>
```

## Workflow B: Improve CAM using outside repos

```bash
cam mine /path/to/repo-folder --target /Users/o2satz/multiclaw --max-repos 4
cam kb insights
cam kb search "small model finetuning"
cam synergies
```

## Workflow C: Use CAM to help design a new non-CAM app

```bash
cam mine /path/to/repo-folder --target /path/to/new-app --max-repos 4
cam ideate /path/to/repo-folder --ideas 3
cam create /path/to/new-app --repo-mode new --request "Build the selected concept"
cam validate --spec-file data/create_specs/<spec-file>.json
```

## Workflow D: Export CAM knowledge to a standalone tool

```bash
cam forge-export --out data/cam_knowledge_pack.jsonl
cam benchmark --knowledge-pack data/cam_knowledge_pack.jsonl
```

## Practical Notes

- `cam mine` is for learning from repos. It does not itself create the new app.
- `cam ideate` is for proposing new app concepts from learned knowledge plus repo inputs.
- `cam create` is the command that turns a requested outcome into a task/spec and optional execution.
- `cam validate` should happen before `cam benchmark`.
- `cam benchmark` is about quality/performance measurement, not first-pass correctness.
- `cam forge-export` is the clean bridge from CAM memory into a standalone non-CAM app.

## One-Line Summary Per Command

- `evaluate`: inspect one repo and score it
- `enhance`: run CAM’s improvement loop on one repo
- `fleet-enhance`: run enhancement across many repos
- `results`: show prior task outcomes
- `status`: show system health/status
- `runbook`: inspect a task’s execution plan
- `quickstart`: create a goal and optionally run it fast
- `create`: define and optionally execute a requested repo outcome
- `add-goal`: manually add a task to a repo
- `ideate`: invent app concepts from CAM memory plus candidate repos
- `mine`: learn from a folder of repos
- `forge-export`: export CAM memory for outside use
- `forge-benchmark`: benchmark the standalone Forge path
- `validate`: check whether a created repo meets its saved spec
- `benchmark`: measure Forge quality after validation
- `govern`: inspect and maintain CAM memory governance
- `setup`: configure keys and models
- `synergies`: inspect capability interactions
- `prism-demo`: run the PRISM embedding demo
- `kb insights/search/capability/domains/synergies`: browse CAM’s learned knowledge
