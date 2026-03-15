# CAM Operator Cheat Sheet

This is the short version.

Use this when you do not want the full command reference and just need the commands that matter most in normal operation.

## Start Here

```bash
cd clawamorphosis
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/cam --help
.venv/bin/cam govern stats
```

Preferred mental split:
- core workflow verbs stay top-level: `evaluate`, `enhance`, `mine`, `ideate`, `create`, `validate`
- advanced workflow support is grouped:
  - `cam doctor ...`
  - `cam learn ...`
  - `cam task ...`
  - `cam forge ...`

## 1. Check That CAM Is Healthy

```bash
.venv/bin/cam govern stats
.venv/bin/cam status
```

Use this when:
- you just cloned the repo
- you are not sure the DB/runtime is healthy

## 2. Study One Repo Before Changing It

```bash
.venv/bin/cam evaluate /path/to/repo --mode quick
.venv/bin/cam enhance /path/to/repo --dry-run
```

Use this when:
- you want CAM to inspect before acting
- you want a safe first pass

## 3. Mine Outside Repos For Reusable Patterns

Preview only, no model calls:

```bash
.venv/bin/cam mine /path/to/repo-folder --scan-only --depth 3 --max-repos 5
```

Default behavior:
- unchanged repos are skipped automatically
- changed repos are rescanned automatically
- use `--force-rescan` when you want to ignore the ledger and mine them again anyway
- live mining now validates required provider keys before it starts unless you explicitly use `--no-live-keycheck`

Inspect the folder first:

```bash
.venv/bin/cam mine-report /path/to/repo-folder --depth 3
```

Preflight the real provider path first:

```bash
.venv/bin/cam doctor keycheck --for mine --live
```

Real mining:

```bash
.venv/bin/cam mine /path/to/repo-folder \
  --target /path/to/target-project \
  --max-repos 4 \
  --max-minutes 20
```

Use this when:
- you want CAM to learn from other repos
- you want CAM memory enriched before building something new

## 4. Ask CAM For New App Ideas

```bash
.venv/bin/cam ideate /path/to/repo-folder \
  --focus "Invent useful standalone apps that build, troubleshoot, or create" \
  --ideas 3 \
  --max-repos 4 \
  --max-minutes 10
```

Use this when:
- you want new product directions from CAM memory plus source repos
- you do not want just a summary of the repos

## 5. Turn A Request Into A Real Spec And Task

```bash
.venv/bin/cam create /path/to/target-repo \
  --repo-mode new \
  --request "Build the selected app" \
  --spec "Must be standalone" \
  --check "pytest -q" \
  --max-minutes 20
```

Use this when:
- you want a spec-backed creation task
- you want the requested outcome written down explicitly

If you want CAM to attempt execution immediately:

```bash
.venv/bin/cam create /path/to/target-repo \
  --repo-mode new \
  --request "Build the selected app" \
  --check "pytest -q" \
  --execute \
  --max-minutes 20
```

## 6. Validate Before You Trust

```bash
.venv/bin/cam validate --spec-file data/create_specs/<spec-file>.json --max-minutes 5
```

Use this when:
- CAM said it created something
- you want to know whether the repo actually changed and checks passed

## 7. Benchmark After Validation

```bash
.venv/bin/cam benchmark --max-minutes 5
```

Use this when:
- you already validated correctness
- now you want a quality/performance measure

## 8. Export CAM Knowledge For A Standalone App

```bash
.venv/bin/cam forge export \
  --out data/cam_knowledge_pack.jsonl \
  --max-methodologies 200 \
  --max-tasks 200 \
  --max-minutes 5
```

Use this when:
- you want a non-CAM app to consume CAM’s learned knowledge
- you want a clean bridge instead of importing CAM internals

## 9. Inspect What CAM Already Knows

```bash
.venv/bin/cam kb insights
.venv/bin/cam kb search "repo repair"
.venv/bin/cam kb domains
.venv/bin/cam kb synergies --limit 15
.venv/bin/cam learn report --limit 10
.venv/bin/cam learn delta /path/to/repo-folder --since-hours 24 --latest 10
.venv/bin/cam learn reassess --task "repair broken tests with ast-based refactoring" --limit 10
```

Use this when:
- you want to see whether mining actually produced useful knowledge
- you want to inspect CAM memory before creating something new

## 10. Most Common Real Workflows

### Improve CAM using outside repos

```bash
.venv/bin/cam doctor keycheck --for mine --live
.venv/bin/cam mine /path/to/repo-folder --target /Users/o2satz/multiclaw --max-repos 4 --max-minutes 20
.venv/bin/cam kb insights
```

### Build a new standalone app using outside repos

```bash
.venv/bin/cam doctor keycheck --for mine --live
.venv/bin/cam mine /path/to/repo-folder --target /path/to/new-app --max-repos 4 --max-minutes 20
.venv/bin/cam ideate /path/to/repo-folder --ideas 3 --max-repos 4
.venv/bin/cam create /path/to/new-app --repo-mode new --request "Build the selected concept"
.venv/bin/cam validate --spec-file data/create_specs/<spec-file>.json
```

### Safe first-contact workflow for a single repo

```bash
.venv/bin/cam evaluate /path/to/repo --mode quick
.venv/bin/cam enhance /path/to/repo --dry-run
```

## Rules Of Thumb

- `mine` learns from repos. It does not itself build the new app.
- `ideate` proposes app concepts.
- `create` turns a requested outcome into a spec and task.
- `validate` should happen before `benchmark`.
- `cam forge export` is the clean handoff from CAM into a standalone app.
- if CAM claims success but validation says the repo did not change, trust validation.
