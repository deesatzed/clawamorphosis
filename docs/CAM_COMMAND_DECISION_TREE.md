# CAM Command Decision Tree

Use this when the main question is simple:

> Which CAM command should I use for the job I have right now?

## Fast Rule

- `mine` = learn from outside repos
- `evaluate` = inspect a repo
- `enhance` = improve an existing repo
- `ideate` = invent a new app concept
- `create` = turn a request into a spec-backed build task
- `validate` = verify the result

## Decision Tree

### 1. Do you want CAM to learn from other repos?

If **yes**, start with:

```bash
cam doctor keycheck --for mine --live
cam mine /path/to/repo-folder --target /path/to/target --max-repos 10 --depth 4 --max-minutes 30
```

Use this when:
- you want CAM memory enriched
- you want CAM to assimilate useful patterns from outside repos

Then decide what the real target is:
- CAM itself
- another existing repo
- a brand-new app

### 2. Is the target CAM itself?

If **yes**, use:

```bash
cam doctor keycheck --for mine --live
cam mine /path/to/repo-folder --target /Users/o2satz/multiclaw --max-repos 10 --depth 4 --max-minutes 30
cam learn report --limit 10
cam learn reassess --task "your next CAM improvement task" --limit 10
```

If you want CAM to actually change its own code after learning:

```bash
cam evaluate /Users/o2satz/multiclaw --mode quick
cam enhance /Users/o2satz/multiclaw --dry-run
cam enhance /Users/o2satz/multiclaw --max-tasks 5
```

Meaning:
- `mine` teaches CAM
- `enhance` changes CAM

### 3. Is the target an existing repo that already exists?

If **yes**, start with:

```bash
cam evaluate /path/to/existing-repo --mode quick
cam enhance /path/to/existing-repo --dry-run
```

If the plan looks good:

```bash
cam enhance /path/to/existing-repo --max-tasks 5
```

Use this when you want to:
- modernize structure
- improve security
- future-proof architecture
- troubleshoot or repair an existing codebase

If you want outside-repo learning first:

```bash
cam doctor keycheck --for mine --live
cam mine /path/to/repo-folder --target /path/to/existing-repo --max-repos 10 --depth 4 --max-minutes 30
cam evaluate /path/to/existing-repo --mode quick
cam enhance /path/to/existing-repo --dry-run
```

### 4. Is the target a brand-new app that does not exist yet?

If **yes**, use:

```bash
cam doctor keycheck --for mine --live
cam mine /path/to/repo-folder --target /path/to/new-app --max-repos 10 --depth 4 --max-minutes 30
cam doctor keycheck --for ideate --live
cam ideate /path/to/repo-folder --ideas 3 --max-repos 4
cam create /path/to/new-app --repo-mode new --request "Build the selected concept" --max-minutes 20
cam validate --spec-file data/create_specs/<spec-file>.json --max-minutes 5
```

Use this when you want CAM to:
- synthesize several repos into something new
- generate a standalone tool or app
- create a new repo rather than improve an old one

## Short Examples

### Improve CAM itself

```bash
cam doctor keycheck --for mine --live
cam mine Repo2Eval --target /Users/o2satz/multiclaw --max-repos 20 --depth 4 --max-minutes 30
cam learn report --limit 10
```

### Improve another repo

```bash
cam evaluate /path/to/repo --mode quick
cam enhance /path/to/repo --dry-run
cam enhance /path/to/repo --max-tasks 5
```

### Build a new standalone app

```bash
cam doctor keycheck --for mine --live
cam mine Repo2Eval --target /path/to/new-app --max-repos 10 --depth 4 --max-minutes 30
cam doctor keycheck --for ideate --live
cam ideate Repo2Eval --ideas 3 --max-repos 4
cam create /path/to/new-app --repo-mode new --request "Build the selected concept" --max-minutes 20
cam validate --spec-file data/create_specs/<spec-file>.json --max-minutes 5
```

## Final Mental Model

- If the target already exists: start with `evaluate`
- If the target does not exist yet: start with `mine` + `ideate` + `create`
- If the target is CAM itself: `mine` into CAM first, then `enhance` CAM if you want code changes
- Use `cam doctor ...`, `cam learn ...`, `cam task ...`, and `cam forge ...` as the preferred advanced grouped paths
