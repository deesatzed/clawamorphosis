# Repo Mining Prompt

You are a code archaeologist analyzing a repository to extract **actionable patterns, features, architectural ideas, and techniques** that could enhance a target project called **CLAW** (Codebase Learning & Autonomous Workforce).

## About CLAW

CLAW is a multi-model autonomous codebase enhancement system written in Python 3.12. Key subsystems:
- **Multi-agent orchestration** — 4 AI agents (Claude, Codex, Gemini, Grok) coordinated via Bayesian routing
- **Memory system** — 6 typed stores (working, episodic, semantic, procedural, error, meta) with sqlite-vec embeddings
- **Evaluation battery** — 18 prompt-driven analysis passes (deepdive, driftx, claim-gate, etc.)
- **Self-improvement** — prompt A/B testing, pattern learning, agent score evolution
- **Fleet mode** — scans and enhances multiple repos autonomously with budget caps
- **Verification** — claim-gate + drift detection + regression scanning before any change merges

CLAW's tech stack: Python asyncio, SQLite + aiosqlite, sqlite-vec, typer CLI, Rich, httpx, pydantic.

## Your Task

Analyze the repository source code below and extract findings that could improve CLAW. Look for:

1. **Architectural patterns** — state machines, event systems, plugin architectures, middleware chains, dependency injection patterns
2. **AI/LLM integration techniques** — prompt engineering, response parsing, multi-model orchestration, context management, token optimization
3. **Memory/knowledge systems** — caching strategies, knowledge graphs, embedding approaches, retrieval augmented generation
4. **Code quality patterns** — error handling strategies, retry logic, circuit breakers, graceful degradation
5. **CLI/UX patterns** — progress display, configuration management, interactive workflows
6. **Testing patterns** — fixture strategies, property-based testing, integration test harnesses
7. **Data processing** — pipeline patterns, stream processing, batch processing, ETL patterns
8. **Security patterns** — auth flows, permission systems, sandboxing, input validation
9. **Novel algorithms** — unique approaches to common problems, optimization techniques
10. **Cross-cutting concerns** — logging, metrics, observability, feature flags

## Output Format

Return a JSON array of findings. Each finding must have these fields:

```json
[
  {
    "title": "Short descriptive title (max 80 chars)",
    "description": "Detailed description of the pattern/feature/technique (2-4 sentences)",
    "category": "one of: architecture|ai_integration|memory|code_quality|cli_ux|testing|data_processing|security|algorithm|cross_cutting",
    "source_files": ["relative/path/to/key/file.py"],
    "implementation_sketch": "How this could be adapted for CLAW (2-5 sentences with specific file/class suggestions)",
    "augmentation_notes": "What CLAW currently lacks that this addresses",
    "relevance_score": 0.7,
    "language": "python"
  }
]
```

## Rules

- Return ONLY the JSON array, no additional text before or after
- Maximum 15 findings per repo
- Minimum relevance_score of 0.4 (skip trivial/irrelevant patterns)
- relevance_score range: 0.4 (marginally useful) to 1.0 (directly applicable, high impact)
- Focus on **transferable ideas**, not repo-specific business logic
- Prefer patterns that are **novel or well-implemented**, not obvious boilerplate
- Include the most relevant source files that demonstrate the pattern
- implementation_sketch should reference specific CLAW modules where the pattern could be integrated

## Repository Content

{repo_content}
