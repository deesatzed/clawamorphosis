# Self-Consumption Meta-Analysis Prompt

You are analyzing an autonomous AI coding system's (CLAW) own work products to extract meta-patterns — patterns about how the system learns and works, not patterns from the repos it analyzes.

## What to look for

1. **Approach patterns**: Recurring strategies that lead to success across different task types
2. **Error resolution patterns**: Common failure modes and the approaches that resolved them
3. **Agent routing insights**: Which AI agents perform best on which task types
4. **Evolution patterns**: How methodologies improve across generations
5. **Efficiency patterns**: Task types that consistently require fewer attempts

## What NOT to include

- Patterns that are too generic ("write tests" is not a meta-pattern)
- Patterns already covered by existing methodologies (check the context)
- Patterns from a single observation (need at least 2-3 occurrences)

## Output format

Return a JSON array of meta-pattern objects:

```json
[
  {
    "title": "Short descriptive title (max 100 chars)",
    "description": "Detailed actionable description. What is the pattern? When does it apply? How should the system use it? (max 500 chars)"
  }
]
```

Return 2-5 patterns maximum. Quality over quantity.
