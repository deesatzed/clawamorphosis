You are evaluating the **future potential** of a capability that was recently ingested into an AI knowledge base.

## Capability Details

**Problem/Purpose:** {problem_description}
**Domains:** {domains}
**Inputs:** {inputs}
**Outputs:** {outputs}
**Type:** {capability_type}
**Composability:** {composability}

## Instructions

Rate this capability's **future potential** — not how useful it is RIGHT NOW, but how valuable it COULD become over time. Consider:

1. **Novel Workflow Enablement**: Could this enable entirely new workflows that don't exist yet in the knowledge base?
2. **Domain Bridging**: Could this connect previously disconnected domains (e.g., medical + ML, finance + NLP)?
3. **Composability Amplification**: When combined with other capabilities, could this create emergent value greater than the sum?
4. **Foundation Potential**: Could this serve as a building block that many future capabilities depend on?
5. **Frontier Expansion**: Does this push into territory the system hasn't explored before?

A medical diagnosis capability in a database of web scraping tools should score HIGH — it represents unexplored territory with unknown potential.

A slightly better web scraper in that same database should score LOW — it's incremental, not transformative.

Return ONLY a JSON object:
```json
{"potential_score": 0.X, "reasoning": "Brief explanation of future value"}
```

Score from 0.0 (zero future potential) to 1.0 (transformative potential).
