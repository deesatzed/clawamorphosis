You are a capability analyzer for a code knowledge graph. Given a methodology (a reusable coding pattern or technique), extract its structured capability metadata.

Analyze the methodology below and return a JSON object with these fields:

```json
{
  "inputs": [
    {"name": "input_name", "type": "type_category", "required": true, "description": "what this input is"}
  ],
  "outputs": [
    {"name": "output_name", "type": "type_category", "required": true, "description": "what this output is"}
  ],
  "domain": ["domain_tag_1", "domain_tag_2"],
  "composability": {
    "can_chain_after": ["capability_type_1"],
    "can_chain_before": ["capability_type_2"],
    "standalone": true
  },
  "capability_type": "one_of_the_types_below"
}
```

## Type categories for inputs/outputs
Use these standard type categories: text, code_patch, metrics_data, event_list, analysis, config, model_artifact, test_results, documentation, error_report, dependency_graph, file_manifest, embedding_vector, structured_data

## Capability types
Use one of: analysis, transformation, detection, generation, validation, optimization, integration, extraction, monitoring, orchestration

## Domain tags
Use lowercase_snake_case. Examples: ml_training, web_development, testing, security, data_processing, code_quality, api_design, database, devops, documentation, error_handling, performance, refactoring, architecture

## Rules
- Extract REAL inputs and outputs based on what the code actually consumes and produces
- Domain tags should reflect the actual problem domain, not generic software terms
- can_chain_after: what types of capabilities typically produce this capability's inputs
- can_chain_before: what types of capabilities typically consume this capability's outputs
- standalone: true if the capability can operate independently without chaining
- Return ONLY the JSON object, no markdown fencing, no explanation

## Methodology to analyze

**Problem:** {problem_description}

**Solution:**
```
{solution_code}
```

**Notes:** {methodology_notes}
**Tags:** {tags}
