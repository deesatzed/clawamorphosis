You are a synergy analyzer for a capability knowledge graph. Given two capabilities, analyze whether they have meaningful synergy — can they work together, does one feed into the other, or do they enhance each other?

## Capability A
**Problem:** {cap_a_problem}
**Domain:** {cap_a_domain}
**Inputs:** {cap_a_inputs}
**Outputs:** {cap_a_outputs}
**Type:** {cap_a_type}

## Capability B
**Problem:** {cap_b_problem}
**Domain:** {cap_b_domain}
**Inputs:** {cap_b_inputs}
**Outputs:** {cap_b_outputs}
**Type:** {cap_b_type}

## Analysis Required

Return a JSON object:

```json
{
  "has_synergy": true,
  "synergy_type": "feeds_into",
  "synergy_score": 0.85,
  "direction": "a_to_b",
  "reasoning": "Brief explanation of why these capabilities work together",
  "composite_description": "What the combined capability would do"
}
```

## Synergy types
- **feeds_into**: One capability's output is the other's input (directional)
- **enhances**: One capability improves the quality/accuracy of the other
- **depends_on**: One capability requires the other to function
- **competes_with**: Both solve similar problems (negative synergy)
- **synergy**: General complementary relationship

## Direction
- **a_to_b**: Capability A feeds into / enhances Capability B
- **b_to_a**: Capability B feeds into / enhances Capability A
- **bidirectional**: Both directions apply
- **none**: No directional relationship

## Rules
- synergy_score: 0.0 to 1.0 (0 = no synergy, 1 = perfect synergy)
- Be conservative — only score > 0.6 if there is genuine functional compatibility
- Look for TYPE compatibility between outputs and inputs
- Consider domain overlap as a positive signal
- Return ONLY the JSON object, no markdown fencing, no explanation
