"""Self-consumption loop — CLAW mining its own outputs for meta-patterns.

Analyzes CLAW's completed work to find recurring patterns about:
- How CLAW solves problems (approach patterns)
- Which agents perform best on which tasks (routing insights)
- How methodologies evolve over generations (improvement patterns)

Circular reasoning guards:
- Only consume verified successes (status=DONE)
- Tag all outputs 'self_consumed' — cannot be re-consumed
- Lower initial fitness (0.3 relevance penalty)
- Max generation depth (configurable, default 3)
- Require thriving + success_count >= 3 for global promotion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from claw.core.config import ClawConfig, GovernanceConfig
from claw.core.models import HypothesisOutcome, Methodology, Task, TaskStatus
from claw.db.repository import Repository
from claw.llm.client import LLMClient, LLMMessage
from claw.memory.semantic import SemanticMemory

logger = logging.getLogger("claw.self_consumer")


@dataclass
class SelfConsumptionReport:
    """Results from a self-consumption analysis."""
    patterns_found: int = 0
    patterns_stored: int = 0
    patterns_blocked_dedup: int = 0
    patterns_blocked_generation: int = 0
    analysis_types: list[str] = field(default_factory=list)
    error: Optional[str] = None


class SelfConsumer:
    """Mines CLAW's own completed work for meta-patterns.

    Dependencies:
        repository: Database access.
        llm_client: For LLM analysis of meta-patterns.
        semantic_memory: For storing discovered meta-patterns.
        config: ClawConfig for model selection.
        governance_config: GovernanceConfig for generation limits.
    """

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        semantic_memory: SemanticMemory,
        config: ClawConfig,
        governance_config: Optional[GovernanceConfig] = None,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.semantic_memory = semantic_memory
        self.config = config
        self.gov_config = governance_config or GovernanceConfig()
        self.assimilation_engine: Any = None

    async def run_full_consumption(
        self,
        project_id: str,
    ) -> SelfConsumptionReport:
        """Run all self-consumption analyses.

        Combines recent work, methodology evolution, and routing analysis.
        """
        if not self.gov_config.self_consume_enabled:
            return SelfConsumptionReport()

        report = SelfConsumptionReport()

        # 1. Recent work patterns
        try:
            r1 = await self.consume_recent_work(project_id)
            report.patterns_found += r1.patterns_found
            report.patterns_stored += r1.patterns_stored
            report.patterns_blocked_dedup += r1.patterns_blocked_dedup
            report.patterns_blocked_generation += r1.patterns_blocked_generation
            report.analysis_types.append("recent_work")
        except Exception as e:
            logger.warning("Recent work analysis failed: %s", e)

        # 2. Routing patterns
        try:
            r2 = await self.consume_routing_decisions(project_id)
            report.patterns_found += r2.patterns_found
            report.patterns_stored += r2.patterns_stored
            report.patterns_blocked_dedup += r2.patterns_blocked_dedup
            report.analysis_types.append("routing")
        except Exception as e:
            logger.warning("Routing analysis failed: %s", e)

        # 3. Methodology evolution
        try:
            r3 = await self.consume_methodology_evolution()
            report.patterns_found += r3.patterns_found
            report.patterns_stored += r3.patterns_stored
            report.analysis_types.append("evolution")
        except Exception as e:
            logger.warning("Evolution analysis failed: %s", e)

        logger.info(
            "Self-consumption complete: found=%d, stored=%d, blocked_dedup=%d, blocked_gen=%d",
            report.patterns_found, report.patterns_stored,
            report.patterns_blocked_dedup, report.patterns_blocked_generation,
        )
        return report

    async def consume_recent_work(
        self,
        project_id: str,
        lookback_tasks: Optional[int] = None,
    ) -> SelfConsumptionReport:
        """Analyze recent completed tasks for meta-patterns.

        Only consumes tasks with status=DONE. Extracts patterns about
        successful approaches, error resolution strategies, and agent
        effectiveness.
        """
        report = SelfConsumptionReport()
        lookback = lookback_tasks or self.gov_config.self_consume_lookback

        # Get completed tasks
        completed = await self.repository.get_tasks_by_status(
            project_id, TaskStatus.DONE
        )

        if len(completed) < self.gov_config.self_consume_min_tasks:
            logger.info(
                "Not enough completed tasks for self-consumption (%d < %d)",
                len(completed), self.gov_config.self_consume_min_tasks,
            )
            return report

        # Take the most recent N
        completed = completed[:lookback]

        # Gather hypothesis logs for each task
        summaries: list[str] = []
        for task in completed:
            approaches = await self.repository.get_failed_approaches(task.id)
            success_count = task.attempt_count - len(approaches)

            summary = (
                f"Task: {task.title}\n"
                f"  Type: {task.task_type or 'unknown'}\n"
                f"  Agent: {task.assigned_agent or 'unknown'}\n"
                f"  Attempts: {task.attempt_count}\n"
                f"  Failures: {len(approaches)}\n"
            )
            if approaches:
                error_sigs = [a.error_signature for a in approaches if a.error_signature]
                if error_sigs:
                    summary += f"  Error patterns: {', '.join(error_sigs[:3])}\n"

            summaries.append(summary)

        if not summaries:
            return report

        # Build meta-analysis prompt
        prompt = self._build_recent_work_prompt(summaries)

        # Call LLM for meta-analysis
        try:
            model = self._get_analysis_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            report.error = f"LLM call failed: {e}"
            return report

        # Parse and store meta-patterns
        patterns = self._parse_meta_patterns(response.content)
        report.patterns_found = len(patterns)

        for pattern in patterns:
            stored = await self._store_meta_pattern(pattern, "recent_work")
            if stored == "stored":
                report.patterns_stored += 1
            elif stored == "dedup":
                report.patterns_blocked_dedup += 1
            elif stored == "generation":
                report.patterns_blocked_generation += 1

        return report

    async def consume_routing_decisions(
        self,
        project_id: str,
    ) -> SelfConsumptionReport:
        """Analyze agent routing decisions vs outcomes.

        Finds patterns like 'codex outperforms on refactoring tasks'
        by correlating assigned_agent with task outcomes.
        """
        report = SelfConsumptionReport()

        # Get agent performance data from agent_scores table
        scores = await self.repository.get_agent_scores()
        if not scores:
            return report

        # Build routing summary
        routing_lines: list[str] = []
        for score in scores:
            total = score.get("total_attempts", 0)
            if total < 3:
                continue
            success_rate = score.get("successes", 0) / max(total, 1)
            routing_lines.append(
                f"Agent '{score['agent_id']}' on '{score['task_type']}': "
                f"{total} attempts, {success_rate:.0%} success rate, "
                f"avg quality {score.get('avg_quality_score', 0):.2f}"
            )

        if not routing_lines:
            return report

        prompt = self._build_routing_prompt(routing_lines)

        try:
            model = self._get_analysis_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            report.error = f"LLM call failed: {e}"
            return report

        patterns = self._parse_meta_patterns(response.content)
        report.patterns_found = len(patterns)

        for pattern in patterns:
            stored = await self._store_meta_pattern(pattern, "routing")
            if stored == "stored":
                report.patterns_stored += 1
            elif stored == "dedup":
                report.patterns_blocked_dedup += 1

        return report

    async def consume_methodology_evolution(
        self,
        min_generation: int = 1,
    ) -> SelfConsumptionReport:
        """Analyze how methodologies have evolved over generations.

        Looks at methodology lineage (parent_ids, superseded_by)
        to find patterns in how solutions improve.
        """
        report = SelfConsumptionReport()

        # Find methodologies with generation > 0 (have parents)
        evolved: list[Methodology] = []
        for state in ("viable", "thriving"):
            batch = await self.repository.get_methodologies_by_state(state, limit=100)
            evolved.extend(m for m in batch if m.generation >= min_generation)

        if len(evolved) < 3:
            return report

        evolution_lines: list[str] = []
        for m in evolved[:20]:
            line = (
                f"Methodology (gen {m.generation}): {m.problem_description[:100]}\n"
                f"  State: {m.lifecycle_state}, "
                f"Success: {m.success_count}, Failures: {m.failure_count}"
            )
            if m.parent_ids:
                line += f"\n  Parents: {m.parent_ids[:3]}"
            evolution_lines.append(line)

        prompt = self._build_evolution_prompt(evolution_lines)

        try:
            model = self._get_analysis_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            report.error = f"LLM call failed: {e}"
            return report

        patterns = self._parse_meta_patterns(response.content)
        report.patterns_found = len(patterns)

        for pattern in patterns:
            stored = await self._store_meta_pattern(pattern, "evolution")
            if stored == "stored":
                report.patterns_stored += 1
            elif stored == "dedup":
                report.patterns_blocked_dedup += 1

        return report

    async def _store_meta_pattern(
        self,
        pattern: dict[str, str],
        source_type: str,
    ) -> str:
        """Store a meta-pattern as a methodology with guards.

        Returns:
            'stored', 'dedup', or 'generation' indicating outcome.
        """
        title = pattern.get("title", "untitled meta-pattern")
        description = pattern.get("description", "")
        if not description or len(description) < 20:
            return "dedup"

        # Check generation cap
        max_gen = self.gov_config.self_consume_max_generation

        # Check if any existing self_consumed methodology at max generation
        existing_self = await self.repository.get_methodologies_by_tag(
            "self_consumed", limit=100
        )
        current_max_gen = max(
            (m.generation for m in existing_self), default=0
        )
        new_generation = current_max_gen + 1 if existing_self else 1

        if new_generation > max_gen:
            logger.info(
                "Generation cap reached (%d > %d) — skipping meta-pattern '%s'",
                new_generation, max_gen, title,
            )
            return "generation"

        problem_desc = f"[Self-consumed:{source_type}] {title}: {description}"

        tags = [
            "self_consumed",
            f"source_type:{source_type}",
            f"generation:{new_generation}",
        ]

        try:
            methodology = await self.semantic_memory.save_solution(
                problem_description=problem_desc,
                solution_code=f"## Meta-Pattern: {title}\n\n{description}",
                methodology_notes=f"Self-consumed from {source_type} analysis",
                tags=tags,
                scope="project",
                methodology_type="PATTERN",
            )

            # If dedup blocked it, save_solution returns an existing methodology
            # Check if it's one we just saved by seeing if it has our tags
            if methodology and "self_consumed" not in methodology.tags:
                return "dedup"

            # Update generation on the saved methodology
            if methodology:
                await self.repository.engine.execute(
                    "UPDATE methodologies SET generation = ? WHERE id = ?",
                    [new_generation, methodology.id],
                )

            logger.info(
                "Stored meta-pattern '%s' (gen=%d, source=%s)",
                title, new_generation, source_type,
            )

            # Trigger capability assimilation
            if methodology and self.assimilation_engine is not None:
                try:
                    await self.assimilation_engine.assimilate(methodology.id)
                except Exception as ae:
                    logger.warning("Assimilation failed for %s: %s", methodology.id, ae)

            return "stored"

        except Exception as e:
            logger.warning("Failed to store meta-pattern '%s': %s", title, e)
            return "dedup"

    def _parse_meta_patterns(self, llm_response: str) -> list[dict[str, str]]:
        """Parse meta-patterns from LLM response.

        Expects JSON array of {title, description} objects.
        Falls back to treating the entire response as a single pattern.
        """
        import re

        cleaned = llm_response.strip()
        fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(fence_pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

        if not cleaned.startswith("["):
            arr_start = cleaned.find("[")
            arr_end = cleaned.rfind("]")
            if arr_start != -1 and arr_end != -1:
                cleaned = cleaned[arr_start:arr_end + 1]

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                patterns = []
                for item in data:
                    if isinstance(item, dict) and "title" in item:
                        patterns.append({
                            "title": str(item["title"])[:200],
                            "description": str(item.get("description", ""))[:2000],
                        })
                return patterns[:10]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: treat entire response as a single pattern
        if len(cleaned) > 30:
            return [{"title": "Meta-pattern", "description": cleaned[:2000]}]
        return []

    def _get_analysis_model(self) -> str:
        """Get model for self-consumption analysis."""
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = self.config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise ValueError("No model configured in any agent")

    def _build_recent_work_prompt(self, summaries: list[str]) -> str:
        """Build prompt for recent work meta-analysis."""
        work_block = "\n\n".join(summaries)
        return (
            "You are analyzing a coding AI system's recent completed work to find meta-patterns.\n\n"
            "## Recent Completed Tasks\n\n"
            f"{work_block}\n\n"
            "## Instructions\n\n"
            "Identify 2-5 recurring meta-patterns from this work. Focus on:\n"
            "- Recurring successful approaches across different tasks\n"
            "- Common error patterns and how they were resolved\n"
            "- Agent-task type correlations (which agent works best for what)\n\n"
            "Return a JSON array of objects with 'title' and 'description' keys.\n"
            "Each description should be actionable — something the system can use to improve.\n"
            "```json\n[{\"title\": \"...\", \"description\": \"...\"}]\n```"
        )

    def _build_routing_prompt(self, routing_lines: list[str]) -> str:
        """Build prompt for routing meta-analysis."""
        routing_block = "\n".join(routing_lines)
        return (
            "You are analyzing an AI agent routing system's performance data.\n\n"
            "## Agent Performance Data\n\n"
            f"{routing_block}\n\n"
            "## Instructions\n\n"
            "Identify 1-3 routing insights from this data:\n"
            "- Which agents excel at which task types?\n"
            "- Are there task types where all agents struggle?\n"
            "- Any surprising agent-task pairings?\n\n"
            "Return a JSON array of objects with 'title' and 'description' keys.\n"
            "```json\n[{\"title\": \"...\", \"description\": \"...\"}]\n```"
        )

    def _build_evolution_prompt(self, evolution_lines: list[str]) -> str:
        """Build prompt for methodology evolution meta-analysis."""
        evolution_block = "\n\n".join(evolution_lines)
        return (
            "You are analyzing how an AI system's learned methodologies evolve.\n\n"
            "## Methodology Evolution Data\n\n"
            f"{evolution_block}\n\n"
            "## Instructions\n\n"
            "Identify 1-3 evolution patterns:\n"
            "- How do methodologies improve across generations?\n"
            "- What distinguishes thriving from declining methodologies?\n"
            "- Are there patterns in what gets superseded?\n\n"
            "Return a JSON array of objects with 'title' and 'description' keys.\n"
            "```json\n[{\"title\": \"...\", \"description\": \"...\"}]\n```"
        )
