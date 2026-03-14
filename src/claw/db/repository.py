"""Data access layer for CLAW.

All SQL queries live here. Agents never write raw SQL — they call Repository
methods that return Pydantic models. This keeps the SQL in one place and
makes the dual-backend (sqlite-vec + FTS5) transparent.
"""

from __future__ import annotations

import json
import struct
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from claw.core.models import (
    ActionTemplate,
    ContextSnapshot,
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    PeerReview,
    Project,
    SynergyExploration,
    Task,
    TaskStatus,
    TokenCostRecord,
)
from claw.db.engine import DatabaseEngine


class Repository:
    """Async data access layer wrapping DatabaseEngine with typed methods."""

    def __init__(self, engine: DatabaseEngine):
        self.engine = engine

    # -------------------------------------------------------------------
    # Projects
    # -------------------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        await self.engine.execute(
            """INSERT INTO projects (id, name, repo_path, tech_stack, project_rules, banned_dependencies)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                project.id,
                project.name,
                project.repo_path,
                json.dumps(project.tech_stack),
                project.project_rules,
                json.dumps(project.banned_dependencies),
            ],
        )
        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        row = await self.engine.fetch_one(
            "SELECT * FROM projects WHERE id = ?", [project_id]
        )
        if row is None:
            return None
        return _row_to_project(row)

    async def list_projects(self) -> list[Project]:
        """List all projects, most recent first."""
        rows = await self.engine.fetch_all(
            "SELECT * FROM projects ORDER BY created_at DESC"
        )
        return [_row_to_project(r) for r in rows]

    async def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get a project by its name."""
        row = await self.engine.fetch_one(
            "SELECT * FROM projects WHERE name = ? LIMIT 1", [name]
        )
        if row is None:
            return None
        return _row_to_project(row)

    # -------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------

    async def create_task(self, task: Task) -> Task:
        await self.engine.execute(
            """INSERT INTO tasks (id, project_id, title, description, status, priority,
               task_type, recommended_agent, assigned_agent, action_template_id,
               execution_steps, acceptance_checks)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                task.id,
                task.project_id,
                task.title,
                task.description,
                task.status.value,
                task.priority,
                task.task_type,
                task.recommended_agent,
                task.assigned_agent,
                task.action_template_id,
                json.dumps(task.execution_steps),
                json.dumps(task.acceptance_checks),
            ],
        )
        return task

    async def get_next_task(self, project_id: str) -> Optional[Task]:
        """Get the highest-priority PENDING task for a project."""
        row = await self.engine.fetch_one(
            """SELECT * FROM tasks
               WHERE project_id = ? AND status = 'PENDING'
               ORDER BY priority DESC, created_at ASC
               LIMIT 1""",
            [project_id],
        )
        if row is None:
            return None
        return _row_to_task(row)

    async def get_task(self, task_id: str) -> Optional[Task]:
        row = await self.engine.fetch_one("SELECT * FROM tasks WHERE id = ?", [task_id])
        if row is None:
            return None
        return _row_to_task(row)

    async def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        now = datetime.now(UTC).isoformat()
        completed_at = now if status == TaskStatus.DONE else None
        await self.engine.execute(
            "UPDATE tasks SET status = ?, updated_at = ?, completed_at = ? WHERE id = ?",
            [status.value, now, completed_at, task_id],
        )

    async def update_task_agent(self, task_id: str, agent_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            "UPDATE tasks SET assigned_agent = ?, updated_at = ? WHERE id = ?",
            [agent_id, now, task_id],
        )

    async def increment_task_attempt(self, task_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            "UPDATE tasks SET attempt_count = attempt_count + 1, updated_at = ? WHERE id = ?",
            [now, task_id],
        )

    async def increment_task_escalation(self, task_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            "UPDATE tasks SET escalation_count = escalation_count + 1, updated_at = ? WHERE id = ?",
            [now, task_id],
        )

    async def get_tasks_by_status(self, project_id: str, status: TaskStatus) -> list[Task]:
        rows = await self.engine.fetch_all(
            "SELECT * FROM tasks WHERE project_id = ? AND status = ? ORDER BY priority DESC",
            [project_id, status.value],
        )
        return [_row_to_task(r) for r in rows]

    async def get_in_progress_tasks(self) -> list[Task]:
        rows = await self.engine.fetch_all(
            "SELECT * FROM tasks WHERE status IN ('EVALUATING', 'PLANNING', 'DISPATCHED', 'CODING', 'REVIEWING')"
        )
        return [_row_to_task(r) for r in rows]

    async def list_tasks(self, project_id: str, include_done: bool = True) -> list[Task]:
        if include_done:
            rows = await self.engine.fetch_all(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at DESC",
                [project_id],
            )
        else:
            rows = await self.engine.fetch_all(
                "SELECT * FROM tasks WHERE project_id = ? AND status != 'DONE' ORDER BY created_at DESC",
                [project_id],
            )
        return [_row_to_task(r) for r in rows]

    async def get_project_results(self, project_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """Get tasks with their latest hypothesis entry for results display.

        Returns a list of dicts with task + hypothesis fields joined together.
        """
        if project_id:
            rows = await self.engine.fetch_all(
                """SELECT t.id AS task_id, t.title, t.status, t.task_type,
                          t.assigned_agent, t.attempt_count, t.created_at AS task_created,
                          t.completed_at,
                          h.approach_summary, h.outcome AS hypothesis_outcome,
                          h.error_signature, h.files_changed, h.duration_seconds,
                          h.model_used, h.agent_id, h.created_at AS hypothesis_created
                   FROM tasks t
                   LEFT JOIN hypothesis_log h ON h.task_id = t.id
                       AND h.attempt_number = (
                           SELECT MAX(h2.attempt_number)
                           FROM hypothesis_log h2
                           WHERE h2.task_id = t.id
                       )
                   WHERE t.project_id = ?
                   ORDER BY t.created_at DESC
                   LIMIT ?""",
                [project_id, limit],
            )
        else:
            rows = await self.engine.fetch_all(
                """SELECT t.id AS task_id, t.title, t.status, t.task_type,
                          t.assigned_agent, t.attempt_count, t.created_at AS task_created,
                          t.completed_at,
                          h.approach_summary, h.outcome AS hypothesis_outcome,
                          h.error_signature, h.files_changed, h.duration_seconds,
                          h.model_used, h.agent_id, h.created_at AS hypothesis_created
                   FROM tasks t
                   LEFT JOIN hypothesis_log h ON h.task_id = t.id
                       AND h.attempt_number = (
                           SELECT MAX(h2.attempt_number)
                           FROM hypothesis_log h2
                           WHERE h2.task_id = t.id
                       )
                   ORDER BY t.created_at DESC
                   LIMIT ?""",
                [limit],
            )
        return [dict(r) for r in rows]

    async def get_task_status_summary(self, project_id: Optional[str] = None) -> dict[str, int]:
        if project_id is None:
            rows = await self.engine.fetch_all(
                "SELECT status, COUNT(*) AS cnt FROM tasks GROUP BY status"
            )
        else:
            rows = await self.engine.fetch_all(
                "SELECT status, COUNT(*) AS cnt FROM tasks WHERE project_id = ? GROUP BY status",
                [project_id],
            )
        return {str(row["status"]): int(row["cnt"]) for row in rows}

    async def get_next_hypothesis_attempt(self, task_id: str) -> int:
        row = await self.engine.fetch_one(
            "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt FROM hypothesis_log WHERE task_id = ?",
            [task_id],
        )
        return int(row["next_attempt"]) if row else 1

    # -------------------------------------------------------------------
    # Action Templates
    # -------------------------------------------------------------------

    async def create_action_template(self, template: ActionTemplate) -> ActionTemplate:
        await self.engine.execute(
            """INSERT INTO action_templates
               (id, title, problem_pattern, execution_steps, acceptance_checks,
                rollback_steps, preconditions, source_methodology_id, source_repo,
                confidence, success_count, failure_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                template.id,
                template.title,
                template.problem_pattern,
                json.dumps(template.execution_steps),
                json.dumps(template.acceptance_checks),
                json.dumps(template.rollback_steps),
                json.dumps(template.preconditions),
                template.source_methodology_id,
                template.source_repo,
                template.confidence,
                template.success_count,
                template.failure_count,
                template.created_at.isoformat(),
                template.updated_at.isoformat(),
            ],
        )
        return template

    async def get_action_template(self, template_id: str) -> Optional[ActionTemplate]:
        row = await self.engine.fetch_one(
            "SELECT * FROM action_templates WHERE id = ?",
            [template_id],
        )
        if row is None:
            return None
        return _row_to_action_template(row)

    async def list_action_templates(
        self,
        source_repo: Optional[str] = None,
        limit: int = 50,
    ) -> list[ActionTemplate]:
        if source_repo:
            rows = await self.engine.fetch_all(
                """SELECT * FROM action_templates
                   WHERE source_repo = ?
                   ORDER BY confidence DESC, updated_at DESC
                   LIMIT ?""",
                [source_repo, limit],
            )
        else:
            rows = await self.engine.fetch_all(
                """SELECT * FROM action_templates
                   ORDER BY confidence DESC, updated_at DESC
                   LIMIT ?""",
                [limit],
            )
        return [_row_to_action_template(r) for r in rows]

    async def update_action_template_outcome(self, template_id: str, success: bool) -> None:
        now = datetime.now(UTC).isoformat()
        if success:
            await self.engine.execute(
                """UPDATE action_templates
                   SET success_count = success_count + 1,
                       confidence = MIN(1.0, confidence + 0.03),
                       updated_at = ?
                   WHERE id = ?""",
                [now, template_id],
            )
        else:
            await self.engine.execute(
                """UPDATE action_templates
                   SET failure_count = failure_count + 1,
                       confidence = MAX(0.0, confidence - 0.05),
                       updated_at = ?
                   WHERE id = ?""",
                [now, template_id],
            )

    # -------------------------------------------------------------------
    # Hypothesis Log
    # -------------------------------------------------------------------

    async def log_hypothesis(self, entry: HypothesisEntry) -> HypothesisEntry:
        await self.engine.execute(
            """INSERT INTO hypothesis_log
               (id, task_id, attempt_number, approach_summary, outcome,
                error_signature, error_full, files_changed, duration_seconds, model_used, agent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                entry.id,
                entry.task_id,
                entry.attempt_number,
                entry.approach_summary,
                entry.outcome.value,
                entry.error_signature,
                entry.error_full,
                json.dumps(entry.files_changed),
                entry.duration_seconds,
                entry.model_used,
                entry.agent_id,
            ],
        )
        return entry

    async def get_failed_approaches(self, task_id: str) -> list[HypothesisEntry]:
        rows = await self.engine.fetch_all(
            """SELECT * FROM hypothesis_log
               WHERE task_id = ? AND outcome = 'FAILURE'
               ORDER BY attempt_number ASC""",
            [task_id],
        )
        return [_row_to_hypothesis(r) for r in rows]

    async def get_hypothesis_count(self, task_id: str) -> int:
        row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM hypothesis_log WHERE task_id = ?",
            [task_id],
        )
        return row["cnt"] if row else 0

    async def has_duplicate_error(self, task_id: str, error_signature: str) -> bool:
        row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM hypothesis_log WHERE task_id = ? AND error_signature = ? AND outcome = 'FAILURE'",
            [task_id, error_signature],
        )
        return (row["cnt"] if row else 0) > 0

    async def count_error_signature(self, task_id: str, error_signature: str) -> int:
        row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM hypothesis_log WHERE task_id = ? AND error_signature = ?",
            [task_id, error_signature],
        )
        return int(row["cnt"]) if row else 0

    async def get_hypothesis_error_stats(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Get error signature statistics across tasks."""
        if project_id:
            rows = await self.engine.fetch_all(
                """SELECT h.error_signature, COUNT(*) as cnt
                   FROM hypothesis_log h
                   JOIN tasks t ON h.task_id = t.id
                   WHERE t.project_id = ? AND h.error_signature IS NOT NULL
                   GROUP BY h.error_signature
                   ORDER BY cnt DESC
                   LIMIT 20""",
                [project_id],
            )
        else:
            rows = await self.engine.fetch_all(
                """SELECT error_signature, COUNT(*) as cnt
                   FROM hypothesis_log
                   WHERE error_signature IS NOT NULL
                   GROUP BY error_signature
                   ORDER BY cnt DESC
                   LIMIT 20"""
            )
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------
    # Methodologies
    # -------------------------------------------------------------------

    async def save_methodology(self, methodology: Methodology) -> Methodology:
        await self.engine.execute(
            """INSERT INTO methodologies
               (id, problem_description, solution_code, methodology_notes,
                source_task_id, tags, language, scope, methodology_type, files_affected,
                lifecycle_state, generation, fitness_vector, parent_ids, superseded_by,
                prism_data, capability_data, novelty_score, potential_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                methodology.id,
                methodology.problem_description,
                methodology.solution_code,
                methodology.methodology_notes,
                methodology.source_task_id,
                json.dumps(methodology.tags),
                methodology.language,
                methodology.scope,
                methodology.methodology_type,
                json.dumps(methodology.files_affected),
                methodology.lifecycle_state,
                methodology.generation,
                json.dumps(methodology.fitness_vector),
                json.dumps(methodology.parent_ids),
                methodology.superseded_by,
                json.dumps(methodology.prism_data) if methodology.prism_data else None,
                json.dumps(methodology.capability_data) if methodology.capability_data else None,
                methodology.novelty_score,
                methodology.potential_score,
            ],
        )

        # Insert into FTS5 index
        await self.engine.execute(
            "INSERT INTO methodology_fts (methodology_id, problem_description, methodology_notes, tags) VALUES (?, ?, ?, ?)",
            [
                methodology.id,
                methodology.problem_description,
                methodology.methodology_notes or "",
                json.dumps(methodology.tags),
            ],
        )

        # Insert embedding into sqlite-vec if available
        if methodology.problem_embedding:
            vec_bytes = struct.pack(f"<{len(methodology.problem_embedding)}f", *methodology.problem_embedding)
            await self.engine.execute(
                "INSERT INTO methodology_embeddings (methodology_id, embedding) VALUES (?, ?)",
                [methodology.id, vec_bytes],
            )

        return methodology

    async def find_similar_methodologies(
        self, embedding: list[float], limit: int = 3
    ) -> list[tuple[Methodology, float]]:
        """Find methodologies by vector similarity. Returns (methodology, similarity) pairs."""
        vec_bytes = struct.pack(f"<{len(embedding)}f", *embedding)
        rows = await self.engine.fetch_all(
            """SELECT methodology_id, distance
               FROM methodology_embeddings
               WHERE embedding MATCH ?
               ORDER BY distance ASC
               LIMIT ?""",
            [vec_bytes, limit],
        )

        results = []
        for row in rows:
            mid = row["methodology_id"]
            distance = row["distance"]
            similarity = 1.0 - distance
            meth = await self.get_methodology(mid)
            if meth:
                results.append((meth, similarity))
        return results

    async def search_methodologies_text(self, query: str, limit: int = 5) -> list[Methodology]:
        """Full-text search on methodologies using FTS5."""
        rows = await self.engine.fetch_all(
            """SELECT methodology_id, rank
               FROM methodology_fts
               WHERE methodology_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            [query, limit],
        )

        results = []
        for row in rows:
            meth = await self.get_methodology(row["methodology_id"])
            if meth:
                results.append(meth)
        return results

    async def get_methodology(self, methodology_id: str) -> Optional[Methodology]:
        row = await self.engine.fetch_one(
            "SELECT * FROM methodologies WHERE id = ?", [methodology_id]
        )
        if row is None:
            return None
        return _row_to_methodology(row)

    async def get_methodologies_by_state(self, state: str, limit: int = 50) -> list[Methodology]:
        rows = await self.engine.fetch_all(
            "SELECT * FROM methodologies WHERE lifecycle_state = ? LIMIT ?",
            [state, limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def update_methodology_retrieval(self, methodology_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            "UPDATE methodologies SET retrieval_count = retrieval_count + 1, last_retrieved_at = ? WHERE id = ?",
            [now, methodology_id],
        )

    async def update_methodology_outcome(self, methodology_id: str, success: bool) -> None:
        if success:
            await self.engine.execute(
                "UPDATE methodologies SET success_count = success_count + 1 WHERE id = ?",
                [methodology_id],
            )
        else:
            await self.engine.execute(
                "UPDATE methodologies SET failure_count = failure_count + 1 WHERE id = ?",
                [methodology_id],
            )

    async def update_methodology_fitness(self, methodology_id: str, fitness_vector: dict[str, float]) -> None:
        await self.engine.execute(
            "UPDATE methodologies SET fitness_vector = ? WHERE id = ?",
            [json.dumps(fitness_vector), methodology_id],
        )

    async def update_methodology_lifecycle(self, methodology_id: str, new_state: str) -> None:
        await self.engine.execute(
            "UPDATE methodologies SET lifecycle_state = ? WHERE id = ?",
            [new_state, methodology_id],
        )

        # Update vmf_kappa in stored PRISM data to match new lifecycle
        row = await self.engine.fetch_one(
            "SELECT prism_data FROM methodologies WHERE id = ?", [methodology_id]
        )
        if row and row.get("prism_data"):
            try:
                from claw.embeddings.prism import _DEFAULT_KAPPA, _LIFECYCLE_KAPPA
                prism_dict = json.loads(row["prism_data"])
                prism_dict["vmf_kappa"] = _LIFECYCLE_KAPPA.get(new_state, _DEFAULT_KAPPA)
                await self.engine.execute(
                    "UPDATE methodologies SET prism_data = ? WHERE id = ?",
                    [json.dumps(prism_dict), methodology_id],
                )
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupt prism_data — leave as-is

    async def update_methodology_prism_data(self, methodology_id: str, prism_data: dict) -> None:
        """Store or replace the PRISM embedding for an existing methodology."""
        await self.engine.execute(
            "UPDATE methodologies SET prism_data = ? WHERE id = ?",
            [json.dumps(prism_data), methodology_id],
        )

    async def count_methodologies(self) -> int:
        row = await self.engine.fetch_one("SELECT COUNT(*) as cnt FROM methodologies")
        return row["cnt"] if row else 0

    async def count_active_methodologies(self) -> int:
        """Count non-dead methodologies."""
        row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM methodologies WHERE lifecycle_state != 'dead'"
        )
        return int(row["cnt"]) if row else 0

    async def count_methodologies_by_state(self) -> dict[str, int]:
        """Count methodologies grouped by lifecycle state."""
        rows = await self.engine.fetch_all(
            "SELECT lifecycle_state, COUNT(*) as cnt FROM methodologies GROUP BY lifecycle_state"
        )
        return {str(r["lifecycle_state"]): int(r["cnt"]) for r in rows}

    async def get_dead_methodologies(self, limit: int = 100) -> list[Methodology]:
        """Get dead methodologies for garbage collection."""
        rows = await self.engine.fetch_all(
            "SELECT * FROM methodologies WHERE lifecycle_state = 'dead' LIMIT ?",
            [limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def get_lowest_fitness_methodologies(
        self, states: list[str], limit: int = 50
    ) -> list[Methodology]:
        """Get methodologies with lowest fitness in given states, ordered for culling."""
        placeholders = ",".join("?" for _ in states)
        rows = await self.engine.fetch_all(
            f"""SELECT * FROM methodologies
                WHERE lifecycle_state IN ({placeholders})
                ORDER BY
                    CASE lifecycle_state
                        WHEN 'dead' THEN 0
                        WHEN 'dormant' THEN 1
                        WHEN 'declining' THEN 2
                        WHEN 'embryonic' THEN 3
                        WHEN 'viable' THEN 4
                        WHEN 'thriving' THEN 5
                    END ASC,
                    json_extract(fitness_vector, '$.total') ASC
                LIMIT ?""",
            [*states, limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def delete_methodology(self, methodology_id: str) -> bool:
        """Delete a methodology and its associated FTS5, embedding, and synergy entries."""
        existing = await self.get_methodology(methodology_id)
        if existing is None:
            return False

        await self.engine.execute(
            "DELETE FROM methodology_embeddings WHERE methodology_id = ?",
            [methodology_id],
        )
        await self.engine.execute(
            "DELETE FROM methodology_fts WHERE methodology_id = ?",
            [methodology_id],
        )
        await self.engine.execute(
            "DELETE FROM methodology_links WHERE source_id = ? OR target_id = ?",
            [methodology_id, methodology_id],
        )
        # Mark synergy explorations as stale rather than deleting
        await self.engine.execute(
            """UPDATE synergy_exploration_log SET result = 'stale'
               WHERE cap_a_id = ? OR cap_b_id = ?""",
            [methodology_id, methodology_id],
        )
        await self.engine.execute(
            "DELETE FROM methodologies WHERE id = ?",
            [methodology_id],
        )
        return True

    async def get_db_size_bytes(self) -> int:
        """Get the SQLite database file size in bytes."""
        row = await self.engine.fetch_one(
            "SELECT page_count * page_size as size FROM pragma_page_count, pragma_page_size"
        )
        return int(row["size"]) if row else 0

    async def get_methodologies_by_tag(self, tag: str, limit: int = 50) -> list[Methodology]:
        """Get methodologies containing a specific tag."""
        rows = await self.engine.fetch_all(
            "SELECT * FROM methodologies WHERE tags LIKE ? LIMIT ?",
            [f'%"{tag}"%', limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def log_governance_action(
        self,
        action_type: str,
        methodology_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a governance action for audit trail."""
        action_id = str(uuid.uuid4())
        await self.engine.execute(
            """INSERT INTO governance_log (id, action_type, methodology_id, details)
               VALUES (?, ?, ?, ?)""",
            [action_id, action_type, methodology_id, json.dumps(details or {})],
        )
        return action_id

    async def count_episodes(self) -> int:
        """Count total episodes."""
        row = await self.engine.fetch_one("SELECT COUNT(*) as cnt FROM episodes")
        return int(row["cnt"]) if row else 0

    async def delete_old_episodes(self, before_date: str) -> int:
        """Delete episodes older than the given ISO date. Returns count deleted."""
        row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM episodes WHERE created_at < ?",
            [before_date],
        )
        count = int(row["cnt"]) if row else 0
        if count > 0:
            await self.engine.execute(
                "DELETE FROM episodes WHERE created_at < ?",
                [before_date],
            )
        return count

    # -------------------------------------------------------------------
    # Methodology Links (Stigmergic co-retrieval)
    # -------------------------------------------------------------------

    async def upsert_methodology_link(
        self, source_id: str, target_id: str, link_type: str = "co_retrieval", strength: float = 1.0
    ) -> None:
        link_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            """INSERT INTO methodology_links (id, source_id, target_id, link_type, strength, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_id, target_id, link_type)
               DO UPDATE SET strength = strength + ?, updated_at = ?""",
            [link_id, source_id, target_id, link_type, strength, now, now, strength, now],
        )

    async def get_methodology_links(self, methodology_id: str) -> list[dict[str, Any]]:
        rows = await self.engine.fetch_all(
            "SELECT * FROM methodology_links WHERE source_id = ? OR target_id = ?",
            [methodology_id, methodology_id],
        )
        return [dict(r) for r in rows]

    async def get_methodology_links_by_type(
        self, methodology_id: str, link_type: str
    ) -> list[dict[str, Any]]:
        """Get links of a specific type for a methodology."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodology_links
               WHERE (source_id = ? OR target_id = ?) AND link_type = ?""",
            [methodology_id, methodology_id, link_type],
        )
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------
    # Capability Data
    # -------------------------------------------------------------------

    async def update_methodology_capability_data(
        self, methodology_id: str, capability_data: dict
    ) -> None:
        """Store or replace structured capability_data for a methodology."""
        await self.engine.execute(
            "UPDATE methodologies SET capability_data = ? WHERE id = ?",
            [json.dumps(capability_data), methodology_id],
        )

    async def get_methodologies_with_capabilities(self, limit: int = 100) -> list[Methodology]:
        """Get methodologies that have capability_data populated."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodologies
               WHERE capability_data IS NOT NULL AND lifecycle_state != 'dead'
               LIMIT ?""",
            [limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def get_methodologies_without_capability_data(self, limit: int = 50) -> list[Methodology]:
        """Get methodologies missing capability_data for enrichment."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodologies
               WHERE capability_data IS NULL AND lifecycle_state != 'dead'
               ORDER BY created_at ASC
               LIMIT ?""",
            [limit],
        )
        return [_row_to_methodology(r) for r in rows]

    # -------------------------------------------------------------------
    # Novelty Scoring
    # -------------------------------------------------------------------

    async def update_methodology_novelty_scores(
        self, methodology_id: str, novelty: float, potential: float
    ) -> None:
        """Persist novelty and potential scores for a methodology."""
        await self.engine.execute(
            "UPDATE methodologies SET novelty_score = ?, potential_score = ? WHERE id = ?",
            [novelty, potential, methodology_id],
        )

    async def get_most_novel_methodologies(
        self, limit: int = 10, min_novelty: float = 0.0
    ) -> list[Methodology]:
        """Get methodologies ordered by novelty_score DESC."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodologies
               WHERE novelty_score IS NOT NULL AND novelty_score >= ?
                 AND lifecycle_state != 'dead'
               ORDER BY novelty_score DESC
               LIMIT ?""",
            [min_novelty, limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def get_high_potential_methodologies(
        self, limit: int = 10, min_potential: float = 0.0
    ) -> list[Methodology]:
        """Get methodologies ordered by potential_score DESC."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodologies
               WHERE potential_score IS NOT NULL AND potential_score >= ?
                 AND lifecycle_state != 'dead'
               ORDER BY potential_score DESC
               LIMIT ?""",
            [min_potential, limit],
        )
        return [_row_to_methodology(r) for r in rows]

    async def get_embedding_centroid(self) -> list[float]:
        """Compute mean embedding vector from all methodology_embeddings.

        Returns a 384-dimensional centroid vector, or empty list if no embeddings.
        """
        rows = await self.engine.fetch_all(
            "SELECT embedding FROM methodology_embeddings"
        )
        if not rows:
            return []

        dim = 384
        centroid = [0.0] * dim
        count = 0
        for row in rows:
            raw = row["embedding"]
            if raw is None:
                continue
            vec = list(struct.unpack(f"<{dim}f", raw))
            for i in range(dim):
                centroid[i] += vec[i]
            count += 1

        if count == 0:
            return []
        return [c / count for c in centroid]

    async def get_domain_distribution(self) -> dict[str, int]:
        """Count occurrences of each domain tag across all capability_data.

        Parses the domain list from capability_data JSON for each methodology.
        """
        rows = await self.engine.fetch_all(
            """SELECT capability_data FROM methodologies
               WHERE capability_data IS NOT NULL AND lifecycle_state != 'dead'"""
        )
        dist: dict[str, int] = {}
        for row in rows:
            raw = row["capability_data"]
            if not raw:
                continue
            try:
                cap = json.loads(raw) if isinstance(raw, str) else raw
                for domain in cap.get("domain", []):
                    dist[domain] = dist.get(domain, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue
        return dist

    async def get_type_distribution(self) -> dict[str, int]:
        """Count occurrences of each capability_type across methodologies."""
        rows = await self.engine.fetch_all(
            """SELECT capability_data FROM methodologies
               WHERE capability_data IS NOT NULL AND lifecycle_state != 'dead'"""
        )
        dist: dict[str, int] = {}
        for row in rows:
            raw = row["capability_data"]
            if not raw:
                continue
            try:
                cap = json.loads(raw) if isinstance(raw, str) else raw
                ctype = cap.get("capability_type", "transformation")
                dist[ctype] = dist.get(ctype, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue
        return dist

    # -------------------------------------------------------------------
    # Synergy Exploration Log
    # -------------------------------------------------------------------

    async def record_synergy_exploration(self, exploration: SynergyExploration) -> None:
        """Record an explored capability pair. Canonical ordering enforced by caller."""
        await self.engine.execute(
            """INSERT INTO synergy_exploration_log
               (id, cap_a_id, cap_b_id, explored_at, result,
                synergy_score, synergy_type, edge_id, exploration_method, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cap_a_id, cap_b_id) DO UPDATE SET
                   result = excluded.result,
                   synergy_score = excluded.synergy_score,
                   synergy_type = excluded.synergy_type,
                   edge_id = excluded.edge_id,
                   exploration_method = excluded.exploration_method,
                   details = excluded.details,
                   explored_at = excluded.explored_at""",
            [
                exploration.id,
                exploration.cap_a_id,
                exploration.cap_b_id,
                exploration.explored_at.isoformat() if exploration.explored_at else None,
                exploration.result,
                exploration.synergy_score,
                exploration.synergy_type,
                exploration.edge_id,
                exploration.exploration_method,
                json.dumps(exploration.details),
            ],
        )

    async def get_synergy_exploration(
        self, cap_a_id: str, cap_b_id: str
    ) -> Optional[SynergyExploration]:
        """Get exploration record for a canonical pair (a < b alphabetically)."""
        a, b = (cap_a_id, cap_b_id) if cap_a_id < cap_b_id else (cap_b_id, cap_a_id)
        row = await self.engine.fetch_one(
            "SELECT * FROM synergy_exploration_log WHERE cap_a_id = ? AND cap_b_id = ?",
            [a, b],
        )
        if row is None:
            return None
        return _row_to_synergy_exploration(row)

    async def get_unexplored_pairs(
        self, cap_id: str, candidate_ids: list[str]
    ) -> list[str]:
        """Filter candidate_ids to only those NOT yet explored with cap_id."""
        if not candidate_ids:
            return []
        unexplored = []
        for cid in candidate_ids:
            a, b = (cap_id, cid) if cap_id < cid else (cid, cap_id)
            row = await self.engine.fetch_one(
                "SELECT 1 FROM synergy_exploration_log WHERE cap_a_id = ? AND cap_b_id = ?",
                [a, b],
            )
            if row is None:
                unexplored.append(cid)
        return unexplored

    async def get_synergy_stats(self) -> dict[str, Any]:
        """Get aggregate stats from the synergy exploration log."""
        total_row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM synergy_exploration_log"
        )
        total = int(total_row["cnt"]) if total_row else 0

        by_result = await self.engine.fetch_all(
            "SELECT result, COUNT(*) as cnt FROM synergy_exploration_log GROUP BY result"
        )
        result_counts = {str(r["result"]): int(r["cnt"]) for r in by_result}

        avg_row = await self.engine.fetch_one(
            "SELECT AVG(synergy_score) as avg_score FROM synergy_exploration_log WHERE synergy_score IS NOT NULL"
        )
        avg_score = float(avg_row["avg_score"]) if avg_row and avg_row["avg_score"] else 0.0

        edge_row = await self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM methodology_links WHERE link_type != 'co_retrieval'"
        )
        synergy_edges = int(edge_row["cnt"]) if edge_row else 0

        return {
            "total_explored": total,
            "by_result": result_counts,
            "avg_synergy_score": round(avg_score, 4),
            "synergy_edges": synergy_edges,
        }

    async def mark_stale_explorations(self, methodology_id: str) -> int:
        """Mark explorations as stale when a methodology is deleted."""
        rows = await self.engine.fetch_all(
            """SELECT id FROM synergy_exploration_log
               WHERE cap_a_id = ? OR cap_b_id = ?""",
            [methodology_id, methodology_id],
        )
        count = len(rows)
        if count > 0:
            await self.engine.execute(
                """UPDATE synergy_exploration_log SET result = 'stale'
                   WHERE cap_a_id = ? OR cap_b_id = ?""",
                [methodology_id, methodology_id],
            )
        return count

    # -------------------------------------------------------------------
    # Capability Graph Traversal
    # -------------------------------------------------------------------

    async def get_synergy_graph(
        self, methodology_id: str, depth: int = 2
    ) -> dict[str, Any]:
        """BFS traversal of synergy edges from a starting methodology.

        depth=1 means follow one hop (root + direct neighbors).
        Returns a dict with 'nodes' (set of methodology IDs) and
        'edges' (list of (source, target, link_type, strength) tuples).
        """
        visited: set[str] = set()
        edges: list[tuple[str, str, str, float]] = []
        current_level = {methodology_id}

        for _ in range(depth + 1):
            if not current_level:
                break
            next_level: set[str] = set()
            for node_id in current_level:
                if node_id in visited:
                    continue
                visited.add(node_id)
                links = await self.engine.fetch_all(
                    """SELECT source_id, target_id, link_type, strength
                       FROM methodology_links
                       WHERE source_id = ? OR target_id = ?""",
                    [node_id, node_id],
                )
                for link in links:
                    src = link["source_id"]
                    tgt = link["target_id"]
                    edge_tuple = (src, tgt, link["link_type"], link["strength"])
                    if edge_tuple not in edges:
                        edges.append(edge_tuple)
                    neighbor = tgt if src == node_id else src
                    if neighbor not in visited:
                        next_level.add(neighbor)
            current_level = next_level

        return {
            "nodes": visited,
            "edges": edges,
        }

    async def get_complementary_capabilities(
        self, methodology_id: str
    ) -> list[Methodology]:
        """Follow feeds_into, enhances, and synergy edges to find complementary capabilities."""
        complementary_ids: set[str] = set()
        target_link_types = ("feeds_into", "enhances", "synergy")

        for lt in target_link_types:
            links = await self.engine.fetch_all(
                """SELECT source_id, target_id FROM methodology_links
                   WHERE (source_id = ? OR target_id = ?) AND link_type = ?""",
                [methodology_id, methodology_id, lt],
            )
            for link in links:
                neighbor = link["target_id"] if link["source_id"] == methodology_id else link["source_id"]
                complementary_ids.add(neighbor)

        results = []
        for cid in complementary_ids:
            meth = await self.get_methodology(cid)
            if meth and meth.lifecycle_state != "dead":
                results.append(meth)
        return results

    # -------------------------------------------------------------------
    # Knowledge Browser Queries
    # -------------------------------------------------------------------

    async def get_top_synergy_edges(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get top synergy edges ordered by score, with capability summaries."""
        rows = await self.engine.fetch_all(
            """SELECT sel.cap_a_id, sel.cap_b_id, sel.synergy_score,
                      sel.synergy_type, sel.details
               FROM synergy_exploration_log sel
               WHERE sel.result = 'synergy'
               ORDER BY sel.synergy_score DESC
               LIMIT ?""",
            [limit],
        )
        results = []
        for row in rows:
            cap_a = await self.get_methodology(row["cap_a_id"])
            cap_b = await self.get_methodology(row["cap_b_id"])
            details_raw = row["details"]
            details = json.loads(details_raw) if isinstance(details_raw, str) else (details_raw or {})
            results.append({
                "cap_a_id": row["cap_a_id"],
                "cap_b_id": row["cap_b_id"],
                "cap_a_summary": (cap_a.problem_description[:80] if cap_a else "(deleted)"),
                "cap_b_summary": (cap_b.problem_description[:80] if cap_b else "(deleted)"),
                "cap_a_domains": (cap_a.capability_data or {}).get("domain", []) if cap_a else [],
                "cap_b_domains": (cap_b.capability_data or {}).get("domain", []) if cap_b else [],
                "synergy_score": row["synergy_score"] or 0.0,
                "synergy_type": row["synergy_type"] or "",
                "details": details,
            })
        return results

    async def get_novelty_potential_distribution(self) -> dict[str, Any]:
        """Get summary statistics for novelty and potential scores."""
        row = await self.engine.fetch_one(
            """SELECT
                  COUNT(*) as total,
                  AVG(novelty_score) as avg_novelty,
                  MAX(novelty_score) as max_novelty,
                  MIN(novelty_score) as min_novelty,
                  AVG(potential_score) as avg_potential,
                  MAX(potential_score) as max_potential,
                  MIN(potential_score) as min_potential
               FROM methodologies
               WHERE novelty_score IS NOT NULL"""
        )
        if row is None or row["total"] == 0:
            return {
                "total": 0, "avg_novelty": 0.0, "max_novelty": 0.0,
                "min_novelty": 0.0, "avg_potential": 0.0,
                "max_potential": 0.0, "min_potential": 0.0,
            }
        return {
            "total": int(row["total"]),
            "avg_novelty": round(float(row["avg_novelty"] or 0), 4),
            "max_novelty": round(float(row["max_novelty"] or 0), 4),
            "min_novelty": round(float(row["min_novelty"] or 0), 4),
            "avg_potential": round(float(row["avg_potential"] or 0), 4),
            "max_potential": round(float(row["max_potential"] or 0), 4),
            "min_potential": round(float(row["min_potential"] or 0), 4),
        }

    async def get_cross_domain_capabilities(
        self, min_domains: int = 2, limit: int = 20
    ) -> list[Methodology]:
        """Get capabilities spanning multiple knowledge domains (bridge capabilities)."""
        rows = await self.engine.fetch_all(
            """SELECT * FROM methodologies
               WHERE capability_data IS NOT NULL AND lifecycle_state != 'dead'"""
        )
        bridges = []
        for row in rows:
            raw = row["capability_data"]
            if not raw:
                continue
            try:
                cap = json.loads(raw) if isinstance(raw, str) else raw
                domains = cap.get("domain", [])
                if len(domains) >= min_domains:
                    bridges.append(_row_to_methodology(row))
            except (json.JSONDecodeError, TypeError):
                continue
        # Sort by number of domains descending, then by novelty_score descending
        bridges.sort(
            key=lambda m: (
                len((m.capability_data or {}).get("domain", [])),
                m.novelty_score or 0,
            ),
            reverse=True,
        )
        return bridges[:limit]

    async def get_methodology_by_prefix(self, prefix: str) -> Optional[Methodology]:
        """Find a methodology by ID prefix (first 6+ chars)."""
        rows = await self.engine.fetch_all(
            "SELECT * FROM methodologies WHERE id LIKE ? LIMIT 2",
            [f"{prefix}%"],
        )
        if len(rows) == 1:
            return _row_to_methodology(rows[0])
        if len(rows) > 1:
            # Ambiguous prefix — try exact match first
            for row in rows:
                if row["id"] == prefix:
                    return _row_to_methodology(row)
            # Return first match as best effort
            return _row_to_methodology(rows[0])
        return None

    # -------------------------------------------------------------------
    # Peer Reviews
    # -------------------------------------------------------------------

    async def save_peer_review(self, review: PeerReview) -> PeerReview:
        await self.engine.execute(
            """INSERT INTO peer_reviews
               (id, task_id, model_used, diagnosis, recommended_approach, reasoning)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                review.id,
                review.task_id,
                review.model_used,
                review.diagnosis,
                review.recommended_approach,
                review.reasoning,
            ],
        )
        return review

    async def get_peer_reviews(self, task_id: str) -> list[PeerReview]:
        rows = await self.engine.fetch_all(
            "SELECT * FROM peer_reviews WHERE task_id = ? ORDER BY created_at DESC",
            [task_id],
        )
        return [_row_to_peer_review(r) for r in rows]

    # -------------------------------------------------------------------
    # Context Snapshots
    # -------------------------------------------------------------------

    async def save_context_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        await self.engine.execute(
            """INSERT INTO context_snapshots
               (id, task_id, attempt_number, git_ref, file_manifest)
               VALUES (?, ?, ?, ?, ?)""",
            [
                snapshot.id,
                snapshot.task_id,
                snapshot.attempt_number,
                snapshot.git_ref,
                json.dumps(snapshot.file_manifest) if snapshot.file_manifest else None,
            ],
        )
        return snapshot

    async def get_latest_snapshot(self, task_id: str) -> Optional[ContextSnapshot]:
        row = await self.engine.fetch_one(
            "SELECT * FROM context_snapshots WHERE task_id = ? ORDER BY attempt_number DESC LIMIT 1",
            [task_id],
        )
        if row is None:
            return None
        return _row_to_context_snapshot(row)

    # -------------------------------------------------------------------
    # Token Costs
    # -------------------------------------------------------------------

    async def save_token_cost(self, record: TokenCostRecord) -> TokenCostRecord:
        await self.engine.execute(
            """INSERT INTO token_costs
               (id, task_id, run_id, agent_role, agent_id, model_used,
                input_tokens, output_tokens, total_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                record.id,
                record.task_id,
                record.run_id,
                record.agent_role,
                record.agent_id,
                record.model_used,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.cost_usd,
            ],
        )
        return record

    async def get_token_cost_summary(self, task_id: Optional[str] = None) -> dict[str, Any]:
        if task_id:
            row = await self.engine.fetch_one(
                """SELECT COUNT(*) as calls, SUM(input_tokens) as input_tok,
                   SUM(output_tokens) as output_tok, SUM(total_tokens) as total_tok,
                   SUM(cost_usd) as total_cost
                   FROM token_costs WHERE task_id = ?""",
                [task_id],
            )
        else:
            row = await self.engine.fetch_one(
                """SELECT COUNT(*) as calls, SUM(input_tokens) as input_tok,
                   SUM(output_tokens) as output_tok, SUM(total_tokens) as total_tok,
                   SUM(cost_usd) as total_cost
                   FROM token_costs"""
            )
        if row is None:
            return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}
        return {
            "calls": row["calls"] or 0,
            "input_tokens": row["input_tok"] or 0,
            "output_tokens": row["output_tok"] or 0,
            "total_tokens": row["total_tok"] or 0,
            "total_cost_usd": row["total_cost"] or 0.0,
        }

    # -------------------------------------------------------------------
    # CLAW-specific: Agent Scores
    # -------------------------------------------------------------------

    async def get_agent_scores(self, agent_id: Optional[str] = None) -> list[dict[str, Any]]:
        if agent_id:
            rows = await self.engine.fetch_all(
                "SELECT * FROM agent_scores WHERE agent_id = ?", [agent_id]
            )
        else:
            rows = await self.engine.fetch_all("SELECT * FROM agent_scores")
        return [dict(r) for r in rows]

    async def update_agent_score(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        duration_seconds: float = 0.0,
        quality_score: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        score_id = str(uuid.uuid4())

        # Upsert: update if exists, insert if not
        existing = await self.engine.fetch_one(
            "SELECT * FROM agent_scores WHERE agent_id = ? AND task_type = ?",
            [agent_id, task_type],
        )

        if existing:
            total = existing["total_attempts"] + 1
            new_avg_dur = (existing["avg_duration_seconds"] * existing["total_attempts"] + duration_seconds) / total
            new_avg_qual = (existing["avg_quality_score"] * existing["total_attempts"] + quality_score) / total
            new_avg_cost = (existing["avg_cost_usd"] * existing["total_attempts"] + cost_usd) / total

            await self.engine.execute(
                """UPDATE agent_scores SET
                   successes = successes + ?, failures = failures + ?,
                   total_attempts = total_attempts + 1,
                   avg_duration_seconds = ?, avg_quality_score = ?, avg_cost_usd = ?,
                   last_used_at = ?, updated_at = ?
                   WHERE agent_id = ? AND task_type = ?""",
                [
                    1 if success else 0,
                    0 if success else 1,
                    new_avg_dur, new_avg_qual, new_avg_cost,
                    now, now,
                    agent_id, task_type,
                ],
            )
        else:
            await self.engine.execute(
                """INSERT INTO agent_scores
                   (id, agent_id, task_type, successes, failures, total_attempts,
                    avg_duration_seconds, avg_quality_score, avg_cost_usd,
                    last_used_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                [
                    score_id, agent_id, task_type,
                    1 if success else 0,
                    0 if success else 1,
                    duration_seconds, quality_score, cost_usd,
                    now, now, now,
                ],
            )

    # -------------------------------------------------------------------
    # CLAW-specific: Prompt Variants
    # -------------------------------------------------------------------

    async def save_prompt_variant(
        self,
        prompt_name: str,
        variant_label: str,
        content: str,
        agent_id: Optional[str] = None,
        is_active: bool = False,
    ) -> str:
        variant_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        await self.engine.execute(
            """INSERT INTO prompt_variants
               (id, prompt_name, variant_label, content, agent_id, is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [variant_id, prompt_name, variant_label, content, agent_id, 1 if is_active else 0, now, now],
        )
        return variant_id

    # -------------------------------------------------------------------
    # CLAW-specific: Fleet Repos
    # -------------------------------------------------------------------

    async def get_fleet_repos(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        if status:
            rows = await self.engine.fetch_all(
                "SELECT * FROM fleet_repos WHERE status = ? ORDER BY priority DESC",
                [status],
            )
        else:
            rows = await self.engine.fetch_all(
                "SELECT * FROM fleet_repos ORDER BY priority DESC"
            )
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------
    # CLAW-specific: Episodes
    # -------------------------------------------------------------------

    async def log_episode(
        self,
        session_id: str,
        event_type: str,
        event_data: dict[str, Any],
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        cycle_level: Optional[str] = None,
    ) -> str:
        episode_id = str(uuid.uuid4())
        await self.engine.execute(
            """INSERT INTO episodes
               (id, project_id, session_id, event_type, event_data, agent_id, task_id, cycle_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [episode_id, project_id, session_id, event_type, json.dumps(event_data), agent_id, task_id, cycle_level],
        )
        return episode_id


# ---------------------------------------------------------------------------
# Row → Model converters
# ---------------------------------------------------------------------------

def _row_to_project(row: dict[str, Any]) -> Project:
    tech_stack = row.get("tech_stack", "{}")
    if isinstance(tech_stack, str):
        tech_stack = json.loads(tech_stack)
    banned = row.get("banned_dependencies", "[]")
    if isinstance(banned, str):
        banned = json.loads(banned)
    return Project(
        id=row["id"],
        name=row["name"],
        repo_path=row["repo_path"],
        tech_stack=tech_stack,
        project_rules=row.get("project_rules"),
        banned_dependencies=banned,
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
    )


def _row_to_task(row: dict[str, Any]) -> Task:
    execution_steps = row.get("execution_steps", "[]")
    if isinstance(execution_steps, str):
        execution_steps = json.loads(execution_steps)

    acceptance_checks = row.get("acceptance_checks", "[]")
    if isinstance(acceptance_checks, str):
        acceptance_checks = json.loads(acceptance_checks)

    return Task(
        id=row["id"],
        project_id=row["project_id"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        priority=row.get("priority", 0),
        task_type=row.get("task_type"),
        recommended_agent=row.get("recommended_agent"),
        assigned_agent=row.get("assigned_agent"),
        action_template_id=row.get("action_template_id"),
        execution_steps=execution_steps,
        acceptance_checks=acceptance_checks,
        context_snapshot_id=row.get("context_snapshot_id"),
        attempt_count=row.get("attempt_count", 0),
        escalation_count=row.get("escalation_count", 0),
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
        completed_at=_parse_dt(row.get("completed_at")),
    )


def _row_to_hypothesis(row: dict[str, Any]) -> HypothesisEntry:
    files = row.get("files_changed", "[]")
    if isinstance(files, str):
        files = json.loads(files)
    return HypothesisEntry(
        id=row["id"],
        task_id=row["task_id"],
        attempt_number=row["attempt_number"],
        approach_summary=row["approach_summary"],
        outcome=HypothesisOutcome(row["outcome"]),
        error_signature=row.get("error_signature"),
        error_full=row.get("error_full"),
        files_changed=files,
        duration_seconds=row.get("duration_seconds"),
        model_used=row.get("model_used"),
        agent_id=row.get("agent_id"),
        created_at=_parse_dt(row.get("created_at")),
    )


def _row_to_methodology(row: dict[str, Any]) -> Methodology:
    tags = row.get("tags", "[]")
    if isinstance(tags, str):
        tags = json.loads(tags)
    files = row.get("files_affected", "[]")
    if isinstance(files, str):
        files = json.loads(files)
    fv = row.get("fitness_vector", "{}")
    if isinstance(fv, str):
        fv = json.loads(fv)
    parents = row.get("parent_ids", "[]")
    if isinstance(parents, str):
        parents = json.loads(parents)

    raw_prism = row.get("prism_data")
    prism_data = json.loads(raw_prism) if isinstance(raw_prism, str) else None

    raw_cap = row.get("capability_data")
    capability_data = json.loads(raw_cap) if isinstance(raw_cap, str) else None

    return Methodology(
        id=row["id"],
        problem_description=row["problem_description"],
        solution_code=row["solution_code"],
        methodology_notes=row.get("methodology_notes"),
        source_task_id=row.get("source_task_id"),
        tags=tags,
        language=row.get("language"),
        scope=row.get("scope", "project"),
        methodology_type=row.get("methodology_type"),
        files_affected=files,
        created_at=_parse_dt(row.get("created_at")),
        lifecycle_state=row.get("lifecycle_state", "viable"),
        retrieval_count=row.get("retrieval_count", 0),
        success_count=row.get("success_count", 0),
        failure_count=row.get("failure_count", 0),
        last_retrieved_at=_parse_dt(row.get("last_retrieved_at")),
        generation=row.get("generation", 0),
        fitness_vector=fv,
        parent_ids=parents,
        superseded_by=row.get("superseded_by"),
        prism_data=prism_data,
        capability_data=capability_data,
        novelty_score=row.get("novelty_score"),
        potential_score=row.get("potential_score"),
    )


def _row_to_action_template(row: dict[str, Any]) -> ActionTemplate:
    execution_steps = row.get("execution_steps", "[]")
    if isinstance(execution_steps, str):
        execution_steps = json.loads(execution_steps)

    acceptance_checks = row.get("acceptance_checks", "[]")
    if isinstance(acceptance_checks, str):
        acceptance_checks = json.loads(acceptance_checks)

    rollback_steps = row.get("rollback_steps", "[]")
    if isinstance(rollback_steps, str):
        rollback_steps = json.loads(rollback_steps)

    preconditions = row.get("preconditions", "[]")
    if isinstance(preconditions, str):
        preconditions = json.loads(preconditions)

    return ActionTemplate(
        id=row["id"],
        title=row["title"],
        problem_pattern=row["problem_pattern"],
        execution_steps=execution_steps,
        acceptance_checks=acceptance_checks,
        rollback_steps=rollback_steps,
        preconditions=preconditions,
        source_methodology_id=row.get("source_methodology_id"),
        source_repo=row.get("source_repo"),
        confidence=float(row.get("confidence", 0.5) or 0.5),
        success_count=int(row.get("success_count", 0) or 0),
        failure_count=int(row.get("failure_count", 0) or 0),
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
    )


def _row_to_peer_review(row: dict[str, Any]) -> PeerReview:
    return PeerReview(
        id=row["id"],
        task_id=row["task_id"],
        model_used=row["model_used"],
        diagnosis=row["diagnosis"],
        recommended_approach=row.get("recommended_approach"),
        reasoning=row.get("reasoning"),
        created_at=_parse_dt(row.get("created_at")),
    )


def _row_to_context_snapshot(row: dict[str, Any]) -> ContextSnapshot:
    manifest = row.get("file_manifest")
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    return ContextSnapshot(
        id=row["id"],
        task_id=row["task_id"],
        attempt_number=row["attempt_number"],
        git_ref=row["git_ref"],
        file_manifest=manifest,
        created_at=_parse_dt(row.get("created_at")),
    )


def _row_to_synergy_exploration(row: dict[str, Any]) -> SynergyExploration:
    details = row.get("details", "{}")
    if isinstance(details, str):
        details = json.loads(details)
    return SynergyExploration(
        id=row["id"],
        cap_a_id=row["cap_a_id"],
        cap_b_id=row["cap_b_id"],
        explored_at=_parse_dt(row.get("explored_at")),
        result=row.get("result", "pending"),
        synergy_score=row.get("synergy_score"),
        synergy_type=row.get("synergy_type"),
        edge_id=row.get("edge_id"),
        exploration_method=row.get("exploration_method"),
        details=details,
    )


def _parse_dt(val: Any) -> Optional[datetime]:
    """Parse ISO-8601 datetime string from SQLite TEXT column."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None
