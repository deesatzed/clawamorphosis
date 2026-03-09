"""Memory governance — quotas, pruning, GC, dedup, and monitoring.

Central governor that orchestrates all memory hygiene operations.
Called periodically (every N cycles) and on startup to keep the
methodology store lean and free of dead weight.

Operations (in order during a sweep):
1. Lifecycle sweep — run_periodic_sweep() from lifecycle.py
2. Garbage collect dead methodologies — remove from all 3 stores
3. Enforce quota — cull lowest-fitness if over limit
4. Prune episodes — apply retention policy
5. Log storage stats — audit trail in governance_log
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from claw.core.config import GovernanceConfig
from claw.db.repository import Repository

logger = logging.getLogger("claw.memory.governance")


@dataclass
class StorageStats:
    """Memory system storage statistics."""
    total_methodologies: int = 0
    by_state: dict[str, int] = field(default_factory=dict)
    total_embeddings: int = 0
    total_episodes: int = 0
    db_size_bytes: int = 0
    quota_limit: int = 0
    quota_used_pct: float = 0.0


@dataclass
class GovernanceReport:
    """Results of a governance sweep."""
    dead_collected: int = 0
    quota_culled: int = 0
    episodes_pruned: int = 0
    lifecycle_transitions: dict[str, int] = field(default_factory=dict)
    storage_stats: Optional[StorageStats] = None
    duplicates_blocked: int = 0
    sweep_duration_seconds: float = 0.0


class MemoryGovernor:
    """Central memory governance: quotas, pruning, GC, monitoring.

    Dependencies:
        repository: Database access.
        config: GovernanceConfig for thresholds.
    """

    def __init__(
        self,
        repository: Repository,
        config: Optional[GovernanceConfig] = None,
    ):
        self.repository = repository
        self.config = config or GovernanceConfig()
        self._cycle_count: int = 0
        self._duplicates_blocked: int = 0

    async def run_full_sweep(self) -> GovernanceReport:
        """Execute all governance operations in sequence.

        Order matters:
        1. Lifecycle sweep (transition time-based states)
        2. Garbage collect dead methodologies
        3. Enforce quotas (if over limit, cull lowest-fitness)
        4. Prune episodic memory
        5. Compute and log storage stats
        """
        start = time.monotonic()
        report = GovernanceReport()

        # 1. Lifecycle sweep
        from claw.memory.lifecycle import run_periodic_sweep
        transitions = await run_periodic_sweep(self.repository)
        report.lifecycle_transitions = transitions

        # 2. Garbage collect dead
        if self.config.gc_dead_on_sweep:
            report.dead_collected = await self.garbage_collect_dead()

        # 3. Enforce quota
        report.quota_culled = await self.enforce_methodology_quota()

        # 4. Prune episodes
        report.episodes_pruned = await self._prune_episodes()

        # 5. Storage stats
        report.storage_stats = await self.get_storage_stats()
        report.duplicates_blocked = self._duplicates_blocked
        report.sweep_duration_seconds = time.monotonic() - start

        # Log to governance_log
        await self.repository.log_governance_action(
            action_type="sweep",
            details={
                "dead_collected": report.dead_collected,
                "quota_culled": report.quota_culled,
                "episodes_pruned": report.episodes_pruned,
                "lifecycle_transitions": report.lifecycle_transitions,
                "duration_seconds": round(report.sweep_duration_seconds, 3),
            },
        )

        logger.info(
            "Governance sweep complete: gc=%d, culled=%d, episodes=%d, transitions=%s (%.2fs)",
            report.dead_collected,
            report.quota_culled,
            report.episodes_pruned,
            report.lifecycle_transitions or "none",
            report.sweep_duration_seconds,
        )
        return report

    async def garbage_collect_dead(self) -> int:
        """Delete dead methodologies from DB, FTS5, and sqlite-vec.

        Logs each deletion to governance_log before removing.
        """
        dead = await self.repository.get_dead_methodologies(limit=500)
        if not dead:
            return 0

        deleted = 0
        for m in dead:
            await self.repository.log_governance_action(
                action_type="gc_dead",
                methodology_id=m.id,
                details={
                    "problem_description": m.problem_description[:200],
                    "lifecycle_state": m.lifecycle_state,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                },
            )
            success = await self.repository.delete_methodology(m.id)
            if success:
                deleted += 1

        if deleted > 0:
            logger.info("Garbage collected %d dead methodologies", deleted)
        return deleted

    async def enforce_methodology_quota(self) -> int:
        """If total active methodologies exceed quota, cull lowest-fitness.

        Cull order: dormant, declining, embryonic.
        Never culls thriving or viable methodologies.
        """
        active_count = await self.repository.count_active_methodologies()
        quota = self.config.max_methodologies

        if active_count <= quota:
            # Check warning threshold
            if active_count >= quota * self.config.quota_warning_pct:
                logger.warning(
                    "Methodology quota warning: %d/%d (%.0f%%)",
                    active_count, quota, (active_count / quota) * 100,
                )
            return 0

        # Need to cull (active_count - quota) methodologies
        to_cull = active_count - quota
        logger.warning(
            "Methodology quota exceeded: %d/%d — culling %d lowest-fitness",
            active_count, quota, to_cull,
        )

        # Get candidates in cull order (dormant, declining, embryonic)
        candidates = await self.repository.get_lowest_fitness_methodologies(
            states=["dormant", "declining", "embryonic"],
            limit=to_cull,
        )

        culled = 0
        for m in candidates:
            if culled >= to_cull:
                break
            await self.repository.log_governance_action(
                action_type="quota_cull",
                methodology_id=m.id,
                details={
                    "problem_description": m.problem_description[:200],
                    "lifecycle_state": m.lifecycle_state,
                    "fitness": m.fitness_vector.get("total", 0.0) if m.fitness_vector else 0.0,
                },
            )
            success = await self.repository.delete_methodology(m.id)
            if success:
                culled += 1

        if culled < to_cull:
            logger.warning(
                "Could only cull %d/%d — remaining %d are thriving/viable (protected)",
                culled, to_cull, to_cull - culled,
            )
        return culled

    async def check_pre_save_dedup(
        self,
        problem_description: str,
        embedding: Optional[list[float]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if a methodology should be saved or is a near-duplicate.

        Called BEFORE save_solution() to catch duplicates at insertion time.

        Returns:
            (should_save, existing_id). If should_save is False,
            existing_id is the matching methodology that already covers this.
        """
        if not self.config.dedup_enabled:
            return True, None

        if embedding is None:
            return True, None

        threshold = similarity_threshold or self.config.dedup_similarity_threshold

        try:
            similar_pairs = await self.repository.find_similar_methodologies(
                embedding=embedding, limit=5,
            )
            for existing, similarity in similar_pairs:
                if similarity >= threshold:
                    # Skip dead/dormant — they don't count as duplicates
                    if existing.lifecycle_state in ("dead", "dormant"):
                        continue
                    self._duplicates_blocked += 1
                    logger.info(
                        "Pre-save dedup: blocked (sim=%.3f >= %.3f) — existing=%s",
                        similarity, threshold, existing.id,
                    )
                    await self.repository.log_governance_action(
                        action_type="dedup_block",
                        methodology_id=existing.id,
                        details={
                            "blocked_description": problem_description[:200],
                            "similarity": round(similarity, 4),
                            "threshold": threshold,
                        },
                    )
                    return False, existing.id
        except Exception as e:
            logger.warning("Pre-save dedup check failed (allowing save): %s", e)

        return True, None

    async def get_storage_stats(self) -> StorageStats:
        """Compute storage statistics for monitoring."""
        by_state = await self.repository.count_methodologies_by_state()
        total = sum(by_state.values())
        active = total - by_state.get("dead", 0)

        episode_count = await self.repository.count_episodes()
        db_size = await self.repository.get_db_size_bytes()

        quota = self.config.max_methodologies
        pct = (active / quota * 100) if quota > 0 else 0.0

        stats = StorageStats(
            total_methodologies=total,
            by_state=by_state,
            total_episodes=episode_count,
            db_size_bytes=db_size,
            quota_limit=quota,
            quota_used_pct=round(pct, 1),
        )

        # DB size warning
        max_bytes = self.config.max_db_size_mb * 1024 * 1024
        if db_size > max_bytes:
            logger.warning(
                "DB size exceeds limit: %d MB > %d MB",
                db_size // (1024 * 1024),
                self.config.max_db_size_mb,
            )

        return stats

    async def maybe_run_sweep(self) -> Optional[GovernanceReport]:
        """Conditionally run governance sweep based on cycle count.

        Called after every MicroClaw cycle. Runs full sweep every
        N cycles (configured via sweep_interval_cycles).
        """
        self._cycle_count += 1
        if self._cycle_count % self.config.sweep_interval_cycles != 0:
            return None
        return await self.run_full_sweep()

    async def _prune_episodes(self) -> int:
        """Prune old episodes using the configured retention days."""
        from datetime import UTC, datetime, timedelta
        cutoff = (
            datetime.now(UTC) - timedelta(days=self.config.episodic_retention_days)
        ).isoformat()
        return await self.repository.delete_old_episodes(cutoff)
