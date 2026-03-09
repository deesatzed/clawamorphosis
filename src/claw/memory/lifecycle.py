"""Methodology lifecycle state machine and competitive exclusion.

Implements Gause's competitive exclusion principle for memory management:
two memories cannot indefinitely occupy the same niche. Low-fitness memories
decline through natural selection while high-fitness memories thrive.

State transitions:
    embryonic -> viable    : on first successful outcome
    viable -> thriving     : fitness_score > 0.7 AND success_count >= 3
    thriving -> declining  : fitness_score drops below 0.4
    viable -> declining    : failure_count > success_count AND retrieval_count >= 3
    declining -> dormant   : not retrieved for 180 days
    dormant -> dead        : not retrieved for 365 days
    declining -> viable    : fitness recovers above 0.5 (rehabilitation)

Adapted from xplurx's implicit tournament selection -- no forced deletion,
just fitness-driven natural decline.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Optional

from claw.core.models import LifecycleState, Methodology
from claw.db.repository import Repository
from claw.memory.fitness import get_fitness_score

logger = logging.getLogger("claw.memory.lifecycle")

# Thresholds
THRIVING_FITNESS_THRESHOLD = 0.7
THRIVING_SUCCESS_MINIMUM = 3
DECLINING_FITNESS_THRESHOLD = 0.4
REHABILITATION_FITNESS_THRESHOLD = 0.5
DORMANT_DAYS = 180
DEAD_DAYS = 365
NICHE_COLLISION_SIMILARITY = 0.92


def evaluate_transition(
    methodology: Methodology,
    now: Optional[datetime] = None,
    novelty_protection_threshold: float = 0.7,
    novelty_protection_days: int = 90,
) -> Optional[str]:
    """Determine if a methodology should transition to a new lifecycle state.

    Pure computation -- no database access.

    Args:
        methodology: The methodology to evaluate.
        now: Current time. Defaults to utcnow.
        novelty_protection_threshold: Min novelty_score for decay protection.
        novelty_protection_days: Max age in days for novelty protection.

    Returns:
        The new lifecycle state string, or None if no transition.
    """
    if now is None:
        now = datetime.now(UTC)

    current = methodology.lifecycle_state
    fitness = get_fitness_score(methodology)

    # Dead is terminal
    if current == LifecycleState.DEAD.value:
        return None

    # Novelty protection: novel capabilities are shielded from decay transitions
    # for a configurable period (default 90 days) to give them time to prove value
    novel_protected = _is_novelty_protected(
        methodology, now, novelty_protection_threshold, novelty_protection_days
    )

    # Dormant -> dead (365 days without retrieval)
    if current == LifecycleState.DORMANT.value:
        if novel_protected:
            return None  # Protected from death
        if _days_since_retrieval(methodology, now) >= DEAD_DAYS:
            return LifecycleState.DEAD.value
        return None

    # Declining -> dormant (180 days without retrieval)
    if current == LifecycleState.DECLINING.value:
        if novel_protected:
            return None  # Protected from going dormant
        if _days_since_retrieval(methodology, now) >= DORMANT_DAYS:
            return LifecycleState.DORMANT.value
        # Rehabilitation: fitness recovers
        if fitness >= REHABILITATION_FITNESS_THRESHOLD:
            return LifecycleState.VIABLE.value
        return None

    # Embryonic -> viable (first success)
    if current == LifecycleState.EMBRYONIC.value:
        if methodology.success_count >= 1:
            return LifecycleState.VIABLE.value
        return None

    # Viable -> thriving or declining
    if current == LifecycleState.VIABLE.value:
        if (
            fitness >= THRIVING_FITNESS_THRESHOLD
            and methodology.success_count >= THRIVING_SUCCESS_MINIMUM
        ):
            return LifecycleState.THRIVING.value
        if (
            methodology.failure_count > methodology.success_count
            and methodology.retrieval_count >= 3
        ):
            return LifecycleState.DECLINING.value
        return None

    # Thriving -> declining (fitness drop)
    if current == LifecycleState.THRIVING.value:
        if fitness < DECLINING_FITNESS_THRESHOLD:
            return LifecycleState.DECLINING.value
        return None

    return None


async def apply_transition(
    methodology: Methodology,
    repository: Repository,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Evaluate and apply lifecycle transition if warranted.

    Args:
        methodology: The methodology to evaluate.
        repository: Database access for persisting the transition.
        now: Current time.

    Returns:
        The new state if transitioned, or None.
    """
    new_state = evaluate_transition(methodology, now=now)
    if new_state is None:
        return None

    logger.info(
        "Lifecycle transition: %s -> %s for methodology %s",
        methodology.lifecycle_state,
        new_state,
        methodology.id,
    )
    await repository.update_methodology_lifecycle(methodology.id, new_state)
    methodology.lifecycle_state = new_state
    return new_state


async def run_periodic_sweep(
    repository: Repository,
    now: Optional[datetime] = None,
) -> dict[str, int]:
    """Run lifecycle evaluation on all active methodologies.

    Called periodically (e.g., once per run_loop iteration) to catch
    time-based transitions (dormant -> dead, declining -> dormant).

    Returns:
        Dict mapping state transitions to counts.
    """
    if now is None:
        now = datetime.now(UTC)

    transitions: dict[str, int] = {}

    # Fetch active methodologies: viable, thriving, declining, dormant
    active: list[Methodology] = []
    for state in (
        LifecycleState.VIABLE.value,
        LifecycleState.THRIVING.value,
        LifecycleState.DECLINING.value,
        LifecycleState.DORMANT.value,
        LifecycleState.EMBRYONIC.value,
    ):
        batch = await repository.get_methodologies_by_state(state, limit=500)
        active.extend(batch)

    for m in active:
        old_state = m.lifecycle_state
        new_state = await apply_transition(m, repository, now=now)
        if new_state:
            key = f"{old_state}->{new_state}"
            transitions[key] = transitions.get(key, 0) + 1

    if transitions:
        logger.info("Periodic sweep transitions: %s", transitions)
    return transitions


async def check_niche_collision(
    new_methodology: Methodology,
    repository: Repository,
) -> list[Methodology]:
    """Check for competitive exclusion niche collisions.

    When a new methodology is saved, finds near-identical methodologies
    (cosine > 0.92, same type, overlapping files) and transitions the
    lower-fitness ones toward declining.

    Args:
        new_methodology: The newly saved methodology.
        repository: Database access.

    Returns:
        List of methodologies that were transitioned to declining.
    """
    if new_methodology.problem_embedding is None:
        return []

    # Use vector search to find similar methodologies
    similar_pairs = await repository.find_similar_methodologies(
        embedding=new_methodology.problem_embedding,
        limit=10,
    )

    if not similar_pairs:
        return []

    # Filter to niche collisions: high similarity, same type, overlapping files
    collisions: list[tuple[Methodology, float]] = []
    for existing, similarity in similar_pairs:
        if existing.id == new_methodology.id:
            continue
        if similarity < NICHE_COLLISION_SIMILARITY:
            continue
        if (
            new_methodology.methodology_type
            and existing.methodology_type
            and new_methodology.methodology_type != existing.methodology_type
        ):
            continue
        collisions.append((existing, similarity))

    if not collisions:
        return []

    new_fitness = get_fitness_score(new_methodology)
    demoted: list[Methodology] = []

    for existing, similarity in collisions:
        # Check files overlap
        if not _has_file_overlap(new_methodology, existing):
            continue

        existing_fitness = get_fitness_score(existing)

        if existing_fitness < new_fitness:
            # Existing is weaker -- demote it
            if existing.lifecycle_state not in (
                LifecycleState.DECLINING.value,
                LifecycleState.DORMANT.value,
                LifecycleState.DEAD.value,
            ):
                logger.info(
                    "Niche collision: demoting %s (fit=%.3f)"
                    " for %s (fit=%.3f, sim=%.3f)",
                    existing.id, existing_fitness,
                    new_methodology.id, new_fitness, similarity,
                )
                await repository.update_methodology_lifecycle(
                    existing.id, LifecycleState.DECLINING.value
                )
                existing.lifecycle_state = LifecycleState.DECLINING.value
                demoted.append(existing)
        elif new_fitness < existing_fitness:
            # New one is weaker -- demote the new one
            if new_methodology.lifecycle_state not in (
                LifecycleState.DECLINING.value,
                LifecycleState.DORMANT.value,
                LifecycleState.DEAD.value,
            ):
                logger.info(
                    "Niche collision: demoting new %s (fit=%.3f)"
                    " -- existing %s stronger (fit=%.3f)",
                    new_methodology.id, new_fitness,
                    existing.id, existing_fitness,
                )
                await repository.update_methodology_lifecycle(
                    new_methodology.id, LifecycleState.DECLINING.value
                )
                new_methodology.lifecycle_state = LifecycleState.DECLINING.value
                break  # No need to check further -- new one is already demoted

    return demoted


def _days_since_retrieval(methodology: Methodology, now: datetime) -> float:
    """Calculate days since last retrieval (or creation if never retrieved)."""
    ref = methodology.last_retrieved_at or methodology.created_at
    delta = now - ref
    return delta.total_seconds() / 86400.0


def _has_file_overlap(a: Methodology, b: Methodology) -> bool:
    """Check if two methodologies have overlapping files_affected."""
    if not a.files_affected or not b.files_affected:
        return True  # No file constraints means possible overlap
    a_set = {f.lower() for f in a.files_affected}
    b_set = {f.lower() for f in b.files_affected}
    return bool(a_set & b_set)


def _is_novelty_protected(
    methodology: Methodology,
    now: datetime,
    threshold: float,
    max_age_days: int,
) -> bool:
    """Check if a methodology is protected from decay by novelty score.

    A methodology is protected if:
    1. It has a novelty_score >= threshold
    2. It was created less than max_age_days ago
    """
    if methodology.novelty_score is None:
        return False
    if methodology.novelty_score < threshold:
        return False
    age_days = (now - methodology.created_at).total_seconds() / 86400.0
    return age_days < max_age_days
