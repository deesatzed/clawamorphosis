"""Two-Key hybrid search for methodology retrieval.

Combines sqlite-vec semantic similarity with SQLite FTS5 full-text search,
then merges and deduplicates results. Adapted from HMLR's hybrid search
pattern and GrokFlow's EnhancedGUKS merge strategy (P13).

The two search backends:
1. Vector search: Repository.find_similar_methodologies() -- sqlite-vec cosine distance
2. Text search: Repository.search_methodologies_text() -- FTS5 + rank

This module orchestrates both, normalizes scores, and returns top-K merged results.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from claw.core.models import LifecycleState, Methodology
from claw.db.embeddings import EmbeddingEngine
from claw.db.repository import Repository
from claw.memory.fitness import get_fitness_score

logger = logging.getLogger("claw.memory.hybrid_search")


class HybridSearchResult:
    """A single hybrid search result with combined score."""

    def __init__(
        self,
        methodology: Methodology,
        vector_score: float = 0.0,
        text_score: float = 0.0,
        combined_score: float = 0.0,
        confidence_score: float = 0.0,
        conflict_score: float = 0.0,
        source: str = "hybrid",
    ):
        self.methodology = methodology
        self.vector_score = vector_score
        self.text_score = text_score
        self.combined_score = combined_score
        self.confidence_score = confidence_score
        self.conflict_score = conflict_score
        self.source = source

    def __repr__(self) -> str:
        return (
            f"HybridSearchResult(id={self.methodology.id}, "
            f"combined={self.combined_score:.3f}, "
            f"vec={self.vector_score:.3f}, txt={self.text_score:.3f}, "
            f"conf={self.confidence_score:.3f}, conflict={self.conflict_score:.3f})"
        )


class HybridSearch:
    """Two-Key search combining vector similarity and full-text search.

    Injected dependencies:
        repository: Database access for both search backends.
        embedding_engine: Encodes query text to vectors for similarity search.

    Configuration:
        vector_weight: Weight for vector similarity score (0.0-1.0).
        text_weight: Weight for text search score (0.0-1.0).
        min_score: Minimum combined score to include in results.
    """

    def __init__(
        self,
        repository: Repository,
        embedding_engine: EmbeddingEngine,
        vector_weight: float = 0.6,
        text_weight: float = 0.4,
        min_score: float = 0.1,
        max_conflict_score: float = 0.85,
        mmr_enabled: bool = True,
        mmr_lambda: float = 0.7,
        prism_engine: Any = None,
        novelty_retrieval_boost: float = 0.0,
        potential_retrieval_boost: float = 0.0,
    ):
        self.repository = repository
        self.embedding_engine = embedding_engine
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.min_score = min_score
        self.max_conflict_score = max_conflict_score
        self._mmr_enabled = mmr_enabled
        self._mmr_lambda = mmr_lambda
        self.prism_engine = prism_engine
        self.novelty_retrieval_boost = novelty_retrieval_boost
        self.potential_retrieval_boost = potential_retrieval_boost

    async def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        file_paths: Optional[list[str]] = None,
        scope: Optional[str] = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search combining vector and text results.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            language: Optional filter by programming language.
            tags: Optional filter by tags.
            file_paths: Optional filter by files_affected overlap (Item 4).
            scope: Optional scope filter -- "project", "global", or None for both.

        Returns:
            List of HybridSearchResult sorted by combined score descending.
        """
        # Fetch more candidates than needed for better merge quality
        fetch_limit = limit * 3

        # 1. Vector search
        vector_results = await self._vector_search(query, fetch_limit)

        # 2. Text search
        text_results = await self._text_search(query, fetch_limit)

        # 3. Merge and deduplicate (sync -- operates on in-memory data)
        merged = self._merge_results(vector_results, text_results, query=query)

        # 4. Apply scope filter (Item 2)
        if scope:
            merged = [r for r in merged if r.methodology.scope == scope]

        # 5. Apply context filters
        if language or tags or file_paths:
            merged = self._apply_filters(merged, language, tags, file_paths)

        # 6. Sort by combined score and limit
        merged.sort(key=lambda r: r.combined_score, reverse=True)

        # 7. Apply minimum score threshold
        merged = [r for r in merged if r.combined_score >= self.min_score]
        merged = [r for r in merged if r.conflict_score <= self.max_conflict_score]

        # 8. MMR re-ranking for diversity (OpenClaw transfer)
        merged = self._apply_mmr(merged, limit)

        return merged[:limit]

    async def _vector_search(
        self, query: str, limit: int
    ) -> list[HybridSearchResult]:
        """Execute vector similarity search."""
        try:
            embedding = self.embedding_engine.encode(query)
            raw_results = await self.repository.find_similar_methodologies(embedding, limit=limit)

            results = []
            for methodology, similarity in raw_results:
                results.append(
                    HybridSearchResult(
                        methodology=methodology,
                        vector_score=max(0.0, similarity),  # Clamp to non-negative
                        source="vector",
                    )
                )
            logger.debug("Vector search returned %d results for: %s", len(results), query[:50])
            return results

        except Exception as e:
            logger.warning("Vector search failed (falling back to text-only): %s", e)
            return []

    async def _text_search(self, query: str, limit: int) -> list[HybridSearchResult]:
        """Execute full-text search."""
        try:
            raw_results = await self.repository.search_methodologies_text(query, limit=limit)

            if not raw_results:
                return []

            # Normalize text scores to 0.0-1.0 range
            # Since we don't get rank scores back from repository, assign
            # descending scores based on position (first result = highest rank)
            results = []
            for i, methodology in enumerate(raw_results):
                # Linear decay: first result gets 1.0, last gets something > 0
                text_score = 1.0 - (i / max(len(raw_results), 1))
                results.append(
                    HybridSearchResult(
                        methodology=methodology,
                        text_score=text_score,
                        source="text",
                    )
                )
            logger.debug("Text search returned %d results for: %s", len(results), query[:50])
            return results

        except Exception as e:
            logger.warning("Text search failed (falling back to vector-only): %s", e)
            return []

    def _merge_results(
        self,
        vector_results: list[HybridSearchResult],
        text_results: list[HybridSearchResult],
        query: str = "",
    ) -> list[HybridSearchResult]:
        """Merge and deduplicate results from both search backends.

        When a methodology appears in both result sets, its scores are combined
        using the configured weights. Unique results keep their single-source score.

        This is sync -- operates entirely on in-memory data.
        """
        # Index by methodology ID for deduplication
        merged: dict[str, HybridSearchResult] = {}

        # Add vector results
        for r in vector_results:
            mid = r.methodology.id
            merged[mid] = HybridSearchResult(
                methodology=r.methodology,
                vector_score=r.vector_score,
                text_score=0.0,
                source="vector",
            )

        # Merge text results
        for r in text_results:
            mid = r.methodology.id
            if mid in merged:
                # Methodology found in both -- merge scores
                existing = merged[mid]
                existing.text_score = r.text_score
                existing.source = "hybrid"
            else:
                merged[mid] = HybridSearchResult(
                    methodology=r.methodology,
                    vector_score=0.0,
                    text_score=r.text_score,
                    source="text",
                )

        # Compute query-side PRISM embedding once for the entire merge pass
        query_prism_emb = None
        if self.prism_engine and query:
            try:
                query_emb = self.embedding_engine.encode(query)
                query_prism_emb = self.prism_engine.enhance(
                    query_emb, {"lifecycle_state": "query"}
                )
            except Exception:
                query_prism_emb = None

        # Calculate combined scores with fitness-weighted tournament boost
        for result in merged.values():
            similarity_score = (
                self.vector_weight * result.vector_score
                + self.text_weight * result.text_score
            )

            # PRISM enhancement: use stored PRISM data when available
            if query_prism_emb and result.vector_score > 0:
                try:
                    stored_prism = getattr(result.methodology, "prism_data", None)
                    if stored_prism is not None:
                        # Fast path: deserialize stored PRISM data — no recomputation
                        from claw.embeddings.prism import PrismEmbedding
                        prism_b = PrismEmbedding.from_dict(stored_prism)
                        prism_score = self.prism_engine.similarity(
                            query_prism_emb, prism_b
                        )
                        similarity_score = (
                            self.vector_weight * prism_score.combined
                            + self.text_weight * result.text_score
                        )
                    else:
                        # Slow fallback for old methodologies without stored PRISM data
                        problem_emb = getattr(
                            result.methodology, "problem_embedding", None
                        )
                        if problem_emb:
                            result_meta = {
                                "lifecycle_state": result.methodology.lifecycle_state
                                or "viable"
                            }
                            prism_b = self.prism_engine.enhance(
                                problem_emb, result_meta
                            )
                            prism_score = self.prism_engine.similarity(
                                query_prism_emb, prism_b
                            )
                            similarity_score = (
                                self.vector_weight * prism_score.combined
                                + self.text_weight * result.text_score
                            )
                except Exception:
                    logger.debug(
                        "PRISM enhancement skipped for %s",
                        result.methodology.id,
                    )

            # MEE: fitness-weighted tournament selection
            # Blend similarity (60%) with fitness (40%) for final ranking
            fitness = get_fitness_score(result.methodology)
            result.combined_score = similarity_score * 0.6 + fitness * 0.4

            # Novelty & potential retrieval boost: surface novel/high-potential items
            if self.novelty_retrieval_boost > 0 and result.methodology.novelty_score is not None:
                result.combined_score += self.novelty_retrieval_boost * result.methodology.novelty_score
            if self.potential_retrieval_boost > 0 and result.methodology.potential_score is not None:
                result.combined_score += self.potential_retrieval_boost * result.methodology.potential_score

            result.confidence_score, result.conflict_score = self._derive_memory_signals(result)

        # Filter out dead/dormant methodologies from retrieval
        alive = {
            mid: r for mid, r in merged.items()
            if r.methodology.lifecycle_state not in (
                LifecycleState.DEAD.value,
                LifecycleState.DORMANT.value,
            )
        }

        return list(alive.values())

    def summarize_signals(self, results: list[HybridSearchResult]) -> dict[str, float | int | list[str]]:
        """Aggregate confidence/conflict signals across retrieval results."""
        if not results:
            return {
                "retrieval_confidence": 0.0,
                "conflict_count": 0,
                "conflicts": [],
                "hybrid_hits": 0,
            }

        confidence = sum(r.confidence_score for r in results) / len(results)
        conflicts: list[str] = []
        hybrid_hits = 0
        for r in results:
            if r.source == "hybrid":
                hybrid_hits += 1
            if r.conflict_score >= 0.60:
                conflicts.append(
                    f"{r.methodology.problem_description[:100]} (conflict={r.conflict_score:.2f})"
                )

        return {
            "retrieval_confidence": round(confidence, 3),
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:3],
            "hybrid_hits": hybrid_hits,
        }

    def _apply_mmr(
        self,
        results: list[HybridSearchResult],
        limit: int,
    ) -> list[HybridSearchResult]:
        """Re-rank results using Maximal Marginal Relevance for diversity.

        MMR(candidate) = lambda * norm_relevance - (1-lambda) * max_similarity_to_selected

        Similarity metric: Jaccard on tokenized text (problem_description + methodology_notes).
        Adapted from OpenClaw's retrieval diversity pattern.
        """
        if not self._mmr_enabled or len(results) <= 1:
            return results[:limit]

        def _tokenize(r: HybridSearchResult) -> set[str]:
            text = (r.methodology.problem_description or "") + " " + (r.methodology.methodology_notes or "")
            return set(text.lower().split())

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        tokens = {id(r): _tokenize(r) for r in results}
        scores = {id(r): r.combined_score for r in results}

        # Normalize scores to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        min_score = min(scores.values()) if scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        selected: list[HybridSearchResult] = []
        remaining = list(results)

        while remaining and len(selected) < limit:
            best_mmr = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                norm_rel = (scores[id(candidate)] - min_score) / score_range

                max_sim = 0.0
                for sel in selected:
                    sim = _jaccard(tokens[id(candidate)], tokens[id(sel)])
                    if sim > max_sim:
                        max_sim = sim

                mmr = self._mmr_lambda * norm_rel - (1 - self._mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _apply_filters(
        self,
        results: list[HybridSearchResult],
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        file_paths: Optional[list[str]] = None,
    ) -> list[HybridSearchResult]:
        """Apply context filters to search results.

        Args:
            results: Search results to filter.
            language: Optional filter by programming language.
            tags: Optional filter by tags.
            file_paths: Optional filter by files_affected overlap (Item 4).
        """
        filtered = results

        if language:
            filtered = [
                r for r in filtered
                if r.methodology.language and r.methodology.language.lower() == language.lower()
            ]

        if tags:
            tag_set = {t.lower() for t in tags}
            filtered = [
                r for r in filtered
                if tag_set & {t.lower() for t in r.methodology.tags}
            ]

        if file_paths:
            path_set = {p.lower() for p in file_paths}
            filtered = [
                r for r in filtered
                if path_set & {f.lower() for f in (r.methodology.files_affected or [])}
            ]

        return filtered

    @staticmethod
    def _derive_memory_signals(result: HybridSearchResult) -> tuple[float, float]:
        """Infer retrieval confidence and conflict from score agreement."""
        if result.source == "hybrid":
            agreement = 1.0 - min(1.0, abs(result.vector_score - result.text_score))
            confidence = 0.50 + 0.50 * agreement
            conflict = 1.0 - agreement
            return confidence, conflict

        primary = max(result.vector_score, result.text_score)
        confidence = 0.30 + 0.70 * max(0.0, min(1.0, primary))
        return confidence, 0.0
