"""Capability Assimilation Engine for CWM (Clawamorphosis).

When CWM ingests ANY new capability (via mining, self-consumption, or task
completion), this engine:
  1. Extracts structured metadata (inputs, outputs, domain, composability)
  2. Discovers synergies with existing capabilities (SMART — never re-explores)
  3. Creates typed edges (feeds_into, enhances, synergy, etc.)
  4. Auto-composes high-confidence synergies into composite capabilities

Three components:
  - CapabilityExtractor: LLM-powered structured capability analysis
  - SynergyDiscoverer: 4-signal weighted scoring with exploration dedup
  - CapabilityComposer: Auto-creates composite methodologies from synergies

Facade:
  - CapabilityAssimilationEngine: Orchestrates all three; single assimilate() entry point
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any, Optional

from claw.core.config import AssimilationConfig, ClawConfig
from claw.core.models import (
    CapabilityData,
    CapabilityIO,
    ComposabilityInterface,
    Methodology,
    SynergyExploration,
)
from claw.db.repository import Repository
from claw.llm.client import LLMClient, LLMMessage

logger = logging.getLogger("claw.evolution.assimilation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_pair(id_a: str, id_b: str) -> tuple[str, str]:
    """Return (a, b) in canonical order (alphabetically) for dedup."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


def _parse_capability_json(raw: str) -> Optional[dict]:
    """Parse LLM response into a capability_data dict, tolerant of fencing."""
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
        # Validate required keys
        if not isinstance(data, dict):
            return None
        # Ensure inputs/outputs are lists
        if not isinstance(data.get("inputs"), list):
            data["inputs"] = []
        if not isinstance(data.get("outputs"), list):
            data["outputs"] = []
        if not isinstance(data.get("domain"), list):
            data["domain"] = []
        # Validate capability_data through Pydantic
        cd = CapabilityData(**data)
        return cd.model_dump()
    except (json.JSONDecodeError, TypeError, Exception) as e:
        logger.warning("Failed to parse capability JSON: %s", e)
        return None


def _parse_synergy_json(raw: str) -> Optional[dict]:
    """Parse LLM synergy analysis response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse synergy JSON")
        return None


# ---------------------------------------------------------------------------
# CapabilityExtractor
# ---------------------------------------------------------------------------

class CapabilityExtractor:
    """Enriches methodologies with structured capability_data via LLM analysis."""

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        config: ClawConfig,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.config = config
        self._prompt_template: Optional[str] = None

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            from claw.core.config import PromptLoader
            loader = PromptLoader()
            self._prompt_template = loader.load(
                "capability-extract.md",
                default="Extract capability metadata as JSON from the methodology below.",
            )
        return self._prompt_template

    def _get_model(self) -> str:
        """Get an available model from agent configs."""
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = self.config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise ValueError("No model configured in any agent")

    async def extract_capability(self, methodology: Methodology) -> Optional[dict]:
        """Extract structured capability_data from a methodology via LLM.

        Returns the parsed capability_data dict, or None on failure.
        """
        template = self._load_prompt()
        prompt = template.replace("{problem_description}", methodology.problem_description)
        prompt = prompt.replace("{solution_code}", methodology.solution_code or "")
        prompt = prompt.replace("{methodology_notes}", methodology.methodology_notes or "")
        prompt = prompt.replace("{tags}", json.dumps(methodology.tags))

        try:
            model = self._get_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.2,
                max_tokens=1024,
            )
            return _parse_capability_json(response.content)
        except Exception as e:
            logger.error("Capability extraction failed for %s: %s", methodology.id, e)
            return None

    async def enrich_methodology(self, methodology_id: str) -> bool:
        """Extract and store capability_data for a methodology.

        Returns True if enrichment succeeded.
        """
        methodology = await self.repository.get_methodology(methodology_id)
        if methodology is None:
            logger.warning("Methodology %s not found for enrichment", methodology_id)
            return False

        if methodology.capability_data is not None:
            logger.debug("Methodology %s already enriched", methodology_id)
            return True

        cap_data = await self.extract_capability(methodology)
        if cap_data is None:
            return False

        await self.repository.update_methodology_capability_data(methodology_id, cap_data)
        logger.info("Enriched methodology %s with capability_data", methodology_id)
        return True


# ---------------------------------------------------------------------------
# SynergyDiscoverer
# ---------------------------------------------------------------------------

class SynergyDiscoverer:
    """Discovers synergies between capabilities using 4-signal weighted scoring.

    Signals:
      - IO Compatibility (30%): output types of A match input types of B
      - Domain Overlap (20%): Jaccard similarity on domain tags
      - Embedding Similarity (30%): cosine similarity from sqlite-vec
      - LLM Analysis (20%): deep analysis, only if fast signals > 0.4
    """

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        config: ClawConfig,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.config = config
        self.assimilation_config: AssimilationConfig = config.assimilation
        self._synergy_prompt: Optional[str] = None

    def _load_synergy_prompt(self) -> str:
        if self._synergy_prompt is None:
            from claw.core.config import PromptLoader
            loader = PromptLoader()
            self._synergy_prompt = loader.load(
                "synergy-analysis.md",
                default="Analyze synergy between two capabilities and return JSON.",
            )
        return self._synergy_prompt

    def _get_model(self) -> str:
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = self.config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise ValueError("No model configured in any agent")

    def _check_io_compatibility(self, cap_a: dict, cap_b: dict) -> float:
        """Score IO compatibility: do A's outputs match B's inputs and vice versa?"""
        a_output_types = {o.get("type", "") for o in cap_a.get("outputs", [])}
        b_input_types = {i.get("type", "") for i in cap_b.get("inputs", [])}
        b_output_types = {o.get("type", "") for o in cap_b.get("outputs", [])}
        a_input_types = {i.get("type", "") for i in cap_a.get("inputs", [])}

        if not (a_output_types or b_output_types):
            return 0.0

        # Forward: A's outputs → B's inputs
        forward = len(a_output_types & b_input_types) / max(len(a_output_types | b_input_types), 1)
        # Reverse: B's outputs → A's inputs
        reverse = len(b_output_types & a_input_types) / max(len(b_output_types | a_input_types), 1)

        return max(forward, reverse)

    def _check_domain_overlap(self, cap_a: dict, cap_b: dict) -> float:
        """Jaccard similarity on domain tag lists."""
        a_domains = set(cap_a.get("domain", []))
        b_domains = set(cap_b.get("domain", []))
        if not a_domains and not b_domains:
            return 0.0
        union = a_domains | b_domains
        if not union:
            return 0.0
        return len(a_domains & b_domains) / len(union)

    async def _get_embedding_similarity(
        self, methodology_a: Methodology, methodology_b: Methodology
    ) -> float:
        """Get embedding similarity between two methodologies from sqlite-vec."""
        try:
            # Fetch embeddings from the vec table
            row_a = await self.repository.engine.fetch_one(
                "SELECT embedding FROM methodology_embeddings WHERE methodology_id = ?",
                [methodology_a.id],
            )
            row_b = await self.repository.engine.fetch_one(
                "SELECT embedding FROM methodology_embeddings WHERE methodology_id = ?",
                [methodology_b.id],
            )
            if row_a is None or row_b is None:
                return 0.0

            # Compute cosine similarity from raw float32 bytes
            dim = self.config.embeddings.dimension
            vec_a = struct.unpack(f"<{dim}f", row_a["embedding"])
            vec_b = struct.unpack(f"<{dim}f", row_b["embedding"])

            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            mag_a = sum(a * a for a in vec_a) ** 0.5
            mag_b = sum(b * b for b in vec_b) ** 0.5
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return max(0.0, dot / (mag_a * mag_b))
        except Exception as e:
            logger.debug("Embedding similarity failed: %s", e)
            return 0.0

    async def _llm_synergy_analysis(
        self, meth_a: Methodology, meth_b: Methodology
    ) -> tuple[float, Optional[str], Optional[str]]:
        """Deep LLM analysis of synergy. Returns (score, synergy_type, composite_desc)."""
        cap_a = meth_a.capability_data or {}
        cap_b = meth_b.capability_data or {}

        template = self._load_synergy_prompt()
        prompt = template
        prompt = prompt.replace("{cap_a_problem}", meth_a.problem_description)
        prompt = prompt.replace("{cap_a_domain}", json.dumps(cap_a.get("domain", [])))
        prompt = prompt.replace("{cap_a_inputs}", json.dumps(cap_a.get("inputs", [])))
        prompt = prompt.replace("{cap_a_outputs}", json.dumps(cap_a.get("outputs", [])))
        prompt = prompt.replace("{cap_a_type}", cap_a.get("capability_type", "unknown"))
        prompt = prompt.replace("{cap_b_problem}", meth_b.problem_description)
        prompt = prompt.replace("{cap_b_domain}", json.dumps(cap_b.get("domain", [])))
        prompt = prompt.replace("{cap_b_inputs}", json.dumps(cap_b.get("inputs", [])))
        prompt = prompt.replace("{cap_b_outputs}", json.dumps(cap_b.get("outputs", [])))
        prompt = prompt.replace("{cap_b_type}", cap_b.get("capability_type", "unknown"))

        try:
            model = self._get_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.2,
                max_tokens=512,
            )
            result = _parse_synergy_json(response.content)
            if result is None:
                return 0.0, None, None

            score = float(result.get("synergy_score", 0.0))
            stype = result.get("synergy_type")
            composite = result.get("composite_description")
            return score, stype, composite
        except Exception as e:
            logger.error("LLM synergy analysis failed: %s", e)
            return 0.0, None, None

    async def discover_synergies(
        self, methodology_id: str
    ) -> list[SynergyExploration]:
        """Discover synergies between a new capability and existing ones.

        Returns list of exploration records (all recorded in DB regardless of outcome).
        """
        methodology = await self.repository.get_methodology(methodology_id)
        if methodology is None or methodology.capability_data is None:
            return []

        cfg = self.assimilation_config

        # Find candidates: methodologies with capability_data
        candidates = await self.repository.get_methodologies_with_capabilities(
            limit=cfg.synergy_candidate_limit
        )
        candidate_ids = [c.id for c in candidates if c.id != methodology_id]

        # SMART dedup: filter out already-explored pairs
        unexplored_ids = await self.repository.get_unexplored_pairs(
            methodology_id, candidate_ids
        )

        if not unexplored_ids:
            logger.debug("No unexplored pairs for %s", methodology_id)
            return []

        explorations: list[SynergyExploration] = []
        candidate_map = {c.id: c for c in candidates}

        for cid in unexplored_ids:
            candidate = candidate_map.get(cid)
            if candidate is None or candidate.capability_data is None:
                continue

            cap_a = methodology.capability_data
            cap_b = candidate.capability_data

            # Fast signals
            io_score = self._check_io_compatibility(cap_a, cap_b)
            domain_score = self._check_domain_overlap(cap_a, cap_b)
            embed_score = await self._get_embedding_similarity(methodology, candidate)

            fast_score = (
                cfg.io_compatibility_weight * io_score
                + cfg.domain_overlap_weight * domain_score
                + cfg.embedding_similarity_weight * embed_score
            )

            # LLM analysis only if fast signals show promise
            llm_score = 0.0
            synergy_type = None
            composite_desc = None
            llm_triggered = False

            if fast_score > 0.4 * (1.0 - cfg.llm_analysis_weight):
                llm_score, synergy_type, composite_desc = await self._llm_synergy_analysis(
                    methodology, candidate
                )
                llm_triggered = True

            # Weighted total
            total_score = (
                cfg.io_compatibility_weight * io_score
                + cfg.domain_overlap_weight * domain_score
                + cfg.embedding_similarity_weight * embed_score
                + cfg.llm_analysis_weight * llm_score
            )

            # Determine result
            result = "no_match"
            if total_score >= cfg.synergy_score_threshold:
                result = "synergy"

            # Record canonical pair
            a_id, b_id = _canonical_pair(methodology_id, cid)
            exploration = SynergyExploration(
                cap_a_id=a_id,
                cap_b_id=b_id,
                result=result,
                synergy_score=round(total_score, 4),
                synergy_type=synergy_type,
                exploration_method="4-signal" if llm_triggered else "fast-only",
                details={
                    "io_score": round(io_score, 4),
                    "domain_score": round(domain_score, 4),
                    "embedding_score": round(embed_score, 4),
                    "llm_score": round(llm_score, 4),
                    "llm_triggered": llm_triggered,
                },
            )
            await self.repository.record_synergy_exploration(exploration)

            # Create edge if synergy found
            if result == "synergy" and synergy_type:
                await self.repository.upsert_methodology_link(
                    methodology_id, cid, synergy_type, total_score
                )
                exploration.edge_id = f"{methodology_id}->{cid}"

            explorations.append(exploration)

        logger.info(
            "Synergy discovery for %s: %d explored, %d synergies found",
            methodology_id,
            len(explorations),
            sum(1 for e in explorations if e.result == "synergy"),
        )
        return explorations


# ---------------------------------------------------------------------------
# CapabilityComposer
# ---------------------------------------------------------------------------

class CapabilityComposer:
    """Auto-composes high-confidence synergies into new composite methodologies."""

    def __init__(
        self,
        repository: Repository,
        config: ClawConfig,
    ):
        self.repository = repository
        self.config = config
        self.assimilation_config: AssimilationConfig = config.assimilation

    def _merge_capability_data(self, cap_a: dict, cap_b: dict) -> dict:
        """Merge two capability_data dicts into a composite."""
        # Merge inputs: take all unique inputs from both
        all_inputs = {(i["name"], i["type"]): i for i in cap_a.get("inputs", [])}
        for inp in cap_b.get("inputs", []):
            key = (inp["name"], inp["type"])
            if key not in all_inputs:
                all_inputs[key] = inp
        # Remove inputs that are satisfied by the other's outputs
        a_output_types = {o["type"] for o in cap_a.get("outputs", [])}
        b_output_types = {o["type"] for o in cap_b.get("outputs", [])}
        internal_types = a_output_types | b_output_types
        external_inputs = [
            inp for inp in all_inputs.values()
            if inp["type"] not in internal_types
        ]

        # Merge outputs: take all unique outputs from both
        all_outputs = {(o["name"], o["type"]): o for o in cap_a.get("outputs", [])}
        for out in cap_b.get("outputs", []):
            key = (out["name"], out["type"])
            if key not in all_outputs:
                all_outputs[key] = out

        # Merge domains (union, deduplicated)
        domains = list(set(cap_a.get("domain", []) + cap_b.get("domain", [])))

        # Merge composability
        comp_a = cap_a.get("composability", {})
        comp_b = cap_b.get("composability", {})
        merged_comp = ComposabilityInterface(
            can_chain_after=list(set(
                comp_a.get("can_chain_after", []) + comp_b.get("can_chain_after", [])
            )),
            can_chain_before=list(set(
                comp_a.get("can_chain_before", []) + comp_b.get("can_chain_before", [])
            )),
            standalone=comp_a.get("standalone", True) and comp_b.get("standalone", True),
        )

        return CapabilityData(
            inputs=[CapabilityIO(**i) for i in external_inputs] if external_inputs else [
                CapabilityIO(**i) for i in all_inputs.values()
            ],
            outputs=[CapabilityIO(**o) for o in all_outputs.values()],
            domain=domains,
            composability=merged_comp,
            capability_type="composite",
        ).model_dump()

    async def compose(
        self,
        meth_a: Methodology,
        meth_b: Methodology,
        synergy_type: Optional[str] = None,
        composite_description: Optional[str] = None,
    ) -> Optional[str]:
        """Create a composite methodology from two synergistic capabilities.

        Returns the new methodology ID, or None if composition is blocked.
        """
        max_gen = self.config.governance.self_consume_max_generation
        new_gen = max(meth_a.generation, meth_b.generation) + 1
        if new_gen > max_gen:
            logger.info(
                "Composition blocked: generation %d > max %d",
                new_gen, max_gen,
            )
            return None

        # Check methodology count
        count = await self.repository.count_active_methodologies()
        if count >= self.config.governance.max_methodologies:
            logger.info("Composition blocked: at methodology quota (%d)", count)
            return None

        cap_a = meth_a.capability_data or {}
        cap_b = meth_b.capability_data or {}
        merged_cap = self._merge_capability_data(cap_a, cap_b)

        problem = composite_description or (
            f"Composite: {meth_a.problem_description} + {meth_b.problem_description}"
        )

        composite = Methodology(
            problem_description=problem,
            solution_code=(
                f"# Composite of:\n"
                f"# - {meth_a.id}: {meth_a.problem_description[:80]}\n"
                f"# - {meth_b.id}: {meth_b.problem_description[:80]}\n"
                f"\n{meth_a.solution_code}\n\n# ---\n\n{meth_b.solution_code}"
            ),
            methodology_notes=(
                f"Auto-composed from synergy ({synergy_type or 'general'}). "
                f"Parents: {meth_a.id}, {meth_b.id}"
            ),
            tags=list(set(meth_a.tags + meth_b.tags + ["composite", "auto_composed"])),
            language=meth_a.language or meth_b.language,
            scope="global",
            methodology_type="composite",
            lifecycle_state="embryonic",
            generation=new_gen,
            parent_ids=[meth_a.id, meth_b.id],
            capability_data=merged_cap,
        )

        await self.repository.save_methodology(composite)
        logger.info(
            "Composed new methodology %s (gen %d) from %s + %s",
            composite.id, new_gen, meth_a.id, meth_b.id,
        )
        return composite.id


# ---------------------------------------------------------------------------
# NoveltyScorer
# ---------------------------------------------------------------------------

# Generic IO types that can connect to many things
_GENERIC_IO_TYPES = frozenset({
    "text", "json", "code", "code_patch", "event_list", "metrics_data",
    "config", "string", "dict", "list", "file_path", "url", "data",
    "html", "markdown", "csv", "yaml", "xml",
})


class NoveltyScorer:
    """Computes novelty and potential scores for capabilities.

    Novelty: How different is this from everything we already know?
      4 signals: nearest-neighbor distance, domain uniqueness, type rarity, centroid distance

    Potential: How composable, generalizable, and domain-bridging could this become?
      5 signals: IO generality, composability richness, domain breadth, standalone, LLM assessment

    Caches distributions and centroid, refreshing when KB grows by >10%.
    """

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        config: ClawConfig,
    ):
        self.repository = repository
        self.llm_client = llm_client
        self.config = config
        self.cfg: AssimilationConfig = config.assimilation
        # Cache
        self._centroid: Optional[list[float]] = None
        self._domain_dist: Optional[dict[str, int]] = None
        self._type_dist: Optional[dict[str, int]] = None
        self._kb_size_at_cache: int = 0
        self._novelty_prompt: Optional[str] = None

    # -- Cache management --

    async def _refresh_cache_if_needed(self) -> None:
        """Refresh cached distributions if KB has grown by >10%."""
        current_size = await self.repository.count_active_methodologies()
        if (
            self._centroid is not None
            and self._kb_size_at_cache > 0
            and current_size < self._kb_size_at_cache * 1.10
        ):
            return  # Cache still valid

        self._centroid = await self.repository.get_embedding_centroid()
        self._domain_dist = await self.repository.get_domain_distribution()
        self._type_dist = await self.repository.get_type_distribution()
        self._kb_size_at_cache = current_size
        logger.debug(
            "Novelty cache refreshed: kb_size=%d, domains=%d, types=%d",
            current_size,
            len(self._domain_dist),
            len(self._type_dist),
        )

    def invalidate_cache(self) -> None:
        """Force cache refresh on next score() call."""
        self._centroid = None
        self._domain_dist = None
        self._type_dist = None
        self._kb_size_at_cache = 0

    # -- Novelty signals --

    async def _nearest_neighbor_novelty(self, methodology: Methodology) -> float:
        """Average distance to K nearest neighbors. Far from everything = novel."""
        if methodology.problem_embedding is None:
            return 0.5  # Unknown = moderately novel

        k = self.cfg.novelty_nearest_neighbor_k
        neighbors = await self.repository.find_similar_methodologies(
            methodology.problem_embedding, limit=k + 1
        )
        # Exclude self from results
        distances = [
            1.0 - sim for meth, sim in neighbors if meth.id != methodology.id
        ][:k]

        if not distances:
            return 1.0  # No neighbors at all = maximally novel

        avg_distance = sum(distances) / len(distances)
        # Clamp to [0, 1]
        return min(1.0, max(0.0, avg_distance))

    def _domain_uniqueness(self, cap_data: dict) -> float:
        """Fraction of capability's domains that are rare (<3 occurrences) or new."""
        domains = cap_data.get("domain", [])
        if not domains:
            return 0.5  # No domain info = moderate
        dist = self._domain_dist or {}
        rare_count = sum(1 for d in domains if dist.get(d, 0) < 3)
        return rare_count / len(domains)

    def _type_rarity(self, cap_data: dict) -> float:
        """Inverse frequency of capability_type in knowledge base."""
        ctype = cap_data.get("capability_type", "transformation")
        dist = self._type_dist or {}
        total = sum(dist.values()) if dist else 0
        if total == 0:
            return 1.0
        type_count = dist.get(ctype, 0)
        if type_count == 0:
            return 1.0  # Never seen before
        # Inverse frequency, normalized
        return 1.0 - (type_count / total)

    async def _centroid_distance_novelty(self, methodology: Methodology) -> float:
        """Cosine distance from mean embedding of all methodologies."""
        if methodology.problem_embedding is None or not self._centroid:
            return 0.5

        vec = methodology.problem_embedding
        centroid = self._centroid
        dim = len(vec)
        if dim != len(centroid):
            return 0.5

        dot = sum(a * b for a, b in zip(vec, centroid))
        mag_v = sum(a * a for a in vec) ** 0.5
        mag_c = sum(b * b for b in centroid) ** 0.5
        if mag_v == 0 or mag_c == 0:
            return 0.5

        cosine_sim = dot / (mag_v * mag_c)
        distance = 1.0 - cosine_sim
        return min(1.0, max(0.0, distance))

    # -- Potential signals --

    def _io_generality(self, cap_data: dict) -> float:
        """Fraction of IO types that are generic (text, json, etc.)."""
        all_types: list[str] = []
        for io in cap_data.get("inputs", []):
            all_types.append(io.get("type", ""))
        for io in cap_data.get("outputs", []):
            all_types.append(io.get("type", ""))
        if not all_types:
            return 0.5
        generic_count = sum(1 for t in all_types if t.lower() in _GENERIC_IO_TYPES)
        return generic_count / len(all_types)

    def _composability_richness(self, cap_data: dict) -> float:
        """Number of can_chain_after/before entries (diminishing returns)."""
        comp = cap_data.get("composability", {})
        after = len(comp.get("can_chain_after", []))
        before = len(comp.get("can_chain_before", []))
        total = after + before
        if total == 0:
            return 0.0
        # Diminishing returns: log-like scaling, caps at ~1.0 for 10+ chains
        import math
        return min(1.0, math.log1p(total) / math.log1p(10))

    def _domain_breadth(self, cap_data: dict) -> float:
        """Number of domains (bridge potential). Multi-domain = bridge node."""
        domains = cap_data.get("domain", [])
        n = len(domains)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.2
        if n == 2:
            return 0.5
        if n == 3:
            return 0.75
        return 1.0  # 4+ domains

    def _standalone_score(self, cap_data: dict) -> float:
        """Can function independently? 1.0 if standalone, 0.3 if not."""
        comp = cap_data.get("composability", {})
        return 1.0 if comp.get("standalone", True) else 0.3

    async def _llm_potential_assessment(self, methodology: Methodology) -> float:
        """LLM judges future value: could this become transformative?"""
        if self._novelty_prompt is None:
            from claw.core.config import PromptLoader
            loader = PromptLoader()
            self._novelty_prompt = loader.load(
                "novelty-potential.md",
                default=(
                    "Rate the future potential of this capability on a scale of 0.0 to 1.0. "
                    "Consider: could it enable entirely new workflows? Could it bridge "
                    "previously disconnected domains? Return ONLY a JSON object: "
                    '{\"potential_score\": 0.X, \"reasoning\": \"...\"}'
                ),
            )

        cap = methodology.capability_data or {}
        prompt = self._novelty_prompt
        prompt = prompt.replace("{problem_description}", methodology.problem_description)
        prompt = prompt.replace("{domains}", json.dumps(cap.get("domain", [])))
        prompt = prompt.replace("{inputs}", json.dumps(cap.get("inputs", [])))
        prompt = prompt.replace("{outputs}", json.dumps(cap.get("outputs", [])))
        prompt = prompt.replace("{capability_type}", cap.get("capability_type", "unknown"))
        prompt = prompt.replace("{composability}", json.dumps(cap.get("composability", {})))

        try:
            model = self._get_model()
            response = await self.llm_client.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=0.2,
                max_tokens=256,
            )
            parsed = _parse_synergy_json(response.content)
            if parsed and "potential_score" in parsed:
                return min(1.0, max(0.0, float(parsed["potential_score"])))
            return 0.5
        except Exception as e:
            logger.warning("LLM potential assessment failed: %s", e)
            return 0.5

    def _get_model(self) -> str:
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = self.config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise ValueError("No model configured in any agent")

    # -- Public API --

    async def compute_novelty(self, methodology: Methodology) -> float:
        """Compute novelty score (0.0-1.0) from 4 weighted signals."""
        cap = methodology.capability_data or {}
        cfg = self.cfg

        nn = await self._nearest_neighbor_novelty(methodology)
        du = self._domain_uniqueness(cap)
        tr = self._type_rarity(cap)
        cd = await self._centroid_distance_novelty(methodology)

        score = (
            cfg.novelty_nn_weight * nn
            + cfg.novelty_domain_uniqueness_weight * du
            + cfg.novelty_type_rarity_weight * tr
            + cfg.novelty_centroid_distance_weight * cd
        )
        return round(min(1.0, max(0.0, score)), 4)

    async def compute_potential(self, methodology: Methodology) -> float:
        """Compute potential score (0.0-1.0) from 5 weighted signals."""
        cap = methodology.capability_data or {}
        cfg = self.cfg

        io_gen = self._io_generality(cap)
        comp_rich = self._composability_richness(cap)
        db = self._domain_breadth(cap)
        sa = self._standalone_score(cap)

        fast_score = (
            cfg.potential_io_generality_weight * io_gen
            + cfg.potential_composability_weight * comp_rich
            + cfg.potential_domain_breadth_weight * db
            + cfg.potential_standalone_weight * sa
        )

        # Gate LLM on fast threshold
        llm = 0.5  # default neutral
        if fast_score >= cfg.potential_llm_threshold * (1.0 - cfg.potential_llm_weight):
            llm = await self._llm_potential_assessment(methodology)

        score = (
            cfg.potential_io_generality_weight * io_gen
            + cfg.potential_composability_weight * comp_rich
            + cfg.potential_domain_breadth_weight * db
            + cfg.potential_standalone_weight * sa
            + cfg.potential_llm_weight * llm
        )
        return round(min(1.0, max(0.0, score)), 4)

    async def score(self, methodology_id: str) -> dict[str, float]:
        """Compute both novelty and potential scores, persist to DB.

        Returns dict with 'novelty_score' and 'potential_score'.
        """
        methodology = await self.repository.get_methodology(methodology_id)
        if methodology is None:
            return {"novelty_score": 0.0, "potential_score": 0.0}

        if methodology.capability_data is None:
            return {"novelty_score": 0.0, "potential_score": 0.0}

        await self._refresh_cache_if_needed()

        novelty = await self.compute_novelty(methodology)
        potential = await self.compute_potential(methodology)

        await self.repository.update_methodology_novelty_scores(
            methodology_id, novelty, potential
        )

        logger.info(
            "Novelty scored %s: novelty=%.3f, potential=%.3f",
            methodology_id, novelty, potential,
        )
        return {"novelty_score": novelty, "potential_score": potential}


# ---------------------------------------------------------------------------
# CapabilityAssimilationEngine (Facade)
# ---------------------------------------------------------------------------

class CapabilityAssimilationEngine:
    """Orchestrates capability extraction, synergy discovery, and composition.

    Single entry point: assimilate(methodology_id)
    """

    def __init__(
        self,
        repository: Repository,
        llm_client: LLMClient,
        config: ClawConfig,
    ):
        self.repository = repository
        self.config = config
        self.extractor = CapabilityExtractor(repository, llm_client, config)
        self.discoverer = SynergyDiscoverer(repository, llm_client, config)
        self.composer = CapabilityComposer(repository, config)
        self.novelty_scorer = NoveltyScorer(repository, llm_client, config)
        self._compositions_this_cycle = 0

    async def assimilate(self, methodology_id: str) -> dict[str, Any]:
        """Full assimilation pipeline for a methodology.

        1. Extract capability_data (if missing)
        2. Discover synergies with existing capabilities
        3. Auto-compose high-confidence synergies

        Returns a summary dict of what happened.
        """
        if not self.config.assimilation.enabled:
            return {"status": "disabled"}

        result: dict[str, Any] = {
            "methodology_id": methodology_id,
            "enriched": False,
            "novelty_score": None,
            "potential_score": None,
            "synergies_explored": 0,
            "synergies_found": 0,
            "compositions_created": 0,
        }

        # Step 1: Enrich with capability_data
        enriched = await self.extractor.enrich_methodology(methodology_id)
        result["enriched"] = enriched
        if not enriched:
            return result

        # Step 2: Novelty & potential scoring
        if self.config.assimilation.novelty_enabled:
            try:
                scores = await self.novelty_scorer.score(methodology_id)
                result["novelty_score"] = scores["novelty_score"]
                result["potential_score"] = scores["potential_score"]
            except Exception as e:
                logger.warning("Novelty scoring failed for %s: %s", methodology_id, e)

        # Step 3: Discover synergies
        explorations = await self.discoverer.discover_synergies(methodology_id)
        result["synergies_explored"] = len(explorations)
        synergies = [e for e in explorations if e.result == "synergy"]
        result["synergies_found"] = len(synergies)

        # Step 4: Auto-compose high-confidence synergies
        cfg = self.config.assimilation
        max_compositions = cfg.max_compositions_per_cycle
        composed_ids: list[str] = []

        for exp in synergies:
            if self._compositions_this_cycle >= max_compositions:
                break
            if (exp.synergy_score or 0) < cfg.auto_compose_threshold:
                continue

            # Get both methodologies
            meth_a = await self.repository.get_methodology(exp.cap_a_id)
            meth_b = await self.repository.get_methodology(exp.cap_b_id)
            if meth_a is None or meth_b is None:
                continue

            composite_id = await self.composer.compose(
                meth_a, meth_b,
                synergy_type=exp.synergy_type,
            )
            if composite_id:
                composed_ids.append(composite_id)
                self._compositions_this_cycle += 1

        result["compositions_created"] = len(composed_ids)
        result["composed_ids"] = composed_ids
        return result

    def reset_cycle_counter(self) -> None:
        """Reset the per-cycle composition counter (call at start of each NanoClaw cycle)."""
        self._compositions_this_cycle = 0
        self.novelty_scorer.invalidate_cache()
