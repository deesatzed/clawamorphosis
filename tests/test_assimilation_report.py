from __future__ import annotations

from claw.cli import _classify_assimilation_stage, _is_future_candidate
from claw.core.models import Methodology


class TestAssimilationContinuumHelpers:
    def test_stage_stored(self):
        meth = Methodology(problem_description="p", solution_code="s")
        assert _classify_assimilation_stage(meth) == "stored"

    def test_stage_enriched(self):
        meth = Methodology(
            problem_description="p",
            solution_code="s",
            capability_data={"domain": ["ml"]},
            potential_score=0.5,
        )
        assert _classify_assimilation_stage(meth) == "enriched"

    def test_stage_retrieved(self):
        meth = Methodology(problem_description="p", solution_code="s", retrieval_count=2)
        assert _classify_assimilation_stage(meth) == "retrieved"

    def test_stage_operationalized(self):
        meth = Methodology(problem_description="p", solution_code="s", retrieval_count=0)
        assert _classify_assimilation_stage(meth, template_count=1) == "operationalized"

    def test_stage_proven_from_methodology_success(self):
        meth = Methodology(problem_description="p", solution_code="s", success_count=1)
        assert _classify_assimilation_stage(meth) == "proven"

    def test_stage_proven_from_template_success(self):
        meth = Methodology(problem_description="p", solution_code="s")
        assert _classify_assimilation_stage(meth, template_count=1, template_successes=1) == "proven"

    def test_future_candidate_from_high_potential(self):
        meth = Methodology(problem_description="p", solution_code="s", potential_score=0.8)
        assert _is_future_candidate(meth, potential_threshold=0.65) is True

    def test_future_candidate_false_after_direct_success(self):
        meth = Methodology(problem_description="p", solution_code="s", potential_score=0.8, success_count=1)
        assert _is_future_candidate(meth, potential_threshold=0.65) is False

    def test_future_candidate_from_capability_plus_template(self):
        meth = Methodology(
            problem_description="p",
            solution_code="s",
            capability_data={"domain": ["systems"]},
        )
        assert _is_future_candidate(meth, potential_threshold=0.95, template_count=1) is True
