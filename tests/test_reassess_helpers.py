from __future__ import annotations

from claw.cli import _derive_activation_triggers, _score_methodology_for_task, _tokenize_reassessment_text
from claw.core.models import Methodology


class TestReassessHelpers:
    def test_tokenize_reassessment_text_filters_stopwords(self):
        tokens = _tokenize_reassessment_text("Build a frontend app with testing and validation")
        assert "frontend" in tokens
        assert "testing" in tokens
        assert "build" not in tokens

    def test_derive_activation_triggers_from_capability_and_tags(self):
        meth = Methodology(
            problem_description="Repair failing backend API tests and validation flow",
            solution_code="...",
            tags=["source:demo", "category:testing"],
            capability_data={
                "domain": ["backend", "testing"],
                "capability_type": "validation",
                "inputs": [{"type": "jsonl", "name": "dataset"}],
                "outputs": [{"type": "report", "name": "result"}],
            },
            potential_score=0.8,
        )
        triggers = _derive_activation_triggers(meth, template_count=1)
        assert "backend" in triggers
        assert "testing" in triggers
        assert "validation" in triggers
        assert "data_pipeline" in triggers
        assert "has_action_template" in triggers
        assert "high_future_value" in triggers

    def test_score_methodology_for_task_prefers_matching_task_overlap(self):
        meth = Methodology(
            problem_description="AST-based code transformation for repo repair and test regeneration",
            solution_code="...",
            tags=["category:code_quality", "source:agenticSeek"],
            capability_data={
                "domain": ["refactoring", "testing"],
                "capability_type": "transformation",
            },
            potential_score=0.9,
            novelty_score=0.6,
            retrieval_count=2,
        )
        score, reasons, triggers = _score_methodology_for_task(
            meth,
            task_tokens=_tokenize_reassessment_text("repair broken tests with ast-based refactoring"),
            repo_tokens=_tokenize_reassessment_text("pytest tests src backend"),
            template_count=1,
            template_successes=0,
        )
        assert score >= 0.4
        assert any("overlap" in reason for reason in reasons)
        assert any("potential" in reason for reason in reasons)
        assert "testing" in triggers or "repo_repair" in triggers

    def test_score_methodology_rewards_success_evidence(self):
        meth = Methodology(
            problem_description="evaluation harness for model benchmarking",
            solution_code="...",
            capability_data={"domain": ["evaluation"], "capability_type": "validation"},
            success_count=1,
            potential_score=0.7,
        )
        score, reasons, _ = _score_methodology_for_task(
            meth,
            task_tokens=_tokenize_reassessment_text("benchmark evaluation pipeline"),
            repo_tokens=set(),
            template_count=0,
            template_successes=1,
        )
        assert score > 0.2
        assert any("success evidence" in reason for reason in reasons)
