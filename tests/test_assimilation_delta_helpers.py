from __future__ import annotations

from datetime import UTC, datetime

from claw.cli import _infer_feature_opportunities, _recently_created_near_mine, _summarize_new_capabilities
from claw.core.models import Methodology


class TestAssimilationDeltaHelpers:
    def test_summarize_new_capabilities_counts_domains_types_and_sources(self):
        methods = [
            Methodology(
                problem_description="fine-tuning pipeline",
                solution_code="...",
                tags=["source:mlx-tune-main"],
                capability_data={"domain": ["ml", "training"], "capability_type": "generation"},
            ),
            Methodology(
                problem_description="evaluation harness",
                solution_code="...",
                tags=["source:mlx-tune-main", "source:Repo2Eval"],
                capability_data={"domain": ["ml", "evaluation"], "capability_type": "validation"},
            ),
        ]
        summary = _summarize_new_capabilities(methods)
        assert summary["domains"][0] == ("ml", 2)
        assert ("generation", 1) in summary["capability_types"]
        assert ("validation", 1) in summary["capability_types"]
        assert summary["source_repos"][0][0] == "mlx-tune-main"

    def test_infer_feature_opportunities_uses_trigger_counts(self):
        methods = [
            Methodology(
                id="m1",
                problem_description="LoRA fine-tuning and evaluation workflow",
                solution_code="...",
                capability_data={"domain": ["evaluation"], "capability_type": "validation"},
                potential_score=0.8,
            ),
            Methodology(
                id="m2",
                problem_description="repair failing tests with backend validation",
                solution_code="...",
                capability_data={"domain": ["backend", "testing"], "capability_type": "validation"},
                potential_score=0.7,
            ),
        ]
        opportunities = _infer_feature_opportunities(
            methods,
            methodology_ids_with_templates={"m1"},
            limit=4,
        )
        triggers = {item["trigger"] for item in opportunities}
        assert "finetuning" in triggers
        assert "evaluation" in triggers
        assert "backend" in triggers or "repo_repair" in triggers or "testing" in triggers

    def test_recently_created_near_mine_accepts_close_timestamps(self):
        mine_ts = datetime(2026, 3, 15, 12, 0, tzinfo=UTC).timestamp()
        created = datetime(2026, 3, 15, 11, 30, tzinfo=UTC)
        assert _recently_created_near_mine(created, mine_ts) is True

    def test_recently_created_near_mine_rejects_old_items(self):
        mine_ts = datetime(2026, 3, 15, 12, 0, tzinfo=UTC).timestamp()
        created = datetime(2026, 3, 14, 1, 0, tzinfo=UTC)
        assert _recently_created_near_mine(created, mine_ts) is False
