"""Comprehensive tests for the CLAW Verifier — 7-check audit gate.

Tests every check independently using real objects (no mocks, no placeholders).
Each test creates a Verifier instance with embedding_engine=None and llm_client=None
to isolate rule-based checks from heavy dependencies.

Coverage:
  - Check 1: Dependency Jail (clean, banned import, destructive op)
  - Check 2: Style Match (clean, mixed tabs/spaces, wildcard import)
  - Check 3: Chaos Check (clean, bare except, eval, hardcoded creds)
  - Check 4: Placeholder Scan (clean, TODO, NotImplementedError, pass # placeholder)
  - Check 5: Drift Alignment (skipped when no embedding engine, high similarity)
  - Check 6: Claim Validation (no claims, production ready, tested w/ pass/fail)
  - Main verify() (clean pass, placeholder rejection, quality score)
  - Regression scan (no regression, regression detected)
"""

from __future__ import annotations

import asyncio
import os

import pytest

from claw.core.models import Task, TaskContext, TaskOutcome, VerificationResult
from claw.verifier import Verifier


# ---------------------------------------------------------------------------
# Helpers — build real model objects for tests
# ---------------------------------------------------------------------------

def _make_task(title: str = "Refactor auth module", description: str = "Improve error handling in auth") -> Task:
    """Create a real Task model instance."""
    return Task(
        project_id="proj-001",
        title=title,
        description=description,
        priority=5,
        task_type="refactoring",
    )


def _make_task_context(
    title: str = "Refactor auth module",
    description: str = "Improve error handling in auth",
) -> TaskContext:
    """Create a real TaskContext with embedded Task."""
    task = _make_task(title=title, description=description)
    return TaskContext(task=task)


def _make_outcome(
    diff: str = "",
    approach_summary: str = "Refactored the auth module for better error handling.",
    tests_passed: bool = True,
    files_changed: list[str] | None = None,
    self_audit: str = "",
) -> TaskOutcome:
    """Create a real TaskOutcome model instance."""
    return TaskOutcome(
        files_changed=files_changed or ["src/auth.py"],
        test_output="5 passed",
        tests_passed=tests_passed,
        diff=diff,
        approach_summary=approach_summary,
        self_audit=self_audit,
    )


def _make_verifier(
    banned_dependencies: list[str] | None = None,
    drift_threshold: float = 0.40,
) -> Verifier:
    """Create a Verifier with no embedding engine and no LLM client."""
    return Verifier(
        embedding_engine=None,
        banned_dependencies=banned_dependencies,
        drift_threshold=drift_threshold,
        llm_client=None,
    )


# ---------------------------------------------------------------------------
# Realistic diff strings
# ---------------------------------------------------------------------------

CLEAN_DIFF = """\
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,12 @@ class AuthService:
+    def validate_token(self, token: str) -> bool:
+        if not token:
+            raise ValueError("Token cannot be empty")
+        decoded = jwt.decode(token, self.secret, algorithms=["HS256"])
+        return decoded is not None
"""

BANNED_IMPORT_DIFF = """\
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,4 +1,5 @@
+import flask
+from flask import Flask
 import jwt
 from datetime import datetime
"""

DESTRUCTIVE_OP_DIFF = """\
--- a/src/cleanup.py
+++ b/src/cleanup.py
@@ -5,6 +5,8 @@ class Cleanup:
+    def purge_old_records(self):
+        self.db.delete(table="users", where="active=False")
"""

MIXED_INDENT_DIFF = """\
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,4 +1,6 @@
+\tdef helper():
+\t\treturn True
+    def another():
+        return False
"""

WILDCARD_IMPORT_DIFF = """\
--- a/src/views.py
+++ b/src/views.py
@@ -1,3 +1,4 @@
+from os.path import *
 import sys
"""

BARE_EXCEPT_DIFF = """\
--- a/src/handler.py
+++ b/src/handler.py
@@ -5,6 +5,10 @@ def process():
+    try:
+        result = do_work()
+    except:
+        pass
"""

EVAL_DIFF = """\
--- a/src/dynamic.py
+++ b/src/dynamic.py
@@ -1,3 +1,5 @@
+def run_dynamic(code_str):
+    return eval(code_str)
"""

HARDCODED_CREDS_DIFF = """\
--- a/src/config.py
+++ b/src/config.py
@@ -1,3 +1,5 @@
+DB_HOST = "localhost"
+api_key = "sk-live-abc123def456"
"""

TODO_DIFF = """\
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
+def new_feature():
+    # TODO: implement this properly
+    return None
"""

NOT_IMPLEMENTED_DIFF = """\
--- a/src/service.py
+++ b/src/service.py
@@ -1,3 +1,5 @@
+def process_payment(amount):
+    raise NotImplementedError
"""

PLACEHOLDER_PASS_DIFF = """\
--- a/src/stub.py
+++ b/src/stub.py
@@ -1,3 +1,4 @@
+def placeholder_method():
+    pass  # placeholder
"""


# ===========================================================================
# Check 1: Dependency Jail
# ===========================================================================

class TestDependencyJail:
    """Tests for _check_dependency_jail — unauthorized imports and destructive ops."""

    async def test_clean_diff_no_violations(self):
        """Case 1: Clean diff with no banned imports or destructive operations."""
        verifier = _make_verifier(banned_dependencies=["flask", "django"])
        violations, recommendations = await verifier._check_dependency_jail(CLEAN_DIFF)

        assert len(violations) == 0
        assert len(recommendations) == 0

    async def test_banned_import_detected(self):
        """Case 2: Diff contains an import of a banned package."""
        verifier = _make_verifier(banned_dependencies=["flask", "django"])
        violations, recommendations = await verifier._check_dependency_jail(BANNED_IMPORT_DIFF)

        assert len(violations) >= 1
        ban_violations = [v for v in violations if "flask" in v["detail"].lower()]
        assert len(ban_violations) >= 1
        assert ban_violations[0]["check"] == "dependency_jail"

    async def test_destructive_operation_detected(self):
        """Case 3: Diff contains a destructive function call (delete)."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_dependency_jail(DESTRUCTIVE_OP_DIFF)

        assert len(violations) >= 1
        destructive_violations = [v for v in violations if "destructive" in v["detail"].lower()]
        assert len(destructive_violations) >= 1
        assert destructive_violations[0]["check"] == "dependency_jail"

    async def test_empty_diff_no_violations(self):
        """Empty diff should produce no violations."""
        verifier = _make_verifier(banned_dependencies=["flask"])
        violations, recommendations = await verifier._check_dependency_jail("")

        assert len(violations) == 0
        assert len(recommendations) == 0


# ===========================================================================
# Check 2: Style Match
# ===========================================================================

class TestStyleMatch:
    """Tests for _check_style_match — tabs/spaces, wildcard imports, line length."""

    async def test_clean_code_no_violations(self):
        """Case 4: Clean code with consistent style."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_style_match(CLEAN_DIFF)

        assert len(violations) == 0

    async def test_mixed_tabs_and_spaces_detected(self):
        """Case 5: Mixed tabs and spaces trigger a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_style_match(MIXED_INDENT_DIFF)

        assert len(violations) >= 1
        tab_violations = [v for v in violations if "tab" in v["detail"].lower()]
        assert len(tab_violations) == 1
        assert tab_violations[0]["check"] == "style_match"

    async def test_wildcard_import_detected(self):
        """Case 6: Wildcard import (from X import *) triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_style_match(WILDCARD_IMPORT_DIFF)

        assert len(violations) >= 1
        wildcard_violations = [v for v in violations if "wildcard" in v["detail"].lower()]
        assert len(wildcard_violations) == 1
        assert wildcard_violations[0]["check"] == "style_match"

    async def test_long_line_recommendation(self):
        """Lines exceeding 200 chars produce a recommendation, not a violation."""
        long_line_diff = "+    result = some_function(" + "x" * 200 + ")\n"
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_style_match(long_line_diff)

        assert len(violations) == 0
        assert len(recommendations) >= 1
        assert "200" in recommendations[0]


# ===========================================================================
# Check 3: Chaos Check
# ===========================================================================

class TestChaosCheck:
    """Tests for _check_chaos — bare except, eval/exec, hardcoded credentials."""

    async def test_clean_code_no_violations(self):
        """Case 7: Clean code with proper error handling."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_chaos(CLEAN_DIFF)

        assert len(violations) == 0

    async def test_bare_except_detected(self):
        """Case 8: Bare `except:` (no exception type) triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_chaos(BARE_EXCEPT_DIFF)

        assert len(violations) >= 1
        except_violations = [v for v in violations if "except:" in v["detail"].lower()]
        assert len(except_violations) == 1
        assert except_violations[0]["check"] == "chaos_check"

    async def test_eval_detected(self):
        """Case 9: eval() call triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_chaos(EVAL_DIFF)

        assert len(violations) >= 1
        eval_violations = [v for v in violations if "eval" in v["detail"].lower()]
        assert len(eval_violations) == 1
        assert eval_violations[0]["check"] == "chaos_check"

    async def test_hardcoded_credentials_detected(self):
        """Case 10: Hardcoded api_key string triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_chaos(HARDCODED_CREDS_DIFF)

        assert len(violations) >= 1
        cred_violations = [v for v in violations if "credential" in v["detail"].lower()]
        assert len(cred_violations) == 1
        assert cred_violations[0]["check"] == "chaos_check"


# ===========================================================================
# Check 4: Placeholder Scan
# ===========================================================================

class TestPlaceholderScan:
    """Tests for _check_placeholders — TODOs, stubs, NotImplementedError."""

    async def test_clean_code_no_violations(self):
        """Case 11: Clean code with no placeholder patterns."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_placeholders(CLEAN_DIFF)

        assert len(violations) == 0

    async def test_todo_detected(self):
        """Case 12: TODO comment in new code triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_placeholders(TODO_DIFF)

        assert len(violations) >= 1
        todo_violations = [v for v in violations if "TODO" in v["detail"]]
        assert len(todo_violations) >= 1
        assert todo_violations[0]["check"] == "placeholder_scan"

    async def test_not_implemented_error_detected(self):
        """Case 13: raise NotImplementedError triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_placeholders(NOT_IMPLEMENTED_DIFF)

        assert len(violations) >= 1
        not_impl_violations = [
            v for v in violations if "NotImplementedError" in v["detail"]
        ]
        assert len(not_impl_violations) == 1
        assert not_impl_violations[0]["check"] == "placeholder_scan"

    async def test_pass_placeholder_detected(self):
        """Case 14: `pass  # placeholder` triggers a violation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_placeholders(PLACEHOLDER_PASS_DIFF)

        assert len(violations) >= 1
        pass_violations = [v for v in violations if "pass" in v["detail"].lower()]
        assert len(pass_violations) >= 1
        assert pass_violations[0]["check"] == "placeholder_scan"

    async def test_fixme_detected(self):
        """FIXME pattern also triggers a violation."""
        fixme_diff = "+    # FIXME: this needs proper error handling\n"
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_placeholders(fixme_diff)

        assert len(violations) >= 1
        fixme_violations = [v for v in violations if "FIXME" in v["detail"]]
        assert len(fixme_violations) >= 1


# ===========================================================================
# Check 5: Drift Alignment
# ===========================================================================

class TestDriftAlignment:
    """Tests for _check_drift_alignment — semantic similarity checking."""

    async def test_skipped_when_no_embedding_engine(self):
        """Case 15: When embedding_engine=None, drift check is skipped gracefully."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_drift_alignment(
            task_description="Refactor the auth module",
            approach_summary="I refactored the authentication module",
        )

        assert len(violations) == 0
        assert len(recommendations) >= 1
        assert "skipped" in recommendations[0].lower()
        assert "embedding" in recommendations[0].lower()

    async def test_skipped_when_task_description_empty(self):
        """When task_description is empty, drift check is skipped with recommendation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_drift_alignment(
            task_description="",
            approach_summary="Some approach",
        )

        assert len(violations) == 0
        assert len(recommendations) >= 1
        assert "skipped" in recommendations[0].lower()

    async def test_skipped_when_approach_empty(self):
        """When approach_summary is empty, drift check is skipped with recommendation."""
        verifier = _make_verifier()
        violations, recommendations = await verifier._check_drift_alignment(
            task_description="Some task",
            approach_summary="",
        )

        assert len(violations) == 0
        assert len(recommendations) >= 1
        assert "skipped" in recommendations[0].lower()

    async def test_high_similarity_passes_with_real_embedding_engine(self):
        """Case 16: If an embedding engine is available, identical text passes.

        This test uses the real EmbeddingEngine if sentence-transformers is
        installed. It loads the model (may take a moment), encodes both texts,
        and verifies high cosine similarity produces no violations.
        """
        if os.getenv("CLAW_RUN_HEAVY_EMBEDDING_TESTS", "0") != "1":
            pytest.skip("Set CLAW_RUN_HEAVY_EMBEDDING_TESTS=1 to run heavy embedding integration test")

        try:
            from claw.db.embeddings import EmbeddingEngine

            engine = EmbeddingEngine()
            verifier = Verifier(
                embedding_engine=engine,
                drift_threshold=0.40,
                llm_client=None,
            )

            task_text = "Refactor the authentication module to improve error handling"
            approach_text = "Refactored authentication module with improved error handling"

            violations, recommendations = await verifier._check_drift_alignment(
                task_description=task_text,
                approach_summary=approach_text,
            )

            assert len(violations) == 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_drift_severity_critical(self):
        """Similarity below 0.1 is CRITICAL severity."""
        verifier = _make_verifier()
        assert verifier._drift_severity(0.05) == "CRITICAL"

    def test_drift_severity_high(self):
        """Similarity 0.1-0.2 is HIGH severity."""
        verifier = _make_verifier()
        assert verifier._drift_severity(0.15) == "HIGH"

    def test_drift_severity_medium(self):
        """Similarity 0.2-0.3 is MEDIUM severity."""
        verifier = _make_verifier()
        assert verifier._drift_severity(0.25) == "MEDIUM"

    def test_drift_severity_low(self):
        """Similarity 0.3+ is LOW severity."""
        verifier = _make_verifier()
        assert verifier._drift_severity(0.35) == "LOW"

    def test_drift_guidance_returns_string_for_all_severities(self):
        """All severity levels produce non-empty guidance strings."""
        verifier = _make_verifier()
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            guidance = verifier._drift_guidance(0.0, severity)
            assert isinstance(guidance, str)
            assert len(guidance) > 0


# ===========================================================================
# Check 6: Claim Validation
# ===========================================================================

class TestClaimValidation:
    """Tests for _check_claims — detecting unsubstantiated claims."""

    async def test_no_claims_in_text_passes(self):
        """Case 17: Approach summary with no detectable claims passes cleanly."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="Refactored the auth module for clarity.",
            tests_passed=True,
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) == 0

    async def test_production_ready_claim_blocked(self):
        """Case 18: 'production ready' claim is always blocked."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="The code is now production ready and fully functional.",
            tests_passed=True,
            files_changed=["src/auth.py"],
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) >= 1
        prod_violations = [v for v in violations if "production ready" in v["detail"].lower()]
        assert len(prod_violations) == 1
        assert prod_violations[0]["check"] == "claim_validation"

    async def test_tested_claim_with_tests_passed_true(self):
        """Case 19: 'tested' claim with tests_passed=True passes."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="Changes have been tested and verified.",
            tests_passed=True,
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        tested_violations = [v for v in violations if "tested" in v["detail"].lower()]
        assert len(tested_violations) == 0

    async def test_tested_claim_with_tests_passed_false(self):
        """Case 20: 'tested' claim with tests_passed=False is BLOCKED."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="Changes have been tested and verified.",
            tests_passed=False,
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) >= 1
        tested_violations = [v for v in violations if "tested" in v["detail"].lower()]
        assert len(tested_violations) >= 1
        assert tested_violations[0]["check"] == "claim_validation"

    async def test_tests_pass_claim_with_true(self):
        """'tests pass' claim with tests_passed=True passes."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="All tests pass after the refactoring.",
            tests_passed=True,
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        tests_pass_violations = [v for v in violations if "tests pass" in v["detail"].lower()]
        assert len(tests_pass_violations) == 0

    async def test_tests_pass_claim_with_false(self):
        """'tests pass' claim with tests_passed=False is BLOCKED."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="All tests pass after the refactoring.",
            tests_passed=False,
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) >= 1
        tests_violations = [v for v in violations if "tests pass" in v["detail"].lower()]
        assert len(tests_violations) >= 1

    async def test_fixed_claim_with_tests_and_files(self):
        """'fixed' claim with tests_passed=True and files_changed produces PARTIAL."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="The authentication bug has been fixed.",
            tests_passed=True,
            files_changed=["src/auth.py"],
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        # PARTIAL results in a recommendation, not a violation
        fixed_violations = [v for v in violations if "fixed" in v["detail"].lower()]
        assert len(fixed_violations) == 0
        fixed_recs = [r for r in recommendations if "fixed" in r.lower()]
        assert len(fixed_recs) >= 1

    async def test_fixed_claim_without_evidence_blocked(self):
        """'fixed' claim with tests_passed=False is BLOCKED."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            approach_summary="The authentication bug has been fixed.",
            tests_passed=False,
            files_changed=[],
        )

        violations, recommendations = await verifier._check_claims(
            approach_summary=outcome.approach_summary,
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) >= 1
        fixed_violations = [v for v in violations if "fixed" in v["detail"].lower()]
        assert len(fixed_violations) >= 1

    async def test_empty_approach_summary_passes(self):
        """Empty approach summary triggers no claim checks."""
        verifier = _make_verifier()
        outcome = _make_outcome(approach_summary="", tests_passed=False)

        violations, recommendations = await verifier._check_claims(
            approach_summary="",
            outcome=outcome,
            self_audit="",
        )

        assert len(violations) == 0


# ===========================================================================
# Main verify() method
# ===========================================================================

class TestVerifyMain:
    """Tests for the main verify() entry point."""

    async def test_clean_outcome_passes_all_checks(self):
        """Case 21: Clean outcome with no violations passes verification."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            diff=CLEAN_DIFF,
            approach_summary="Improved error handling in the auth module with proper validation.",
            tests_passed=True,
            files_changed=["src/auth.py"],
        )
        task_context = _make_task_context()

        result = await verifier.verify(outcome, task_context)

        assert isinstance(result, VerificationResult)
        assert result.approved is True
        assert len(result.violations) == 0
        assert result.quality_score is not None
        assert result.quality_score > 0.0

    async def test_outcome_with_placeholder_rejected(self):
        """Case 22: Outcome containing a TODO placeholder is rejected."""
        verifier = _make_verifier()
        outcome = _make_outcome(
            diff=TODO_DIFF,
            approach_summary="Added a new feature with some areas to fill in.",
            tests_passed=True,
            files_changed=["src/feature.py"],
        )
        task_context = _make_task_context()

        result = await verifier.verify(outcome, task_context)

        assert isinstance(result, VerificationResult)
        assert result.approved is False
        assert len(result.violations) >= 1
        placeholder_violations = [
            v for v in result.violations if v["check"] == "placeholder_scan"
        ]
        assert len(placeholder_violations) >= 1

    async def test_quality_score_calculated(self):
        """Case 23: Quality score is computed based on violations and recommendations."""
        verifier = _make_verifier()

        # Clean outcome — should get a high quality score
        clean_outcome = _make_outcome(
            diff=CLEAN_DIFF,
            approach_summary="Improved auth module.",
            tests_passed=True,
        )
        task_context = _make_task_context()

        result = await verifier.verify(clean_outcome, task_context)
        assert result.quality_score is not None
        # With embedding_engine=None, there will be a drift-skipped recommendation
        # so the score will not be exactly 1.0, but should be close
        assert result.quality_score >= 0.90

        # Outcome with violations — lower quality score
        bad_outcome = _make_outcome(
            diff=BARE_EXCEPT_DIFF + "\n" + TODO_DIFF + "\n" + HARDCODED_CREDS_DIFF,
            approach_summary="Quick fix.",
            tests_passed=True,
        )

        bad_result = await verifier.verify(bad_outcome, task_context)
        assert bad_result.quality_score is not None
        assert bad_result.quality_score < result.quality_score

    async def test_multiple_violations_accumulated(self):
        """Multiple checks can fail simultaneously, accumulating violations."""
        verifier = _make_verifier(banned_dependencies=["flask"])

        # Diff that triggers multiple check types
        combined_diff = (
            "+import flask\n"
            "+from os.path import *\n"
            "+    except:\n"
            "+        pass\n"
            "+    # TODO: handle this better\n"
        )
        outcome = _make_outcome(
            diff=combined_diff,
            approach_summary="Quick refactor.",
            tests_passed=True,
        )
        task_context = _make_task_context()

        result = await verifier.verify(outcome, task_context)

        assert result.approved is False
        # Should have violations from multiple checks
        check_names = {v["check"] for v in result.violations}
        assert "dependency_jail" in check_names
        assert "placeholder_scan" in check_names

    async def test_verify_returns_verification_result_type(self):
        """verify() always returns a VerificationResult model."""
        verifier = _make_verifier()
        outcome = _make_outcome(diff="", approach_summary="", tests_passed=True)
        task_context = _make_task_context()

        result = await verifier.verify(outcome, task_context)

        assert isinstance(result, VerificationResult)
        assert isinstance(result.approved, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.recommendations, list)

    async def test_verify_with_banned_dep_and_clean_diff(self):
        """Banned dependency list has no effect when diff contains no banned imports."""
        verifier = _make_verifier(banned_dependencies=["requests", "urllib3"])
        outcome = _make_outcome(
            diff=CLEAN_DIFF,
            approach_summary="Improved auth validation.",
            tests_passed=True,
        )
        task_context = _make_task_context()

        result = await verifier.verify(outcome, task_context)

        assert result.approved is True
        dep_violations = [v for v in result.violations if v["check"] == "dependency_jail"]
        assert len(dep_violations) == 0


# ===========================================================================
# Acceptance checks
# ===========================================================================

class TestAcceptanceChecks:
    async def test_validate_acceptance_command_blocks_shell_chaining(self):
        tokens, reason = Verifier._validate_acceptance_command("pytest -q && echo done")
        assert tokens is None
        assert reason is not None
        assert "shell chaining" in reason

    async def test_run_acceptance_checks_passes_on_allowlisted_command(self, tmp_path):
        verifier = _make_verifier()
        violations, recommendations = await verifier._run_acceptance_checks(
            workspace_dir=str(tmp_path),
            acceptance_checks=["python3 --version"],
        )
        assert violations == []
        assert recommendations == []

    async def test_run_acceptance_checks_blocks_disallowed_command(self, tmp_path):
        verifier = _make_verifier()
        violations, recommendations = await verifier._run_acceptance_checks(
            workspace_dir=str(tmp_path),
            acceptance_checks=["rm -rf /tmp/anything"],
        )
        assert len(violations) == 1
        assert violations[0]["check"] == "acceptance_checks"
        assert "not allowlisted" in violations[0]["detail"]

    async def test_verify_fails_when_acceptance_check_command_fails(self, tmp_path):
        verifier = _make_verifier()
        task_context = _make_task_context()
        task_context.task.acceptance_checks = ["python3 -m module_that_does_not_exist"]
        outcome = _make_outcome(diff=CLEAN_DIFF, tests_passed=True)

        result = await verifier.verify(
            outcome=outcome,
            task_context=task_context,
            workspace_dir=str(tmp_path),
        )
        assert result.approved is False
        checks = [v for v in result.violations if v["check"] == "acceptance_checks"]
        assert len(checks) == 1


# ===========================================================================
# Regression Scan
# ===========================================================================

class TestRegressionScan:
    """Tests for regression_scan — comparing before/after test counts."""

    async def test_no_regression_equal_counts(self):
        """Case 24: Tests after >= tests before means no regression."""
        verifier = _make_verifier()
        regression, message = await verifier.regression_scan(tests_before=10, tests_after=10)

        assert regression is False
        assert "no regression" in message.lower()

    async def test_no_regression_tests_increased(self):
        """More tests after than before is not a regression."""
        verifier = _make_verifier()
        regression, message = await verifier.regression_scan(tests_before=10, tests_after=15)

        assert regression is False
        assert "no regression" in message.lower()

    async def test_regression_detected_tests_decreased(self):
        """Case 25: Tests after < tests before is a regression."""
        verifier = _make_verifier()
        regression, message = await verifier.regression_scan(tests_before=10, tests_after=7)

        assert regression is True
        assert "regression" in message.lower()
        assert "3" in message  # 10 - 7 = 3 tests lost

    async def test_regression_all_tests_disappeared(self):
        """All tests disappearing is a critical regression."""
        verifier = _make_verifier()
        regression, message = await verifier.regression_scan(tests_before=10, tests_after=0)

        assert regression is True
        assert "regression" in message.lower()

    async def test_regression_from_zero_to_zero(self):
        """Zero tests before and after is not a regression (no tests existed)."""
        verifier = _make_verifier()
        regression, message = await verifier.regression_scan(tests_before=0, tests_after=0)

        assert regression is False


# ===========================================================================
# Quality Score computation
# ===========================================================================

class TestQualityScore:
    """Tests for _compute_quality_score — scoring logic."""

    def test_perfect_score_no_violations_no_recommendations(self):
        """No violations and no recommendations yields 1.0."""
        verifier = _make_verifier()
        score = verifier._compute_quality_score(violations=[], recommendations=[])
        assert score == 1.0

    def test_violations_deduct_score(self):
        """Each violation deducts 0.15 from the score."""
        verifier = _make_verifier()
        violations = [{"check": "test", "detail": "v1"}, {"check": "test", "detail": "v2"}]
        score = verifier._compute_quality_score(violations=violations, recommendations=[])
        assert score == pytest.approx(0.70, abs=0.01)

    def test_recommendations_deduct_score(self):
        """Each recommendation deducts 0.03 from the score."""
        verifier = _make_verifier()
        recommendations = ["r1", "r2", "r3"]
        score = verifier._compute_quality_score(violations=[], recommendations=recommendations)
        assert score == pytest.approx(0.91, abs=0.01)

    def test_score_clamped_at_zero(self):
        """Score never goes below 0.0 even with many violations."""
        verifier = _make_verifier()
        violations = [{"check": "test", "detail": f"v{i}"} for i in range(20)]
        score = verifier._compute_quality_score(violations=violations, recommendations=[])
        assert score == 0.0

    def test_combined_violations_and_recommendations(self):
        """Both violations and recommendations reduce the score."""
        verifier = _make_verifier()
        violations = [{"check": "test", "detail": "v1"}]
        recommendations = ["r1", "r2"]
        score = verifier._compute_quality_score(
            violations=violations, recommendations=recommendations
        )
        # 1.0 - 0.15 - 0.06 = 0.79
        assert score == pytest.approx(0.79, abs=0.01)


# ===========================================================================
# Test count parsing
# ===========================================================================

class TestParseTestCount:
    """Tests for _parse_test_count — parsing pytest output."""

    def test_simple_passed(self):
        assert Verifier._parse_test_count("5 passed") == 5

    def test_passed_and_failed(self):
        assert Verifier._parse_test_count("3 passed, 1 failed") == 4

    def test_complex_output(self):
        assert Verifier._parse_test_count("10 passed, 2 failed, 1 error") == 13

    def test_with_skipped(self):
        assert Verifier._parse_test_count("8 passed, 2 skipped") == 10

    def test_no_tests_ran(self):
        assert Verifier._parse_test_count("no tests ran") == 0

    def test_empty_output(self):
        assert Verifier._parse_test_count("") == 0

    def test_full_pytest_output(self):
        output = """\
============================= test session starts ==============================
collected 15 items

tests/test_auth.py ....
tests/test_db.py .....F

========================= 14 passed, 1 failed =================================
"""
        assert Verifier._parse_test_count(output) == 15


# ===========================================================================
# Claim validation helper (_validate_claim)
# ===========================================================================

class TestValidateClaim:
    """Tests for _validate_claim — individual claim verdicts."""

    async def test_tested_claim_pass(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=True)
        verdict = await verifier._validate_claim("tested", outcome)
        assert verdict == "PASS"

    async def test_tested_claim_block(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=False)
        verdict = await verifier._validate_claim("tested", outcome)
        assert verdict == "BLOCK"

    async def test_production_ready_always_block(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=True, files_changed=["a.py"])
        verdict = await verifier._validate_claim("production ready", outcome)
        assert verdict == "BLOCK"

    async def test_fixed_with_evidence_partial(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=True, files_changed=["src/fix.py"])
        verdict = await verifier._validate_claim("fixed", outcome)
        assert verdict == "PARTIAL"

    async def test_fixed_without_evidence_block(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=False, files_changed=[])
        verdict = await verifier._validate_claim("fixed", outcome)
        assert verdict == "BLOCK"

    async def test_done_with_tests_and_files_partial(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=True, files_changed=["src/service.py"])
        verdict = await verifier._validate_claim("done", outcome)
        assert verdict == "PARTIAL"

    async def test_done_without_evidence_block(self):
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=False, files_changed=[])
        verdict = await verifier._validate_claim("done", outcome)
        assert verdict == "BLOCK"

    async def test_unknown_claim_partial(self):
        """Claims not matching any known set default to PARTIAL."""
        verifier = _make_verifier()
        outcome = _make_outcome(tests_passed=True)
        verdict = await verifier._validate_claim("refactored", outcome)
        assert verdict == "PARTIAL"


# ===========================================================================
# Cross-check self-audit
# ===========================================================================

class TestCrossCheckSelfAudit:
    """Tests for _cross_check_self_audit — detecting contradictions."""

    async def test_no_contradiction_when_audit_empty(self):
        """Empty self-audit produces no additional violations."""
        verifier = _make_verifier()
        violations: list[dict[str, str]] = []
        recommendations: list[str] = []

        await verifier._cross_check_self_audit("", _make_outcome(), violations, recommendations)

        assert len(violations) == 0

    async def test_placeholder_contradiction_detected(self):
        """Agent claims no placeholders but placeholder_scan found violations."""
        verifier = _make_verifier()
        violations: list[dict[str, str]] = [
            {"check": "placeholder_scan", "detail": "TODO found on line 5"},
        ]
        recommendations: list[str] = []

        self_audit = "Yes, no placeholder or TODO remains in the code."

        await verifier._cross_check_self_audit(
            self_audit, _make_outcome(), violations, recommendations
        )

        contradiction_violations = [
            v for v in violations if v["check"] == "claim_validation"
        ]
        assert len(contradiction_violations) >= 1
        assert "contradiction" in contradiction_violations[0]["detail"].lower()

    async def test_bare_except_contradiction_detected(self):
        """Agent claims error handling but bare except was detected."""
        verifier = _make_verifier()
        violations: list[dict[str, str]] = [
            {"check": "chaos_check", "detail": "Bare 'except:' found. Catch specific exceptions."},
        ]
        recommendations: list[str] = []

        self_audit = "Yes, proper error handling is in place throughout."

        await verifier._cross_check_self_audit(
            self_audit, _make_outcome(), violations, recommendations
        )

        contradiction_violations = [
            v for v in violations if v["check"] == "claim_validation"
        ]
        assert len(contradiction_violations) >= 1
        assert "contradiction" in contradiction_violations[0]["detail"].lower()
