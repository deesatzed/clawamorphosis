"""CLI UX surface tests for recently added user-facing commands/options."""

from __future__ import annotations

import inspect


def _command_map():
    from claw.cli import app

    mapping = {}
    for cmd in app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "")
        if cmd.callback is not None:
            mapping[name] = cmd.callback
    return mapping


def _group_map():
    from claw.cli import app

    mapping = {}
    for group in getattr(app, "registered_groups", []):
        info = group.typer_instance.info
        name = group.name
        if hasattr(name, "value"):
            name = name.value
        if not name:
            name = info.name
        mapping[name] = group.typer_instance
    return mapping


class TestCLIUXSurface:
    def test_cli_root_dir_points_to_repo(self):
        from claw.cli import ROOT_DIR

        assert (ROOT_DIR / "scripts" / "export_cam_knowledge_pack.py").exists()
        assert (ROOT_DIR / "apps" / "embedding_forge" / "benchmark_regression.py").exists()

    def test_quickstart_command_registered(self):
        commands = _command_map()
        assert "quickstart" in commands

    def test_forge_export_command_registered(self):
        commands = _command_map()
        assert "forge-export" in commands

    def test_create_command_registered(self):
        commands = _command_map()
        assert "create" in commands

    def test_ideate_command_registered(self):
        commands = _command_map()
        assert "ideate" in commands

    def test_mine_report_command_registered(self):
        commands = _command_map()
        assert "mine-report" in commands

    def test_assimilation_report_command_registered(self):
        commands = _command_map()
        assert "assimilation-report" in commands

    def test_assimilation_delta_command_registered(self):
        commands = _command_map()
        assert "assimilation-delta" in commands

    def test_reassess_command_registered(self):
        commands = _command_map()
        assert "reassess" in commands

    def test_keycheck_command_registered(self):
        commands = _command_map()
        assert "keycheck" in commands

    def test_grouped_workflow_namespaces_registered(self):
        groups = _group_map()
        assert "learn" in groups
        assert "task" in groups
        assert "forge" in groups
        assert "doctor" in groups
        assert "kb" in groups

    def test_grouped_namespace_commands_exist(self):
        groups = _group_map()

        learn_names = {
            cmd.name or (cmd.callback.__name__ if cmd.callback else "")
            for cmd in groups["learn"].registered_commands
        }
        assert {"report", "delta", "reassess", "synergies"} <= learn_names

        task_names = {
            cmd.name or (cmd.callback.__name__ if cmd.callback else "")
            for cmd in groups["task"].registered_commands
        }
        assert {"add", "quickstart", "runbook", "results"} <= task_names

        forge_names = {
            cmd.name or (cmd.callback.__name__ if cmd.callback else "")
            for cmd in groups["forge"].registered_commands
        }
        assert {"export", "benchmark"} <= forge_names

        doctor_names = {
            cmd.name or (cmd.callback.__name__ if cmd.callback else "")
            for cmd in groups["doctor"].registered_commands
        }
        assert {"keycheck", "status"} <= doctor_names

    def test_benchmark_command_registered(self):
        commands = _command_map()
        assert "benchmark" in commands

    def test_validate_command_registered(self):
        commands = _command_map()
        assert "validate" in commands

    def test_forge_benchmark_command_registered(self):
        commands = _command_map()
        assert "forge-benchmark" in commands

    def test_runbook_command_registered(self):
        commands = _command_map()
        assert "runbook" in commands

    def test_enhance_has_dry_run_option(self):
        commands = _command_map()
        enhance_cb = commands["enhance"]
        sig = inspect.signature(enhance_cb)
        assert "dry_run" in sig.parameters

    def test_mine_has_time_guardrail_option(self):
        commands = _command_map()
        mine_cb = commands["mine"]
        sig = inspect.signature(mine_cb)
        assert "max_minutes" in sig.parameters
        assert "skip_known" in sig.parameters
        assert "force_rescan" in sig.parameters
        assert "changed_only" in sig.parameters
        assert "live_keycheck" in sig.parameters

    def test_create_has_repo_mode_and_time_guardrail(self):
        commands = _command_map()
        cb = commands["create"]
        sig = inspect.signature(cb)
        assert "repo_mode" in sig.parameters
        assert "max_minutes" in sig.parameters

    def test_ideate_has_focus_promote_and_time_guardrail(self):
        commands = _command_map()
        cb = commands["ideate"]
        sig = inspect.signature(cb)
        assert "focus" in sig.parameters
        assert "promote" in sig.parameters
        assert "target_repo" in sig.parameters
        assert "max_minutes" in sig.parameters

    def test_benchmark_has_spec_file_and_time_guardrail(self):
        commands = _command_map()
        cb = commands["validate"]
        sig = inspect.signature(cb)
        assert "spec_file" in sig.parameters
        assert "max_minutes" in sig.parameters

    def test_benchmark_has_time_guardrail(self):
        commands = _command_map()
        cb = commands["benchmark"]
        sig = inspect.signature(cb)
        assert "max_minutes" in sig.parameters

    def test_add_goal_has_step_and_check_options(self):
        commands = _command_map()
        add_goal_cb = commands["add-goal"]
        sig = inspect.signature(add_goal_cb)
        assert "step" in sig.parameters
        assert "check" in sig.parameters

    def test_quickstart_has_preview_and_execute_options(self):
        commands = _command_map()
        quickstart_cb = commands["quickstart"]
        sig = inspect.signature(quickstart_cb)
        assert "preview" in sig.parameters
        assert "execute" in sig.parameters

    def test_forge_export_has_time_guardrail_option(self):
        commands = _command_map()
        cb = commands["forge-export"]
        sig = inspect.signature(cb)
        assert "max_minutes" in sig.parameters

    def test_forge_benchmark_has_time_guardrail_option(self):
        commands = _command_map()
        cb = commands["forge-benchmark"]
        sig = inspect.signature(cb)
        assert "max_minutes" in sig.parameters

    def test_keycheck_has_for_command_option(self):
        commands = _command_map()
        cb = commands["keycheck"]
        sig = inspect.signature(cb)
        assert "for_command" in sig.parameters
        assert "live" in sig.parameters
