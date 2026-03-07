"""CLAW CLI — Typer-based command line interface.

Commands:
  evaluate <repo>        — structural analysis + 18-prompt evaluation battery
  enhance <repo>         — full pipeline: evaluate -> plan -> dispatch -> verify -> learn
  fleet-enhance <dir>    — multi-repo fleet processing with ranking and budget allocation
  add-goal <repo>        — manually add a task/goal for a repository
  results                — show past task results from the database
  status                 — show system status
  setup                  — interactive API key, model, and agent configuration
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import time as _time

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="claw",
    help="CLAW — Codebase Learning & Autonomous Workforce",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def evaluate(
    repo: str = typer.Argument(..., help="Path to the repository to evaluate"),
    mode: str = typer.Option(
        "auto", "--mode", "-m",
        help="Evaluation mode: full (all 18 prompts), quick (orientation + deep_analysis only), structural (no agent calls), auto (full if agents configured, else structural)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Evaluate a repository for enhancement potential.

    Runs structural analysis and the 18-prompt evaluation battery on the
    target repo, storing results in SQLite.

    Modes:
      full       — structural analysis + all 18 evaluation prompts via agents
      quick      — structural analysis + orientation and deep_analysis prompts only
      structural — structural analysis only (no agent calls)
      auto       — uses 'full' if agents are configured, otherwise 'structural'
    """
    _setup_logging(verbose)

    valid_modes = ("full", "quick", "structural", "auto")
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}. Use: {', '.join(valid_modes)}[/red]")
        raise typer.Exit(1)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    asyncio.run(_evaluate_async(repo_path, config, mode))


async def _evaluate_async(repo_path: Path, config_path: Optional[str], mode: str) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project, Task

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repo_path)

    try:
        # Create or get project
        project = Project(
            name=repo_path.name,
            repo_path=str(repo_path),
        )
        await ctx.repository.create_project(project)

        # Resolve "auto" mode based on whether agents are available
        effective_mode = mode
        if mode == "auto":
            effective_mode = "full" if ctx.agents else "structural"

        console.print(f"\n[bold]CLAW Evaluation: {repo_path.name}[/bold]")
        console.print(f"  Repository: {repo_path}")
        console.print(f"  Project ID: {project.id}")
        console.print(f"  Database: {ctx.config.database.db_path}")
        console.print(f"  Mode: {effective_mode}")
        if effective_mode != "structural":
            console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")

        # ---------------------------------------------------------------
        # Phase 1: Basic structural analysis (always runs)
        # ---------------------------------------------------------------
        console.print(f"\n[cyan]Phase 1: Structural Analysis[/cyan]")
        analysis = await _analyze_repo(repo_path)

        # Create evaluation task
        eval_task = Task(
            project_id=project.id,
            title=f"Evaluate {repo_path.name}",
            description=f"Structural analysis of {repo_path.name}",
            task_type="analysis",
            priority=10,
        )
        await ctx.repository.create_task(eval_task)

        # Log episode
        await ctx.repository.log_episode(
            session_id="cli-evaluate",
            event_type="evaluation_started",
            event_data={"repo_path": str(repo_path), "analysis": analysis, "mode": effective_mode},
            project_id=project.id,
        )

        # Display structural results
        _display_analysis(analysis, repo_path.name)

        # Store structural results
        await ctx.repository.log_episode(
            session_id="cli-evaluate",
            event_type="structural_analysis_completed",
            event_data=analysis,
            project_id=project.id,
        )

        # ---------------------------------------------------------------
        # Phase 2: Evaluation Battery (if mode is "full" or "quick")
        # ---------------------------------------------------------------
        if effective_mode in ("full", "quick"):
            from claw.evaluator import Evaluator

            # Determine battery mode for the Evaluator
            battery_mode = effective_mode  # "full" or "quick"

            # Use the dispatcher if agents are available, otherwise None
            # (Evaluator records prompts as pending when dispatcher is None)
            dispatcher = ctx.dispatcher if ctx.agents else None

            evaluator = Evaluator(
                repository=ctx.repository,
                dispatcher=dispatcher,
            )

            agent_status = f"dispatching to {', '.join(ctx.agents.keys())}" if ctx.agents else "no agents (prompts will be recorded as pending)"
            console.print(f"\n[cyan]Phase 2: Evaluation Battery ({battery_mode})[/cyan]")
            console.print(f"  {agent_status}")

            # Run the battery with a live progress indicator
            battery_start = _time.monotonic()
            report = await evaluator.run_battery(
                project_id=project.id,
                repo_path=str(repo_path),
                mode=battery_mode,
            )
            battery_elapsed = _time.monotonic() - battery_start

            # Display the evaluation report
            _display_evaluation_report(report)

            # Log the battery summary as an episode
            await ctx.repository.log_episode(
                session_id="cli-evaluate",
                event_type="evaluation_battery_summary",
                event_data={
                    "mode": battery_mode,
                    "total_prompts": report.total_prompts,
                    "successful_prompts": report.successful_prompts,
                    "failed_prompts": report.failed_prompts,
                    "total_duration_seconds": report.total_duration_seconds,
                    "phases_completed": len(report.phases),
                    "agents_used": list({
                        pr.agent_id
                        for phase in report.phases
                        for pr in phase.prompt_results
                        if pr.agent_id is not None
                    }),
                },
                project_id=project.id,
            )

        # Final status
        await ctx.repository.log_episode(
            session_id="cli-evaluate",
            event_type="evaluation_completed",
            event_data={"mode": effective_mode, "analysis": analysis},
            project_id=project.id,
        )

        console.print(f"\n[green]Evaluation stored in {ctx.config.database.db_path}[/green]")

    finally:
        await ctx.close()


def _display_evaluation_report(report) -> None:
    """Display the evaluation battery report as a Rich table.

    Shows each prompt with its phase, agent, status, and duration,
    followed by a summary line.
    """
    from claw.evaluator import EvaluationReport

    if not isinstance(report, EvaluationReport):
        return

    console.print()

    table = Table(title="Evaluation Battery Results")
    table.add_column("Phase", style="cyan", max_width=22)
    table.add_column("Prompt", style="bold", max_width=22)
    table.add_column("Agent", style="yellow", width=10)
    table.add_column("Status", width=10)
    table.add_column("Duration", justify="right", width=10)

    for phase in report.phases:
        for pr in phase.prompt_results:
            # Phase name (cleaned up for display)
            phase_display = phase.phase_name.replace("_", " ").title()

            # Agent
            agent_display = pr.agent_id or ("pending" if pr.error is None else "-")

            # Status: green check for success, red x for failure, yellow dash for pending
            if pr.error is not None:
                status_display = "[red]FAILED[/red]"
            elif pr.agent_id is not None:
                status_display = "[green]OK[/green]"
            else:
                # No error, but no agent -- prompt was recorded as pending
                status_display = "[yellow]PENDING[/yellow]"

            # Duration
            dur = pr.duration_seconds
            if dur >= 60:
                mins = int(dur // 60)
                secs = int(dur % 60)
                dur_str = f"{mins}m {secs:02d}s"
            elif dur >= 0.01:
                dur_str = f"{dur:.2f}s"
            else:
                dur_str = "<0.01s"

            table.add_row(
                phase_display,
                pr.prompt_name,
                agent_display,
                status_display,
                dur_str,
            )

    console.print(table)

    # Summary line
    total = report.total_prompts
    succeeded = report.successful_prompts
    failed = report.failed_prompts
    pending = total - succeeded - failed

    parts = [f"{succeeded}/{total} prompts completed"]
    if failed > 0:
        parts.append(f"[red]{failed} failed[/red]")
    if pending > 0:
        parts.append(f"[yellow]{pending} pending[/yellow]")

    # Show unique agents used
    agents_used = sorted({
        pr.agent_id
        for phase in report.phases
        for pr in phase.prompt_results
        if pr.agent_id is not None
    })
    if agents_used:
        parts.append(f"agents: {', '.join(agents_used)}")

    # Total duration
    dur = report.total_duration_seconds
    if dur >= 60:
        mins = int(dur // 60)
        secs = int(dur % 60)
        dur_str = f"{mins}m {secs:02d}s"
    else:
        dur_str = f"{dur:.2f}s"
    parts.append(f"total: {dur_str}")

    console.print(f"\n  {' | '.join(parts)}")


async def _analyze_repo(repo_path: Path) -> dict:
    """Perform basic structural analysis of a repository."""
    analysis = {
        "has_git": (repo_path / ".git").exists(),
        "has_readme": any(
            (repo_path / f).exists() for f in ["README.md", "readme.md", "README"]
        ),
        "has_tests": any(
            (repo_path / d).exists() for d in ["tests", "test", "spec", "__tests__"]
        ),
        "file_counts": {},
        "total_files": 0,
        "languages_detected": [],
    }

    # Count files by extension
    ext_counts: dict[str, int] = {}
    total = 0
    for f in repo_path.rglob("*"):
        if f.is_file() and ".git" not in f.parts:
            total += 1
            ext = f.suffix.lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    analysis["file_counts"] = dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:20])
    analysis["total_files"] = total

    # Detect languages from extensions
    lang_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".rs": "Rust", ".go": "Go", ".java": "Java", ".rb": "Ruby",
        ".cpp": "C++", ".c": "C", ".cs": "C#", ".swift": "Swift",
        ".kt": "Kotlin", ".scala": "Scala", ".php": "PHP",
    }
    langs = []
    for ext, lang in lang_map.items():
        if ext in ext_counts:
            langs.append(lang)
    analysis["languages_detected"] = langs

    # Check for config files
    config_files = [
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
        "pom.xml", "build.gradle", "Gemfile", "Makefile",
        "docker-compose.yml", "Dockerfile",
    ]
    analysis["config_files"] = [f for f in config_files if (repo_path / f).exists()]

    return analysis


def _display_analysis(analysis: dict, name: str) -> None:
    """Display analysis results using Rich."""
    console.print()

    # Summary table
    table = Table(title=f"Repository Analysis: {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files", str(analysis["total_files"]))
    table.add_row("Git Repository", "Yes" if analysis["has_git"] else "No")
    table.add_row("Has README", "Yes" if analysis["has_readme"] else "No")
    table.add_row("Has Tests", "Yes" if analysis["has_tests"] else "No")
    table.add_row("Languages", ", ".join(analysis["languages_detected"]) or "None detected")
    table.add_row("Config Files", ", ".join(analysis["config_files"]) or "None")

    console.print(table)

    # File breakdown
    if analysis["file_counts"]:
        ft = Table(title="File Type Breakdown (Top 10)")
        ft.add_column("Extension", style="cyan")
        ft.add_column("Count", style="yellow", justify="right")

        for ext, count in list(analysis["file_counts"].items())[:10]:
            ft.add_row(ext, str(count))

        console.print(ft)


@app.command()
def enhance(
    repo: str = typer.Argument(..., help="Path to the repository to enhance"),
    mode: str = typer.Option("attended", "--mode", "-m", help="Mode: attended, supervised, autonomous"),
    max_tasks: int = typer.Option(10, "--max-tasks", help="Maximum number of tasks to process"),
    battery: bool = typer.Option(False, "--battery", "-b", help="Use full evaluation battery (MesoClaw) instead of structural analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Enhance a repository: evaluate, plan, dispatch, verify, learn.

    Runs the full MesoClaw pipeline on the target repo.
    Use --battery to run the full 18-prompt evaluation battery.
    """
    _setup_logging(verbose)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    if mode not in ("attended", "supervised", "autonomous"):
        console.print(f"[red]Invalid mode: {mode}. Use attended, supervised, or autonomous.[/red]")
        raise typer.Exit(1)

    if battery:
        asyncio.run(_enhance_battery_async(repo_path, config, mode, max_tasks))
    else:
        asyncio.run(_enhance_async(repo_path, config, mode, max_tasks))


async def _enhance_async(
    repo_path: Path,
    config_path: Optional[str],
    mode: str,
    max_tasks: int,
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project
    from claw.cycle import MicroClaw
    from claw.planner import EvaluationResult, Planner

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repo_path)

    try:
        # Create or get project
        project = Project(
            name=repo_path.name,
            repo_path=str(repo_path),
        )
        await ctx.repository.create_project(project)

        console.print(f"\n[bold]CLAW Enhancement: {repo_path.name}[/bold]")
        console.print(f"  Repository: {repo_path}")
        console.print(f"  Mode: {mode}")
        console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")

        if not ctx.agents:
            console.print("[red]No agents available. Enable at least one agent in claw.toml.[/red]")
            return

        # Phase 1: Evaluate
        console.print("\n[cyan]Phase 1: Evaluating repository...[/cyan]")
        analysis = await _analyze_repo(repo_path)
        _display_analysis(analysis, repo_path.name)

        # Phase 2: Plan — convert analysis into tasks
        console.print("\n[cyan]Phase 2: Planning enhancements...[/cyan]")
        planner = Planner(project_id=project.id, repository=ctx.repository)

        eval_results = _analysis_to_eval_results(analysis, repo_path.name)
        tasks = await planner.analyze_gaps(eval_results)

        if not tasks:
            console.print("[green]No enhancement tasks identified. Repository looks good![/green]")
            return

        tasks = tasks[:max_tasks]
        console.print(f"  Generated {len(tasks)} enhancement tasks")

        # Store tasks in DB
        for task in tasks:
            await ctx.repository.create_task(task)

        # Phase 3: Execute — run MicroClaw cycles
        console.print(f"\n[cyan]Phase 3: Executing {len(tasks)} tasks...[/cyan]")
        micro = MicroClaw(ctx=ctx, project_id=project.id)

        completed = 0
        failed = 0
        for i in range(len(tasks)):
            task_label = tasks[i].title[:60] if i < len(tasks) else "task"
            console.print(f"\n  [bold]Task {i + 1}/{len(tasks)}:[/bold] {task_label}")

            # Progress state shared with the callback
            progress_state = {"step": "starting", "detail": "", "start": _time.monotonic()}

            def on_step(step: str, detail: str) -> None:
                progress_state["step"] = step
                progress_state["detail"] = detail

            async def run_with_progress():
                """Run the cycle while updating a live spinner."""
                cycle_task = asyncio.create_task(micro.run_cycle(on_step=on_step))
                step_icons = {
                    "grab": "[cyan]grab[/cyan]",
                    "evaluate": "[cyan]evaluate[/cyan]",
                    "decide": "[yellow]decide[/yellow]",
                    "act": "[bold green]act[/bold green]",
                    "verify": "[magenta]verify[/magenta]",
                    "learn": "[blue]learn[/blue]",
                    "done": "[green]done[/green]",
                }
                with Live(console=console, refresh_per_second=2, transient=True) as live:
                    while not cycle_task.done():
                        elapsed = _time.monotonic() - progress_state["start"]
                        step = progress_state["step"]
                        icon = step_icons.get(step, step)
                        detail = progress_state["detail"]
                        mins = int(elapsed // 60)
                        secs = int(elapsed % 60)
                        time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                        live.update(
                            Text.from_markup(
                                f"    [{time_str}] {icon}  {detail}"
                            )
                        )
                        await asyncio.sleep(0.5)
                return cycle_task.result()

            cycle_result = await run_with_progress()

            if cycle_result.success:
                completed += 1
                duration = cycle_result.duration_seconds or 0
                console.print(f"    [green]completed[/green] ({duration:.1f}s)")
            else:
                failed += 1
                console.print(f"    [yellow]failed[/yellow]")

            # Show what the agent did
            _display_task_result(cycle_result)

            if mode == "attended":
                response = console.input("  Continue? [y/n] ")
                if response.lower() != "y":
                    console.print("  [yellow]Paused by user.[/yellow]")
                    break

        # Summary
        console.print(f"\n[bold]Enhancement Summary[/bold]")
        console.print(f"  Completed: {completed}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Results stored in {ctx.config.database.db_path}")

    finally:
        await ctx.close()


async def _enhance_battery_async(
    repo_path: Path,
    config_path: Optional[str],
    mode: str,
    max_tasks: int,
) -> None:
    """Run enhance using the full MesoClaw pipeline with evaluation battery."""
    from claw.core.factory import ClawFactory
    from claw.core.models import Project
    from claw.cycle import MesoClaw

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repo_path)

    try:
        project = Project(
            name=repo_path.name,
            repo_path=str(repo_path),
        )
        await ctx.repository.create_project(project)

        console.print(f"\n[bold]CLAW Enhancement (Battery Mode): {repo_path.name}[/bold]")
        console.print(f"  Repository: {repo_path}")
        console.print(f"  Mode: {mode}")
        console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")

        if not ctx.agents:
            console.print("[red]No agents available. Enable at least one agent in claw.toml.[/red]")
            return

        # Run MesoClaw which handles: evaluate -> plan -> dispatch -> verify -> learn
        meso = MesoClaw(
            ctx=ctx,
            project_id=project.id,
            repo_path=str(repo_path),
        )

        console.print("\n[cyan]Running MesoClaw pipeline (evaluate -> plan -> execute -> learn)...[/cyan]")

        progress_state = {"step": "starting", "detail": "", "start": _time.monotonic()}

        def on_step(step: str, detail: str) -> None:
            progress_state["step"] = step
            progress_state["detail"] = detail

        async def run_with_progress():
            cycle_task = asyncio.create_task(meso.run_cycle(on_step=on_step))
            step_icons = {
                "grab": "[cyan]grab[/cyan]",
                "evaluate": "[cyan]evaluate[/cyan]",
                "decide": "[yellow]decide[/yellow]",
                "act": "[bold green]act[/bold green]",
                "verify": "[magenta]verify[/magenta]",
                "learn": "[blue]learn[/blue]",
                "done": "[green]done[/green]",
            }
            with Live(console=console, refresh_per_second=2, transient=True) as live:
                while not cycle_task.done():
                    elapsed = _time.monotonic() - progress_state["start"]
                    step = progress_state["step"]
                    icon = step_icons.get(step, step)
                    detail = progress_state["detail"]
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                    live.update(
                        Text.from_markup(
                            f"    [{time_str}] {icon}  {detail}"
                        )
                    )
                    await asyncio.sleep(0.5)
            return cycle_task.result()

        result = await run_with_progress()

        # Display results
        console.print(f"\n[bold]Enhancement Summary (Battery Mode)[/bold]")
        console.print(f"  Success: {result.success}")
        console.print(f"  Tasks processed: {result.outcome.approach_summary[:200] if result.outcome.approach_summary else 'N/A'}")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")
        console.print(f"  Tokens: {result.tokens_used}")
        console.print(f"  Cost: ${result.cost_usd:.4f}")
        console.print(f"  Results stored in {ctx.config.database.db_path}")

    finally:
        await ctx.close()


def _analysis_to_eval_results(analysis: dict, name: str) -> list:
    """Convert structural analysis into EvaluationResult objects for the Planner."""
    from claw.planner import EvaluationResult

    results = []

    if not analysis.get("has_tests"):
        results.append(EvaluationResult(
            prompt_name="structural_analysis",
            findings=[f"{name} has no test directory — add test infrastructure"],
            severity="high",
            category="testing",
        ))

    if not analysis.get("has_readme"):
        results.append(EvaluationResult(
            prompt_name="structural_analysis",
            findings=[f"{name} is missing a README — add documentation"],
            severity="medium",
            category="docs",
        ))

    if not analysis.get("has_git"):
        results.append(EvaluationResult(
            prompt_name="structural_analysis",
            findings=[f"{name} is not a git repository — initialize git"],
            severity="low",
            category="architecture",
        ))

    if not analysis.get("config_files"):
        results.append(EvaluationResult(
            prompt_name="structural_analysis",
            findings=[f"{name} has no build/config files — add project manifest"],
            severity="medium",
            category="architecture",
        ))

    # If the analysis looks healthy, add a general enhancement task
    if not results:
        results.append(EvaluationResult(
            prompt_name="structural_analysis",
            findings=[f"General code quality review for {name}"],
            severity="low",
            category="analysis",
        ))

    return results


def _display_task_result(cycle_result) -> None:
    """Display the outcome of a single task cycle."""
    from claw.core.models import CycleResult

    if not isinstance(cycle_result, CycleResult):
        return

    outcome = cycle_result.outcome
    verification = cycle_result.verification

    # Agent and cost
    agent = cycle_result.agent_id or "unknown"
    cost = cycle_result.cost_usd
    tokens = cycle_result.tokens_used

    info_parts = [f"Agent: {agent}"]
    if cost > 0:
        info_parts.append(f"Cost: ${cost:.4f}")
    if tokens > 0:
        info_parts.append(f"Tokens: {tokens:,}")
    console.print(f"    {' | '.join(info_parts)}")

    # Approach summary (truncated for display)
    if outcome and outcome.approach_summary:
        summary = outcome.approach_summary
        if len(summary) > 200:
            summary = summary[:200] + "..."
        console.print(f"    [dim]Summary:[/dim] {summary}")

    # Files changed
    if outcome and outcome.files_changed:
        files_str = ", ".join(outcome.files_changed[:5])
        extra = f" (+{len(outcome.files_changed) - 5} more)" if len(outcome.files_changed) > 5 else ""
        console.print(f"    [dim]Files:[/dim] {files_str}{extra}")

    # Verification
    if verification:
        if verification.approved:
            console.print(f"    [green]Verified[/green] (quality: {verification.quality_score or 0:.2f})")
        else:
            v_count = len(verification.violations)
            console.print(f"    [red]Rejected[/red] ({v_count} violation{'s' if v_count != 1 else ''})")
            for v in verification.violations[:3]:
                check = v.get("check", "")
                detail = v.get("detail", "")
                console.print(f"      - {check}: {detail}")

    # Failure reason
    if outcome and outcome.failure_reason and not cycle_result.success:
        console.print(f"    [yellow]Failure:[/yellow] {outcome.failure_reason}")
        if outcome.failure_detail:
            detail = outcome.failure_detail[:150]
            console.print(f"    [dim]{detail}[/dim]")


# ---------------------------------------------------------------------------
# fleet-enhance command
# ---------------------------------------------------------------------------


@app.command(name="fleet-enhance")
def fleet_enhance(
    repos_dir: str = typer.Argument(..., help="Directory containing repositories to enhance"),
    mode: str = typer.Option("supervised", "--mode", "-m", help="Mode: attended, supervised, autonomous"),
    max_repos: int = typer.Option(10, "--max-repos", help="Maximum number of repos to process"),
    max_tasks_per_repo: int = typer.Option(5, "--max-tasks", help="Maximum tasks per repo"),
    budget: float = typer.Option(50.0, "--budget", "-b", help="Total budget in USD"),
    strategy: str = typer.Option("proportional", "--strategy", help="Budget strategy: proportional or equal"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Enhance multiple repositories across a fleet.

    Scans a directory for git repositories, ranks them by enhancement potential,
    allocates a budget, and runs the MesoClaw pipeline on each repo in ranked order.
    All agent work goes to enhancement branches -- never directly to main.
    """
    _setup_logging(verbose)

    repos_path = Path(repos_dir).resolve()
    if not repos_path.exists():
        console.print(f"[red]Repos directory does not exist: {repos_path}[/red]")
        raise typer.Exit(1)
    if not repos_path.is_dir():
        console.print(f"[red]Path is not a directory: {repos_path}[/red]")
        raise typer.Exit(1)

    if mode not in ("attended", "supervised", "autonomous"):
        console.print(f"[red]Invalid mode: {mode}. Use attended, supervised, or autonomous.[/red]")
        raise typer.Exit(1)

    if strategy not in ("proportional", "equal"):
        console.print(f"[red]Invalid strategy: {strategy}. Use proportional or equal.[/red]")
        raise typer.Exit(1)

    if budget < 0:
        console.print(f"[red]Budget must be non-negative, got {budget}[/red]")
        raise typer.Exit(1)

    if max_repos < 1:
        console.print(f"[red]--max-repos must be at least 1[/red]")
        raise typer.Exit(1)

    asyncio.run(_fleet_enhance_async(
        repos_path, config, mode, max_repos, max_tasks_per_repo, budget, strategy,
    ))


async def _fleet_enhance_async(
    repos_path: Path,
    config_path: Optional[str],
    mode: str,
    max_repos: int,
    max_tasks_per_repo: int,
    budget: float,
    strategy: str,
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project
    from claw.cycle import MicroClaw
    from claw.fleet import FleetOrchestrator
    from claw.planner import Planner

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repos_path)

    try:
        fleet = FleetOrchestrator(repository=ctx.repository, config=ctx.config)

        console.print(f"\n[bold]CLAW Fleet Enhancement[/bold]")
        console.print(f"  Repos directory: {repos_path}")
        console.print(f"  Mode: {mode}")
        console.print(f"  Budget: ${budget:.2f} ({strategy})")
        console.print(f"  Max repos: {max_repos}")
        console.print(f"  Max tasks per repo: {max_tasks_per_repo}")
        console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")
        console.print(f"  Database: {ctx.config.database.db_path}")

        # ---------------------------------------------------------------
        # Phase 1: Scan for repositories
        # ---------------------------------------------------------------
        console.print(f"\n[cyan]Phase 1: Scanning for repositories...[/cyan]")
        discovered = await fleet.scan_repos(str(repos_path))

        if not discovered:
            console.print("[yellow]No git repositories found in the directory.[/yellow]")
            return

        # Display discovered repos
        disc_table = Table(title=f"Discovered Repositories ({len(discovered)} found)")
        disc_table.add_column("#", style="dim", width=4)
        disc_table.add_column("Name", style="cyan", max_width=30)
        disc_table.add_column("Path", style="dim", max_width=50)
        disc_table.add_column("Branch", style="yellow", width=16)
        disc_table.add_column("Last Commit", style="green", width=22)

        for i, repo_info in enumerate(discovered, 1):
            branch = repo_info.get("default_branch") or "-"
            last_commit = repo_info.get("last_commit_date") or "-"
            # Truncate the ISO timestamp for display
            if last_commit != "-" and len(last_commit) > 19:
                last_commit = last_commit[:19]
            disc_table.add_row(
                str(i),
                repo_info["name"],
                repo_info["path"],
                branch,
                last_commit,
            )

        console.print(disc_table)

        # ---------------------------------------------------------------
        # Phase 2: Register repos
        # ---------------------------------------------------------------
        console.print(f"\n[cyan]Phase 2: Registering repositories...[/cyan]")
        repo_ids: dict[str, str] = {}  # repo_path -> repo_id
        for repo_info in discovered:
            repo_id = await fleet.register_repo(
                repo_path=repo_info["path"],
                repo_name=repo_info["name"],
            )
            repo_ids[repo_info["path"]] = repo_id
        console.print(f"  Registered {len(repo_ids)} repositories")

        # ---------------------------------------------------------------
        # Phase 3: Rank repos
        # ---------------------------------------------------------------
        console.print(f"\n[cyan]Phase 3: Ranking repositories...[/cyan]")
        ranked = await fleet.rank_repos()

        if not ranked:
            console.print("[yellow]No repos eligible for ranking (all completed or skipped).[/yellow]")
            return

        rank_table = Table(title="Repository Ranking")
        rank_table.add_column("Rank", style="bold", width=5)
        rank_table.add_column("Name", style="cyan", max_width=30)
        rank_table.add_column("Priority", justify="right", width=9)
        rank_table.add_column("Score", style="green", justify="right", width=8)
        rank_table.add_column("Status", width=12)

        for i, repo in enumerate(ranked[:max_repos], 1):
            status_val = repo.get("status", "pending")
            if status_val == "pending":
                status_display = "[yellow]pending[/yellow]"
            elif status_val == "completed":
                status_display = "[green]completed[/green]"
            elif status_val == "failed":
                status_display = "[red]failed[/red]"
            else:
                status_display = status_val
            rank_table.add_row(
                str(i),
                repo["repo_name"],
                f"{repo['priority']:.2f}",
                f"{repo['rank_score']:.4f}",
                status_display,
            )

        console.print(rank_table)

        # ---------------------------------------------------------------
        # Phase 4: Allocate budget
        # ---------------------------------------------------------------
        console.print(f"\n[cyan]Phase 4: Allocating budget...[/cyan]")
        allocation_result = await fleet.allocate_budget(
            total_budget_usd=budget,
            strategy=strategy,
        )

        if allocation_result["allocations"]:
            budget_table = Table(title=f"Budget Allocation (strategy: {strategy})")
            budget_table.add_column("Repo", style="cyan", max_width=30)
            budget_table.add_column("Allocated", style="green", justify="right", width=12)

            for alloc in allocation_result["allocations"]:
                budget_table.add_row(
                    alloc["repo_name"],
                    f"${alloc['allocated_usd']:.4f}",
                )

            budget_table.add_section()
            budget_table.add_row(
                "[bold]Total Allocated[/bold]",
                f"[bold]${allocation_result['allocated_usd']:.4f}[/bold]",
            )
            budget_table.add_row(
                "[dim]Unallocated[/dim]",
                f"[dim]${budget - allocation_result['allocated_usd']:.4f}[/dim]",
            )

            console.print(budget_table)
        else:
            console.print("  [yellow]No repos eligible for budget allocation.[/yellow]")

        # ---------------------------------------------------------------
        # Phase 5: Process repos
        # ---------------------------------------------------------------
        repos_to_process = ranked[:max_repos]
        console.print(
            f"\n[cyan]Phase 5: Processing {len(repos_to_process)} "
            f"repositor{'y' if len(repos_to_process) == 1 else 'ies'}...[/cyan]"
        )

        fleet_completed = 0
        fleet_failed = 0
        fleet_skipped = 0
        total_tasks_created = 0
        total_tasks_completed = 0

        for repo_idx, repo_row in enumerate(repos_to_process, 1):
            repo_name = repo_row["repo_name"]
            repo_path_str = repo_row["repo_path"]
            repo_id = repo_row["id"]
            repo_path_obj = Path(repo_path_str)

            console.print(
                f"\n{'=' * 60}\n"
                f"[bold]Repo {repo_idx}/{len(repos_to_process)}: {repo_name}[/bold]\n"
                f"  Path: {repo_path_str}"
            )

            if not repo_path_obj.exists():
                console.print(f"  [red]Path no longer exists, skipping.[/red]")
                await fleet.update_repo_status(repo_id, "skipped")
                fleet_skipped += 1
                continue

            try:
                # 5a: Create enhancement branch
                console.print(f"  [dim]Creating enhancement branch...[/dim]")
                try:
                    branch_name = await fleet.create_enhancement_branch(repo_path_str)
                    console.print(f"  Branch: [green]{branch_name}[/green]")
                    await fleet.update_repo_status(
                        repo_id, "enhancing",
                        enhancement_branch=branch_name,
                    )
                except RuntimeError as branch_err:
                    console.print(f"  [yellow]Branch creation failed: {branch_err}[/yellow]")
                    console.print(f"  [dim]Continuing on current branch.[/dim]")
                    await fleet.update_repo_status(repo_id, "enhancing")

                # 5b: Create project in DB
                project = Project(
                    name=repo_name,
                    repo_path=repo_path_str,
                )
                await ctx.repository.create_project(project)

                # 5c: Run structural analysis
                console.print(f"  [dim]Analyzing repository...[/dim]")
                analysis = await _analyze_repo(repo_path_obj)
                _display_analysis(analysis, repo_name)

                # Update evaluation timestamp
                from datetime import UTC, datetime
                await fleet.update_repo_status(
                    repo_id, "enhancing",
                    last_evaluated_at=datetime.now(UTC).isoformat(),
                )

                # Log episode
                await ctx.repository.log_episode(
                    session_id=f"fleet-{repo_id}",
                    event_type="fleet_repo_evaluated",
                    event_data={"repo_name": repo_name, "analysis": analysis},
                    project_id=project.id,
                )

                # 5d: Plan and execute tasks (requires agents)
                if ctx.agents:
                    console.print(f"  [dim]Planning enhancements...[/dim]")
                    planner = Planner(project_id=project.id, repository=ctx.repository)
                    eval_results = _analysis_to_eval_results(analysis, repo_name)
                    tasks = await planner.analyze_gaps(eval_results)

                    if not tasks:
                        console.print(f"  [green]No enhancement tasks for {repo_name}.[/green]")
                        await fleet.update_repo_status(
                            repo_id, "completed",
                            tasks_created=0,
                            tasks_completed=0,
                        )
                        fleet_completed += 1
                        continue

                    tasks = tasks[:max_tasks_per_repo]
                    for task in tasks:
                        await ctx.repository.create_task(task)

                    repo_tasks_created = len(tasks)
                    total_tasks_created += repo_tasks_created

                    await fleet.update_repo_status(
                        repo_id, "enhancing",
                        tasks_created=repo_tasks_created,
                    )

                    console.print(f"  Generated {repo_tasks_created} tasks")

                    # 5e: Run MicroClaw cycles
                    micro = MicroClaw(ctx=ctx, project_id=project.id)
                    repo_completed = 0
                    repo_failed = 0

                    for task_idx in range(len(tasks)):
                        task_label = tasks[task_idx].title[:60]
                        console.print(
                            f"\n  [bold]Task {task_idx + 1}/{len(tasks)}:[/bold] {task_label}"
                        )

                        progress_state = {
                            "step": "starting",
                            "detail": "",
                            "start": _time.monotonic(),
                        }

                        def on_step(step: str, detail: str) -> None:
                            progress_state["step"] = step
                            progress_state["detail"] = detail

                        async def run_with_progress():
                            """Run the cycle while updating a live spinner."""
                            cycle_task = asyncio.create_task(
                                micro.run_cycle(on_step=on_step)
                            )
                            step_icons = {
                                "grab": "[cyan]grab[/cyan]",
                                "evaluate": "[cyan]evaluate[/cyan]",
                                "decide": "[yellow]decide[/yellow]",
                                "act": "[bold green]act[/bold green]",
                                "verify": "[magenta]verify[/magenta]",
                                "learn": "[blue]learn[/blue]",
                                "done": "[green]done[/green]",
                            }
                            with Live(
                                console=console,
                                refresh_per_second=2,
                                transient=True,
                            ) as live:
                                while not cycle_task.done():
                                    elapsed = _time.monotonic() - progress_state["start"]
                                    step = progress_state["step"]
                                    icon = step_icons.get(step, step)
                                    detail = progress_state["detail"]
                                    mins = int(elapsed // 60)
                                    secs = int(elapsed % 60)
                                    time_str = (
                                        f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                                    )
                                    live.update(
                                        Text.from_markup(
                                            f"    [{time_str}] {icon}  {detail}"
                                        )
                                    )
                                    await asyncio.sleep(0.5)
                            return cycle_task.result()

                        cycle_result = await run_with_progress()

                        if cycle_result.success:
                            repo_completed += 1
                            duration = cycle_result.duration_seconds or 0
                            console.print(
                                f"    [green]completed[/green] ({duration:.1f}s)"
                            )
                        else:
                            repo_failed += 1
                            console.print(f"    [yellow]failed[/yellow]")

                        _display_task_result(cycle_result)

                        if mode == "attended":
                            response = console.input("  Continue to next task? [y/n] ")
                            if response.lower() != "y":
                                console.print(
                                    "  [yellow]Skipping remaining tasks for this repo.[/yellow]"
                                )
                                break

                    total_tasks_completed += repo_completed

                    # 5f: Display per-repo results
                    console.print(f"\n  [bold]{repo_name} Results:[/bold]")
                    console.print(f"    Tasks completed: {repo_completed}/{repo_tasks_created}")
                    console.print(f"    Tasks failed: {repo_failed}")

                    # 5g: Update repo status
                    final_status = "completed" if repo_failed == 0 else "failed"
                    await fleet.update_repo_status(
                        repo_id, final_status,
                        tasks_completed=repo_completed,
                    )

                    if final_status == "completed":
                        fleet_completed += 1
                    else:
                        fleet_failed += 1

                else:
                    # No agents available -- scan and rank only
                    console.print(
                        f"  [yellow]No agents available. "
                        f"Analysis stored but no tasks executed.[/yellow]"
                    )
                    await fleet.update_repo_status(
                        repo_id, "completed",
                        tasks_created=0,
                        tasks_completed=0,
                    )
                    fleet_completed += 1

            except Exception as repo_err:
                console.print(f"  [red]Error processing {repo_name}: {repo_err}[/red]")
                logging.getLogger("claw.cli").error(
                    "Fleet repo %s failed: %s", repo_name, repo_err, exc_info=True,
                )
                try:
                    await fleet.update_repo_status(repo_id, "failed")
                except Exception:
                    pass  # Best-effort status update on error path
                fleet_failed += 1

            # In attended mode, ask before moving to the next repo
            if mode == "attended" and repo_idx < len(repos_to_process):
                response = console.input("\n  Continue to next repo? [y/n] ")
                if response.lower() != "y":
                    console.print("  [yellow]Fleet processing paused by user.[/yellow]")
                    break

        # ---------------------------------------------------------------
        # Phase 6: Fleet summary
        # ---------------------------------------------------------------
        console.print(f"\n{'=' * 60}")
        summary = await fleet.get_fleet_summary()
        _display_fleet_summary(summary, fleet_completed, fleet_failed, fleet_skipped)

        console.print(f"\n[dim]Results stored in {ctx.config.database.db_path}[/dim]")

    finally:
        await ctx.close()


def _display_fleet_summary(
    summary: dict,
    repos_completed: int,
    repos_failed: int,
    repos_skipped: int,
) -> None:
    """Display the fleet processing summary using Rich tables."""

    summary_table = Table(title="Fleet Enhancement Summary", show_lines=True)
    summary_table.add_column("Metric", style="cyan", width=28)
    summary_table.add_column("Value", style="green", justify="right", width=20)

    summary_table.add_row(
        "Total Repos in Fleet",
        str(summary.get("total_repos", 0)),
    )
    summary_table.add_row("Repos Completed", f"[green]{repos_completed}[/green]")
    summary_table.add_row(
        "Repos Failed",
        f"[red]{repos_failed}[/red]" if repos_failed else "0",
    )
    summary_table.add_row(
        "Repos Skipped",
        f"[yellow]{repos_skipped}[/yellow]" if repos_skipped else "0",
    )

    # Status breakdown from DB
    by_status = summary.get("by_status", {})
    if by_status:
        status_parts = []
        for status_name, count in sorted(by_status.items()):
            status_parts.append(f"{status_name}: {count}")
        summary_table.add_row("Status Breakdown", ", ".join(status_parts))

    # Budget
    allocated = summary.get("total_budget_allocated_usd", 0.0)
    used = summary.get("total_budget_used_usd", 0.0)
    summary_table.add_row("Budget Allocated", f"${allocated:.4f}")
    summary_table.add_row("Budget Used", f"${used:.4f}")
    if allocated > 0:
        usage_pct = (used / allocated) * 100.0
        summary_table.add_row("Budget Usage", f"{usage_pct:.1f}%")

    # Tasks
    tasks_created = summary.get("total_tasks_created", 0)
    tasks_completed = summary.get("total_tasks_completed", 0)
    summary_table.add_row("Tasks Created", str(tasks_created))
    summary_table.add_row("Tasks Completed", str(tasks_completed))

    completion_rate = summary.get("completion_rate", 0.0)
    if tasks_created > 0:
        rate_str = f"{completion_rate * 100:.1f}%"
        if completion_rate >= 0.8:
            rate_display = f"[green]{rate_str}[/green]"
        elif completion_rate >= 0.5:
            rate_display = f"[yellow]{rate_str}[/yellow]"
        else:
            rate_display = f"[red]{rate_str}[/red]"
    else:
        rate_display = "-"
    summary_table.add_row("Completion Rate", rate_display)

    console.print(summary_table)


# ---------------------------------------------------------------------------
# results command
# ---------------------------------------------------------------------------


@app.command()
def results(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of results to show"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project ID"),
) -> None:
    """Show past task results from the database."""
    _setup_logging(False)
    asyncio.run(_results_async(config, limit, project))


async def _results_async(config_path: Optional[str], limit: int, project_id: Optional[str]) -> None:
    from claw.core.factory import ClawFactory

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p)

    try:
        rows = await ctx.repository.get_project_results(project_id=project_id, limit=limit)

        if not rows:
            console.print("\n[yellow]No task results found.[/yellow]")
            return

        console.print(f"\n[bold]CLAW Task Results[/bold] ({len(rows)} shown)\n")

        table = Table(show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Task", style="cyan", max_width=40)
        table.add_column("Status", width=10)
        table.add_column("Agent", style="yellow", width=8)
        table.add_column("Outcome", width=9)
        table.add_column("Duration", justify="right", width=8)
        table.add_column("Summary", max_width=50)

        for i, row in enumerate(rows, 1):
            title = (row.get("title") or "")[:40]
            status_val = row.get("status", "")
            agent = row.get("agent_id") or row.get("assigned_agent") or "-"
            hypothesis_outcome = row.get("hypothesis_outcome") or "-"
            duration = row.get("duration_seconds")
            summary = (row.get("approach_summary") or "")[:50]

            # Color status
            if status_val == "DONE":
                status_display = "[green]DONE[/green]"
            elif status_val == "PENDING":
                status_display = "[yellow]PENDING[/yellow]"
            elif status_val in ("CODING", "REVIEWING", "DISPATCHED"):
                status_display = f"[cyan]{status_val}[/cyan]"
            else:
                status_display = status_val

            # Color outcome
            if hypothesis_outcome == "SUCCESS":
                outcome_display = "[green]SUCCESS[/green]"
            elif hypothesis_outcome == "FAILURE":
                outcome_display = "[red]FAILURE[/red]"
            else:
                outcome_display = hypothesis_outcome

            # Format duration
            if duration:
                mins = int(duration // 60)
                secs = int(duration % 60)
                dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            else:
                dur_str = "-"

            table.add_row(
                str(i), title, status_display, agent,
                outcome_display, dur_str, summary,
            )

        console.print(table)

        # Quick summary stats
        total = len(rows)
        successes = sum(1 for r in rows if r.get("hypothesis_outcome") == "SUCCESS")
        failures = sum(1 for r in rows if r.get("hypothesis_outcome") == "FAILURE")
        pending = sum(1 for r in rows if r.get("status") == "PENDING")
        console.print(f"\n  Total: {total} | Success: {successes} | Failed: {failures} | Pending: {pending}")

    finally:
        await ctx.close()


@app.command()
def status(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Show CLAW system status."""
    _setup_logging(False)
    asyncio.run(_status_async(config))


async def _status_async(config_path: Optional[str]) -> None:
    from claw.core.factory import ClawFactory

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p)

    try:
        console.print("\n[bold]CLAW System Status[/bold]")
        console.print(f"  Database: {ctx.config.database.db_path}")
        console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")

        # Check agent health
        for name, agent in ctx.agents.items():
            health = await agent.health_check()
            status_str = "[green]available[/green]" if health.available else f"[red]unavailable: {health.error}[/red]"
            console.print(f"  {name}: {status_str}")

        # Task summary
        summary = await ctx.repository.get_task_status_summary()
        if summary:
            console.print("\n  Task Summary:")
            for status, count in summary.items():
                console.print(f"    {status}: {count}")
        else:
            console.print("  No tasks yet.")

    finally:
        await ctx.close()


@app.command(name="add-goal")
def add_goal(
    repo: str = typer.Argument(..., help="Path to the repository this goal is for"),
    title: str = typer.Option(..., "--title", "-t", prompt="Goal title", help="Short title for the goal"),
    description: str = typer.Option(
        ..., "--description", "-d", prompt="Goal description (what should the agent do?)",
        help="Detailed description of what should be accomplished",
    ),
    priority: str = typer.Option(
        "medium", "--priority", "-p",
        help="Priority: critical, high, medium, low",
    ),
    task_type: str = typer.Option(
        "analysis", "--type",
        help="Task type: analysis, testing, documentation, security, refactoring, bug_fix, architecture, dependency_analysis",
    ),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a",
        help="Preferred agent: claude, codex, gemini, grok (or leave blank for auto-routing)",
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Add a custom goal/task for a repository.

    Creates a task that will be picked up by `claw enhance` on the next run.
    """
    _setup_logging(False)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    valid_priorities = {"critical": 10, "high": 8, "medium": 5, "low": 2}
    if priority.lower() not in valid_priorities:
        console.print(f"[red]Invalid priority '{priority}'. Use: critical, high, medium, low[/red]")
        raise typer.Exit(1)

    valid_types = [
        "analysis", "testing", "documentation", "security", "refactoring",
        "bug_fix", "architecture", "dependency_analysis",
    ]
    if task_type not in valid_types:
        console.print(f"[red]Invalid task type '{task_type}'. Use: {', '.join(valid_types)}[/red]")
        raise typer.Exit(1)

    if agent and agent not in ("claude", "codex", "gemini", "grok"):
        console.print(f"[red]Invalid agent '{agent}'. Use: claude, codex, gemini, grok[/red]")
        raise typer.Exit(1)

    asyncio.run(_add_goal_async(
        repo_path, title, description, priority.lower(), task_type, agent, config,
    ))


async def _add_goal_async(
    repo_path: Path,
    title: str,
    description: str,
    priority: str,
    task_type: str,
    agent: Optional[str],
    config_path: Optional[str],
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project, Task, TaskStatus
    from claw.dispatcher import DEFAULT_AGENT, STATIC_ROUTING

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repo_path)

    priority_map = {"critical": 10, "high": 8, "medium": 5, "low": 2}

    try:
        # Find or create project for this repo
        project = await ctx.repository.get_project_by_name(repo_path.name)
        if project is None:
            project = Project(name=repo_path.name, repo_path=str(repo_path))
            await ctx.repository.create_project(project)
            console.print(f"  Created new project: {project.name} ({project.id})")

        # Determine recommended agent
        recommended = agent or STATIC_ROUTING.get(task_type, DEFAULT_AGENT)

        task = Task(
            project_id=project.id,
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority_map[priority],
            task_type=task_type,
            recommended_agent=recommended,
        )
        await ctx.repository.create_task(task)

        console.print(f"\n[green]Goal added successfully![/green]")
        console.print(f"  Title: {title}")
        console.print(f"  Project: {project.name}")
        console.print(f"  Priority: {priority} ({priority_map[priority]})")
        console.print(f"  Type: {task_type}")
        console.print(f"  Agent: {recommended}")
        console.print(f"  Task ID: {task.id}")
        console.print(f"\nRun [bold]claw enhance {repo_path}[/bold] to execute this goal.")

    finally:
        await ctx.close()


@app.command()
def setup(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Interactive setup for API keys, models, and agent configuration.

    Walks you through configuring each agent with API keys and model preferences,
    then writes the updated configuration to claw.toml.
    """
    import toml as _toml

    config_path = Path(config) if config else Path(__file__).parent.parent.parent / "claw.toml"
    config_path = config_path.resolve()

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("[dim]Run from the multiclaw directory or pass --config path/to/claw.toml[/dim]")
        raise typer.Exit(1)

    console.print(f"\n[bold]CLAW Setup[/bold]")
    console.print(f"  Config: {config_path}\n")

    # Load current config
    with open(config_path) as f:
        raw = _toml.load(f)

    agents_section = raw.setdefault("agents", {})
    changed = False

    # --- Agent configuration ---
    agent_info = {
        "claude": {
            "label": "Claude Code (Anthropic)",
            "key_env": "ANTHROPIC_API_KEY",
            "default_mode": "cli",
            "model_hint": "e.g. claude-sonnet-4-6, claude-opus-4-6",
        },
        "codex": {
            "label": "Codex (OpenAI)",
            "key_env": "OPENAI_API_KEY",
            "default_mode": "cli",
            "model_hint": "e.g. codex-mini-latest, o4-mini",
        },
        "gemini": {
            "label": "Gemini (Google)",
            "key_env": "GOOGLE_API_KEY",
            "default_mode": "api",
            "model_hint": "e.g. gemini-2.5-pro, gemini-2.5-flash",
        },
        "grok": {
            "label": "Grok (xAI)",
            "key_env": "XAI_API_KEY",
            "default_mode": "api",
            "model_hint": "e.g. grok-3, grok-3-mini",
        },
    }

    for agent_name, info in agent_info.items():
        console.print(f"[bold cyan]--- {info['label']} ---[/bold cyan]")

        current = agents_section.get(agent_name, {})
        current_enabled = current.get("enabled", False)
        current_model = current.get("model")
        current_budget = current.get("max_budget_usd", 1.0)

        # Check if API key is set in environment
        import os
        key_env = info["key_env"]
        key_present = bool(os.getenv(key_env, ""))
        key_status = "[green]set[/green]" if key_present else "[red]not set[/red]"
        console.print(f"  API key ({key_env}): {key_status}")

        if not key_present:
            console.print(f"  [dim]Set it with: export {key_env}=your-key-here[/dim]")

        # Enable/disable
        enable_str = console.input(
            f"  Enable {agent_name}? [{'Y/n' if current_enabled else 'y/N'}] "
        ).strip().lower()

        if enable_str == "":
            enable = current_enabled
        else:
            enable = enable_str in ("y", "yes")

        if not enable:
            agents_section.setdefault(agent_name, {})["enabled"] = False
            if enable != current_enabled:
                changed = True
            console.print(f"  [dim]{agent_name}: disabled[/dim]\n")
            continue

        # Model selection
        console.print(f"  Model ({info['model_hint']}):")
        model_input = console.input(
            f"  Model [{current_model or 'none'}]: "
        ).strip()

        model = model_input if model_input else current_model

        # Budget
        budget_input = console.input(
            f"  Max budget per task USD [{current_budget}]: "
        ).strip()

        try:
            budget = float(budget_input) if budget_input else current_budget
        except ValueError:
            console.print(f"  [yellow]Invalid budget, keeping {current_budget}[/yellow]")
            budget = current_budget

        # Mode
        current_mode = current.get("mode", info["default_mode"])
        mode_input = console.input(
            f"  Mode (cli/api) [{current_mode}]: "
        ).strip().lower()
        mode = mode_input if mode_input in ("cli", "api", "cloud") else current_mode

        # Write to config
        agent_section = agents_section.setdefault(agent_name, {})
        new_values = {
            "enabled": True,
            "mode": mode,
            "api_key_env": key_env,
            "max_concurrent": current.get("max_concurrent", 2),
            "timeout": current.get("timeout", 600 if agent_name in ("claude", "gemini") else 300),
            "max_budget_usd": budget,
        }
        if model:
            new_values["model"] = model

        if new_values != {k: current.get(k) for k in new_values}:
            changed = True

        agent_section.update(new_values)

        status_parts = [f"enabled", f"mode={mode}"]
        if model:
            status_parts.append(f"model={model}")
        status_parts.append(f"budget=${budget:.2f}")
        console.print(f"  [green]{agent_name}: {', '.join(status_parts)}[/green]\n")

    # --- OpenRouter API key (used by LLM client for verification/planning) ---
    console.print(f"[bold cyan]--- OpenRouter (LLM Client) ---[/bold cyan]")
    import os
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    or_status = "[green]set[/green]" if or_key else "[red]not set[/red]"
    console.print(f"  API key (OPENROUTER_API_KEY): {or_status}")
    if not or_key:
        console.print(f"  [dim]Set it with: export OPENROUTER_API_KEY=your-key-here[/dim]")
    console.print()

    # --- Write config ---
    if changed:
        with open(config_path, "w") as f:
            _toml.dump(raw, f)
        console.print(f"[green]Configuration saved to {config_path}[/green]")
    else:
        console.print(f"[dim]No changes made to {config_path}[/dim]")

    # --- Summary ---
    enabled_agents = [
        name for name, cfg in agents_section.items()
        if isinstance(cfg, dict) and cfg.get("enabled")
    ]
    console.print(f"\n[bold]Setup Complete[/bold]")
    console.print(f"  Enabled agents: {', '.join(enabled_agents) or 'none'}")

    # Check for missing keys
    missing_keys = []
    for name in enabled_agents:
        cfg = agents_section[name]
        key_env_name = cfg.get("api_key_env", "")
        if key_env_name and not os.getenv(key_env_name, ""):
            missing_keys.append(f"  export {key_env_name}=your-key-here")

    if missing_keys:
        console.print(f"\n[yellow]Missing API keys — add these to your shell profile:[/yellow]")
        for line in missing_keys:
            console.print(line)

    console.print(f"\n[dim]Next steps:[/dim]")
    console.print(f"  claw status              — verify agent connectivity")
    console.print(f"  claw evaluate <repo>     — analyze a repository")
    console.print(f"  claw add-goal <repo>     — add a custom task")
    console.print(f"  claw enhance <repo>      — run the full pipeline")
    console.print(f"  claw fleet-enhance <dir> — process a fleet of repos")


def app_main() -> None:
    """Entry point for the installed CLI."""
    app()


if __name__ == "__main__":
    app_main()
