"""CAM CLI — Typer-based command line interface for Clawamorphosis.

Primary workflows:
  evaluate <repo>        — inspect one repo and score improvement potential
  enhance <repo>         — improve one existing repo in a bounded loop
  mine <dir>             — learn from outside repos into CAM memory
  ideate <dir>           — invent standalone app concepts from mined knowledge
  create <repo>          — create or augment a repo from a requested outcome
  validate               — verify a created repo against its saved spec/checks

Advanced groups:
  learn <subcommand>     — learning continuum, delta, reassessment, synergies
  task <subcommand>      — goal/task setup, runbooks, and task results
  forge <subcommand>     — standalone Forge export and benchmark workflow
  doctor <subcommand>    — preflight and environment diagnostics
  kb <subcommand>        — low-level knowledge browser
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import time as _time

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="cam",
    help="CAM — inspect repos, learn from repos, create from that learning, and validate outcomes",
    no_args_is_help=True,
)
console = Console()

ROOT_DIR = Path(__file__).resolve().parents[2]
_IDEA_DIR = ROOT_DIR / "data" / "ideation"

learn_app = typer.Typer(
    name="learn",
    help="Learning lifecycle tools — delta, continuum report, reassessment, synergies",
    no_args_is_help=True,
)
task_app = typer.Typer(
    name="task",
    help="Task/operator tools — add goals, quickstart, runbooks, results",
    no_args_is_help=True,
)
forge_app = typer.Typer(
    name="forge",
    help="Standalone Forge subsystem — export knowledge packs and benchmark them",
    no_args_is_help=True,
)
doctor_app = typer.Typer(
    name="doctor",
    help="Preflight and diagnostics — key checks and system health",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_python_script_with_timeout(script_path: Path, args: list[str], max_minutes: int) -> subprocess.CompletedProcess[str]:
    if max_minutes <= 0:
        raise typer.BadParameter("max-minutes must be greater than 0")

    cmd = [sys.executable, str(script_path), *args]
    timeout_seconds = max_minutes * 60
    try:
        return subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        console.print(
            f"[red]Timed out after {max_minutes} minute(s) while running {script_path.name}[/red]"
        )
        if exc.stdout:
            console.print("[dim]Partial stdout:[/dim]")
            console.print(exc.stdout.strip())
        raise typer.Exit(124)


def _uses_remote_gemini_embeddings(config: Any) -> bool:
    model_name = str(getattr(config.embeddings, "model", "") or "")
    required_model = str(getattr(config.embeddings, "required_model", "") or "")
    return model_name.startswith("gemini-embedding") or required_model.startswith("gemini-embedding")


def _required_api_keys_for_command(config: Any, command_name: str) -> list[tuple[str, str]]:
    command = command_name.strip().lower()
    requirements: list[tuple[str, str]] = []

    if command in {"mine", "ideate"}:
        requirements.append(("OPENROUTER_API_KEY", "OpenRouter LLM access"))

    if command == "mine" and _uses_remote_gemini_embeddings(config):
        key_name = getattr(config.embeddings, "api_key_env", "") or "GOOGLE_API_KEY"
        requirements.append((str(key_name), "Gemini embeddings for methodology persistence"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key_name, reason in requirements:
        if key_name in seen:
            continue
        seen.add(key_name)
        deduped.append((key_name, reason))
    return deduped


def _select_live_llm_model(config: Any, command_name: str) -> str:
    command = command_name.strip().lower()
    if command == "mine":
        for agent_name in ("claude", "gemini", "codex", "grok"):
            agent_cfg = config.agents.get(agent_name)
            if agent_cfg and agent_cfg.enabled and agent_cfg.model:
                return agent_cfg.model
        raise typer.BadParameter("No enabled agent model is configured for mining")
    return _select_ideation_model(config)


def _print_api_key_check(config: Any, command_name: str) -> list[str]:
    requirements = _required_api_keys_for_command(config, command_name)
    console.print(f"\n[bold]CAM API Key Check[/bold]")
    console.print(f"  Command: {command_name}")

    if not requirements:
        console.print("  No API keys required for this command path.")
        return []

    table = Table(title="Required Keys")
    table.add_column("Key", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Why", style="dim", max_width=44)

    missing: list[str] = []
    for key_name, reason in requirements:
        present = bool(os.getenv(key_name, ""))
        if present:
            status = "[green]set[/green]"
        else:
            status = "[red]missing[/red]"
            missing.append(key_name)
        table.add_row(key_name, status, reason)

    console.print(table)
    return missing


def _fail_if_missing_api_keys(config: Any, command_name: str) -> None:
    missing = _print_api_key_check(config, command_name)
    if not missing:
        return

    console.print("\n[red]Required API keys are missing. Refusing to start live work.[/red]")
    for key_name in missing:
        console.print(f"  export {key_name}=your-key-here")
    raise typer.Exit(1)


async def _run_live_key_checks(config: Any, command_name: str) -> list[dict[str, str]]:
    from claw.db.embeddings import EmbeddingEngine
    from claw.llm.client import LLMClient, LLMMessage

    command = command_name.strip().lower()
    results: list[dict[str, str]] = []

    if any(key == "OPENROUTER_API_KEY" for key, _ in _required_api_keys_for_command(config, command)):
        model = _select_live_llm_model(config, command)
        llm_client = LLMClient(config.llm)
        try:
            response = await llm_client.complete(
                messages=[LLMMessage(role="user", content="Reply with OK only.")],
                model=model,
                temperature=0.0,
                max_tokens=8,
            )
            content = (response.content or "").strip().replace("\n", " ")
            results.append({
                "service": "OpenRouter",
                "status": "ok",
                "detail": f"model={model} reply={content[:60] or 'non-empty'}",
            })
        except Exception as exc:
            results.append({
                "service": "OpenRouter",
                "status": "failed",
                "detail": str(exc),
            })
        finally:
            await llm_client.close()

    if command == "mine" and _uses_remote_gemini_embeddings(config):
        try:
            engine = EmbeddingEngine(config.embeddings)
            vector = engine.encode("cam keycheck live probe")
            results.append({
                "service": "Gemini embeddings",
                "status": "ok",
                "detail": f"model={engine.model_name} dim={len(vector)}",
            })
        except Exception as exc:
            results.append({
                "service": "Gemini embeddings",
                "status": "failed",
                "detail": str(exc),
            })

    return results


def _render_live_key_check_results(results: list[dict[str, str]]) -> bool:
    console.print("\n[bold]CAM Live API Validation[/bold]")
    table = Table(title="Provider Checks")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail", max_width=56)

    failed = False
    for item in results:
        status = item["status"]
        rendered_status = "[green]ok[/green]" if status == "ok" else "[red]failed[/red]"
        if status != "ok":
            failed = True
        table.add_row(item["service"], rendered_status, item["detail"])

    console.print(table)
    return not failed


def _fail_if_live_key_checks_fail(config: Any, command_name: str) -> None:
    try:
        live_results = asyncio.run(_run_live_key_checks(config, command_name))
    except Exception as exc:
        console.print(f"\n[red]Live preflight failed before provider validation: {exc}[/red]")
        raise typer.Exit(1)

    if not _render_live_key_check_results(live_results):
        raise typer.Exit(1)


def _build_create_spec(
    repo_path: Path,
    request: str,
    repo_mode: str,
    title: str,
    task_type: str,
    execution_steps: list[str],
    acceptance_checks: list[str],
    spec_items: list[str],
) -> dict[str, Any]:
    baseline_snapshot = _snapshot_repo_state(repo_path)
    return {
        "version": 1,
        "title": title,
        "request": request,
        "repo_mode": repo_mode,
        "target_repo": str(repo_path),
        "task_type": task_type,
        "spec_items": spec_items,
        "baseline_snapshot": baseline_snapshot,
        "execution_steps": execution_steps,
        "acceptance_checks": acceptance_checks,
        "validation": {
            "require_repo_exists": True,
            "require_nonempty_repo": True,
        },
        "benchmark": {
            "catastrophic_floor_pct": -35.0,
            "require_non_negative_lift": False,
        },
        "created_at_epoch": int(_time.time()),
    }


def _write_create_spec(spec: dict[str, Any]) -> Path:
    spec_dir = ROOT_DIR / "data" / "create_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d-%H%M%S", _time.localtime(spec["created_at_epoch"]))
    repo_slug = Path(spec["target_repo"]).name or "repo"
    filename = f"{timestamp}-{repo_slug}-create-spec.json"
    out_path = spec_dir / filename
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return out_path


def _build_create_description(request: str, repo_mode: str, spec_path: Path, spec_items: list[str]) -> str:
    lines = [
        f"Creation mode: {repo_mode}",
        "",
        "Requested outcome:",
        request.strip(),
        "",
        f"Spec file: {spec_path}",
    ]
    if spec_items:
        lines.extend(["", "Initial specs:"])
        lines.extend([f"- {item}" for item in spec_items])
    lines.extend(
        [
            "",
            "Requirement: use prior mined/assimilated CAM knowledge where relevant.",
            "Outcome target: produce the requested repo state, not just analysis.",
        ]
    )
    return "\n".join(lines)


def _select_ideation_model(config: Any, preferred_agent: Optional[str] = None) -> str:
    agent_order = [preferred_agent] if preferred_agent else ["claude", "gemini", "codex", "grok"]
    if not preferred_agent:
        agent_order = ["claude", "gemini", "codex", "grok"]

    for agent_name in agent_order:
        if not agent_name:
            continue
        agent_cfg = config.agents.get(agent_name)
        if agent_cfg and agent_cfg.enabled and agent_cfg.model:
            return agent_cfg.model
    raise typer.BadParameter("No enabled agent with a configured model is available for ideation")


def _summarize_repo_tree(repo_path: Path, max_files: int = 10) -> dict[str, Any]:
    from claw.miner import _CODE_EXTENSIONS, _SKIP_DIRS

    marker_names = {
        "README.md", "README.rst", "README.txt",
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
        "requirements.txt", "setup.py", "Makefile", "Dockerfile",
    }

    sample_files: list[str] = []
    marker_files: list[str] = []
    top_dirs: list[str] = []

    try:
        for entry in sorted(repo_path.iterdir(), key=lambda p: p.name):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and entry.name not in _SKIP_DIRS and len(top_dirs) < 8:
                top_dirs.append(entry.name)
            elif entry.is_file() and entry.name in marker_names and len(marker_files) < 8:
                marker_files.append(entry.name)
    except OSError:
        pass

    try:
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_path)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if path.name in marker_names or path.suffix.lower() in _CODE_EXTENSIONS:
                sample_files.append(str(rel))
            if len(sample_files) >= max_files:
                break
    except OSError:
        pass

    return {
        "name": repo_path.name,
        "path": str(repo_path),
        "marker_files": marker_files,
        "top_dirs": top_dirs,
        "sample_files": sample_files,
    }


def _summarize_methodology(meth: Any) -> dict[str, Any]:
    return {
        "id": getattr(meth, "id", ""),
        "problem": getattr(meth, "problem_description", "")[:240],
        "notes": (getattr(meth, "methodology_notes", "") or "")[:240],
        "tags": list(getattr(meth, "tags", []) or [])[:8],
        "novelty_score": getattr(meth, "novelty_score", None),
        "potential_score": getattr(meth, "potential_score", None),
    }


def _classify_assimilation_stage(
    meth: Any,
    *,
    template_count: int = 0,
    template_successes: int = 0,
) -> str:
    """Classify a methodology along the learning/usefulness continuum."""
    if getattr(meth, "success_count", 0) > 0 or template_successes > 0:
        return "proven"
    if template_count > 0:
        return "operationalized"
    if getattr(meth, "retrieval_count", 0) > 0:
        return "retrieved"
    if (
        getattr(meth, "capability_data", None) is not None
        or getattr(meth, "novelty_score", None) is not None
        or getattr(meth, "potential_score", None) is not None
    ):
        return "enriched"
    return "stored"


def _is_future_candidate(
    meth: Any,
    *,
    potential_threshold: float,
    template_count: int = 0,
) -> bool:
    """Estimate whether a methodology looks promising for future use."""
    if getattr(meth, "success_count", 0) > 0:
        return False
    potential = getattr(meth, "potential_score", None)
    if potential is not None and potential >= potential_threshold:
        return True
    capability_data = getattr(meth, "capability_data", None) or {}
    domains = capability_data.get("domain", []) if isinstance(capability_data, dict) else []
    if capability_data and domains and template_count > 0:
        return True
    return False


_TRIGGER_KEYWORDS: dict[str, set[str]] = {
    "frontend": {"frontend", "ui", "ux", "react", "component", "design"},
    "backend": {"backend", "api", "server", "service", "endpoint"},
    "finetuning": {"finetune", "fine-tuning", "training", "adapter", "lora", "sft"},
    "evaluation": {"eval", "evaluation", "benchmark", "grade", "metrics", "score"},
    "validation": {"validate", "validation", "check", "verify", "assert"},
    "repo_repair": {"repair", "fix", "debug", "failing", "broken", "regression"},
    "testing": {"test", "pytest", "unit", "integration", "coverage"},
    "data_pipeline": {"dataset", "jsonl", "ingest", "embedding", "pipeline", "packing"},
    "security": {"security", "privacy", "secret", "auth", "permission"},
    "deployment": {"deploy", "deployment", "release", "packaging", "ops", "ci", "cd"},
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using", "use",
    "build", "create", "make", "repo", "project", "task", "app", "tool", "system",
    "your", "their", "there", "will", "have", "has", "was", "are", "its", "not",
}


def _tokenize_reassessment_text(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_+-]{3,}", text.lower()))
    return {t for t in tokens if t not in _STOPWORDS}


def _derive_activation_triggers(meth: Any, *, template_count: int = 0) -> list[str]:
    """Derive lightweight trigger metadata from existing methodology fields."""
    text_parts = [
        getattr(meth, "problem_description", "") or "",
        getattr(meth, "methodology_notes", "") or "",
        " ".join(getattr(meth, "tags", []) or []),
    ]
    capability_data = getattr(meth, "capability_data", None) or {}
    if isinstance(capability_data, dict):
        text_parts.extend(capability_data.get("domain", []) or [])
        ctype = capability_data.get("capability_type")
        if ctype:
            text_parts.append(str(ctype))
        for io_key in ("inputs", "outputs"):
            for item in capability_data.get(io_key, []) or []:
                if isinstance(item, dict):
                    if item.get("type"):
                        text_parts.append(str(item["type"]))
                    if item.get("name"):
                        text_parts.append(str(item["name"]))

    text = " ".join(text_parts).lower()
    triggers: list[str] = []
    for name, keywords in _TRIGGER_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            triggers.append(name)
    if template_count > 0:
        triggers.append("has_action_template")
    if getattr(meth, "potential_score", None) is not None and (meth.potential_score or 0) >= 0.75:
        triggers.append("high_future_value")
    return sorted(set(triggers))


def _score_methodology_for_task(
    meth: Any,
    *,
    task_tokens: set[str],
    repo_tokens: set[str],
    template_count: int = 0,
    template_successes: int = 0,
) -> tuple[float, list[str], list[str]]:
    """Heuristic task-conditioned reassessment score and explanation."""
    tags = set(getattr(meth, "tags", []) or [])
    capability_data = getattr(meth, "capability_data", None) or {}
    domains = set(capability_data.get("domain", []) if isinstance(capability_data, dict) else [])
    triggers = _derive_activation_triggers(meth, template_count=template_count)
    trigger_set = set(triggers)

    text_tokens = _tokenize_reassessment_text(
        " ".join([
            getattr(meth, "problem_description", "") or "",
            getattr(meth, "methodology_notes", "") or "",
            " ".join(tags),
            " ".join(domains),
            " ".join(triggers),
        ])
    )

    overlap_tokens = sorted((task_tokens | repo_tokens) & text_tokens)
    score = 0.0
    reasons: list[str] = []

    if overlap_tokens:
        overlap_score = min(0.45, 0.06 * len(overlap_tokens))
        score += overlap_score
        reasons.append("task/repo overlap: " + ", ".join(overlap_tokens[:5]))

    potential = getattr(meth, "potential_score", None)
    if potential is not None:
        score += min(0.2, potential * 0.2)
        if potential >= 0.65:
            reasons.append(f"high potential {potential:.2f}")

    novelty = getattr(meth, "novelty_score", None)
    if novelty is not None and novelty >= 0.45:
        score += min(0.08, novelty * 0.08)
        reasons.append(f"novelty {novelty:.2f}")

    retrieval_count = getattr(meth, "retrieval_count", 0) or 0
    if retrieval_count > 0:
        score += min(0.1, 0.02 * retrieval_count)
        reasons.append(f"retrieved {retrieval_count}x")

    direct_success = getattr(meth, "success_count", 0) or 0
    if direct_success > 0 or template_successes > 0:
        combined_success = direct_success + template_successes
        score += min(0.18, 0.05 * combined_success)
        reasons.append(f"success evidence {combined_success}")

    if template_count > 0:
        score += min(0.12, 0.04 * template_count)
        reasons.append(f"{template_count} action template(s)")

    if trigger_set & task_tokens:
        score += 0.08
        reasons.append("activation trigger matched task")

    return score, reasons, triggers


_TRIGGER_OPPORTUNITY_MAP: dict[str, str] = {
    "finetuning": "Task-specific small-model training or adapter pipelines",
    "evaluation": "Benchmark and evaluation harnesses with measurable pass/fail criteria",
    "validation": "Spec-backed validation and acceptance-check workflows",
    "repo_repair": "Automated repo repair, regression triage, and fix suggestions",
    "testing": "Test generation, stabilization, and coverage-improvement workflows",
    "data_pipeline": "Dataset, embedding, ingestion, or packing pipelines",
    "frontend": "Frontend scaffolding, UI modernization, and usability improvements",
    "backend": "Service/API modernization and backend capability upgrades",
    "security": "Security hardening, secret handling, and permission boundary improvements",
    "deployment": "CI/CD, packaging, and deployment automation",
}


def _summarize_new_capabilities(methodologies: list[Any]) -> dict[str, Any]:
    """Summarize domains, capability types, and source repos for newly mined methodologies."""
    domains: dict[str, int] = {}
    capability_types: dict[str, int] = {}
    source_repos: dict[str, int] = {}

    for meth in methodologies:
        capability_data = getattr(meth, "capability_data", None) or {}
        if isinstance(capability_data, dict):
            for domain in capability_data.get("domain", []) or []:
                domains[str(domain)] = domains.get(str(domain), 0) + 1
            cap_type = capability_data.get("capability_type")
            if cap_type:
                capability_types[str(cap_type)] = capability_types.get(str(cap_type), 0) + 1

        for tag in getattr(meth, "tags", []) or []:
            if isinstance(tag, str) and tag.startswith("source:"):
                source_repo = tag.split(":", 1)[1]
                source_repos[source_repo] = source_repos.get(source_repo, 0) + 1

    return {
        "domains": sorted(domains.items(), key=lambda item: (-item[1], item[0])),
        "capability_types": sorted(capability_types.items(), key=lambda item: (-item[1], item[0])),
        "source_repos": sorted(source_repos.items(), key=lambda item: (-item[1], item[0])),
    }


def _infer_feature_opportunities(
    methodologies: list[Any],
    *,
    methodology_ids_with_templates: Optional[set[str]] = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Infer likely next-step features or updates from newly mined methodologies."""
    trigger_counts: dict[str, int] = {}
    template_ids = methodology_ids_with_templates or set()
    for meth in methodologies:
        template_count = 1 if getattr(meth, "id", "") in template_ids else 0
        for trigger in _derive_activation_triggers(meth, template_count=template_count):
            if trigger in _TRIGGER_OPPORTUNITY_MAP:
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

    ranked = sorted(trigger_counts.items(), key=lambda item: (-item[1], item[0]))
    opportunities: list[dict[str, Any]] = []
    for trigger, count in ranked[:limit]:
        opportunities.append({
            "trigger": trigger,
            "count": count,
            "description": _TRIGGER_OPPORTUNITY_MAP[trigger],
        })
    return opportunities


def _recently_created_near_mine(
    created_at: Any,
    mine_ts: float,
    *,
    lookback_hours: float = 6.0,
) -> bool:
    """Best-effort fallback for older ledger entries that do not record created IDs."""
    if created_at is None:
        return False
    try:
        created_ts = created_at.timestamp()
    except AttributeError:
        return False
    return (mine_ts - (lookback_hours * 3600)) <= created_ts <= (mine_ts + 300)


def _build_ideation_prompt(
    focus: str,
    repo_contexts: list[dict[str, Any]],
    repo_findings: dict[str, list[dict[str, Any]]],
    cam_memory: dict[str, list[dict[str, Any]]],
    idea_count: int,
) -> str:
    goal = focus.strip() or (
        "Propose novel standalone applications that combine CAM's strongest existing knowledge "
        "with the most useful mechanisms visible in the candidate repos."
    )

    return (
        "You are CAM's product ideation engine.\n"
        "Use the candidate repo context plus CAM's existing knowledge to propose novel, useful, "
        "non-demo application ideas.\n\n"
        "Rules:\n"
        "- Prefer ideas that build, troubleshoot, create, validate, or automate real work.\n"
        "- Do not propose generic chat apps or vague agents.\n"
        "- Each idea must clearly combine CAM knowledge with one or more candidate repos.\n"
        "- Favor standalone apps, not modifications to CAM itself.\n"
        "- Return strict JSON only.\n\n"
        f"User focus:\n{goal}\n\n"
        "Candidate repo summaries:\n"
        f"{json.dumps(repo_contexts, indent=2)}\n\n"
        "Existing mined findings by repo:\n"
        f"{json.dumps(repo_findings, indent=2)}\n\n"
        "CAM memory highlights:\n"
        f"{json.dumps(cam_memory, indent=2)}\n\n"
        f"Return a JSON object with key 'ideas' containing exactly {idea_count} items.\n"
        "Each idea must contain:\n"
        "- title\n"
        "- tagline\n"
        "- problem\n"
        "- why_valuable\n"
        "- novelty\n"
        "- repos_used (array)\n"
        "- cam_knowledge_used (array)\n"
        "- app_request\n"
        "- spec_items (array)\n"
        "- execution_steps (array)\n"
        "- acceptance_checks (array)\n"
        "- repo_mode\n"
        "- build_confidence (0.0 to 1.0)\n"
    )


def _normalize_ideation_payload(payload: dict[str, Any], idea_count: int) -> list[dict[str, Any]]:
    raw_ideas = payload.get("ideas", [])
    if not isinstance(raw_ideas, list):
        return []

    ideas: list[dict[str, Any]] = []
    for idx, idea in enumerate(raw_ideas[:idea_count], start=1):
        if not isinstance(idea, dict):
            continue
        title = str(idea.get("title", "")).strip() or f"Idea {idx}"
        try:
            build_confidence = float(idea.get("build_confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            build_confidence = 0.5
        normalized = {
            "title": title,
            "tagline": str(idea.get("tagline", "")).strip(),
            "problem": str(idea.get("problem", "")).strip(),
            "why_valuable": str(idea.get("why_valuable", "")).strip(),
            "novelty": str(idea.get("novelty", "")).strip(),
            "repos_used": [str(x).strip() for x in idea.get("repos_used", []) if str(x).strip()],
            "cam_knowledge_used": [str(x).strip() for x in idea.get("cam_knowledge_used", []) if str(x).strip()],
            "app_request": str(idea.get("app_request", "")).strip() or title,
            "spec_items": [str(x).strip() for x in idea.get("spec_items", []) if str(x).strip()],
            "execution_steps": [str(x).strip() for x in idea.get("execution_steps", []) if str(x).strip()],
            "acceptance_checks": [str(x).strip() for x in idea.get("acceptance_checks", []) if str(x).strip()],
            "repo_mode": str(idea.get("repo_mode", "new")).strip() or "new",
            "build_confidence": build_confidence,
        }
        ideas.append(normalized)
    return ideas


def _render_ideation_markdown(
    focus: str,
    source_dir: Path,
    ideas: list[dict[str, Any]],
) -> str:
    lines = [
        "# CAM Ideation Report",
        "",
        f"- Source directory: `{source_dir}`",
        f"- Focus: {focus or 'general'}",
        f"- Ideas generated: {len(ideas)}",
        "",
    ]
    for idx, idea in enumerate(ideas, start=1):
        lines.extend(
            [
                f"## {idx}. {idea['title']}",
                "",
                idea["tagline"] or "_No tagline provided._",
                "",
                f"Problem: {idea['problem']}",
                "",
                f"Why valuable: {idea['why_valuable']}",
                "",
                f"Novelty: {idea['novelty']}",
                "",
                f"Repos used: {', '.join(idea['repos_used']) or 'n/a'}",
                "",
                f"CAM knowledge used: {', '.join(idea['cam_knowledge_used']) or 'n/a'}",
                "",
                f"Build confidence: {idea['build_confidence']:.2f}",
                "",
                "Spec items:",
            ]
        )
        if idea["spec_items"]:
            lines.extend([f"- {item}" for item in idea["spec_items"]])
        else:
            lines.append("- n/a")
        lines.extend(["", "Acceptance checks:"])
        if idea["acceptance_checks"]:
            lines.extend([f"- {item}" for item in idea["acceptance_checks"]])
        else:
            lines.append("- n/a")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_ideation_artifacts(
    source_dir: Path,
    focus: str,
    ideas: list[dict[str, Any]],
    raw_payload: dict[str, Any],
) -> tuple[Path, Path]:
    _IDEA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d-%H%M%S", _time.localtime())
    slug = source_dir.name or "ideas"
    json_path = _IDEA_DIR / f"{timestamp}-{slug}-ideas.json"
    md_path = _IDEA_DIR / f"{timestamp}-{slug}-ideas.md"

    json_path.write_text(
        json.dumps(
            {
                "focus": focus,
                "source_dir": str(source_dir),
                "ideas": ideas,
                "raw_payload": raw_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_render_ideation_markdown(focus, source_dir, ideas), encoding="utf-8")
    return json_path, md_path


def _run_validation_check(command: str, cwd: Path, timeout_seconds: float) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_seconds),
            check=False,
        )
        return {
            "command": command,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "ok": False,
            "returncode": 124,
            "stdout": (exc.stdout or "").strip() if exc.stdout else "",
            "stderr": (exc.stderr or "").strip() if exc.stderr else "",
            "timeout": True,
        }


def _snapshot_repo_state(repo_path: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not repo_path.exists():
        return snapshot

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_path)
        if ".git" in rel.parts:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        snapshot[str(rel)] = json.dumps(
            {
                "size": len(data),
                "sha1": hashlib.sha1(data).hexdigest(),
            },
            sort_keys=True,
        )
    return snapshot


def _looks_like_shell_command(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    shell_markers = ("&&", "||", "|", ">", "<", ";", "./", "../")
    if any(marker in text for marker in shell_markers):
        return True
    first = text.split()[0].lower()
    known_commands = {
        "python", "python3", "pytest", "uv", "npm", "npx", "node",
        "cargo", "go", "make", "ruff", "mypy", "bash", "sh", "git",
        "ls", "cat", "echo",
    }
    return first in known_commands


def _validate_create_spec(spec: dict[str, Any], max_minutes: int) -> tuple[bool, dict[str, Any]]:
    start = _time.monotonic()
    findings: list[str] = []
    checks: list[dict[str, Any]] = []
    manual_checks: list[str] = []

    repo_path = Path(str(spec.get("target_repo", ""))).resolve()
    validation_cfg = spec.get("validation", {}) if isinstance(spec.get("validation"), dict) else {}
    acceptance_checks = spec.get("acceptance_checks", []) if isinstance(spec.get("acceptance_checks"), list) else []

    require_repo_exists = bool(validation_cfg.get("require_repo_exists", True))
    require_nonempty_repo = bool(validation_cfg.get("require_nonempty_repo", True))

    if require_repo_exists and not repo_path.exists():
        findings.append(f"target repo does not exist: {repo_path}")
    elif repo_path.exists() and require_nonempty_repo:
        has_files = any(p.is_file() for p in repo_path.rglob("*"))
        if not has_files:
            findings.append(f"target repo has no files: {repo_path}")

    baseline_snapshot = spec.get("baseline_snapshot", {}) if isinstance(spec.get("baseline_snapshot"), dict) else {}
    if repo_path.exists() and baseline_snapshot:
        current_snapshot = _snapshot_repo_state(repo_path)
        if current_snapshot == baseline_snapshot:
            findings.append("target repo is unchanged since create spec was written")

    deadline = start + (max_minutes * 60)
    for command in acceptance_checks:
        if not _looks_like_shell_command(str(command)):
            manual_checks.append(str(command))
            continue
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            findings.append("validation timed out before all acceptance checks completed")
            break
        check_result = _run_validation_check(str(command), cwd=repo_path, timeout_seconds=remaining)
        checks.append(check_result)
        if not check_result["ok"]:
            findings.append(f"acceptance check failed: {command}")

    summary = {
        "repo": str(repo_path),
        "title": spec.get("title", ""),
        "request": spec.get("request", ""),
        "repo_mode": spec.get("repo_mode", ""),
        "checks_run": len(checks),
        "checks": checks,
        "manual_checks": manual_checks,
        "findings": findings,
    }
    return len(findings) == 0, summary


def _validate_benchmark_against_spec(summary: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, list[str]]:
    benchmark = spec.get("benchmark", {}) if isinstance(spec, dict) else {}
    best = summary.get("best", {}) if isinstance(summary, dict) else {}
    findings: list[str] = []

    catastrophic_floor = float(benchmark.get("catastrophic_floor_pct", -35.0))
    lift_pct = float(best.get("hit_rate_lift_pct", 0.0))
    if lift_pct < catastrophic_floor:
        findings.append(
            f"lift {lift_pct:.2f}% is below catastrophic floor {catastrophic_floor:.2f}%"
        )

    require_non_negative = bool(benchmark.get("require_non_negative_lift", False))
    if require_non_negative and lift_pct < 0:
        findings.append(f"lift {lift_pct:.2f}% is negative but spec requires non-negative lift")

    return len(findings) == 0, findings


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
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview planned tasks without writing tasks or executing agents"),
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
        asyncio.run(_enhance_battery_async(repo_path, config, mode, max_tasks, dry_run))
    else:
        asyncio.run(_enhance_async(repo_path, config, mode, max_tasks, dry_run))


async def _enhance_async(
    repo_path: Path,
    config_path: Optional[str],
    mode: str,
    max_tasks: int,
    dry_run: bool = False,
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
        console.print(f"  Dry run: {'yes' if dry_run else 'no'}")
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

        if dry_run:
            _display_planned_tasks(tasks, title=f"Planned Tasks (dry-run): {repo_path.name}")
            console.print("\n[yellow]Dry run enabled: no tasks written, no agents executed.[/yellow]")
            return

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
    dry_run: bool = False,
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
        console.print(f"  Dry run: {'yes' if dry_run else 'no'}")
        console.print(f"  Agents: {', '.join(ctx.agents.keys()) or 'none'}")

        if not ctx.agents:
            console.print("[red]No agents available. Enable at least one agent in claw.toml.[/red]")
            return

        if dry_run:
            meso_preview = MesoClaw(
                ctx=ctx,
                project_id=project.id,
                repo_path=str(repo_path),
            )
            console.print("\n[cyan]Dry-run: evaluating and planning only (no task execution)...[/cyan]")
            evaluation = await meso_preview.evaluate(str(repo_path))
            tasks = await meso_preview.decide(evaluation)
            tasks = tasks[:max_tasks]
            _display_planned_tasks(tasks, title=f"Planned Tasks (battery dry-run): {repo_path.name}")
            console.print("\n[yellow]Dry run enabled: no tasks written, no agents executed.[/yellow]")
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


def _display_planned_tasks(tasks: list, title: str = "Planned Tasks") -> None:
    """Show a concise preview of planned tasks for dry-run workflows."""
    if not tasks:
        console.print("\n[yellow]No tasks were planned.[/yellow]")
        return

    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Priority", justify="right", style="green", width=8)
    table.add_column("Agent", style="yellow", width=10)
    table.add_column("Type", style="cyan", width=18)
    table.add_column("Title", style="white", max_width=46)
    table.add_column("Runbook", style="magenta", width=12)

    for i, task in enumerate(tasks, 1):
        steps = len(getattr(task, "execution_steps", []) or [])
        checks = len(getattr(task, "acceptance_checks", []) or [])
        runbook_label = f"{steps} step/{checks} check"
        if steps != 1:
            runbook_label = f"{steps} steps/{checks} checks"

        table.add_row(
            str(i),
            str(getattr(task, "priority", 0)),
            getattr(task, "recommended_agent", None) or "-",
            (getattr(task, "task_type", None) or "general")[:18],
            (getattr(task, "title", "") or "")[:46],
            runbook_label,
        )

    console.print()
    console.print(table)


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


@app.command(hidden=True)
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


def _display_runbook_details(task, project_name: str, action_template=None) -> None:
    """Render runbook sections for a task with optional template fallback."""
    execution_steps = list(task.execution_steps)
    acceptance_checks = list(task.acceptance_checks)
    preconditions: list[str] = []
    rollback_steps: list[str] = []

    if action_template is not None:
        if not execution_steps:
            execution_steps = list(action_template.execution_steps)
        if not acceptance_checks:
            acceptance_checks = list(action_template.acceptance_checks)
        preconditions = list(action_template.preconditions)
        rollback_steps = list(action_template.rollback_steps)

    console.print(f"\n[bold]Task Runbook[/bold]")
    console.print(f"  Task: {task.title}")
    console.print(f"  Task ID: {task.id}")
    console.print(f"  Project: {project_name}")
    console.print(f"  Status: {task.status.value}")
    console.print(f"  Agent: {task.recommended_agent or task.assigned_agent or '-'}")
    if action_template is not None:
        console.print(
            f"  Template: {action_template.title} "
            f"(confidence={action_template.confidence:.2f}, "
            f"S/F={action_template.success_count}/{action_template.failure_count})"
        )

    if preconditions:
        console.print("\n[cyan]Preconditions[/cyan]")
        for item in preconditions:
            console.print(f"  - {item}")

    if execution_steps:
        console.print("\n[cyan]Execution Steps[/cyan]")
        for i, step in enumerate(execution_steps, 1):
            console.print(f"  {i}. {step}")
    else:
        console.print("\n[yellow]No execution steps defined yet.[/yellow]")

    if acceptance_checks:
        console.print("\n[cyan]Acceptance Checks[/cyan]")
        for i, check in enumerate(acceptance_checks, 1):
            console.print(f"  {i}. {check}")
    else:
        console.print("\n[yellow]No acceptance checks defined yet.[/yellow]")

    if rollback_steps:
        console.print("\n[cyan]Rollback Steps[/cyan]")
        for i, step in enumerate(rollback_steps, 1):
            console.print(f"  {i}. {step}")


@app.command(hidden=True)
def runbook(
    task_id: str = typer.Argument(..., help="Task ID to inspect"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Display execution steps and acceptance checks for a task."""
    _setup_logging(False)
    asyncio.run(_runbook_async(task_id, config))


async def _runbook_async(task_id: str, config_path: Optional[str]) -> None:
    from claw.core.factory import ClawFactory

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p)

    try:
        task = await ctx.repository.get_task(task_id)
        if task is None:
            console.print(f"[red]Task not found: {task_id}[/red]")
            raise typer.Exit(1)

        project = await ctx.repository.get_project(task.project_id)
        action_template = None
        if task.action_template_id:
            action_template = await ctx.repository.get_action_template(task.action_template_id)
        _display_runbook_details(
            task=task,
            project_name=project.name if project else task.project_id,
            action_template=action_template,
        )

        console.print("\n[dim]Use `cam enhance <repo> --dry-run` to preview execution without running agents.[/dim]")

    finally:
        await ctx.close()


@app.command(hidden=True)
def quickstart(
    repo: str = typer.Argument(..., help="Path to the repository this goal is for"),
    title: str = typer.Option(..., "--title", "-t", prompt="Goal title", help="Short title for the goal"),
    description: str = typer.Option(
        ..., "--description", "-d", prompt="Goal description (what should be done?)",
        help="Detailed goal description",
    ),
    priority: str = typer.Option("high", "--priority", "-p", help="Priority: critical, high, medium, low"),
    task_type: str = typer.Option(
        "bug_fix",
        "--type",
        help="Task type: analysis, testing, documentation, security, refactoring, bug_fix, architecture, dependency_analysis",
    ),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Preferred agent: claude, codex, gemini, grok (or leave blank for auto-routing)",
    ),
    step: list[str] = typer.Option(
        [],
        "--step",
        help="Execution command to run for this goal (repeat --step for multiple commands)",
    ),
    check: list[str] = typer.Option(
        [],
        "--check",
        help="Acceptance check command for this goal (repeat --check for multiple commands)",
    ),
    preview: bool = typer.Option(
        True,
        "--preview/--no-preview",
        help="Show runbook and dry-run preview after creating the goal",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Immediately execute this exact task after setup",
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Guided one-command setup: add goal + preview runbook (+ optional execution)."""
    _setup_logging(False)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    asyncio.run(_quickstart_async(
        repo_path=repo_path,
        title=title,
        description=description,
        priority=priority.lower(),
        task_type=task_type,
        agent=agent,
        execution_steps=step,
        acceptance_checks=check,
        preview=preview,
        execute=execute,
        config_path=config,
    ))


async def _quickstart_async(
    repo_path: Path,
    title: str,
    description: str,
    priority: str,
    task_type: str,
    agent: Optional[str],
    execution_steps: list[str],
    acceptance_checks: list[str],
    preview: bool,
    execute: bool,
    config_path: Optional[str],
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import CycleResult, Project, Task, TaskStatus
    from claw.cycle import MicroClaw
    from claw.dispatcher import DEFAULT_AGENT, STATIC_ROUTING

    valid_priorities = {"critical": 10, "high": 8, "medium": 5, "low": 2}
    if priority not in valid_priorities:
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

    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=repo_path)

    try:
        project = await ctx.repository.get_project_by_name(repo_path.name)
        if project is None:
            project = Project(name=repo_path.name, repo_path=str(repo_path))
            await ctx.repository.create_project(project)

        recommended = agent or STATIC_ROUTING.get(task_type, DEFAULT_AGENT)
        task = Task(
            project_id=project.id,
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=valid_priorities[priority],
            task_type=task_type,
            recommended_agent=recommended,
            execution_steps=[s.strip() for s in execution_steps if s.strip()],
            acceptance_checks=[s.strip() for s in acceptance_checks if s.strip()],
        )
        await ctx.repository.create_task(task)

        console.print("\n[green]Quickstart goal created.[/green]")
        console.print(f"  Task ID: {task.id}")
        console.print(f"  Project: {project.name}")
        console.print(f"  Agent: {recommended}")
        console.print(f"  Priority: {priority} ({valid_priorities[priority]})")

        if preview:
            _display_runbook_details(task=task, project_name=project.name, action_template=None)
            _display_planned_tasks([task], title="Quickstart Preview")
            console.print("\n[yellow]Preview mode: no execution yet.[/yellow]")

        if execute:
            if not ctx.agents:
                console.print("\n[red]No agents available to execute. Enable at least one agent in claw.toml.[/red]")
                return

            console.print("\n[cyan]Executing quickstart task...[/cyan]")
            micro = MicroClaw(ctx=ctx, project_id=project.id)
            start = _time.monotonic()
            task_ctx = await micro.evaluate(task)
            decision = await micro.decide(task_ctx)
            acted = await micro.act(decision)
            verified = await micro.verify(acted)
            await micro.learn(verified)
            duration = _time.monotonic() - start

            agent_id, _, outcome, verification = verified
            cycle_result = CycleResult(
                cycle_level="micro",
                task_id=task.id,
                project_id=project.id,
                agent_id=agent_id,
                outcome=outcome,
                verification=verification,
                success=verification.approved,
                tokens_used=outcome.tokens_used,
                cost_usd=outcome.cost_usd,
                duration_seconds=duration,
            )
            _display_task_result(cycle_result)
        else:
            console.print("\n[dim]Run `cam quickstart ... --execute` when you're ready to run it.[/dim]")

    finally:
        await ctx.close()


@app.command()
def create(
    repo: str = typer.Argument(..., help="Target repository path to fix, augment, or create"),
    request: str = typer.Option(..., "--request", "-r", prompt="What should CAM create?", help="Plain-language outcome request"),
    repo_mode: str = typer.Option("augment", "--repo-mode", help="Repo mode: fixed, augment, new"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Optional short task title"),
    priority: str = typer.Option("high", "--priority", "-p", help="Priority: critical, high, medium, low"),
    task_type: str = typer.Option("architecture", "--type", help="Task type for routing and execution"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Preferred agent override"),
    spec: list[str] = typer.Option([], "--spec", help="Initial requirement/spec line (repeatable)"),
    step: list[str] = typer.Option([], "--step", help="Suggested execution step (repeatable)"),
    check: list[str] = typer.Option([], "--check", help="Acceptance check / validation rule (repeatable)"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Preview runbook after creating the task"),
    execute: bool = typer.Option(False, "--execute", help="Immediately execute the created task"),
    max_minutes: int = typer.Option(20, "--max-minutes", help="Wall-clock time guardrail for creation/execution"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Create a fixed repo, augmented repo, or new repo from a requested outcome."""
    _setup_logging(False)

    repo_path = Path(repo).resolve()
    if repo_mode not in ("fixed", "augment", "new"):
        console.print("[red]--repo-mode must be one of: fixed, augment, new[/red]")
        raise typer.Exit(1)

    if repo_mode == "new":
        repo_path.mkdir(parents=True, exist_ok=True)
    elif not repo_path.exists():
        console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    if max_minutes < 1:
        console.print("[red]--max-minutes must be at least 1[/red]")
        raise typer.Exit(1)

    task_title = title or request.strip().split("\n")[0][:80]

    try:
        asyncio.run(asyncio.wait_for(
            _create_async(
                repo_path=repo_path,
                request=request,
                repo_mode=repo_mode,
                title=task_title,
                priority=priority.lower(),
                task_type=task_type,
                agent=agent,
                spec_items=spec,
                execution_steps=step,
                acceptance_checks=check,
                preview=preview,
                execute=execute,
                config_path=config,
            ),
            timeout=max_minutes * 60,
        ))
    except TimeoutError:
        console.print(f"[red]Create timed out after {max_minutes} minute(s)[/red]")
        raise typer.Exit(124)


async def _create_async(
    repo_path: Path,
    request: str,
    repo_mode: str,
    title: str,
    priority: str,
    task_type: str,
    agent: Optional[str],
    spec_items: list[str],
    execution_steps: list[str],
    acceptance_checks: list[str],
    preview: bool,
    execute: bool,
    config_path: Optional[str],
) -> None:
    spec_payload = _build_create_spec(
        repo_path=repo_path,
        request=request,
        repo_mode=repo_mode,
        title=title,
        task_type=task_type,
        execution_steps=[s.strip() for s in execution_steps if s.strip()],
        acceptance_checks=[c.strip() for c in acceptance_checks if c.strip()],
        spec_items=[s.strip() for s in spec_items if s.strip()],
    )
    spec_path = _write_create_spec(spec_payload)
    description = _build_create_description(
        request=request,
        repo_mode=repo_mode,
        spec_path=spec_path,
        spec_items=spec_payload["spec_items"],
    )

    console.print("\n[bold]CAM Create[/bold]")
    console.print(f"  Repo: {repo_path}")
    console.print(f"  Mode: {repo_mode}")
    console.print(f"  Spec file: {spec_path}")
    console.print("  Purpose: convert CAM memory + your request into an executable creation task")

    await _quickstart_async(
        repo_path=repo_path,
        title=title,
        description=description,
        priority=priority,
        task_type=task_type,
        agent=agent,
        execution_steps=spec_payload["execution_steps"],
        acceptance_checks=spec_payload["acceptance_checks"],
        preview=preview,
        execute=execute,
        config_path=config_path,
    )
    console.print("\n[dim]Next: run `cam validate --spec-file "
                  f"{spec_path}` then `cam benchmark`.[/dim]")


@app.command(name="add-goal", hidden=True)
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
    step: list[str] = typer.Option(
        [],
        "--step",
        help="Execution command to run for this goal (repeat --step for multiple commands)",
    ),
    check: list[str] = typer.Option(
        [],
        "--check",
        help="Acceptance check command for this goal (repeat --check for multiple commands)",
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
        repo_path,
        title,
        description,
        priority.lower(),
        task_type,
        agent,
        step,
        check,
        config,
    ))


async def _add_goal_async(
    repo_path: Path,
    title: str,
    description: str,
    priority: str,
    task_type: str,
    agent: Optional[str],
    execution_steps: list[str],
    acceptance_checks: list[str],
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
            execution_steps=[s.strip() for s in execution_steps if s.strip()],
            acceptance_checks=[s.strip() for s in acceptance_checks if s.strip()],
        )
        await ctx.repository.create_task(task)

        console.print(f"\n[green]Goal added successfully![/green]")
        console.print(f"  Title: {title}")
        console.print(f"  Project: {project.name}")
        console.print(f"  Priority: {priority} ({priority_map[priority]})")
        console.print(f"  Type: {task_type}")
        console.print(f"  Agent: {recommended}")
        if task.execution_steps:
            console.print(f"  Steps: {len(task.execution_steps)}")
        if task.acceptance_checks:
            console.print(f"  Checks: {len(task.acceptance_checks)}")
        console.print(f"  Task ID: {task.id}")
        console.print(f"\nRun [bold]cam enhance {repo_path}[/bold] to execute this goal.")

    finally:
        await ctx.close()


@app.command()
def ideate(
    directory: str = typer.Argument(..., help="Path to directory containing candidate repos"),
    focus: str = typer.Option("", "--focus", "-f", help="What kind of app should CAM invent?"),
    ideas: int = typer.Option(3, "--ideas", min=1, max=8, help="How many app concepts to generate"),
    max_repos: int = typer.Option(4, "--max-repos", help="Maximum repos to use as ideation inputs"),
    depth: int = typer.Option(3, "--depth", "-d", help="Max directory depth for repo discovery"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Preferred ideation agent: claude, codex, gemini, grok"),
    promote: int = typer.Option(0, "--promote", help="Promote idea N into a real cam create task/spec"),
    target_repo: Optional[str] = typer.Option(None, "--target-repo", help="Target repo path for promoted idea"),
    repo_mode: str = typer.Option("new", "--repo-mode", help="Repo mode for promoted idea: fixed, augment, new"),
    max_minutes: int = typer.Option(10, "--max-minutes", help="Wall-clock time guardrail for ideation"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Generate novel app concepts from CAM memory plus candidate repos."""
    _setup_logging(False)
    from claw.core.config import load_config

    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        console.print(f"[red]Directory does not exist: {dir_path}[/red]")
        raise typer.Exit(1)

    if agent and agent not in ("claude", "codex", "gemini", "grok"):
        console.print(f"[red]Invalid agent '{agent}'. Use: claude, codex, gemini, grok[/red]")
        raise typer.Exit(1)

    if repo_mode not in ("fixed", "augment", "new"):
        console.print("[red]--repo-mode must be one of: fixed, augment, new[/red]")
        raise typer.Exit(1)

    if promote < 0:
        console.print("[red]--promote must be 0 or a 1-based idea index[/red]")
        raise typer.Exit(1)

    if max_minutes < 1:
        console.print("[red]--max-minutes must be at least 1[/red]")
        raise typer.Exit(1)

    cfg = load_config(Path(config) if config else None)
    _fail_if_missing_api_keys(cfg, "ideate")

    try:
        asyncio.run(asyncio.wait_for(
            _ideate_async(
                dir_path=dir_path,
                focus=focus.strip(),
                idea_count=ideas,
                max_repos=max_repos,
                depth=depth,
                preferred_agent=agent,
                promote_index=promote,
                target_repo=Path(target_repo).resolve() if target_repo else None,
                repo_mode=repo_mode,
                config_path=config,
            ),
            timeout=max_minutes * 60,
        ))
    except TimeoutError:
        console.print(f"[red]Ideation timed out after {max_minutes} minute(s)[/red]")
        raise typer.Exit(124)


@app.command(name="keycheck", hidden=True)
def keycheck(
    for_command: str = typer.Option("mine", "--for", help="Command to preflight: mine, ideate"),
    live: bool = typer.Option(False, "--live", help="Also validate the keys with a tiny real provider call"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Quick API-key preflight before running a live command."""
    _setup_logging(False)
    normalized = for_command.strip().lower()
    if normalized not in {"mine", "ideate"}:
        console.print("[red]--for must be one of: mine, ideate[/red]")
        raise typer.Exit(1)

    from claw.core.config import load_config

    cfg = load_config(Path(config) if config else None)
    missing = _print_api_key_check(cfg, normalized)
    if missing:
        console.print("\n[yellow]Set the missing keys before starting live work.[/yellow]")
        for key_name in missing:
            console.print(f"  export {key_name}=your-key-here")
        raise typer.Exit(1)

    if not live:
        console.print("\n[green]Preflight passed.[/green]")
        return

    _fail_if_live_key_checks_fail(cfg, normalized)
    console.print("\n[green]Live preflight passed.[/green]")


async def _ideate_async(
    dir_path: Path,
    focus: str,
    idea_count: int,
    max_repos: int,
    depth: int,
    preferred_agent: Optional[str],
    promote_index: int,
    target_repo: Optional[Path],
    repo_mode: str,
    config_path: Optional[str],
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project, Task, TaskStatus
    from claw.dispatcher import DEFAULT_AGENT, STATIC_ROUTING
    from claw.llm.client import LLMMessage
    from claw.miner import _dedup_iterations, _discover_repos

    candidates = _discover_repos(dir_path, max_depth=depth)
    if not candidates:
        console.print("[yellow]No repositories or source trees found for ideation.[/yellow]")
        return

    candidates, _ = _dedup_iterations(candidates)
    selected = candidates[:max_repos]

    workspace_dir = target_repo if target_repo else ROOT_DIR
    config_p = Path(config_path) if config_path else None
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=workspace_dir)

    try:
        model = _select_ideation_model(ctx.config, preferred_agent)

        repo_contexts = [_summarize_repo_tree(candidate.path) for candidate in selected]
        repo_findings: dict[str, list[dict[str, Any]]] = {}
        for candidate in selected:
            existing = await ctx.repository.get_methodologies_by_tag(f"source:{candidate.name}", limit=6)
            repo_findings[candidate.name] = [_summarize_methodology(m) for m in existing[:6]]

        high_potential = await ctx.repository.get_high_potential_methodologies(limit=8, min_potential=0.35)
        most_novel = await ctx.repository.get_most_novel_methodologies(limit=8, min_novelty=0.35)
        action_templates = await ctx.repository.list_action_templates(limit=8)
        cam_memory = {
            "high_potential": [_summarize_methodology(m) for m in high_potential],
            "most_novel": [_summarize_methodology(m) for m in most_novel],
            "action_templates": [
                {
                    "title": t.title,
                    "pattern": t.problem_pattern[:220],
                    "source_repo": t.source_repo,
                    "confidence": t.confidence,
                }
                for t in action_templates
            ],
        }

        prompt = _build_ideation_prompt(
            focus=focus,
            repo_contexts=repo_contexts,
            repo_findings=repo_findings,
            cam_memory=cam_memory,
            idea_count=idea_count,
        )

        payload = await ctx.llm_client.complete_json(
            messages=[LLMMessage(role="user", content=prompt)],
            model=model,
            temperature=0.4,
        )
        normalized_ideas = _normalize_ideation_payload(payload, idea_count)
        if not normalized_ideas:
            console.print("[red]Ideation returned no usable ideas.[/red]")
            raise typer.Exit(1)

        json_path, md_path = _write_ideation_artifacts(
            source_dir=dir_path,
            focus=focus,
            ideas=normalized_ideas,
            raw_payload=payload,
        )

        console.print("\n[bold]CAM Ideation[/bold]")
        console.print(f"  Source directory: {dir_path}")
        console.print(f"  Repos used: {len(selected)}")
        console.print(f"  Model: {model}")
        console.print(f"  JSON: {json_path}")
        console.print(f"  Markdown: {md_path}")

        table = Table(title="Novel App Concepts")
        table.add_column("#", justify="right", width=3)
        table.add_column("Title", style="cyan", max_width=28)
        table.add_column("Tagline", style="green", max_width=34)
        table.add_column("Repos", style="magenta", max_width=24)
        table.add_column("Confidence", justify="right", style="yellow", width=10)
        for idx, idea in enumerate(normalized_ideas, start=1):
            table.add_row(
                str(idx),
                idea["title"],
                idea["tagline"] or idea["problem"][:60],
                ", ".join(idea["repos_used"][:3]),
                f"{idea['build_confidence']:.2f}",
            )
        console.print(table)

        if promote_index:
            if promote_index < 1 or promote_index > len(normalized_ideas):
                console.print(f"[red]--promote must be between 1 and {len(normalized_ideas)}[/red]")
                raise typer.Exit(1)
            if target_repo is None:
                console.print("[red]--target-repo is required when using --promote[/red]")
                raise typer.Exit(1)

            chosen = normalized_ideas[promote_index - 1]
            if repo_mode == "new":
                target_repo.mkdir(parents=True, exist_ok=True)
            elif not target_repo.exists():
                console.print(f"[red]Target repo does not exist: {target_repo}[/red]")
                raise typer.Exit(1)

            spec_payload = _build_create_spec(
                repo_path=target_repo,
                request=chosen["app_request"],
                repo_mode=repo_mode,
                title=chosen["title"],
                task_type="architecture",
                execution_steps=chosen["execution_steps"],
                acceptance_checks=chosen["acceptance_checks"],
                spec_items=chosen["spec_items"],
            )
            spec_path = _write_create_spec(spec_payload)
            description = _build_create_description(
                request=chosen["app_request"],
                repo_mode=repo_mode,
                spec_path=spec_path,
                spec_items=chosen["spec_items"],
            )

            project = await ctx.repository.get_project_by_name(target_repo.name)
            if project is None:
                project = Project(name=target_repo.name, repo_path=str(target_repo))
                await ctx.repository.create_project(project)

            recommended = STATIC_ROUTING.get("architecture", DEFAULT_AGENT)
            task = Task(
                project_id=project.id,
                title=chosen["title"][:200],
                description=description,
                status=TaskStatus.PENDING,
                priority=8,
                task_type="architecture",
                recommended_agent=recommended,
                execution_steps=chosen["execution_steps"],
                acceptance_checks=chosen["acceptance_checks"],
            )
            await ctx.repository.create_task(task)

            console.print("\n[green]Promoted idea into a create task.[/green]")
            console.print(f"  Idea: {chosen['title']}")
            console.print(f"  Target repo: {target_repo}")
            console.print(f"  Spec file: {spec_path}")
            console.print(f"  Task ID: {task.id}")
            console.print(f"\n[dim]Next: run `cam runbook {task.id}` or `cam enhance {target_repo}`.[/dim]")

    finally:
        await ctx.close()


@app.command()
def mine(
    directory: str = typer.Argument(..., help="Path to directory containing repos to mine"),
    target: str = typer.Option(".", "--target", "-t", help="Target project path (defaults to current directory)"),
    max_repos: int = typer.Option(10, "--max-repos", help="Maximum number of repos to mine"),
    min_relevance: float = typer.Option(0.6, "--min-relevance", help="Minimum relevance score for task generation (0.4-1.0)"),
    tasks: bool = typer.Option(True, "--tasks/--no-tasks", help="Generate enhancement tasks from findings"),
    depth: int = typer.Option(6, "--depth", "-d", help="Max directory depth for repo discovery"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Dedup repo iterations by canonical name"),
    skip_known: bool = typer.Option(True, "--skip-known/--no-skip-known", help="Skip repos already mined when unchanged"),
    force_rescan: bool = typer.Option(False, "--force-rescan", help="Ignore the mining ledger and rescan selected repos"),
    changed_only: bool = typer.Option(False, "--changed-only", help="Only show/mine repos that are new or changed according to the mining ledger"),
    scan_only: bool = typer.Option(False, "--scan-only", help="Preview discovered repos without mining (no LLM calls)"),
    live_keycheck: bool = typer.Option(True, "--live-keycheck/--no-live-keycheck", help="Validate required provider keys with tiny real calls before live mining"),
    max_minutes: int = typer.Option(15, "--max-minutes", help="Wall-clock time guardrail for mining"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Mine local repositories for patterns, features, and ideas.

    Scans a directory for git repos, analyzes each via LLM to extract
    transferable patterns, stores findings in semantic memory, and
    optionally generates enhancement tasks for the target project.

    Use --scan-only to preview what repos would be mined without making
    any LLM calls. Use --no-dedup to include all iterations of each project.
    """
    _setup_logging(verbose)

    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        console.print(f"[red]Directory does not exist: {dir_path}[/red]")
        raise typer.Exit(1)
    if not dir_path.is_dir():
        console.print(f"[red]Path is not a directory: {dir_path}[/red]")
        raise typer.Exit(1)

    if max_repos < 1:
        console.print("[red]--max-repos must be at least 1[/red]")
        raise typer.Exit(1)

    if not (0.4 <= min_relevance <= 1.0):
        console.print("[red]--min-relevance must be between 0.4 and 1.0[/red]")
        raise typer.Exit(1)

    if depth < 1:
        console.print("[red]--depth must be at least 1[/red]")
        raise typer.Exit(1)

    if max_minutes < 1:
        console.print("[red]--max-minutes must be at least 1[/red]")
        raise typer.Exit(1)

    if scan_only:
        _mine_scan_only(dir_path, depth, dedup, max_repos, config, skip_known, force_rescan, changed_only)
        return

    from claw.core.config import load_config

    cfg = load_config(Path(config) if config else None)
    _fail_if_missing_api_keys(cfg, "mine")
    if live_keycheck:
        _fail_if_live_key_checks_fail(cfg, "mine")

    try:
        asyncio.run(asyncio.wait_for(
            _mine_async(
                dir_path, target, max_repos, min_relevance, tasks, config,
                depth, dedup, skip_known, force_rescan, changed_only,
            ),
            timeout=max_minutes * 60,
        ))
    except TimeoutError:
        console.print(f"[red]Mining timed out after {max_minutes} minute(s)[/red]")
        raise typer.Exit(124)


@app.command(name="mine-report", hidden=True)
def mine_report(
    directory: str = typer.Argument(..., help="Path to directory containing repos to inspect"),
    depth: int = typer.Option(6, "--depth", "-d", help="Max directory depth for repo discovery"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Dedup repo iterations by canonical name"),
    changed_only: bool = typer.Option(False, "--changed-only", help="Only show repos that are new or changed"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Show repo mining status from the persistent mining ledger."""
    _setup_logging(False)

    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        console.print(f"[red]Directory does not exist: {dir_path}[/red]")
        raise typer.Exit(1)
    if not dir_path.is_dir():
        console.print(f"[red]Path is not a directory: {dir_path}[/red]")
        raise typer.Exit(1)

    from claw.core.config import load_config
    from claw.miner import RepoScanLedger, _default_scan_ledger_path, _discover_repos, _dedup_iterations

    cfg = load_config(Path(config) if config else None)
    ledger = RepoScanLedger(_default_scan_ledger_path(cfg))
    candidates = _discover_repos(dir_path, max_depth=depth)
    skipped: list = []
    selected = candidates
    if dedup:
        selected, skipped = _dedup_iterations(candidates)

    console.print(f"\n[bold]CAM Mine Report[/bold]")
    console.print(f"  Directory: {dir_path}")
    console.print(f"  Depth: {depth}")
    console.print(f"  Dedup: {dedup}")
    console.print(f"  Changed only: {changed_only}")
    console.print(f"  Ledger: {ledger.path}")

    table = Table(title=f"Mining Status ({len(selected)} selected)")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Name", style="cyan", max_width=30)
    table.add_column("Kind", style="magenta", width=11)
    table.add_column("Status", style="green", max_width=18)
    table.add_column("Last Mined", style="dim", width=18)
    table.add_column("Findings", justify="right", style="yellow", width=8)
    table.add_column("Tokens", justify="right", style="yellow", width=8)

    unchanged = 0
    changed = 0
    new = 0
    rows_added = 0
    for idx, candidate in enumerate(selected, start=1):
        should_mine, reason = ledger.should_mine(candidate, skip_known=True, force_rescan=False)
        record = ledger.get_record(candidate.path)
        if not should_mine:
            status = "unchanged"
            unchanged += 1
        elif reason == "changed":
            status = "changed"
            changed += 1
        else:
            status = "new"
            new += 1

        if changed_only and status == "unchanged":
            continue

        last_mined = "-"
        findings = "-"
        tokens = "-"
        if record is not None:
            from datetime import datetime
            last_mined = datetime.fromtimestamp(record.last_mined_at).strftime("%Y-%m-%d %H:%M")
            findings = str(record.findings_count)
            tokens = str(record.tokens_used)

        table.add_row(
            str(idx),
            candidate.name,
            candidate.source_kind,
            status,
            last_mined,
            findings,
            tokens,
        )
        rows_added += 1

    if rows_added:
        console.print(table)
    else:
        console.print("[yellow]No repos matched the requested report filters.[/yellow]")

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total discovered: {len(candidates)}")
    console.print(f"  Selected after dedup: {len(selected)}")
    console.print(f"  New: {new}")
    console.print(f"  Changed: {changed}")
    console.print(f"  Unchanged: {unchanged}")
    console.print(f"  Dedup skipped: {len(skipped)}")


def _mine_scan_only(
    dir_path: Path,
    depth: int,
    dedup: bool,
    max_repos: int,
    config_path: Optional[str],
    skip_known: bool,
    force_rescan: bool,
    changed_only: bool,
) -> None:
    """Preview discovered repos without mining (no LLM calls, no DB)."""
    from datetime import datetime
    from claw.core.config import load_config
    from claw.miner import RepoScanLedger, _default_scan_ledger_path, _discover_repos, _dedup_iterations

    console.print(f"\n[bold]CLAW Repo Scanner (scan-only)[/bold]")
    console.print(f"  Directory: {dir_path}")
    console.print(f"  Depth: {depth}")
    console.print(f"  Dedup: {dedup}")
    console.print(f"  Skip unchanged repos: {skip_known}")
    console.print(f"  Force rescan: {force_rescan}")
    console.print(f"  Changed only: {changed_only}")
    console.print()

    console.print("[cyan]Scanning for repos...[/cyan]")
    candidates = _discover_repos(dir_path, max_depth=depth)
    cfg = load_config(Path(config_path) if config_path else None)
    ledger = RepoScanLedger(_default_scan_ledger_path(cfg))

    if not candidates:
        console.print("[yellow]No repositories or source trees found.[/yellow]")
        return

    skipped: list = []
    selected = candidates
    if dedup:
        selected, skipped = _dedup_iterations(candidates)

    effective_candidates = selected
    if changed_only:
        effective_candidates = [
            c for c in selected
            if ledger.should_mine(c, skip_known=skip_known, force_rescan=force_rescan)[0]
        ]

    # Build discovery table
    table = Table(title=f"Discovered Repos ({len(candidates)} total, {len(effective_candidates)} eligible)")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Name", style="cyan", max_width=30)
    table.add_column("Canonical", style="blue", max_width=25)
    table.add_column("Files", justify="right", style="green", width=6)
    table.add_column("Size", justify="right", style="dim", width=8)
    table.add_column("Last Modified", style="dim", width=18)
    table.add_column("Depth", justify="right", style="dim", width=5)
    table.add_column("Kind", style="magenta", width=11)
    table.add_column("Status", max_width=20)

    skipped_names = {id(s[0]) for s in skipped}
    ledger_selected = 0

    for i, c in enumerate(candidates, 1):
        if c.total_bytes >= 1024 * 1024:
            size_str = f"{c.total_bytes / (1024 * 1024):.1f}MB"
        elif c.total_bytes >= 1024:
            size_str = f"{c.total_bytes / 1024:.0f}KB"
        else:
            size_str = f"{c.total_bytes}B"

        if c.last_commit_ts > 0:
            ts_str = datetime.fromtimestamp(c.last_commit_ts).strftime("%Y-%m-%d %H:%M")
        else:
            ts_str = "-"

        ledger_should_mine, ledger_reason = ledger.should_mine(
            c,
            skip_known=skip_known,
            force_rescan=force_rescan,
        )

        if id(c) in skipped_names:
            reason = next(r for s, r in skipped if id(s) == id(c))
            status = f"[dim]skipped: {reason[:18]}[/dim]"
        elif not ledger_should_mine:
            status = "[yellow]already mined[/yellow]"
        elif ledger_reason == "changed":
            ledger_selected += 1
            status = "[green]changed -> rescan[/green]"
        elif ledger_reason == "forced":
            ledger_selected += 1
            status = "[green]force rescan[/green]"
        else:
            ledger_selected += 1
            status = "[green]selected[/green]"

        if changed_only and (id(c) in skipped_names or not ledger_should_mine):
            continue

        table.add_row(
            str(i), c.name, c.canonical_name, str(c.file_count),
            size_str, ts_str, str(c.depth), c.source_kind, status,
        )

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total discovered: {len(candidates)}")
    console.print(f"  Eligible after filters: {len(effective_candidates)}")
    console.print(f"  Skipped (dedup): {len(skipped)}")
    if max_repos < ledger_selected:
        console.print(f"  Will mine (--max-repos): {max_repos}")
    else:
        console.print(f"  Will mine: {ledger_selected}")

    # Show dedup groups with multiple iterations
    if dedup and skipped:
        from collections import Counter
        canon_counts = Counter(c.canonical_name for c in candidates)
        multi = {name: count for name, count in canon_counts.items() if count > 1}
        if multi:
            console.print(f"\n[bold]Iteration Groups ({len(multi)} with duplicates)[/bold]")
            group_table = Table()
            group_table.add_column("Canonical Name", style="blue", max_width=30)
            group_table.add_column("Iterations", justify="right", style="yellow", width=10)
            group_table.add_column("Selected", style="green", max_width=35)
            for name, count in sorted(multi.items(), key=lambda x: -x[1])[:20]:
                winner = next((c.name for c in selected if c.canonical_name == name), "?")
                group_table.add_row(name, str(count), winner)
            console.print(group_table)

    console.print(f"\n[dim]Remove --scan-only to mine these repos.[/dim]")


async def _mine_async(
    dir_path: Path,
    target: str,
    max_repos: int,
    min_relevance: float,
    generate_tasks: bool,
    config_path: Optional[str],
    max_depth: int = 6,
    dedup_iterations: bool = True,
    skip_known: bool = True,
    force_rescan: bool = False,
    changed_only: bool = False,
) -> None:
    from claw.core.factory import ClawFactory
    from claw.core.models import Project

    config_p = Path(config_path) if config_path else None
    target_path = Path(target).resolve()
    ctx = await ClawFactory.create(config_path=config_p, workspace_dir=target_path)

    try:
        # Get or create target project
        project_name = target_path.name
        project = await ctx.repository.get_project_by_name(project_name)
        if project is None:
            project = Project(name=project_name, repo_path=str(target_path))
            project = await ctx.repository.create_project(project)

        console.print(f"\n[bold]CLAW Repo Mining[/bold]")
        console.print(f"  Directory: {dir_path}")
        console.print(f"  Target: {project.name} ({target_path})")
        console.print(f"  Max repos: {max_repos}")
        console.print(f"  Min relevance for tasks: {min_relevance}")
        console.print(f"  Generate tasks: {generate_tasks}")
        console.print(f"  Depth: {max_depth}")
        console.print(f"  Dedup: {dedup_iterations}")
        console.print(f"  Skip unchanged repos: {skip_known}")
        console.print(f"  Force rescan: {force_rescan}")
        console.print(f"  Changed only: {changed_only}")
        console.print(f"  Database: {ctx.config.database.db_path}")
        console.print()

        # Progress callback
        def on_repo_complete(repo_name: str, result: Any) -> None:
            n_findings = len(result.findings) if result.findings else 0
            if result.error:
                console.print(f"  [red]x {repo_name}: {result.error}[/red]")
            elif result.skipped:
                console.print(f"  [yellow]- {repo_name}: skipped ({result.skip_reason})[/yellow]")
            else:
                console.print(
                    f"  [green]+ {repo_name}[/green]: "
                    f"{n_findings} findings, {result.files_analyzed} files, "
                    f"{result.tokens_used} tokens, {result.duration_seconds:.1f}s"
                )

        console.print("[cyan]Mining repositories...[/cyan]")
        report = await ctx.miner.mine_directory(
            base_path=dir_path,
            target_project_id=project.id,
            max_repos=max_repos,
            min_relevance=min_relevance,
            generate_tasks=generate_tasks,
            on_repo_complete=on_repo_complete,
            max_depth=max_depth,
            dedup_iterations=dedup_iterations,
            skip_known=skip_known or changed_only,
            force_rescan=force_rescan,
        )

        # Display results table
        console.print()
        results_table = Table(title="Mining Results")
        results_table.add_column("Repo", style="cyan", max_width=25)
        results_table.add_column("Files", justify="right", style="dim", width=6)
        results_table.add_column("Findings", justify="right", style="green", width=9)
        results_table.add_column("Tokens", justify="right", style="yellow", width=8)
        results_table.add_column("Time", justify="right", style="dim", width=8)
        results_table.add_column("Status", max_width=20)

        for result in report.repo_results:
            if result.skipped:
                status = f"[yellow]skipped: {result.skip_reason}[/yellow]"
            else:
                status = "[green]OK[/green]" if not result.error else f"[red]{result.error[:18]}[/red]"
            results_table.add_row(
                result.repo_name,
                str(result.files_analyzed),
                str(len(result.findings)),
                str(result.tokens_used),
                f"{result.duration_seconds:.1f}s",
                status,
            )

        console.print(results_table)

        # Summary
        console.print(f"\n[bold]Summary[/bold]")
        console.print(f"  Repos scanned: {report.repos_scanned}")
        console.print(f"  Repos skipped: {report.repos_skipped}")
        console.print(f"  Total findings: {report.total_findings}")
        console.print(f"  Tasks generated: {report.tasks_generated}")
        console.print(f"  Total tokens: {report.total_tokens}")
        console.print(f"  Total time: {report.total_duration_seconds:.1f}s")

        if report.tasks:
            console.print(f"\n[bold]Generated Tasks[/bold]")
            task_table = Table()
            task_table.add_column("Title", style="cyan", max_width=60)
            task_table.add_column("Priority", justify="right", style="yellow", width=8)
            task_table.add_column("Type", style="dim", width=16)
            task_table.add_column("Agent", style="green", width=8)

            for task in report.tasks:
                task_table.add_row(
                    task.title[:58],
                    str(task.priority),
                    task.task_type or "-",
                    task.recommended_agent or "-",
                )

            console.print(task_table)

        console.print(f"\n[dim]Use 'claw results' to view tasks, 'claw enhance .' to work on them.[/dim]")

    finally:
        await ctx.close()


@app.command(name="forge-export", hidden=True)
def forge_export(
    out: str = typer.Option("data/cam_knowledge_pack.jsonl", "--out", help="Output JSONL knowledge pack path"),
    db: Optional[str] = typer.Option(None, "--db", help="Override CAM database path"),
    max_methodologies: int = typer.Option(300, "--max-methodologies", help="Maximum methodologies to export"),
    max_tasks: int = typer.Option(300, "--max-tasks", help="Maximum tasks to export"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for the export"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Export CAM memory into a standalone Forge knowledge pack."""
    from claw.core.config import load_config

    _setup_logging(verbose)
    cfg = load_config(Path(config) if config else None)
    db_path = db or cfg.database.db_path
    script_path = ROOT_DIR / "scripts" / "export_cam_knowledge_pack.py"

    console.print("\n[bold]CAM Forge Export[/bold]")
    console.print(f"  Database: {db_path}")
    console.print(f"  Out: {out}")
    console.print(f"  Max methodologies: {max_methodologies}")
    console.print(f"  Max tasks: {max_tasks}")
    console.print(f"  Time guardrail: {max_minutes} minute(s)")

    result = _run_python_script_with_timeout(
        script_path=script_path,
        args=[
            "--db", db_path,
            "--out", out,
            "--max-methodologies", str(max_methodologies),
            "--max-tasks", str(max_tasks),
        ],
        max_minutes=max_minutes,
    )

    if result.returncode != 0:
        console.print(f"[red]Export failed with exit code {result.returncode}[/red]")
        if result.stderr.strip():
            console.print(result.stderr.strip())
        raise typer.Exit(result.returncode)

    payload = json.loads(result.stdout)
    console.print("\n[green]Knowledge pack exported.[/green]")
    console.print(f"  Total: {payload['total']}")
    console.print(f"  Methodologies: {payload['methodologies']}")
    console.print(f"  Tasks: {payload['tasks']}")
    console.print(f"  File: {payload['out']}")


@app.command(name="forge-benchmark", hidden=True)
def forge_benchmark(
    repo: str = typer.Option("tests/fixtures/embedding_forge/repo", "--repo", help="Fixture or target repo path"),
    note: str = typer.Option("tests/fixtures/embedding_forge/note.md", "--note", help="Note path"),
    knowledge_pack: str = typer.Option(
        "tests/fixtures/embedding_forge/knowledge_pack.jsonl",
        "--knowledge-pack",
        help="Knowledge pack JSONL path",
    ),
    out: str = typer.Option("data/forge_benchmark_fixture", "--out", help="Output benchmark directory"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for the benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the standalone Forge regression benchmark with a wall-clock limit."""
    _setup_logging(verbose)
    script_path = ROOT_DIR / "apps" / "embedding_forge" / "benchmark_regression.py"

    console.print("\n[bold]CAM Forge Benchmark[/bold]")
    console.print(f"  Repo: {repo}")
    console.print(f"  Note: {note}")
    console.print(f"  Knowledge pack: {knowledge_pack}")
    console.print(f"  Out: {out}")
    console.print(f"  Time guardrail: {max_minutes} minute(s)")

    result = _run_python_script_with_timeout(
        script_path=script_path,
        args=[
            "--repo", repo,
            "--note", note,
            "--knowledge-pack", knowledge_pack,
            "--out", out,
        ],
        max_minutes=max_minutes,
    )

    if result.returncode != 0:
        console.print(f"[red]Benchmark failed with exit code {result.returncode}[/red]")
        if result.stderr.strip():
            console.print(result.stderr.strip())
        raise typer.Exit(result.returncode)

    payload = json.loads(result.stdout)
    best = payload["best"]
    console.print("\n[green]Benchmark complete.[/green]")
    console.print(f"  Status: {payload['status']}")
    console.print(f"  Docs: {payload['docs_total']}")
    console.print(f"  Best lift: {best['hit_rate_lift_pct']:.2f}%")
    console.print(
        "  Best config: "
        f"anchor_dim={best['anchor_dim']} residual_dim={best['residual_dim']} "
        f"anchor_weight={best['anchor_weight']} residual_weight={best['residual_weight']}"
    )
    console.print(f"  Summary: {Path(out) / 'benchmark_summary.json'}")


@app.command()
def validate(
    spec_file: str = typer.Option(..., "--spec-file", help="Creation spec JSON to validate against"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for validation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Validate a created repo against its saved spec and acceptance checks."""
    _setup_logging(verbose)

    if max_minutes < 1:
        console.print("[red]--max-minutes must be at least 1[/red]")
        raise typer.Exit(1)

    spec_path = Path(spec_file).resolve()
    if not spec_path.exists():
        console.print(f"[red]Spec file does not exist: {spec_path}[/red]")
        raise typer.Exit(1)

    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    passed, summary = _validate_create_spec(spec_payload, max_minutes=max_minutes)

    console.print("\n[bold]CAM Validate[/bold]")
    console.print(f"  Spec file: {spec_path}")
    console.print(f"  Repo: {summary['repo']}")
    console.print(f"  Checks run: {summary['checks_run']}")
    if summary["manual_checks"]:
        console.print(f"  Manual checks: {len(summary['manual_checks'])}")

    if passed:
        console.print("\n[green]Validation passed.[/green]")
    else:
        console.print("\n[red]Validation failed.[/red]")
        for finding in summary["findings"]:
            console.print(f"  - {finding}")

    for check in summary["checks"]:
        status = "[green]OK[/green]" if check["ok"] else "[red]FAIL[/red]"
        console.print(f"  {status} {check['command']}")

    for check in summary["manual_checks"]:
        console.print(f"  [yellow]MANUAL[/yellow] {check}")

    if not passed:
        raise typer.Exit(2)


@app.command()
def benchmark(
    repo: str = typer.Option("tests/fixtures/embedding_forge/repo", "--repo", help="Fixture or target repo path"),
    note: str = typer.Option("tests/fixtures/embedding_forge/note.md", "--note", help="Note path"),
    knowledge_pack: str = typer.Option(
        "tests/fixtures/embedding_forge/knowledge_pack.jsonl",
        "--knowledge-pack",
        help="Knowledge pack JSONL path",
    ),
    out: str = typer.Option("data/forge_benchmark_fixture", "--out", help="Output benchmark directory"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for the benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Benchmark Forge output."""
    _setup_logging(verbose)
    script_path = ROOT_DIR / "apps" / "embedding_forge" / "benchmark_regression.py"

    console.print("\n[bold]CAM Benchmark[/bold]")
    console.print(f"  Repo: {repo}")
    console.print(f"  Note: {note}")
    console.print(f"  Knowledge pack: {knowledge_pack}")
    console.print(f"  Out: {out}")
    console.print(f"  Time guardrail: {max_minutes} minute(s)")

    result = _run_python_script_with_timeout(
        script_path=script_path,
        args=[
            "--repo", repo,
            "--note", note,
            "--knowledge-pack", knowledge_pack,
            "--out", out,
        ],
        max_minutes=max_minutes,
    )

    if result.returncode != 0:
        console.print(f"[red]Benchmark failed with exit code {result.returncode}[/red]")
        if result.stderr.strip():
            console.print(result.stderr.strip())
        raise typer.Exit(result.returncode)

    payload = json.loads(result.stdout)
    best = payload["best"]
    console.print("\n[green]Benchmark complete.[/green]")
    console.print(f"  Status: {payload['status']}")
    console.print(f"  Best lift: {best['hit_rate_lift_pct']:.2f}%")
    console.print(f"  Summary: {Path(out) / 'benchmark_summary.json'}")


@app.command(hidden=True)
def govern(
    action: str = typer.Argument(
        "stats",
        help="Action: stats, sweep, gc, quota, prune",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to claw.toml"),
) -> None:
    """Memory governance — sweep, stats, GC, quota enforcement, episode pruning.

    Actions:
      stats  — Show methodology counts by state, DB size, quota usage
      sweep  — Run a full governance sweep (lifecycle + GC + quota + prune)
      gc     — Garbage-collect dead methodologies only
      quota  — Enforce methodology quota only
      prune  — Prune old episodes only
    """
    _setup_logging(verbose)
    asyncio.run(_govern_async(action, config))


async def _govern_async(action: str, config_path: Optional[str]) -> None:
    """Run governance action."""
    from claw.core.config import load_config
    from claw.db.engine import DatabaseEngine
    from claw.db.repository import Repository
    from claw.memory.governance import MemoryGovernor

    cfg = load_config(Path(config_path) if config_path else None)

    engine = DatabaseEngine(cfg.database)
    await engine.connect()
    await engine.apply_migrations()
    await engine.initialize_schema()
    repository = Repository(engine)

    governor = MemoryGovernor(repository=repository, config=cfg.governance)

    try:
        if action == "stats":
            stats = await governor.get_storage_stats()
            active_methodologies = sum(v for k, v in stats.by_state.items() if k != "dead")
            console.print("\n[bold]Memory Governance Stats[/bold]")
            console.print(f"  Total methodologies:  {stats.total_methodologies}")
            console.print(f"  Active (non-dead):    {active_methodologies}")

            table = Table(title="Methodologies by State")
            table.add_column("State", style="bold")
            table.add_column("Count", justify="right")
            for state, count in sorted(stats.by_state.items()):
                style = {
                    "thriving": "green",
                    "viable": "cyan",
                    "embryonic": "yellow",
                    "declining": "magenta",
                    "dormant": "dim",
                    "dead": "red",
                }.get(state, "")
                table.add_row(f"[{style}]{state}[/{style}]" if style else state, str(count))
            console.print(table)

            quota = cfg.governance.max_methodologies
            usage_pct = (active_methodologies / quota * 100) if quota else 0
            bar_style = "green" if usage_pct < 80 else ("yellow" if usage_pct < 100 else "red")
            console.print(f"  Quota: {active_methodologies}/{quota} ({usage_pct:.1f}%) [{bar_style}]")
            console.print(f"  DB size: {stats.db_size_bytes / 1024 / 1024:.2f} MB")
            console.print(f"  Episodes: {stats.total_episodes}")

        elif action == "sweep":
            console.print("[bold]Running full governance sweep...[/bold]")
            report = await governor.run_full_sweep()
            console.print(f"  Dead collected:    {report.dead_collected}")
            console.print(f"  Quota culled:      {report.quota_culled}")
            console.print(f"  Episodes pruned:   {report.episodes_pruned}")
            console.print(f"  Lifecycle swept:   {report.lifecycle_swept}")
            console.print("[green]Sweep complete.[/green]")

        elif action == "gc":
            console.print("[bold]Garbage-collecting dead methodologies...[/bold]")
            count = await governor.garbage_collect_dead()
            console.print(f"  Removed: {count} dead methodologies")

        elif action == "quota":
            console.print("[bold]Enforcing methodology quota...[/bold]")
            count = await governor.enforce_methodology_quota()
            console.print(f"  Culled: {count} methodologies to stay within quota")

        elif action == "prune":
            console.print("[bold]Pruning old episodes...[/bold]")
            count = await governor._prune_episodes()
            console.print(f"  Pruned: {count} episodes older than {cfg.governance.episodic_retention_days} days")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[dim]Valid actions: stats, sweep, gc, quota, prune[/dim]")

    finally:
        await engine.close()


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
    console.print(f"  claw forge-export        — export CAM memory for standalone Forge")
    console.print(f"  claw forge-benchmark     — benchmark Forge with time guardrails")


@app.command(hidden=True)
def synergies(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed edge list"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to claw.toml"),
):
    """Show capability synergy graph summary, exploration stats, and recent discoveries."""
    _setup_logging(verbose)
    asyncio.run(_synergies_async(verbose))


@app.command(name="assimilation-report", hidden=True)
def assimilation_report(
    limit: int = typer.Option(10, "--limit", "-n", help="Rows to show per section"),
    future_threshold: float = typer.Option(0.65, "--future-threshold", help="Potential score threshold for future-candidate flag"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Show the learning continuum from stored methodologies to proven usefulness."""
    _setup_logging(False)
    asyncio.run(_assimilation_report_async(limit, future_threshold))


@app.command(name="assimilation-delta", hidden=True)
def assimilation_delta(
    directory: Optional[str] = typer.Argument(None, help="Optional repo directory to scope the report"),
    depth: int = typer.Option(6, "--depth", "-d", help="Max directory depth when scoping by directory"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Dedup repo iterations by canonical name"),
    since_hours: float = typer.Option(24.0, "--since-hours", help="Only include repos mined within this many hours"),
    latest: int = typer.Option(10, "--latest", "-n", help="Maximum recently mined repos/methodologies to summarize"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Show what recent mine runs actually added: methodologies, templates, capabilities, and next uses."""
    _setup_logging(False)
    asyncio.run(_assimilation_delta_async(directory, depth, dedup, since_hours, latest, config))


@app.command(hidden=True)
def reassess(
    repo: Optional[str] = typer.Argument(None, help="Optional repository path for additional context"),
    task: str = typer.Option(..., "--task", "-t", help="Task or goal CAM should reassess prior knowledge against"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum recommendations to show"),
    min_score: float = typer.Option(0.2, "--min-score", help="Minimum reassessment score to show"),
    future_threshold: float = typer.Option(0.65, "--future-threshold", help="Potential score threshold for future-candidate flag"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Re-score prior methodologies against a new task and explain why they matter now."""
    _setup_logging(False)
    asyncio.run(_reassess_async(repo, task, limit, min_score, future_threshold))


async def _assimilation_delta_async(
    directory: Optional[str],
    depth: int,
    dedup: bool,
    since_hours: float,
    latest: int,
    config: Optional[str],
) -> None:
    from datetime import UTC, datetime
    from rich.panel import Panel
    from claw.core.config import load_config
    from claw.miner import RepoScanLedger, _default_scan_ledger_path, _discover_repos, _dedup_iterations

    cfg = load_config(Path(config) if config else None)
    ledger = RepoScanLedger(_default_scan_ledger_path(cfg))

    scoped_repo_keys: Optional[set[str]] = None
    if directory:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            console.print(f"[red]Directory does not exist: {dir_path}[/red]")
            raise typer.Exit(1)
        if not dir_path.is_dir():
            console.print(f"[red]Path is not a directory: {dir_path}[/red]")
            raise typer.Exit(1)
        candidates = _discover_repos(dir_path, max_depth=depth)
        if dedup:
            candidates, _ = _dedup_iterations(candidates)
        scoped_repo_keys = {ledger.repo_key(candidate.path) for candidate in candidates}

    cutoff_ts = _time.time() - (max(since_hours, 0.0) * 3600.0)
    records = ledger.list_records()
    if scoped_repo_keys is not None:
        records = [record for record in records if record.repo_path in scoped_repo_keys]
    records = [record for record in records if record.last_mined_at >= cutoff_ts]
    records.sort(key=lambda record: record.last_mined_at, reverse=True)
    if latest > 0:
        records = records[:latest]

    if not records:
        console.print("[yellow]No mined repos matched this delta window. Try a larger --since-hours or run cam mine first.[/yellow]")
        return

    engine, repository = await _kb_engine()

    try:
        repo_summaries: list[dict[str, Any]] = []
        all_methodologies: list[Any] = []
        methodology_ids_with_templates: set[str] = set()

        for record in records:
            methodologies: list[Any] = []
            seen_methodology_ids: set[str] = set()
            for methodology_id in record.methodology_ids:
                meth = await repository.get_methodology(methodology_id)
                if meth is not None and meth.id not in seen_methodology_ids:
                    methodologies.append(meth)
                    seen_methodology_ids.add(meth.id)

            if not methodologies:
                fallback_methodologies = await repository.get_methodologies_by_tag(
                    f"source:{record.repo_name}",
                    limit=max(20, record.findings_count * 3 or 20),
                )
                methodologies = [
                    meth for meth in fallback_methodologies
                    if _recently_created_near_mine(meth.created_at, record.last_mined_at)
                ]
                seen_methodology_ids = {meth.id for meth in methodologies}

            action_templates: list[Any] = []
            for template_id in record.action_template_ids:
                template = await repository.get_action_template(template_id)
                if template is not None:
                    action_templates.append(template)
                    if template.source_methodology_id:
                        methodology_ids_with_templates.add(template.source_methodology_id)

            if not action_templates:
                fallback_templates = await repository.list_action_templates(
                    source_repo=record.repo_name,
                    limit=max(10, len(methodologies) * 2 or 10),
                )
                action_templates = [
                    template for template in fallback_templates
                    if _recently_created_near_mine(template.created_at, record.last_mined_at)
                ]
                for template in action_templates:
                    if template.source_methodology_id:
                        methodology_ids_with_templates.add(template.source_methodology_id)

            future_candidates = [
                meth for meth in methodologies
                if _is_future_candidate(
                    meth,
                    potential_threshold=0.65,
                    template_count=1 if meth.id in methodology_ids_with_templates else 0,
                )
            ]

            all_methodologies.extend(methodologies)
            repo_summaries.append({
                "record": record,
                "methodologies": methodologies,
                "templates": action_templates,
                "future_candidates": future_candidates,
            })

        if not all_methodologies:
            console.print("[yellow]Mine records exist, but no stored methodologies could be resolved from them yet.[/yellow]")
            return

        capability_summary = _summarize_new_capabilities(all_methodologies)
        opportunities = _infer_feature_opportunities(
            all_methodologies,
            methodology_ids_with_templates=methodology_ids_with_templates,
            limit=max(3, min(6, latest if latest > 0 else 6)),
        )

        console.print(Panel.fit(
            f"[bold cyan]CAM Assimilation Delta[/bold cyan]\n"
            f"[bold]{len(repo_summaries)}[/bold] mined repo(s) in the last [bold]{since_hours:g}[/bold] hour(s)\n"
            f"[bold]{len(all_methodologies)}[/bold] methodology record(s) resolved from those mine runs\n"
            f"[bold]{sum(len(item['templates']) for item in repo_summaries)}[/bold] action template(s) created",
            border_style="cyan",
        ))

        summary = Table(title="Recently Mined Repos")
        summary.add_column("Repo", style="cyan", max_width=28)
        summary.add_column("Mined At", style="dim", width=18)
        summary.add_column("Meth", justify="right", width=6)
        summary.add_column("Tpl", justify="right", width=5)
        summary.add_column("Future", justify="right", width=7)
        summary.add_column("Top Domains", max_width=28)
        for item in repo_summaries:
            record = item["record"]
            methodologies = item["methodologies"]
            domains: dict[str, int] = {}
            for meth in methodologies:
                capability_data = getattr(meth, "capability_data", None) or {}
                if isinstance(capability_data, dict):
                    for domain in capability_data.get("domain", []) or []:
                        domains[str(domain)] = domains.get(str(domain), 0) + 1
            domain_str = ", ".join(name for name, _count in sorted(domains.items(), key=lambda x: (-x[1], x[0]))[:3]) or "-"
            summary.add_row(
                record.repo_name,
                datetime.fromtimestamp(record.last_mined_at, UTC).strftime("%Y-%m-%d %H:%M"),
                str(len(methodologies)),
                str(len(item["templates"])),
                str(len(item["future_candidates"])),
                domain_str,
            )
        console.print(summary)

        if capability_summary["domains"] or capability_summary["capability_types"]:
            cap_table = Table(title="New Capabilities Surfaced")
            cap_table.add_column("Kind", style="bold", width=16)
            cap_table.add_column("Top Items", max_width=76)
            if capability_summary["domains"]:
                cap_table.add_row(
                    "Domains",
                    ", ".join(f"{name} ({count})" for name, count in capability_summary["domains"][:8]),
                )
            if capability_summary["capability_types"]:
                cap_table.add_row(
                    "Capability types",
                    ", ".join(f"{name} ({count})" for name, count in capability_summary["capability_types"][:8]),
                )
            console.print(cap_table)

        if opportunities:
            opp_table = Table(title="Possible New Features / Updates")
            opp_table.add_column("Signal", style="cyan", width=16)
            opp_table.add_column("Weight", justify="right", width=6)
            opp_table.add_column("What CAM could operationalize next", max_width=72)
            for opp in opportunities:
                opp_table.add_row(opp["trigger"], str(opp["count"]), opp["description"])
            console.print(opp_table)

        top_methodologies = sorted(
            all_methodologies,
            key=lambda meth: (
                1 if meth.id in methodology_ids_with_templates else 0,
                getattr(meth, "potential_score", None) or 0,
                getattr(meth, "novelty_score", None) or 0,
                getattr(meth, "created_at", datetime.min.replace(tzinfo=UTC)),
            ),
            reverse=True,
        )

        top_limit = latest if latest > 0 else 10
        top_table = Table(title=f"New Methodologies / Operationalization Candidates ({min(len(top_methodologies), top_limit)})")
        top_table.add_column("ID", width=8)
        top_table.add_column("Repo", style="cyan", max_width=20)
        top_table.add_column("Description", max_width=40)
        top_table.add_column("Stage", width=17)
        top_table.add_column("Potential", justify="right", width=9)
        top_table.add_column("Novelty", justify="right", width=8)
        top_table.add_column("Triggers", max_width=26)
        for meth in top_methodologies[:top_limit]:
            template_count = 1 if meth.id in methodology_ids_with_templates else 0
            stage = _classify_assimilation_stage(meth, template_count=template_count)
            source_repo = next(
                (tag.split(":", 1)[1] for tag in getattr(meth, "tags", []) or [] if isinstance(tag, str) and tag.startswith("source:")),
                "-",
            )
            triggers = ", ".join(_derive_activation_triggers(meth, template_count=template_count)[:3]) or "-"
            top_table.add_row(
                meth.id[:8],
                source_repo,
                meth.problem_description[:40],
                stage,
                f"{(meth.potential_score or 0):.3f}",
                f"{(meth.novelty_score or 0):.3f}" if meth.novelty_score is not None else "-",
                triggers,
            )
        console.print(top_table)

        console.print(
            "\n[dim]Interpretation: this report answers 'what did the recent mine runs actually add?' "
            "Use 'cam assimilation-report' for lifecycle maturity and 'cam kb synergies' for cross-capability relationships.[/dim]"
        )

    finally:
        await engine.close()


async def _assimilation_report_async(limit: int, future_threshold: float) -> None:
    from rich.panel import Panel

    engine, repository = await _kb_engine()

    try:
        methods = await repository.list_methodologies(limit=5000, include_dead=False)
        if not methods:
            console.print("[yellow]No methodologies in knowledge base. Run 'cam mine <dir>' first.[/yellow]")
            return

        template_rows = await repository.engine.fetch_all(
            """SELECT source_methodology_id,
                      COUNT(*) as template_count,
                      COALESCE(SUM(success_count), 0) as template_successes,
                      COALESCE(SUM(failure_count), 0) as template_failures,
                      MAX(confidence) as max_confidence
               FROM action_templates
               WHERE source_methodology_id IS NOT NULL
               GROUP BY source_methodology_id"""
        )
        template_stats = {
            row["source_methodology_id"]: {
                "count": int(row["template_count"] or 0),
                "successes": int(row["template_successes"] or 0),
                "failures": int(row["template_failures"] or 0),
                "max_confidence": float(row["max_confidence"] or 0.0),
            }
            for row in template_rows
            if row.get("source_methodology_id")
        }

        stage_counts = {
            "stored": 0,
            "enriched": 0,
            "retrieved": 0,
            "operationalized": 0,
            "proven": 0,
        }
        future_candidates: list[Any] = []
        proven_items: list[Any] = []
        stored_items: list[Any] = []
        enriched_items: list[Any] = []
        operationalized_items: list[Any] = []

        for meth in methods:
            stats = template_stats.get(meth.id, {})
            template_count = int(stats.get("count", 0))
            template_successes = int(stats.get("successes", 0))
            stage = _classify_assimilation_stage(
                meth,
                template_count=template_count,
                template_successes=template_successes,
            )
            stage_counts[stage] += 1

            if _is_future_candidate(
                meth,
                potential_threshold=future_threshold,
                template_count=template_count,
            ):
                future_candidates.append((meth, template_count, template_successes))

            if stage == "proven":
                proven_items.append((meth, template_count, template_successes))
            elif stage == "stored":
                stored_items.append((meth, template_count, template_successes))
            elif stage == "enriched":
                enriched_items.append((meth, template_count, template_successes))
            elif stage == "operationalized":
                operationalized_items.append((meth, template_count, template_successes))

        future_candidates.sort(key=lambda x: ((x[0].potential_score or 0), (x[0].novelty_score or 0)), reverse=True)
        proven_items.sort(key=lambda x: (x[0].success_count + x[2], x[0].retrieval_count), reverse=True)
        stored_items.sort(key=lambda x: x[0].created_at, reverse=True)
        enriched_items.sort(key=lambda x: ((x[0].potential_score or 0), (x[0].novelty_score or 0)), reverse=True)
        operationalized_items.sort(key=lambda x: (x[1], x[0].retrieval_count), reverse=True)

        console.print(Panel.fit(
            f"[bold cyan]CAM Assimilation Continuum[/bold cyan]\n"
            f"[bold]{len(methods):,}[/bold] active methodologies tracked across the learning continuum",
            border_style="cyan",
        ))

        summary = Table(title="Continuum Stages")
        summary.add_column("Stage", style="bold", width=18)
        summary.add_column("Count", justify="right", width=8)
        summary.add_column("Meaning", max_width=52)
        summary.add_row("stored", str(stage_counts["stored"]), "Filed in memory, not yet enriched or retrieved")
        summary.add_row("enriched", str(stage_counts["enriched"]), "Structured metadata exists, but not yet in active use")
        summary.add_row("retrieved", str(stage_counts["retrieved"]), "CAM is pulling it back during later work")
        summary.add_row("operationalized", str(stage_counts["operationalized"]), "Turned into executable action template(s)")
        summary.add_row("proven", str(stage_counts["proven"]), "Has actual success signal from use")
        console.print(summary)

        console.print(
            f"\n[bold]Future candidates:[/bold] {len(future_candidates)} "
            f"[dim](potential >= {future_threshold:.2f}, no direct success yet)[/dim]"
        )

        if future_candidates:
            future_table = Table(title=f"Top Future Candidates ({min(limit, len(future_candidates))})")
            future_table.add_column("ID", width=8)
            future_table.add_column("Description", max_width=44)
            future_table.add_column("Potential", justify="right", width=9, style="bold cyan")
            future_table.add_column("Novelty", justify="right", width=8, style="yellow")
            future_table.add_column("Domains", max_width=24)
            for meth, template_count, _ in future_candidates[:limit]:
                domains = ", ".join(((meth.capability_data or {}).get("domain", [])[:3]))
                future_table.add_row(
                    meth.id[:8],
                    meth.problem_description[:44],
                    f"{(meth.potential_score or 0):.3f}",
                    f"{(meth.novelty_score or 0):.3f}" if meth.novelty_score is not None else "-",
                    domains or ("templates:" + str(template_count) if template_count else "-"),
                )
            console.print(future_table)

        def _print_stage_table(title: str, items: list[Any], *, score_label: str = "") -> None:
            if not items:
                return
            table = Table(title=title)
            table.add_column("ID", width=8)
            table.add_column("Description", max_width=44)
            table.add_column("Retr", justify="right", width=6)
            table.add_column("Succ", justify="right", width=6)
            table.add_column("Tpl", justify="right", width=5)
            if score_label:
                table.add_column(score_label, justify="right", width=9)
            for meth, template_count, template_successes in items[:limit]:
                row = [
                    meth.id[:8],
                    meth.problem_description[:44],
                    str(meth.retrieval_count),
                    str(meth.success_count + template_successes),
                    str(template_count),
                ]
                if score_label == "Potential":
                    row.append(f"{(meth.potential_score or 0):.3f}")
                elif score_label == "Novelty":
                    row.append(f"{(meth.novelty_score or 0):.3f}")
                table.add_row(*row)
            console.print(table)

        _print_stage_table("Proven Use", proven_items, score_label="Potential")
        _print_stage_table("Operationalized But Not Proven", operationalized_items, score_label="Potential")
        _print_stage_table("Enriched But Not Yet Used", enriched_items, score_label="Potential")
        _print_stage_table("Stored Only", stored_items)

        console.print(
            "\n[dim]Interpretation: stored -> enriched -> retrieved -> operationalized -> proven. "
            "Future-candidate is an orthogonal flag for capabilities CAM should keep reconsidering.[/dim]"
        )

    finally:
        await engine.close()


async def _reassess_async(
    repo: Optional[str],
    task: str,
    limit: int,
    min_score: float,
    future_threshold: float,
) -> None:
    from rich.panel import Panel

    repo_tokens: set[str] = set()
    repo_summary: Optional[dict[str, Any]] = None
    if repo:
        repo_path = Path(repo).resolve()
        if not repo_path.exists():
            console.print(f"[red]Repository path does not exist: {repo_path}[/red]")
            raise typer.Exit(1)
        repo_summary = _summarize_repo_tree(repo_path)
        repo_tokens = _tokenize_reassessment_text(
            " ".join(
                repo_summary.get("marker_files", [])
                + repo_summary.get("top_dirs", [])
                + repo_summary.get("sample_files", [])
            )
        )

    task_tokens = _tokenize_reassessment_text(task)
    if not task_tokens and not repo_tokens:
        console.print("[red]Task is too vague for reassessment. Provide a more specific --task.[/red]")
        raise typer.Exit(1)

    engine, repository = await _kb_engine()

    try:
        methods = await repository.list_methodologies(limit=5000, include_dead=False)
        if not methods:
            console.print("[yellow]No methodologies in knowledge base. Run 'cam mine <dir>' first.[/yellow]")
            return

        template_rows = await repository.engine.fetch_all(
            """SELECT source_methodology_id,
                      COUNT(*) as template_count,
                      COALESCE(SUM(success_count), 0) as template_successes
               FROM action_templates
               WHERE source_methodology_id IS NOT NULL
               GROUP BY source_methodology_id"""
        )
        template_stats = {
            row["source_methodology_id"]: {
                "count": int(row["template_count"] or 0),
                "successes": int(row["template_successes"] or 0),
            }
            for row in template_rows
            if row.get("source_methodology_id")
        }

        recommendations: list[dict[str, Any]] = []
        future_watchlist: list[dict[str, Any]] = []
        for meth in methods:
            stats = template_stats.get(meth.id, {})
            template_count = int(stats.get("count", 0))
            template_successes = int(stats.get("successes", 0))
            score, reasons, triggers = _score_methodology_for_task(
                meth,
                task_tokens=task_tokens,
                repo_tokens=repo_tokens,
                template_count=template_count,
                template_successes=template_successes,
            )
            stage = _classify_assimilation_stage(
                meth,
                template_count=template_count,
                template_successes=template_successes,
            )
            payload = {
                "methodology": meth,
                "score": score,
                "reasons": reasons,
                "triggers": triggers,
                "stage": stage,
                "template_count": template_count,
                "template_successes": template_successes,
            }
            if score >= min_score:
                recommendations.append(payload)
            elif _is_future_candidate(meth, potential_threshold=future_threshold, template_count=template_count):
                future_watchlist.append(payload)

        recommendations.sort(
            key=lambda x: (
                x["score"],
                x["methodology"].success_count + x["template_successes"],
                x["methodology"].retrieval_count,
                x["methodology"].potential_score or 0,
            ),
            reverse=True,
        )
        future_watchlist.sort(
            key=lambda x: (
                x["methodology"].potential_score or 0,
                x["methodology"].novelty_score or 0,
            ),
            reverse=True,
        )

        console.print(Panel.fit(
            f"[bold cyan]CAM Reassess[/bold cyan]\n"
            f"Task: {task}\n"
            f"Repo context: {repo_summary['name'] if repo_summary else 'none'}",
            border_style="cyan",
        ))

        if repo_summary:
            console.print(
                f"[dim]Repo markers:[/dim] {', '.join(repo_summary.get('marker_files', [])[:6]) or '-'}"
            )

        if recommendations:
            table = Table(title=f"Recommended Now ({min(limit, len(recommendations))})")
            table.add_column("ID", width=8)
            table.add_column("Stage", width=16)
            table.add_column("Score", justify="right", width=7, style="bold green")
            table.add_column("Description", max_width=36)
            table.add_column("Why Now", max_width=36)
            table.add_column("Triggers", max_width=22)
            for item in recommendations[:limit]:
                meth = item["methodology"]
                table.add_row(
                    meth.id[:8],
                    item["stage"],
                    f"{item['score']:.2f}",
                    meth.problem_description[:36],
                    "; ".join(item["reasons"][:2])[:36],
                    ", ".join(item["triggers"][:3])[:22] or "-",
                )
            console.print(table)
        else:
            console.print("[yellow]No methodologies cleared the reassessment score threshold.[/yellow]")

        if future_watchlist:
            watch = Table(title=f"Future Watchlist ({min(limit, len(future_watchlist))})")
            watch.add_column("ID", width=8)
            watch.add_column("Potential", justify="right", width=9, style="bold cyan")
            watch.add_column("Description", max_width=40)
            watch.add_column("Triggers", max_width=24)
            for item in future_watchlist[:limit]:
                meth = item["methodology"]
                watch.add_row(
                    meth.id[:8],
                    f"{(meth.potential_score or 0):.3f}",
                    meth.problem_description[:40],
                    ", ".join(item["triggers"][:4])[:24] or "-",
                )
            console.print(watch)

        console.print(
            "\n[dim]Use this command when a new task arrives and you want CAM to reactivate prior knowledge "
            "based on current fit, not just historical storage.[/dim]"
        )

    finally:
        await engine.close()


async def _synergies_async(verbose: bool) -> None:
    """Display synergy stats and graph summary."""
    from rich.panel import Panel
    from claw.core.config import DatabaseConfig, load_config
    from claw.db.engine import DatabaseEngine
    from claw.db.repository import Repository

    config = load_config()
    engine = DatabaseEngine(config.database)
    await engine.connect()
    await engine.apply_migrations()
    await engine.initialize_schema()
    repository = Repository(engine)

    try:
        # Synergy exploration stats
        stats = await repository.get_synergy_stats()
        console.print(Panel.fit(
            "[bold cyan]Capability Synergy Graph[/bold cyan]",
            border_style="cyan",
        ))

        table = Table(title="Exploration Stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total explored pairs", str(stats["total_explored"]))
        for result_type, count in stats.get("by_result", {}).items():
            table.add_row(f"  {result_type}", str(count))
        table.add_row("Avg synergy score", f"{stats['avg_synergy_score']:.4f}")
        table.add_row("Synergy edges", str(stats["synergy_edges"]))
        console.print(table)

        # Capabilities with data
        with_caps = await repository.get_methodologies_with_capabilities()
        without_caps = await repository.get_methodologies_without_capability_data()
        console.print(f"\nCapabilities enriched: [bold]{len(with_caps)}[/bold]")
        console.print(f"Capabilities unenriched: [bold]{len(without_caps)}[/bold]")

        if verbose and with_caps:
            cap_table = Table(title="Enriched Capabilities")
            cap_table.add_column("ID", width=8)
            cap_table.add_column("Problem", width=40)
            cap_table.add_column("Type", width=15)
            cap_table.add_column("Domain")
            for m in with_caps[:20]:
                cd = m.capability_data or {}
                cap_table.add_row(
                    m.id[:8],
                    m.problem_description[:40],
                    cd.get("capability_type", "?"),
                    ", ".join(cd.get("domain", [])[:3]),
                )
            console.print(cap_table)

    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# Grouped workflow aliases
# ---------------------------------------------------------------------------


@learn_app.command(name="report")
def learn_report(
    limit: int = typer.Option(10, "--limit", "-n", help="Rows to show per section"),
    future_threshold: float = typer.Option(0.65, "--future-threshold", help="Potential score threshold for future-candidate flag"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam assimilation-report`."""
    assimilation_report(limit=limit, future_threshold=future_threshold, config=config)


@learn_app.command(name="delta")
def learn_delta(
    directory: Optional[str] = typer.Argument(None, help="Optional repo directory to scope the report"),
    depth: int = typer.Option(6, "--depth", "-d", help="Max directory depth when scoping by directory"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Dedup repo iterations by canonical name"),
    since_hours: float = typer.Option(24.0, "--since-hours", help="Only include repos mined within this many hours"),
    latest: int = typer.Option(10, "--latest", "-n", help="Maximum recently mined repos/methodologies to summarize"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam assimilation-delta`."""
    assimilation_delta(
        directory=directory,
        depth=depth,
        dedup=dedup,
        since_hours=since_hours,
        latest=latest,
        config=config,
    )


@learn_app.command(name="reassess")
def learn_reassess(
    repo: Optional[str] = typer.Argument(None, help="Optional repository path for additional context"),
    task: str = typer.Option(..., "--task", "-t", help="Task or goal CAM should reassess prior knowledge against"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum recommendations to show"),
    min_score: float = typer.Option(0.2, "--min-score", help="Minimum reassessment score to show"),
    future_threshold: float = typer.Option(0.65, "--future-threshold", help="Potential score threshold for future-candidate flag"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam reassess`."""
    reassess(
        repo=repo,
        task=task,
        limit=limit,
        min_score=min_score,
        future_threshold=future_threshold,
        config=config,
    )


@learn_app.command(name="synergies")
def learn_synergies(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed edge list"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam synergies`."""
    synergies(verbose=verbose, config=config)


@task_app.command(name="add")
def task_add(
    repo: str = typer.Argument(..., help="Path to the repository this goal is for"),
    title: str = typer.Option(..., "--title", "-t", prompt="Goal title", help="Short title for the goal"),
    description: str = typer.Option(
        ..., "--description", "-d", prompt="Goal description (what should the agent do?)",
        help="Detailed description of what should be accomplished",
    ),
    priority: str = typer.Option("medium", "--priority", "-p", help="Priority: critical, high, medium, low"),
    task_type: str = typer.Option(
        "analysis", "--type",
        help="Task type: analysis, testing, documentation, security, refactoring, bug_fix, architecture, dependency_analysis",
    ),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Preferred agent: claude, codex, gemini, grok"),
    step: list[str] = typer.Option([], "--step", help="Execution command to run for this goal (repeatable)"),
    check: list[str] = typer.Option([], "--check", help="Acceptance check command for this goal (repeatable)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam add-goal`."""
    add_goal(
        repo=repo,
        title=title,
        description=description,
        priority=priority,
        task_type=task_type,
        agent=agent,
        step=step,
        check=check,
        config=config,
    )


@task_app.command(name="quickstart")
def task_quickstart(
    repo: str = typer.Argument(..., help="Path to the repository this goal is for"),
    title: str = typer.Option(..., "--title", "-t", prompt="Goal title", help="Short title for the goal"),
    description: str = typer.Option(
        ..., "--description", "-d", prompt="Goal description (what should be done?)",
        help="Detailed goal description",
    ),
    priority: str = typer.Option("high", "--priority", "-p", help="Priority: critical, high, medium, low"),
    task_type: str = typer.Option(
        "bug_fix",
        "--type",
        help="Task type: analysis, testing, documentation, security, refactoring, bug_fix, architecture, dependency_analysis",
    ),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Preferred agent: claude, codex, gemini, grok"),
    step: list[str] = typer.Option([], "--step", help="Execution command to run for this goal (repeatable)"),
    check: list[str] = typer.Option([], "--check", help="Acceptance check command for this goal (repeatable)"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show runbook and dry-run preview after creating the goal"),
    execute: bool = typer.Option(False, "--execute", help="Immediately execute this exact task after setup"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam quickstart`."""
    quickstart(
        repo=repo,
        title=title,
        description=description,
        priority=priority,
        task_type=task_type,
        agent=agent,
        step=step,
        check=check,
        preview=preview,
        execute=execute,
        config=config,
    )


@task_app.command(name="runbook")
def task_runbook(
    task_id: str = typer.Argument(..., help="Task ID to inspect"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam runbook`."""
    runbook(task_id=task_id, config=config)


@task_app.command(name="results")
def task_results(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of results to show"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project ID"),
) -> None:
    """Preferred grouped alias for `cam results`."""
    results(config=config, limit=limit, project=project)


@forge_app.command(name="export")
def forge_export_grouped(
    out: str = typer.Option("data/cam_knowledge_pack.jsonl", "--out", help="Output JSONL knowledge pack path"),
    db: Optional[str] = typer.Option(None, "--db", help="Override CAM database path"),
    max_methodologies: int = typer.Option(300, "--max-methodologies", help="Maximum methodologies to export"),
    max_tasks: int = typer.Option(300, "--max-tasks", help="Maximum tasks to export"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for the export"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam forge-export`."""
    forge_export(
        out=out,
        db=db,
        max_methodologies=max_methodologies,
        max_tasks=max_tasks,
        max_minutes=max_minutes,
        verbose=verbose,
        config=config,
    )


@forge_app.command(name="benchmark")
def forge_benchmark_grouped(
    repo: str = typer.Option("tests/fixtures/embedding_forge/repo", "--repo", help="Fixture or target repo path"),
    note: str = typer.Option("tests/fixtures/embedding_forge/note.md", "--note", help="Note path"),
    knowledge_pack: str = typer.Option(
        "tests/fixtures/embedding_forge/knowledge_pack.jsonl",
        "--knowledge-pack",
        help="Knowledge pack JSONL path",
    ),
    out: str = typer.Option("data/forge_benchmark_fixture", "--out", help="Output benchmark directory"),
    max_minutes: int = typer.Option(5, "--max-minutes", help="Wall-clock time guardrail for the benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Preferred grouped alias for `cam forge-benchmark`."""
    forge_benchmark(
        repo=repo,
        note=note,
        knowledge_pack=knowledge_pack,
        out=out,
        max_minutes=max_minutes,
        verbose=verbose,
    )


@doctor_app.command(name="keycheck")
def doctor_keycheck(
    for_command: str = typer.Option("mine", "--for", help="Command to preflight: mine, ideate"),
    live: bool = typer.Option(False, "--live", help="Also validate the keys with a tiny real provider call"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam keycheck`."""
    keycheck(for_command=for_command, live=live, config=config)


@doctor_app.command(name="status")
def doctor_status(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Preferred grouped alias for `cam status`."""
    status(config=config)


@app.command(name="prism-demo", hidden=True)
def prism_demo(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed diagnostics"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to claw.toml"),
):
    """Demonstrate PRISM multi-scale embeddings with non-base-10 math."""
    _setup_logging(verbose)
    asyncio.run(_prism_demo_async(verbose))


async def _prism_demo_async(verbose: bool) -> None:
    """Run the PRISM demonstration."""
    import numpy as np
    from rich.panel import Panel
    from claw.embeddings.prism import PrismEngine, PrismEmbedding
    import hashlib

    console.print(Panel.fit(
        "[bold cyan]PRISM[/bold cyan] — P-adic Residue Informed Stochastic Multi-scale Embeddings\n"
        "[dim]Non-base-10 embedding methodology for hierarchical, fault-tolerant, uncertainty-aware similarity[/dim]",
        border_style="cyan",
    ))

    # Use deterministic embedding engine (SHA-384 hash → 384-dim)
    class DemoEmbeddingEngine:
        DIMENSION = 384
        def encode(self, text: str) -> list[float]:
            h = hashlib.sha384(text.encode()).digest()
            raw = [b / 255.0 for b in h] * 8
            return raw[:self.DIMENSION]

    embedding_engine = DemoEmbeddingEngine()
    engine = PrismEngine(embedding_engine=embedding_engine)

    # Sample methodology descriptions with lifecycle states
    samples = [
        ("Refactoring database queries for performance", "thriving"),
        ("Optimizing SQL query execution plans", "thriving"),
        ("Adding JWT authentication to REST API", "viable"),
        ("Implementing OAuth2 flow for user login", "viable"),
        ("Experimental neural code search prototype", "embryonic"),
        ("Legacy XML parser migration to JSON", "declining"),
        ("Deprecated SOAP endpoint removal", "dormant"),
    ]

    # 1. Encode all samples with PRISM
    console.print("\n[bold]1. Encoding samples with PRISM[/bold]")
    embeddings = []
    for text, lifecycle in samples:
        emb = engine.encode_and_enhance(text, {"lifecycle_state": lifecycle})
        embeddings.append((text, lifecycle, emb))
        if verbose:
            console.print(f"  [dim]{lifecycle:10s}[/dim] κ={emb.vmf_kappa:5.1f}  tree={emb.padic_tree}  {text[:50]}")

    # 2. Pairwise comparison table
    console.print("\n[bold]2. PRISM vs Cosine Similarity Matrix[/bold]")
    table = Table(title="Pairwise Similarity (PRISM combined / cosine)")
    table.add_column("", style="dim", width=6)
    for i in range(len(samples)):
        table.add_column(f"S{i}", justify="center", width=12)

    for i, (text_i, _, emb_i) in enumerate(embeddings):
        row = [f"S{i}"]
        for j, (text_j, _, emb_j) in enumerate(embeddings):
            if i == j:
                row.append("[bold]1.00/1.00[/bold]")
            else:
                score = engine.similarity(emb_i, emb_j)
                cos = score.cosine
                prism = score.combined
                # Highlight divergence
                diff = abs(prism - max(0, cos))
                style = "green" if diff > 0.1 else ""
                row.append(f"[{style}]{prism:.2f}/{cos:.2f}[/{style}]" if style else f"{prism:.2f}/{cos:.2f}")
        table.add_row(*row)

    console.print(table)
    console.print("[dim]Format: PRISM/cosine. [green]Green[/green] = divergence > 0.1[/dim]")

    # Legend
    legend = Table(title="Sample Legend", show_header=False)
    legend.add_column("ID", width=4)
    legend.add_column("Lifecycle", width=12)
    legend.add_column("Description")
    for i, (text, lifecycle, _) in enumerate(embeddings):
        legend.add_row(f"S{i}", lifecycle, text)
    console.print(legend)

    # 3. Hierarchy demonstration
    console.print("\n[bold]3. Hierarchical Similarity (P-adic)[/bold]")
    # Same-domain pair vs cross-domain pair
    score_same = engine.similarity(embeddings[0][2], embeddings[1][2])
    score_cross = engine.similarity(embeddings[0][2], embeddings[2][2])
    console.print(f"  Same domain (DB query + SQL optimization):  p-adic={score_same.padic:.3f}  cosine={score_same.cosine:.3f}")
    console.print(f"  Cross domain (DB query + JWT auth):         p-adic={score_cross.padic:.3f}  cosine={score_cross.cosine:.3f}")

    # 4. Fault detection demonstration
    console.print("\n[bold]4. Fault Detection (RNS Channel Voting)[/bold]")
    clean_emb = embeddings[0][2]
    # Create corrupted version
    corrupted_channels = [ch[:] for ch in clean_emb.rns_channels]
    # Corrupt channel 2: shift all values
    corrupted_channels[2] = [(v + 5) % engine.PRIMES[2] for v in corrupted_channels[2]]
    corrupted = PrismEmbedding(
        base_vector=clean_emb.base_vector,
        padic_tree=clean_emb.padic_tree,
        rns_channels=corrupted_channels,
        vmf_kappa=clean_emb.vmf_kappa,
    )
    score_clean = engine.similarity(clean_emb, clean_emb)
    score_corrupt = engine.similarity(clean_emb, corrupted)
    console.print(f"  Clean vs clean:     agreement={score_clean.channel_agreement:.3f}  drift={score_clean.drift_detected}")
    console.print(f"  Clean vs corrupted: agreement={score_corrupt.channel_agreement:.3f}  drift={score_corrupt.drift_detected}")

    # 5. Uncertainty demonstration (vMF)
    console.print("\n[bold]5. Uncertainty Weighting (von Mises-Fisher)[/bold]")
    # Same text, different lifecycle states
    text = "Implementing caching layer for API responses"
    emb_thriving = engine.encode_and_enhance(text, {"lifecycle_state": "thriving"})
    emb_embryonic = engine.encode_and_enhance(text, {"lifecycle_state": "embryonic"})
    emb_viable = engine.encode_and_enhance(text, {"lifecycle_state": "viable"})

    score_tt = engine.similarity(emb_thriving, emb_thriving)
    score_te = engine.similarity(emb_thriving, emb_embryonic)
    score_tv = engine.similarity(emb_thriving, emb_viable)

    console.print(f"  thriving↔thriving (κ=20↔20):   vMF overlap={score_tt.vmf_overlap:.3f}  combined={score_tt.combined:.3f}")
    console.print(f"  thriving↔viable   (κ=20↔5):    vMF overlap={score_tv.vmf_overlap:.3f}  combined={score_tv.combined:.3f}")
    console.print(f"  thriving↔embryonic (κ=20↔2):   vMF overlap={score_te.vmf_overlap:.3f}  combined={score_te.combined:.3f}")

    # 6. Diagnostic breakdown
    if verbose:
        console.print("\n[bold]6. Detailed Diagnostic (S0 vs S1)[/bold]")
        diag = engine.diagnose(embeddings[0][2], embeddings[1][2])
        diag_table = Table(title="Diagnostic Breakdown")
        diag_table.add_column("Component", width=16)
        diag_table.add_column("Raw", justify="right", width=8)
        diag_table.add_column("Weighted", justify="right", width=8)
        diag_table.add_column("Detail")

        diag_table.add_row(
            "Cosine", str(diag["cosine_detail"]["raw"]), str(diag["cosine_detail"]["weighted"]),
            ""
        )
        diag_table.add_row(
            "P-adic", str(diag["padic_detail"]["raw"]), str(diag["padic_detail"]["weighted"]),
            f"shared_depth={diag['padic_detail']['shared_depth']}"
        )
        diag_table.add_row(
            "RNS", str(diag["rns_detail"]["consensus"]), "",
            f"channels={diag['rns_detail']['channel_sims']} agreement={diag['rns_detail']['agreement']}"
        )
        diag_table.add_row(
            "vMF", str(diag["vmf_detail"]["overlap"]), str(diag["vmf_detail"]["weighted"]),
            f"κ_a={diag['vmf_detail']['kappa_a']} κ_b={diag['vmf_detail']['kappa_b']}"
        )
        console.print(diag_table)
        console.print(f"  Dominant: [bold]{diag['dominant_component']}[/bold]")
        console.print(f"  Interpretation: {diag['interpretation']}")

    console.print(f"\n[bold green]PRISM demonstration complete.[/bold green]")
    console.print("[dim]PRISM adds hierarchical (p-adic), fault-tolerant (RNS), and uncertainty-aware (vMF) signals to standard cosine similarity.[/dim]")


# ---------------------------------------------------------------------------
# kb — Knowledge Browser command group
# ---------------------------------------------------------------------------

kb_app = typer.Typer(
    name="kb",
    help="Knowledge browser — explore assimilated capabilities, synergies, and domains",
    no_args_is_help=True,
)
app.add_typer(learn_app, name="learn")
app.add_typer(task_app, name="task")
app.add_typer(forge_app, name="forge")
app.add_typer(doctor_app, name="doctor")
app.add_typer(kb_app, name="kb")


async def _kb_engine():
    """Shared async setup for kb commands — returns (engine, repository)."""
    from claw.core.config import load_config
    from claw.db.engine import DatabaseEngine
    from claw.db.repository import Repository

    config = load_config()
    engine = DatabaseEngine(config.database)
    await engine.connect()
    await engine.apply_migrations()
    await engine.initialize_schema()
    repository = Repository(engine)
    return engine, repository


@kb_app.command()
def insights(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """THE showpiece — top capabilities, domain map, synergy highlights, score distributions."""
    asyncio.run(_kb_insights_async())


async def _kb_insights_async() -> None:
    from rich.panel import Panel

    engine, repository = await _kb_engine()

    try:
        # Header: total count + source repos
        total = await repository.count_methodologies()
        if total == 0:
            console.print("[yellow]No capabilities in knowledge base. Run 'cam mine <dir>' first.[/yellow]")
            return

        state_counts = await repository.count_methodologies_by_state()
        active = sum(v for k, v in state_counts.items() if k != "dead")

        # Count distinct source repos from tags (source:reponame)
        import json as _json
        _all_meths = await repository.engine.fetch_all(
            "SELECT tags FROM methodologies WHERE tags IS NOT NULL"
        )
        _sources = set()
        for _r in _all_meths:
            _tags = _json.loads(_r["tags"]) if isinstance(_r["tags"], str) else (_r["tags"] or [])
            for _t in _tags:
                if isinstance(_t, str) and _t.startswith("source:"):
                    _sources.add(_t[7:])
        repo_count = len(_sources) if _sources else "?"

        console.print(Panel.fit(
            f"[bold cyan]CAM Knowledge Base[/bold cyan]\n"
            f"[bold]{total:,}[/bold] capabilities from [bold]{repo_count}[/bold] repos  |  "
            f"[bold]{active:,}[/bold] active",
            border_style="cyan",
        ))

        # Score distributions
        dist = await repository.get_novelty_potential_distribution()
        if dist["total"] > 0:
            score_table = Table(title="Score Distributions")
            score_table.add_column("Metric", style="bold", width=18)
            score_table.add_column("Avg", justify="right", width=8)
            score_table.add_column("Min", justify="right", width=8)
            score_table.add_column("Max", justify="right", width=8)
            score_table.add_column("Scored", justify="right", width=8)
            score_table.add_row(
                "Novelty",
                f"{dist['avg_novelty']:.3f}",
                f"{dist['min_novelty']:.3f}",
                f"{dist['max_novelty']:.3f}",
                str(dist["total"]),
            )
            score_table.add_row(
                "Potential",
                f"{dist['avg_potential']:.3f}",
                f"{dist['min_potential']:.3f}",
                f"{dist['max_potential']:.3f}",
                str(dist["total"]),
            )
            console.print(score_table)

        # Lifecycle state table
        if state_counts:
            state_table = Table(title="Lifecycle States")
            state_table.add_column("State", style="bold", width=14)
            state_table.add_column("Count", justify="right", width=8)
            state_table.add_column("", width=30)
            state_colors = {
                "thriving": "green", "viable": "cyan", "embryonic": "yellow",
                "declining": "magenta", "dormant": "dim", "dead": "red",
            }
            for state in ["thriving", "viable", "embryonic", "declining", "dormant", "dead"]:
                count = state_counts.get(state, 0)
                if count == 0:
                    continue
                color = state_colors.get(state, "")
                bar_len = min(int(count / max(state_counts.values()) * 25), 25)
                bar = "█" * bar_len
                state_table.add_row(
                    f"[{color}]{state}[/{color}]" if color else state,
                    str(count),
                    f"[{color}]{bar}[/{color}]" if color else bar,
                )
            console.print(state_table)

        # Top 5 Novel
        top_novel = await repository.get_most_novel_methodologies(limit=5)
        if top_novel:
            novel_table = Table(title="Top 5 Novel Capabilities")
            novel_table.add_column("ID", width=8)
            novel_table.add_column("Description", max_width=50)
            novel_table.add_column("Novelty", justify="right", width=8, style="bold yellow")
            novel_table.add_column("Domains", max_width=25)
            for m in top_novel:
                domains = ", ".join((m.capability_data or {}).get("domain", [])[:3])
                score = m.novelty_score or 0
                score_style = "bold green" if score >= 0.7 else ("yellow" if score >= 0.4 else "dim")
                novel_table.add_row(
                    m.id[:8],
                    m.problem_description[:50],
                    f"[{score_style}]{score:.3f}[/{score_style}]",
                    domains,
                )
            console.print(novel_table)

        # Top 5 High-Potential
        top_potential = await repository.get_high_potential_methodologies(limit=5)
        if top_potential:
            pot_table = Table(title="Top 5 High-Potential Capabilities")
            pot_table.add_column("ID", width=8)
            pot_table.add_column("Description", max_width=50)
            pot_table.add_column("Potential", justify="right", width=8, style="bold cyan")
            pot_table.add_column("Domains", max_width=25)
            for m in top_potential:
                domains = ", ".join((m.capability_data or {}).get("domain", [])[:3])
                score = m.potential_score or 0
                score_style = "bold green" if score >= 0.7 else ("cyan" if score >= 0.4 else "dim")
                pot_table.add_row(
                    m.id[:8],
                    m.problem_description[:50],
                    f"[{score_style}]{score:.3f}[/{score_style}]",
                    domains,
                )
            console.print(pot_table)

        # Domain Landscape — Top 15
        domain_dist = await repository.get_domain_distribution()
        if domain_dist:
            sorted_domains = sorted(domain_dist.items(), key=lambda x: -x[1])[:15]
            max_count = sorted_domains[0][1] if sorted_domains else 1
            domain_table = Table(title="Domain Landscape (Top 15)")
            domain_table.add_column("Domain", style="cyan", max_width=25)
            domain_table.add_column("Count", justify="right", width=6)
            domain_table.add_column("", width=30)
            for domain, count in sorted_domains:
                bar_len = min(int(count / max_count * 25), 25)
                bar = "█" * bar_len
                domain_table.add_row(domain, str(count), f"[cyan]{bar}[/cyan]")
            console.print(domain_table)

        # Synergy Highlights — Top 5
        top_edges = await repository.get_top_synergy_edges(limit=5)
        if top_edges:
            syn_table = Table(title="Synergy Highlights (Top 5)")
            syn_table.add_column("Score", justify="right", width=7, style="bold green")
            syn_table.add_column("Type", width=14)
            syn_table.add_column("Capability A", max_width=35)
            syn_table.add_column("Capability B", max_width=35)
            syn_table.add_column("Cross?", width=6)
            for edge in top_edges:
                is_cross = bool(
                    set(edge["cap_a_domains"]) and set(edge["cap_b_domains"])
                    and not set(edge["cap_a_domains"]) & set(edge["cap_b_domains"])
                )
                cross_str = "[bold yellow]YES[/bold yellow]" if is_cross else ""
                syn_table.add_row(
                    f"{edge['synergy_score']:.3f}",
                    edge["synergy_type"][:14],
                    edge["cap_a_summary"][:35],
                    edge["cap_b_summary"][:35],
                    cross_str,
                )
            console.print(syn_table)

        # Capability Type Distribution
        type_dist = await repository.get_type_distribution()
        if type_dist:
            sorted_types = sorted(type_dist.items(), key=lambda x: -x[1])[:10]
            type_table = Table(title="Capability Types (Top 10)")
            type_table.add_column("Type", style="yellow", max_width=25)
            type_table.add_column("Count", justify="right", width=6)
            for ctype, count in sorted_types:
                type_table.add_row(ctype, str(count))
            console.print(type_table)

    finally:
        await engine.close()


@kb_app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Search capabilities with natural language (hybrid vector + FTS5)."""
    asyncio.run(_kb_search_async(query, limit))


async def _kb_search_async(query: str, limit: int) -> None:
    engine, repository = await _kb_engine()

    try:
        total = await repository.count_methodologies()
        if total == 0:
            console.print("[yellow]No capabilities in knowledge base. Run 'cam mine <dir>' first.[/yellow]")
            return

        # Try FTS5 text search (always works, no embedding engine required)
        text_results = await repository.search_methodologies_text(query, limit=limit)

        if not text_results:
            console.print(f"[yellow]No results for '{query}'.[/yellow]")
            return

        console.print(f"\n[bold]Search results for:[/bold] [cyan]{query}[/cyan]  ({len(text_results)} matches)\n")

        table = Table(show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", width=8)
        table.add_column("Description", max_width=50)
        table.add_column("Domains", max_width=20)
        table.add_column("Novelty", justify="right", width=8)
        table.add_column("Potential", justify="right", width=8)
        table.add_column("State", width=10)

        state_colors = {
            "thriving": "green", "viable": "cyan", "embryonic": "yellow",
            "declining": "magenta", "dormant": "dim", "dead": "red",
        }

        for i, m in enumerate(text_results, 1):
            domains = ", ".join((m.capability_data or {}).get("domain", [])[:3])
            novelty_str = f"{m.novelty_score:.3f}" if m.novelty_score is not None else "-"
            potential_str = f"{m.potential_score:.3f}" if m.potential_score is not None else "-"
            color = state_colors.get(m.lifecycle_state, "")
            state_str = f"[{color}]{m.lifecycle_state}[/{color}]" if color else m.lifecycle_state

            table.add_row(
                str(i),
                m.id[:8],
                m.problem_description[:50],
                domains,
                novelty_str,
                potential_str,
                state_str,
            )

        console.print(table)
        console.print(f"\n[dim]Use 'cam kb capability <id>' for full details.[/dim]")

    finally:
        await engine.close()


@kb_app.command()
def capability(
    cap_id: str = typer.Argument(..., help="Capability ID or ID prefix (6+ chars)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Deep dive on a single capability — full data, synergies, and related."""
    asyncio.run(_kb_capability_async(cap_id))


async def _kb_capability_async(cap_id: str) -> None:
    from rich.panel import Panel

    engine, repository = await _kb_engine()

    try:
        # Try prefix match
        m = await repository.get_methodology_by_prefix(cap_id)
        if m is None:
            console.print(f"[red]No capability found matching '{cap_id}'.[/red]")
            console.print("[dim]Provide at least 6 characters of the ID.[/dim]")
            return

        # Header
        state_colors = {
            "thriving": "green", "viable": "cyan", "embryonic": "yellow",
            "declining": "magenta", "dormant": "dim", "dead": "red",
        }
        color = state_colors.get(m.lifecycle_state, "")
        state_str = f"[{color}]{m.lifecycle_state}[/{color}]" if color else m.lifecycle_state

        console.print(Panel.fit(
            f"[bold cyan]Capability Detail[/bold cyan]\n"
            f"ID: [bold]{m.id}[/bold]\n"
            f"State: {state_str}",
            border_style="cyan",
        ))

        # Problem description
        console.print(f"\n[bold]Problem Description[/bold]")
        console.print(f"  {m.problem_description}")

        # Methodology notes
        if m.methodology_notes:
            notes = m.methodology_notes[:500]
            if len(m.methodology_notes) > 500:
                notes += "..."
            console.print(f"\n[bold]Notes[/bold]")
            console.print(f"  {notes}")

        # Scores
        score_table = Table(title="Scores")
        score_table.add_column("Metric", style="bold", width=16)
        score_table.add_column("Value", justify="right", width=10)

        fv = m.fitness_vector
        if fv and "total" in fv:
            score_table.add_row("Fitness (total)", f"{fv['total']:.3f}")
        if m.novelty_score is not None:
            score_table.add_row("Novelty", f"{m.novelty_score:.3f}")
        if m.potential_score is not None:
            score_table.add_row("Potential", f"{m.potential_score:.3f}")
        score_table.add_row("Retrievals", str(m.retrieval_count))
        score_table.add_row("Successes", str(m.success_count))
        score_table.add_row("Failures", str(m.failure_count))
        console.print(score_table)

        # Capability data
        cd = m.capability_data
        if cd:
            cap_table = Table(title="Capability Data")
            cap_table.add_column("Field", style="bold", width=18)
            cap_table.add_column("Value", max_width=50)
            cap_table.add_row("Type", cd.get("capability_type", "-"))
            cap_table.add_row("Domains", ", ".join(cd.get("domain", [])))
            cap_table.add_row("IO Types In", ", ".join(str(t) for t in cd.get("io_types_in", [])))
            cap_table.add_row("IO Types Out", ", ".join(str(t) for t in cd.get("io_types_out", [])))
            cap_table.add_row("Composability", str(cd.get("composability_score", "-")))
            cap_table.add_row("Standalone", str(cd.get("standalone_viable", "-")))
            console.print(cap_table)

        # Metadata
        meta_parts = []
        if m.tags:
            meta_parts.append(f"Tags: {', '.join(m.tags)}")
        if m.language:
            meta_parts.append(f"Language: {m.language}")
        if m.files_affected:
            meta_parts.append(f"Files: {', '.join(m.files_affected[:5])}")
        if m.methodology_type:
            meta_parts.append(f"Type: {m.methodology_type}")
        if m.scope:
            meta_parts.append(f"Scope: {m.scope}")
        if meta_parts:
            console.print(f"\n[bold]Metadata[/bold]")
            for part in meta_parts:
                console.print(f"  {part}")

        # Related synergies
        links = await repository.get_methodology_links(m.id)
        if links:
            link_table = Table(title=f"Related Links ({len(links)})")
            link_table.add_column("Type", width=14)
            link_table.add_column("Linked To", width=10)
            link_table.add_column("Strength", justify="right", width=8)
            for link in links[:10]:
                other_id = link["target_id"] if link["source_id"] == m.id else link["source_id"]
                link_table.add_row(
                    link["link_type"],
                    other_id[:8] + "…",
                    f"{link['strength']:.2f}",
                )
            console.print(link_table)
            if len(links) > 10:
                console.print(f"  [dim]... and {len(links) - 10} more links[/dim]")

    finally:
        await engine.close()


@kb_app.command()
def domains(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Domain landscape — which knowledge domains exist and bridge capabilities."""
    asyncio.run(_kb_domains_async())


async def _kb_domains_async() -> None:
    from rich.panel import Panel

    engine, repository = await _kb_engine()

    try:
        total = await repository.count_methodologies()
        if total == 0:
            console.print("[yellow]No capabilities in knowledge base. Run 'cam mine <dir>' first.[/yellow]")
            return

        domain_dist = await repository.get_domain_distribution()
        if not domain_dist:
            console.print("[yellow]No domain data. Capabilities may not have been enriched yet.[/yellow]")
            return

        sorted_domains = sorted(domain_dist.items(), key=lambda x: -x[1])
        max_count = sorted_domains[0][1] if sorted_domains else 1
        total_domains = len(sorted_domains)
        total_tagged = sum(v for v in domain_dist.values())

        console.print(Panel.fit(
            f"[bold cyan]Domain Landscape[/bold cyan]\n"
            f"[bold]{total_domains}[/bold] domains across [bold]{total:,}[/bold] capabilities\n"
            f"[bold]{total_tagged:,}[/bold] total domain tags (capabilities can span multiple domains)",
            border_style="cyan",
        ))

        # Full domain table
        domain_table = Table(title=f"All Domains ({total_domains})")
        domain_table.add_column("#", style="dim", width=3)
        domain_table.add_column("Domain", style="cyan", max_width=30)
        domain_table.add_column("Caps", justify="right", width=6)
        domain_table.add_column("", width=30)

        for i, (domain, count) in enumerate(sorted_domains, 1):
            bar_len = min(int(count / max_count * 25), 25)
            bar = "█" * bar_len
            domain_table.add_row(str(i), domain, str(count), f"[cyan]{bar}[/cyan]")

        console.print(domain_table)

        # Bridge Capabilities — spanning 3+ domains
        bridges = await repository.get_cross_domain_capabilities(min_domains=3, limit=15)
        if bridges:
            bridge_table = Table(title=f"Bridge Capabilities (3+ domains, showing {len(bridges)})")
            bridge_table.add_column("ID", width=8)
            bridge_table.add_column("Description", max_width=40)
            bridge_table.add_column("Domains", max_width=40)
            bridge_table.add_column("Novelty", justify="right", width=8)

            for m in bridges:
                domains = (m.capability_data or {}).get("domain", [])
                novelty_str = f"{m.novelty_score:.3f}" if m.novelty_score is not None else "-"
                bridge_table.add_row(
                    m.id[:8],
                    m.problem_description[:40],
                    ", ".join(domains),
                    novelty_str,
                )
            console.print(bridge_table)
        else:
            console.print("[dim]No bridge capabilities (spanning 3+ domains) found.[/dim]")

    finally:
        await engine.close()


@kb_app.command(name="synergies")
def kb_synergies(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of top synergy edges"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to claw.toml"),
) -> None:
    """Cross-repo synthesis explorer — top synergy edges and connections."""
    asyncio.run(_kb_synergies_async(limit))


async def _kb_synergies_async(limit: int) -> None:
    from rich.panel import Panel

    engine, repository = await _kb_engine()

    try:
        stats = await repository.get_synergy_stats()
        total_explored = stats["total_explored"]

        if total_explored == 0:
            console.print("[yellow]No synergy data. Run 'cam mine <dir>' to build the synergy graph.[/yellow]")
            return

        by_result = stats.get("by_result", {})
        synergy_count = by_result.get("synergy", 0)

        console.print(Panel.fit(
            f"[bold cyan]Synergy Explorer[/bold cyan]\n"
            f"[bold]{total_explored:,}[/bold] pairs explored  |  "
            f"[bold]{synergy_count:,}[/bold] synergies found  |  "
            f"[bold]{stats['synergy_edges']}[/bold] graph edges\n"
            f"Avg synergy score: [bold]{stats['avg_synergy_score']:.4f}[/bold]",
            border_style="cyan",
        ))

        # Exploration stats
        stats_table = Table(title="Exploration Summary")
        stats_table.add_column("Result", style="bold", width=18)
        stats_table.add_column("Count", justify="right", width=10)
        for result_type, count in sorted(by_result.items(), key=lambda x: -x[1]):
            style = {"synergy": "green", "no_synergy": "dim", "stale": "yellow"}.get(result_type, "")
            stats_table.add_row(
                f"[{style}]{result_type}[/{style}]" if style else result_type,
                str(count),
            )
        console.print(stats_table)

        # Top synergy edges
        top_edges = await repository.get_top_synergy_edges(limit=limit)
        if top_edges:
            edge_table = Table(title=f"Top {len(top_edges)} Synergy Edges")
            edge_table.add_column("#", style="dim", width=3)
            edge_table.add_column("Score", justify="right", width=7, style="bold green")
            edge_table.add_column("Type", width=14)
            edge_table.add_column("Capability A", max_width=35)
            edge_table.add_column("Capability B", max_width=35)
            edge_table.add_column("Cross?", width=6)

            cross_count = 0
            for i, edge in enumerate(top_edges, 1):
                is_cross = bool(
                    set(edge["cap_a_domains"]) and set(edge["cap_b_domains"])
                    and not set(edge["cap_a_domains"]) & set(edge["cap_b_domains"])
                )
                if is_cross:
                    cross_count += 1
                cross_str = "[bold yellow]YES[/bold yellow]" if is_cross else ""
                edge_table.add_row(
                    str(i),
                    f"{edge['synergy_score']:.3f}",
                    edge["synergy_type"][:14],
                    edge["cap_a_summary"][:35],
                    edge["cap_b_summary"][:35],
                    cross_str,
                )
            console.print(edge_table)

            if cross_count > 0:
                console.print(
                    f"\n  [bold yellow]{cross_count}[/bold yellow] cross-domain synergies "
                    f"(capabilities from different domains connected)"
                )
        else:
            console.print("[dim]No synergy edges found.[/dim]")

    finally:
        await engine.close()


def app_main() -> None:
    """Entry point for the installed CLI."""
    app()


if __name__ == "__main__":
    app_main()
