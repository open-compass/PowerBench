from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .bundle import load_bundle
from .config import DEFAULT_CONFIG, DEFAULT_OUTPUT_ROOT, DEFAULT_RAW_ROOT, load_config
from .evaluation import evaluate_runs
from .evaluation.experiments import (
    DEFAULT_MAX_JOB_SECONDS,
    REASONING_EFFORTS,
    build_run_manifest,
)
from .evaluation.report import build_pilot_report
from .evaluation.runner import run_manifest
from .generation import build_benchmark
from .generation.validation import (
    assert_valid,
    summarize_validation,
    validate_generated_records,
)
from .inventory import build_inventory
from .polishing import polish_benchmark_questions
from .polishing.publication import apply_metric_eligibility_clarifications
from .records import expand_gold_records, semantic_tasks_from_records
from .rules import hydrate_supported_rules
from .util import read_jsonl


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _stream_output(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _safe_run_stem(model: str, reasoning_effort: str, bundle_id: str) -> str:
    """Return a readable, path-safe identity for one model run matrix."""

    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("--model must be a non-empty string")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_model.casefold()).strip("-")
    slug = (slug or "model")[:48].rstrip("-")
    identity = f"{normalized_model}\0{reasoning_effort}\0{bundle_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{reasoning_effort}-{digest}"


def _load_qa_ids_file(path: Path) -> tuple[str, ...]:
    """Read a frozen question subset from a JSON array or ``{"qa_ids": [...]}``."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read --qa-ids-file {path}: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("qa_ids")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(qa_id, str) or not qa_id.strip() for qa_id in value)
    ):
        raise ValueError("--qa-ids-file must contain a non-empty JSON qa_ids array")
    qa_ids = tuple(value)
    if len(qa_ids) != len(set(qa_ids)):
        raise ValueError("--qa-ids-file contains duplicate question IDs")
    return qa_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powerbench.tasks",
        description="Executable heterogeneous QA pilot over the frozen PowerBench corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="Build the complete immutable benchmark bundle without API calls"
    )
    _common(build)

    polish = subparsers.add_parser(
        "polish",
        help="Use one generation model to naturalize autonomous questions and freeze them",
    )
    _common(polish)
    polish.add_argument("--model", required=True)
    polish.add_argument("--base-url")
    polish.add_argument("--api-key-env", default="OPENAI_API_KEY")
    polish.add_argument(
        "--per-level",
        dest="per_level",
        type=int,
        help="Polish and commit the next N unpolished questions from each level",
    )
    polish.add_argument("--max-attempts", type=int, default=3)
    polish.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent polishing requests (recommended: 2 or 3)",
    )
    polish.add_argument(
        "--confirm-paid-generation",
        action="store_true",
        help="Required acknowledgement that question polishing sends paid API requests",
    )

    clarify = subparsers.add_parser(
        "clarify-metric-scope",
        help="Clarify only unit-eligibility-affected metric-routing questions without an LLM",
    )
    _common(clarify)
    clarify.add_argument(
        "--expected-count",
        type=int,
        help="Abort unless exactly this many frozen questions require clarification",
    )

    run = subparsers.add_parser("run", help="Execute the benchmark with one evaluated model")
    _common(run)
    run.add_argument("--model", required=True)
    run.add_argument("--api-format", choices=("responses", "chat-completions"), default="responses")
    run.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
    )
    run.add_argument("--base-url")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--index-path", type=Path)
    run.add_argument("--manifest-path", type=Path)
    run.add_argument("--results-path", type=Path)
    run.add_argument("--max-jobs", type=int)
    run.add_argument(
        "--max-job-seconds",
        type=int,
        default=DEFAULT_MAX_JOB_SECONDS,
        help=(
            "Maximum retrieval/reasoning seconds per question; one subsequent no-tools "
            "answer submission is allowed outside this budget"
        ),
    )
    run.add_argument(
        "--task-levels",
        nargs="+",
        choices=("L1", "L2", "L3"),
        help="Run only the selected task levels (default: all levels)",
    )
    run.add_argument(
        "--scope-sizes",
        nargs="+",
        type=int,
        help="Run only selected configured retrieval scope sizes",
    )
    qa_selection = run.add_mutually_exclusive_group()
    qa_selection.add_argument(
        "--qa-ids",
        nargs="+",
        help="Run exactly the selected published question IDs",
    )
    qa_selection.add_argument(
        "--qa-ids-file",
        type=Path,
        help="Read the exact question subset from a JSON array or object with qa_ids",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent evaluation jobs (recommended starting value: 3)",
    )
    run.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Required acknowledgement that this command sends paid external API requests",
    )
    run.add_argument(
        "--reset-circuit-breaker",
        action="store_true",
        help=(
            "After inspecting three consecutive persisted API/infrastructure failures, "
            "explicitly acknowledge reopening the paid run"
        ),
    )
    run.add_argument(
        "--retry-exhausted",
        action="store_true",
        help=(
            "Retry only API/infrastructure jobs that exhausted their previous "
            "three-attempt budget; completed jobs remain skipped"
        ),
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Score saved predictions and observable traces"
    )
    _common(evaluate)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="medium",
    )
    evaluate.add_argument("--results-path", type=Path)
    evaluate.add_argument("--evaluation-root", type=Path)
    evaluate.add_argument("--manifest-path", type=Path)

    validate = subparsers.add_parser(
        "validate", help="Verify the bundle and independently replay generated Gold"
    )
    _common(validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output = args.output_root.resolve()
    questions = output / "public" / "questions.jsonl"
    scopes = output / "public" / "scopes.jsonl"
    tasks = output / "private" / "tasks.jsonl"
    question_generation = output / "private" / "question_generation.jsonl"
    bundle_path = output / "bundle.json"
    index_path = getattr(args, "index_path", None) or output / "public" / "corpus.sqlite"

    if args.command == "build":
        summary = build_benchmark(
            config,
            raw_root=args.raw_root,
            output_root=output,
        )
        _print(summary)
    elif args.command == "clarify-metric-scope":
        _print(
            apply_metric_eligibility_clarifications(
                config,
                raw_root=args.raw_root,
                output_root=output,
                expected_count=args.expected_count,
            )
        )
    elif args.command == "polish":
        if not args.confirm_paid_generation:
            raise PermissionError(
                "Question polishing is a paid external action. Re-run with "
                "--confirm-paid-generation."
            )
        try:
            api_key = os.environ[args.api_key_env]
        except KeyError as exc:
            raise OSError(f"{args.api_key_env} is not set") from exc
        if not api_key:
            raise OSError(f"{args.api_key_env} is empty")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install.
            raise RuntimeError("Install the optional OpenAI dependency with '.[openai]'") from exc
        client_options: dict[str, Any] = {"api_key": api_key, "max_retries": 2, "timeout": 120.0}
        if args.base_url:
            client_options["base_url"] = args.base_url
        client = OpenAI(**client_options)
        _print(
            polish_benchmark_questions(
                config,
                raw_root=args.raw_root,
                output_root=output,
                client=client,
                model=args.model,
                per_level=args.per_level,
                max_attempts=args.max_attempts,
                workers=args.workers,
                progress=lambda message: print(message, flush=True),
                stream_output=_stream_output,
            )
        )
    elif args.command == "run":
        args.reasoning_effort = args.reasoning_effort or (
            "unspecified" if args.api_format == "chat-completions" else "medium"
        )
        if not args.confirm_paid_run:
            raise PermissionError(
                "Model execution is a paid external action. Re-run with --confirm-paid-run."
            )
        bundle = load_bundle(bundle_path)
        selected_qa_ids = (
            _load_qa_ids_file(args.qa_ids_file) if args.qa_ids_file else args.qa_ids
        )
        run_stem = _safe_run_stem(args.model, args.reasoning_effort, bundle.bundle_id)
        if args.api_format != "responses":
            run_stem += "-chat"
        manifest_path = args.manifest_path or output / "runs" / f"{run_stem}_jobs.jsonl"
        results_path = args.results_path or output / "runs" / f"{run_stem}_responses.jsonl"
        build_run_manifest(
            config,
            questions_path=questions,
            scopes_path=scopes,
            gold_path=tasks,
            bundle_path=bundle_path,
            index_path=index_path,
            output_path=manifest_path,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            task_levels=tuple(args.task_levels) if args.task_levels else None,
            scope_sizes=tuple(args.scope_sizes) if args.scope_sizes else None,
            qa_ids=tuple(selected_qa_ids) if selected_qa_ids else None,
            max_job_seconds=args.max_job_seconds,
            api_format=args.api_format,
        )
        _print(
            run_manifest(
                config,
                raw_root=args.raw_root,
                index_path=index_path,
                questions_path=questions,
                scopes_path=scopes,
                jobs_path=manifest_path,
                results_path=results_path,
                gold_path=tasks,
                max_jobs=args.max_jobs,
                workers=args.workers,
                confirm_paid_run=True,
                reset_circuit_breaker=args.reset_circuit_breaker,
                retry_exhausted=args.retry_exhausted,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                progress=lambda message: print(message, flush=True),
            )
        )
    elif args.command == "evaluate":
        bundle = load_bundle(bundle_path)
        run_stem = _safe_run_stem(args.model, args.reasoning_effort, bundle.bundle_id)
        manifest_path = args.manifest_path or output / "runs" / f"{run_stem}_jobs.jsonl"
        results_path = args.results_path or output / "runs" / f"{run_stem}_responses.jsonl"
        evaluation_root = args.evaluation_root or output / "evaluation" / run_stem
        evaluation = evaluate_runs(
            results_path=results_path,
            gold_path=tasks,
            questions_path=questions,
            manifest_path=manifest_path,
            output_root=evaluation_root,
        )
        report = build_pilot_report(
            scores_path=evaluation_root / "per_run_scores.jsonl",
            bundle_path=bundle_path,
            output_path=evaluation_root / "pilot_report.json",
            manifest_path=manifest_path,
        )
        _print({"evaluation": evaluation, "report": report})
    elif args.command == "validate":
        bundle = load_bundle(bundle_path)
        bundle.verify_inputs(config, args.raw_root)
        inventory = build_inventory(args.raw_root)
        hydrate_supported_rules(inventory)
        task_records = read_jsonl(tasks)
        validation = validate_generated_records(
            read_jsonl(questions),
            expand_gold_records(task_records),
            semantic_tasks_from_records(task_records),
            read_jsonl(scopes),
            inventory,
            expected_scope_sizes=config.scope_sizes,
            expected_scope_seeds=config.scope_seeds,
            question_generation=read_jsonl(question_generation),
            require_polished_questions=(
                bundle.data.get("question_generation_status") == "polished"
            ),
        )
        assert_valid(validation)
        _print({"bundle_id": bundle.bundle_id, "validation": summarize_validation(validation)})
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    return 0