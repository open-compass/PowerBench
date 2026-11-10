from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _ensure_root() -> Path:
    from powerbench.paths import repo_root

    root = repo_root(Path(__file__))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main(argv: list[str] | None = None) -> int:
    _ensure_root()
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="powerbench",
        description="PowerBench: data generation framework + HeterogeneousQA evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generation shortcuts ---
    gen = sub.add_parser("generate", help="Run telemetry generation for a device type")
    gen.add_argument(
        "--type-dir",
        type=Path,
        default=None,
        help="Path to data/demo/models/<type> (default: demo 0002)",
    )

    docs = sub.add_parser("documents", help="Generate fault/check/archive Markdown reports")
    docs.add_argument("--type", choices=("fault", "check", "archive"), default="fault")
    docs.add_argument("--model_dir", type=Path, default=None)
    docs.add_argument("--device_index", type=int, default=None)
    docs.add_argument("--jobs", "-j", type=int, default=1)

    # --- tasks (HeterogeneousQA) ---
    tasks = sub.add_parser("tasks", help="HeterogeneousQA: build / polish / run / evaluate / validate")
    tasks.add_argument("tasks_argv", nargs=argparse.REMAINDER, help="Arguments forwarded to powerbench.tasks")

    args, unknown = parser.parse_known_args(argv)

    if args.command == "generate":
        from powerbench.paths import DEFAULT_DEMO_TYPE, demo_models_dir
        from powerbench.generation.telemetry.pipeline_orchestrator import main as gen_main

        type_dir = args.type_dir or (demo_models_dir() / DEFAULT_DEMO_TYPE)
        return int(gen_main(unknown, default_model_dir=Path(type_dir)))

    if args.command == "documents":
        from powerbench.paths import DEFAULT_DEMO_TYPE, demo_models_dir
        from powerbench.generation.documents import run_report

        model_dir = args.model_dir or (demo_models_dir() / DEFAULT_DEMO_TYPE)
        forwarded = [
            "--type",
            args.type,
            "--model_dir",
            str(model_dir),
            "--jobs",
            str(args.jobs),
        ]
        if args.device_index is not None:
            forwarded += ["--device_index", str(args.device_index)]
        forwarded += unknown
        sys.argv = ["run_report", *forwarded]
        return int(run_report.main())

    if args.command == "tasks":
        from powerbench.tasks.cli import main as tasks_main

        # REMAINDER may start with '--' separator
        rest = list(args.tasks_argv)
        if rest and rest[0] == "--":
            rest = rest[1:]
        return int(tasks_main(rest + unknown))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
