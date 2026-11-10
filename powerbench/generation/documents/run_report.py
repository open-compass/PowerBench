from __future__ import annotations
import argparse
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from powerbench.paths import DEFAULT_DEMO_TYPE, demo_models_dir, repo_root


def _load_main(script: Path):
    name = f"pb_report_{script.stem}"
    spec = importlib.util.spec_from_file_location(name, str(script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _discover_device_indices(model_root: Path) -> List[int]:
    per = model_root / "per_device"
    if not per.is_dir():
        return []
    out: List[int] = []
    for p in sorted(per.glob("*.json")):
        if p.name.endswith("_report.json"):
            continue
        head = p.name.split("_", 1)[0]
        if head.isdigit():
            out.append(int(head))
    return out


def _child_env(api_key: str) -> dict:
    env = os.environ.copy()
    key = (api_key or "").strip()
    if key:
        env["OPENAI_API_KEY"] = key
        env["LLM_API_KEY"] = key
    return env


def _run_one(cmd: List[str], idx: int, env: dict) -> Tuple[int, int]:
    print(f"[PARALLEL] start device_index={idx:04d}", flush=True)
    r = subprocess.run(cmd, env=env)
    print(f"[PARALLEL] done  device_index={idx:04d} rc={r.returncode}", flush=True)
    return idx, int(r.returncode)


def main() -> int:
    root = repo_root(Path(__file__))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    this_dir = Path(__file__).resolve().parent
    default_model = demo_models_dir() / DEFAULT_DEMO_TYPE
    ap = argparse.ArgumentParser(description="Generate Markdown reports for a device type (no LLM by default).")
    ap.add_argument("--type", choices=("fault", "check", "archive"), default="fault")
    ap.add_argument("--model_dir", type=str, default=str(default_model), help="model dir under data/demo/models")
    ap.add_argument("--device_index", type=int, default=None)
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--api-key", type=str, default="", help="or set LLM_API_KEY / OPENAI_API_KEY")
    ap.add_argument("--base_url", type=str, default="")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--save_placeholders_copy", action="store_true", default=False)
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="parallel device workers when generating all devices (default 1 = sequential)",
    )
    args, unknown = ap.parse_known_args()

    model_root = Path(args.model_dir).resolve()
    script_map = {
        "fault": this_dir / "generate_fault_report_md.py",
        "check": this_dir / "generate_check_report_md.py",
        "archive": this_dir / "generate_device_archive_md.py",
    }
    script = script_map[args.type]
    key = (args.api_key or "").strip()
    child_env = _child_env(key)

    def _extra_flags() -> List[str]:
        flags: List[str] = ["--model_dir", str(model_root)]
        if args.use_llm:
            flags.append("--use_llm")
            # Pass key via env only (avoid argv / process-list leak).
            if args.base_url.strip():
                flags += ["--base_url", args.base_url.strip()]
            if args.model.strip():
                flags += ["--model", args.model.strip()]
            flags += ["--temperature", str(args.temperature), "--max_new_tokens", str(args.max_new_tokens)]
        if args.save_placeholders_copy:
            flags.append("--save_placeholders_copy")
        flags += list(unknown)
        return flags

    jobs = max(1, int(args.jobs))

    if args.device_index is None and jobs > 1:
        indices = _discover_device_indices(model_root)
        if not indices:
            raise SystemExit(f"no per_device devices under {model_root / 'per_device'}")
        extra = _extra_flags()
        cmds = [
            ([sys.executable, str(script), "--device_index", str(idx)] + extra, idx) for idx in indices
        ]
        print(f"[PARALLEL] type={args.type} devices={len(cmds)} jobs={jobs}", flush=True)
        ok = failed = 0
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_run_one, cmd, idx, child_env) for cmd, idx in cmds]
            for fut in as_completed(futs):
                _idx, rc = fut.result()
                if rc == 0:
                    ok += 1
                else:
                    failed += 1
        print(f"[PARALLEL] done. success={ok}, failed={failed}", flush=True)
        return 0 if failed == 0 else 1

    # In-process: seed env so child main() can read the key without argv.
    if key:
        os.environ["OPENAI_API_KEY"] = key
        os.environ["LLM_API_KEY"] = key

    mod = _load_main(script)
    argv = ["x", "--model_dir", str(model_root)]
    if args.device_index is not None:
        argv += ["--device_index", str(args.device_index)]
    if args.use_llm:
        argv.append("--use_llm")
        if args.base_url.strip():
            argv += ["--base_url", args.base_url.strip()]
        if args.model.strip():
            argv += ["--model", args.model.strip()]
        argv += ["--temperature", str(args.temperature), "--max_new_tokens", str(args.max_new_tokens)]
    if args.save_placeholders_copy:
        argv.append("--save_placeholders_copy")
    if jobs > 1:
        argv += ["--jobs", str(jobs)]
    argv += list(unknown)
    sys.argv = argv
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
