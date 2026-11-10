from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional


def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "powerbench").is_dir():
            return cand
    raise RuntimeError(f"Cannot locate repo root above {start}")


def main(argv: Optional[List[str]] = None) -> int:
    here = Path(__file__).resolve()
    root = _find_repo_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from powerbench.generation.telemetry.pipeline_orchestrator import main as _shared_main

    model_dir = here.parents[1]
    return int(_shared_main(argv, default_model_dir=model_dir))


if __name__ == "__main__":
    raise SystemExit(main())
