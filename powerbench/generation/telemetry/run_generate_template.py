from __future__ import annotations
import sys
from pathlib import Path

from powerbench.paths import repo_root


def main() -> int:
    here = Path(__file__).resolve()
    root = repo_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from powerbench.generation.telemetry.pipeline_orchestrator import main as _shared_main

    return int(_shared_main(default_model_dir=here.parents[1]))


if __name__ == "__main__":
    raise SystemExit(main())
