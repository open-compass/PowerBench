from __future__ import annotations
import importlib.util
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "pyproject.toml").is_file() and (cand / "powerbench").is_dir():
            return cand
    raise RuntimeError(f"Cannot locate repo root above {start}")


def main() -> int:
    p = Path(__file__).resolve()
    root = _find_repo_root(p)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    gen_py = p.parent / "generate_per_device_datasets.py"
    spec = importlib.util.spec_from_file_location("model_local_gen", str(gen_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generator: {gen_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return int(mod.main())  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
