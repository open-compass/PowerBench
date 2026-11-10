from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any


def ensure_repo_root_on_path(model_dir: Path) -> Path:
    from powerbench.paths import repo_root
    root = repo_root(Path(model_dir) if model_dir else Path(__file__))
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


def safe_fs_stem(name: str, *, max_len: int = 160) -> str:
    """Sanitize device names for filenames ( '/', ':' etc. → '_'). Align with per_device stems."""
    s = "" if name is None else str(name).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "device"
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, "_")
    s = re.sub(r"\s+", "_", s).strip("._")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._")
    return s or "device"


def load_rule_functions(model_dir: Path) -> Any:
    """Load pipeline_generate/rule_functions.py (+ derived_metrics_impl if present)."""
    model_dir = Path(model_dir).resolve()
    ensure_repo_root_on_path(model_dir)
    here = model_dir / "pipeline_generate"
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", model_dir.name)[:48] or "model"
    pkg = f"m_{slug}_pipeline_report"
    if pkg not in sys.modules:
        pkg_mod = types.ModuleType(pkg)
        pkg_mod.__path__ = [str(here)]  # type: ignore[attr-defined]
        sys.modules[pkg] = pkg_mod

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    dm = here / "derived_metrics_impl.py"
    if dm.exists():
        _load(f"{pkg}.derived_metrics_impl", dm)
    return _load(f"{pkg}.rule_functions", here / "rule_functions.py")
