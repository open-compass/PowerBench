from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _strip(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def load_derived_name_to_atomics(dm_path: Path) -> Dict[str, List[str]]:
    raw = json.loads(dm_path.read_text(encoding="utf-8"))
    arr = raw.get("derived_metrics")
    if not isinstance(arr, list):
        return {}
    m: Dict[str, List[str]] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        dn = _strip(it.get("derived_name"))
        if not dn:
            continue
        atoms = it.get("atomic_metrics")
        names: List[str] = []
        if isinstance(atoms, list):
            for am in atoms:
                if not isinstance(am, dict):
                    continue
                inn = _strip(am.get("indicator_name"))
                if inn:
                    names.append(inn)
        if names:
            m[dn] = names
    return m


def expand_inputs(inp: Any, dm_map: Dict[str, List[str]]) -> Tuple[List[Any], bool]:
    if not isinstance(inp, list):
        return [], False
    out: List[Any] = []
    changed = False
    for x in inp:
        if not isinstance(x, str):
            out.append(x)
            continue
        s = _strip(x)
        if s in dm_map:
            changed = True
            for a in dm_map[s]:
                if a not in out:
                    out.append(a)
        else:
            out.append(x)
    return out, changed


def process_model(model_dir: Path, dry_run: bool) -> Tuple[int, int]:
    """Returns (number of rules where input_metrics was replaced, total expansion count)"""
    dm_path = model_dir / "derived_metrics.json"
    st_path = model_dir / "rule_metric_stage2.json"
    if not dm_path.is_file() or not st_path.is_file():
        return 0, 0
    dm_map = load_derived_name_to_atomics(dm_path)
    if not dm_map:
        return 0, 0

    data = json.loads(st_path.read_text(encoding="utf-8"))
    rules = data.get("anomaly_rules")
    if not isinstance(rules, list):
        return 0, 0

    rules_changed = 0
    expansions = 0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        inp = rule.get("input_metrics")
        new_inp, ch = expand_inputs(inp, dm_map)
        if not ch:
            continue
        rules_changed += 1
        if isinstance(inp, list):
            expansions += sum(1 for x in inp if isinstance(x, str) and _strip(x) in dm_map)
        rule["input_metrics"] = new_inp

    if rules_changed and not dry_run:
        st_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return rules_changed, expansions


def main() -> int:
    ap = argparse.ArgumentParser(description="Expand stage2 input_metrics using derived metric atomic indicator names")
    ap.add_argument(
        "--models-root",
        type=str,
        default=str(__import__("powerbench.paths", fromlist=["demo_models_dir"]).demo_models_dir()),
        help="models root directory (data/demo/models)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count only, do not write back to files")
    args = ap.parse_args()
    root = Path(args.models_root)
    if not root.is_dir():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 1

    total_rules = 0
    total_models = 0
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        n_rules, n_exp = process_model(p, bool(args.dry_run))
        if n_rules:
            total_models += 1
            total_rules += n_rules
            print(f"{p.name}: updated {n_rules} rule input_metrics (approx. {n_exp} expansions)")
    suffix = " (dry-run, not written)" if args.dry_run else ""
    print(f"Done{suffix}: {total_models} model directories modified, {total_rules} rules affected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
