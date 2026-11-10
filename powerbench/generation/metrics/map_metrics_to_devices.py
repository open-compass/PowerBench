from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trigger_satisfied(row: pd.Series, trigger: Dict[str, Any]) -> bool:
    field = _compact_str(trigger.get("field_name"))
    if not field:
        return False
    if field not in row.index:
        return False
    cell = _compact_str(row.get(field))
    values = trigger.get("field_values", [])
    if not isinstance(values, list) or not values:
        return False
    values_norm = {_compact_str(v) for v in values if _compact_str(v)}
    if not values_norm:
        return False
    return cell in values_norm


def _special_metric_applies(row: pd.Series, metric: Dict[str, Any]) -> bool:
    triggers = metric.get("trigger_conditions", [])
    # If no triggers provided, treat as applies to all (conservative for "special" metrics is debatable,
    # but this avoids accidentally dropping a metric that was meant to be global).
    if not triggers:
        return True
    if not isinstance(triggers, list):
        return False
    return all(_trigger_satisfied(row, t) for t in triggers if isinstance(t, dict))


def _get_metrics(stage1: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    common = []
    for m in stage1.get("common_maintenance_metrics", []) or []:
        if isinstance(m, dict):
            name = _compact_str(m.get("metric_name"))
            if name:
                common.append(name)

    special = []
    for m in stage1.get("special_maintenance_metrics", []) or []:
        if isinstance(m, dict):
            name = _compact_str(m.get("metric_name"))
            if name:
                special.append(m)
    return common, special


def _rule_condition_satisfied(row: pd.Series, cond: Any) -> bool:
    """
    Stage2's applicable_conditions may be:
    - [] or None: applies to all
    - [{"field_name": "...", "field_values": ["..."]}, ...]
    - a legacy string like "cooling_method is forced oil-water cooling" or "applies to all"
    """
    if cond is None or cond == []:
        return True

    # structured array
    if isinstance(cond, list):
        ok = True
        for item in cond:
            if not isinstance(item, dict):
                continue
            if not _trigger_satisfied(row, item):
                ok = False
                break
        return ok

    # legacy string (best-effort)
    if isinstance(cond, str):
        s = cond.strip()
        if not s:
            return True
        if "applies to all" in s.lower() or s.lower() == "all":
            return True
        # Try pattern: field_name is field_value
        if " is " in s:
            left, right = s.split(" is ", 1)
            field = left.strip()
            val = right.strip().strip(".,;")
            if field and field in row.index:
                return _compact_str(row.get(field)) == val
        return False

    return False


def _load_rules(stage2_path: Path) -> List[Dict[str, Any]]:
    try:
        stage2 = _load_json(stage2_path)
    except Exception:
        return []
    rules = stage2.get("anomaly_rules") or stage2.get("maintenance_rules") or []
    if isinstance(rules, list):
        return [r for r in rules if isinstance(r, dict)]
    return []


def _load_metric_catalog_map(catalog_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load metric_catalog.json list and return:
    - metric_name -> db_field_name
    - db_field_name -> metric_name
    """
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(data, list):
        return {}, {}
    name2db: Dict[str, str] = {}
    db2name: Dict[str, str] = {}
    for m in data:
        if not isinstance(m, dict):
            continue
        cn = _compact_str(m.get("metric_name"))
        db = _compact_str(m.get("db_field_name"))
        if cn and db:
            name2db[cn] = db
            db2name[db] = cn
    return name2db, db2name


def _load_derived_metrics(derived_path: Path) -> List[Dict[str, Any]]:
    """
    Load derived_metrics.json and normalize each item as:
    {
      "derived_name": str,
      "derived_key": str,
      "atomic_indicator_names": List[str],
      "atomic_db_fields": List[str],
    }
    """
    try:
        dm = json.loads(derived_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = dm.get("derived_metrics") if isinstance(dm, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        dname = _compact_str(it.get("derived_name"))
        dkey = _compact_str(it.get("derived_key"))
        atomic = it.get("atomic_metrics", []) or []
        atomic_names: List[str] = []
        atomic_db: List[str] = []
        if isinstance(atomic, list):
            for a in atomic:
                if not isinstance(a, dict):
                    continue
                n = _compact_str(a.get("indicator_name"))
                db = _compact_str(a.get("db_field"))
                if n:
                    atomic_names.append(n)
                if db:
                    atomic_db.append(db)
        if dname and dkey:
            out.append(
                {
                    "derived_name": dname,
                    "derived_key": dkey,
                    "atomic_indicator_names": list(dict.fromkeys([x for x in atomic_names if x])),
                    "atomic_db_fields": list(dict.fromkeys([x for x in atomic_db if x])),
                }
            )
    return out


def _extract_rule_derived_names(rule: Dict[str, Any]) -> List[str]:
    """
    Compatibility helper: some stage2 files may list derived metrics under derived_metric_definitions,
    but in the current scheme we primarily judge via "input_metrics" + derived metric computability.
    This function is retained to avoid call-site errors; it returns an empty list by default.
    """
    return []


def _split_semicolon(s: str) -> List[str]:
    s2 = _compact_str(s)
    if not s2:
        return []
    parts = [p.strip() for p in s2.split(";")]
    return [p for p in parts if p]


def process_folder(
    folder: Path,
    stage1_json_name: str,
    stage2_json_name: str,
    sheet_name: Optional[str],
    overwrite: bool,
    backup: bool,
    verbose: bool,
) -> str:
    prefix = f"[{folder.name}]"
    stage1_path = folder / stage1_json_name
    if not stage1_path.exists():
        return f"{prefix} SKIP: missing {stage1_json_name}"

    xlsx_files = sorted(folder.glob("*.xlsx"))
    if not xlsx_files:
        return f"{prefix} SKIP: no xlsx"
    xlsx_path = xlsx_files[0]

    stage1 = _load_json(stage1_path)
    common_names, special_metrics = _get_metrics(stage1)

    # stage2 (rules) optional
    stage2_path = folder / stage2_json_name
    rules = _load_rules(stage2_path) if stage2_path.exists() else []

    # metric catalog + derived metrics (optional)
    name2db: Dict[str, str] = {}
    if (folder / "metric_catalog.json").exists():
        name2db, _ = _load_metric_catalog_map(folder / "metric_catalog.json")
    derived_items: List[Dict[str, Any]] = []
    if (folder / "derived_metrics.json").exists():
        derived_items = _load_derived_metrics(folder / "derived_metrics.json")
    derived_by_name: Dict[str, Dict[str, Any]] = {
        d["derived_name"]: d for d in derived_items if isinstance(d, dict) and d.get("derived_name")
    }

    # read sheet
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name or "Device List")
        used_sheet = sheet_name or "Device List"
    except Exception:
        df = pd.read_excel(xlsx_path, sheet_name=0)
        used_sheet = "<first>"

    if df.empty:
        return f"{prefix} SKIP: empty sheet ({used_sheet})"

    # If this script has been run before, clear "metrics_common" and all columns after it (keep only original columns)
    if "metrics_common" in df.columns:
        idx = df.columns.get_loc("metrics_common")
        df = df.iloc[:, :idx]

    # per-row mapping (rebuild all mapping columns as required)
    common_col = []
    special_col = []
    all_col = []
    all_db_fields_col = []
    derived_all_col = []
    derived_keys_all_col = []
    rule_ids_col = []
    rule_brief_col = []

    for _, row in df.iterrows():
        # 1) Atomic maintenance metrics: from rule_metric_stage1.json
        specials = []
        for sm in special_metrics:
            if _special_metric_applies(row, sm):
                name = _compact_str(sm.get("metric_name"))
                if name:
                    specials.append(name)
        specials = list(dict.fromkeys(specials))  # keep order, dedupe
        all_metrics = common_names + [s for s in specials if s not in common_names]

        common_col.append(";".join(common_names))
        special_col.append(";".join(specials))
        all_col.append(";".join(all_metrics))

        # map atomic metric names -> db fields (best-effort; only those present in metric_catalog.json)
        db_fields = [name2db.get(m, "") for m in all_metrics]
        db_fields = [x for x in db_fields if x]
        all_db_fields_col.append(";".join(list(dict.fromkeys(db_fields))))

        # 2) Derived metrics: only considered "computable" when all required atomic_metrics are in the atomic metrics list
        available_atomic = set(all_metrics)
        derived_names: List[str] = []
        derived_keys: List[str] = []
        for d in derived_items:
            if not isinstance(d, dict):
                continue
            dn = _compact_str(d.get("derived_name"))
            dk = _compact_str(d.get("derived_key"))
            deps = d.get("atomic_indicator_names", []) or []
            if not dn or not dk or not isinstance(deps, list) or not deps:
                continue
            deps_norm = { _compact_str(x) for x in deps if _compact_str(x) }
            if deps_norm and deps_norm.issubset(available_atomic):
                derived_names.append(dn)
                derived_keys.append(dk)
        derived_names = list(dict.fromkeys(derived_names))
        derived_keys = list(dict.fromkeys(derived_keys))
        derived_all_col.append(";".join(derived_names))
        derived_keys_all_col.append(";".join(derived_keys))

        # 3) Rule mapping: must simultaneously satisfy
        #    - applicable_conditions (_rule_condition_satisfied)
        #    - input_metrics fully available: in atomic metrics or derived metrics
        rule_ids: List[str] = []
        rule_briefs: List[str] = []
        for r in rules:
            cond = r.get("applicable_conditions")
            if not _rule_condition_satisfied(row, cond):
                continue
            # check whether input metrics are available
            inputs = r.get("input_metrics", []) or []
            ok_inputs = True
            if isinstance(inputs, list) and inputs:
                for nm in inputs:
                    name_cn = _compact_str(nm)
                    if not name_cn:
                        continue
                    if (name_cn not in available_atomic) and (name_cn not in derived_names):
                        ok_inputs = False
                        break
            if not ok_inputs:
                continue

            rid = _compact_str(r.get("rule_id"))
            obj = _compact_str(r.get("object_name"))
            if rid:
                rule_ids.append(rid)
                rule_briefs.append(f"{rid}:{obj}" if obj else rid)
        # dedupe
        rule_ids = list(dict.fromkeys([x for x in rule_ids if x]))
        rule_briefs = list(dict.fromkeys([x for x in rule_briefs if x]))
        rule_ids_col.append(";".join(rule_ids))
        rule_brief_col.append(";".join(rule_briefs))

    df["metrics_common"] = common_col
    df["metrics_special_by_trigger"] = special_col
    df["metrics_all"] = all_col
    if name2db:
        df["metrics_db_fields_all"] = all_db_fields_col
    if derived_items:
        df["derived_metrics_all_computable"] = derived_all_col
        df["derived_metrics_key_all"] = derived_keys_all_col
    if rules:
        df["anomaly_rule_ids_all"] = rule_ids_col
        df["anomaly_rules_id_and_object"] = rule_brief_col

    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = xlsx_path.with_suffix(f".bak_{ts}.xlsx")
        shutil.copy2(xlsx_path, bak)
        if verbose:
            print(f"{prefix} backup -> {bak.name}")

    if not overwrite:
        out_path = xlsx_path.with_name(xlsx_path.stem + "_with_metrics.xlsx")
    else:
        out_path = xlsx_path

    # write back (single sheet). If your xlsx contains multiple sheets, this will overwrite the workbook.
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=(sheet_name or "Device List"), index=False)

    return f"{prefix} OK: wrote metrics columns to {out_path.name} (rows={len(df)})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Map stage1 metrics onto each device row and append metric columns to xlsx.")
    ap.add_argument(
        "--models_dir",
        type=str,
        default=str(__import__("powerbench.paths", fromlist=["demo_models_dir"]).demo_models_dir()),
        help="models directory (default: data/demo/models)",
    )
    ap.add_argument("--stage1_json", type=str, default="rule_metric_stage1.json", help="stage1 json filename")
    ap.add_argument("--stage2_json", type=str, default="rule_metric_stage2.json", help="stage2 json filename (for rule_id mapping)")
    ap.add_argument("--sheet", type=str, default="Device List", help="xlsx sheet name (default: Device List; fallback: first sheet)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite original xlsx (recommended with --backup)")
    ap.add_argument("--no_backup", action="store_true", help="do not create backup when overwriting")
    ap.add_argument("--only_folder", type=str, default="", help="only process one folder (name or full path)")
    ap.add_argument("--start_from", type=int, default=0, help="skip first N folders (sorted order)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")

    subdirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])

    only = str(args.only_folder).strip()
    if only:
        p = Path(only)
        if p.is_absolute():
            subdirs = [p]
        else:
            subdirs = [models_dir / only]
        if not subdirs[0].exists():
            matches = [d for d in subdirs if d.exists()]
            if not matches:
                matches = [d for d in sorted([p for p in models_dir.iterdir() if p.is_dir()]) if d.name == only]
            if not matches:
                raise SystemExit(f"only_folder not found: {only}")
            subdirs = [matches[0]]
    else:
        start = max(0, int(args.start_from))
        if start:
            subdirs = subdirs[start:]

    backup = not bool(args.no_backup)
    ok = 0
    for d in subdirs:
        msg = process_folder(
            d,
            stage1_json_name=str(args.stage1_json),
            stage2_json_name=str(args.stage2_json),
            sheet_name=str(args.sheet) if args.sheet else None,
            overwrite=bool(args.overwrite),
            backup=backup,
            verbose=bool(args.verbose),
        )
        print(msg)
        if " OK:" in msg:
            ok += 1

    print(f"Done. processed_ok={ok}, total_folders_considered={len(subdirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
