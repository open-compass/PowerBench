from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_catalog_indicator_to_db(model_dir: Path) -> dict[str, str]:
    raw = json.loads((model_dir / "metric_catalog.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = row.get("metric_name")
        db = row.get("database_field")
        if isinstance(name, str) and name and isinstance(db, str) and db:
            out[name] = db
    return out


def _load_rule_id_to_input_indicators(model_dir: Path) -> dict[str, list[str]]:
    stage2 = json.loads((model_dir / "rule_metric_stage2.json").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for r in stage2.get("anomaly_judgment_rules") or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("rule_id", "")).strip().upper()
        if not rid:
            continue
        names = r.get("input_metrics")
        if isinstance(names, list):
            out[rid] = [str(x) for x in names if isinstance(x, str) and x.strip()]
        else:
            out[rid] = []
    return out


def _resolve_db_fields(
    indicators: list[str],
    catalog: dict[str, str],
    *,
    rule_id: str,
    context: str,
) -> list[str]:
    fields: list[str] = []
    for ind in indicators:
        db = catalog.get(ind)
        if db is None:
            raise KeyError(
                f"{context}: rule {rule_id!r} input metric {ind!r} not found in metric_catalog.json metric_name"
            )
        fields.append(db)
    return fields


def _annotate_entries(
    entries: list | None,
    rule_inputs: dict[str, list[str]],
    catalog: dict[str, str],
    *,
    array_name: str,
    path: Path,
) -> int:
    if not isinstance(entries, list):
        return 0
    n = 0
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("rule_id", "")).strip().upper()
        if not rid:
            raise ValueError(f"{path}: {array_name}[{i}] missing rule_id")
        indicators = rule_inputs.get(rid)
        if indicators is None:
            raise KeyError(f"{path}: {array_name}[{i}] rule_id={rid!r} not found in rule_metric_stage2.json")
        fields = _resolve_db_fields(
            indicators,
            catalog,
            rule_id=rid,
            context=f"{path.name} {array_name}[{i}]",
        )
        # Place right after rule_id for readability; remove existing then insert
        entry.pop("input_database_fields", None)
        rid_val = entry.get("rule_id")
        rest = {k: v for k, v in entry.items() if k != "rule_id"}
        entry.clear()
        entry["rule_id"] = rid_val
        entry["input_database_fields"] = fields
        entry.update(rest)
        n += 1
    return n


def process_model_dir(model_dir: Path, *, dry_run: bool) -> None:
    model_dir = model_dir.resolve()
    per_dir = model_dir / "per_device"
    if not per_dir.is_dir():
        print(f"skip (no per_device dir): {model_dir}", file=sys.stderr)
        return

    catalog = _load_catalog_indicator_to_db(model_dir)
    rule_inputs = _load_rule_id_to_input_indicators(model_dir)

    json_files = sorted(per_dir.glob("*.json"))
    if not json_files:
        print(f"skip (no json under per_device): {model_dir}", file=sys.stderr)
        return

    total_sc = 0
    total_nm = 0
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        total_sc += _annotate_entries(
            data.get("scenarios"), rule_inputs, catalog, array_name="scenarios", path=jf
        )
        total_nm += _annotate_entries(
            data.get("near_miss"), rule_inputs, catalog, array_name="near_miss", path=jf
        )
        if not dry_run:
            jf.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"{'[dry-run] ' if dry_run else ''}{jf.name}: annotated scenarios+near_miss entries")

    print(
        f"{'[dry-run] ' if dry_run else ''}done {model_dir.name}: "
        f"scenarios={total_sc}, near_miss={total_nm}, files={len(json_files)}"
    )


def main() -> None:
    from powerbench.paths import DEFAULT_DEMO_TYPE, demo_models_dir

    ap = argparse.ArgumentParser(description="Write input_database_fields for per_device scenarios/near_miss")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=demo_models_dir() / DEFAULT_DEMO_TYPE,
        help="Type directory (requires metric_catalog.json, rule_metric_stage2.json, per_device/)",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate only; do not write files")
    args = ap.parse_args()
    process_model_dir(args.model_dir, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
