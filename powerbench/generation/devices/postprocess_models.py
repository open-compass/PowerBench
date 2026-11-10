from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from powerbench.paths import demo_models_dir

DEFAULT_MODELS_DIR = str(demo_models_dir())
DEFAULT_SHEET_NAME = "Equipment List"


EXCLUDE_COLS_DEFAULT = "Device Name,Belonging Project,Installation Location,System Ownership,Device Number"


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _unique_values(series: pd.Series) -> List[str]:
    vals: List[str] = []
    seen = set()
    for v in series.tolist():
        s = _compact_str(v)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        vals.append(s)
    return vals


def summarize_dimensions(xlsx_path: Path, sheet_name: str, exclude_cols: List[str]) -> Dict[str, Any]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    exclude = set([c.strip() for c in exclude_cols if c.strip()])

    cols = [c for c in df.columns.tolist() if _compact_str(c)]
    keep_cols = [c for c in cols if str(c).strip() not in exclude]

    summary: Dict[str, Any] = {
        "source_xlsx": str(xlsx_path),
        "sheet_name": sheet_name,
        "exclude_columns": list(exclude),
        "dimensions": [],
    }

    for col in keep_cols:
        uniq = _unique_values(df[col])
        summary["dimensions"].append(
            {
                "column": str(col),
                "unique_values": uniq,
                "unique_count": len(uniq),
            }
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Organize models/*.xlsx into subfolders and export dimension JSON summaries.")
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory containing xlsx files")
    ap.add_argument("--sheet", type=str, default=DEFAULT_SHEET_NAME, help="Sheet name to read (default: Equipment List)")
    ap.add_argument(
        "--exclude_cols",
        type=str,
        default=EXCLUDE_COLS_DEFAULT,
        help="Comma-separated columns to exclude (default: Device Name,Belonging Project,Installation Location,System Ownership,Device Number)",
    )
    ap.add_argument(
        "--json_name",
        type=str,
        default="dimensions_summary.json",
        help="JSON filename to write inside each subfolder",
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists() or not models_dir.is_dir():
        raise SystemExit(f"models_dir not found or not a directory: {models_dir}")

    exclude_cols = [c.strip() for c in str(args.exclude_cols).split(",") if c.strip()]

    xlsx_files = sorted(models_dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"[INFO] No xlsx files found in: {models_dir}")
        return 0

    for xlsx in xlsx_files:
        subdir = models_dir / xlsx.stem
        subdir.mkdir(parents=True, exist_ok=True)

        target_xlsx = subdir / xlsx.name
        if xlsx.resolve() != target_xlsx.resolve():
            # Move into subdir; if already exists, keep the existing and skip move
            if not target_xlsx.exists():
                shutil.move(str(xlsx), str(target_xlsx))
            else:
                # If same name exists, keep both by renaming the moved one
                alt = subdir / f"{xlsx.stem}__dup{xlsx.suffix}"
                shutil.move(str(xlsx), str(alt))
                target_xlsx = alt

        try:
            summary = summarize_dimensions(target_xlsx, sheet_name=args.sheet, exclude_cols=exclude_cols)
        except Exception as e:
            # Write an error json for debugging
            err_path = subdir / args.json_name
            err_obj = {
                "source_xlsx": str(target_xlsx),
                "error": str(e),
            }
            err_path.write_text(json.dumps(err_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[FAIL] {target_xlsx.name}: {e}")
            continue

        json_path = subdir / args.json_name
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {target_xlsx.name} -> {subdir.name}\\{args.json_name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
