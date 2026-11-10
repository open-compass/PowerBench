from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from powerbench.generation.devices import prompt as prompt_module


from powerbench.paths import demo_models_dir, taxonomy_dir, DEFAULT_DEVICE_MODELS_XLSX

DEFAULT_INPUT_XLSX = str(taxonomy_dir() / DEFAULT_DEVICE_MODELS_XLSX)
DEFAULT_OUTPUT_DIR = str(demo_models_dir())


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def build_device_type_path_from_row_first4(row: List[Any]) -> str:
    parts = [_compact_str(row[i]) if i < len(row) else "" for i in range(4)]
    parts = [p for p in parts if p]
    return " -> ".join(parts)


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "unnamed"


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("model output is not a string")

    # Fast path: pure JSON
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    # Search for a JSON object by brace matching
    start = t.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1]
                return json.loads(candidate)
    raise ValueError("could not find a complete JSON object in output")


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 120

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _make_stage1_prompt(device_type_path: str) -> str:
    base = _safe_format_prompt(prompt_module.stage_1_prompt, device_type_path=device_type_path)
    constraints = """

## Hard Constraints (please strictly comply)
1. Do not use placeholder words such as "some/certain/XX/example/test" or fictitious engineering names (e.g. "some converter station", "some project").
2. If a concrete name such as a project/converter station is needed, prefer to extract it directly from the user-input "device type path" and reuse the original wording.
3. Output only JSON, with no additional explanation.
""".strip("\n")
    return base + "\n" + constraints + "\n"


def _make_stage2_prompt(device_type: str, dimension_list_json: str) -> str:
    base = _safe_format_prompt(
        prompt_module.stage_2_prompt,
        device_type=device_type,
        dimension_list=dimension_list_json,
    )
    constraints = """

## Hard Constraints (please strictly comply)
1. Do not use placeholder words such as "some/certain/XX/example/test" or fictitious engineering names (e.g. "some converter station", "some project").
2. Device names must use the real name elements already given in the input (e.g. converter station/DC project name, system location, phase, unit number, primary/standby, etc.), producing names like:
   - Jinping Converter Station +/-800kV High-end Converter Transformer Unit 1
   - Xiangjiaba Converter Station +/-800kV High-end Converter Transformer Pole 1 System
   - Fufeng DC Project +/-800kV High-end Converter Transformer Primary Device
   - Xiluodu Converter Station +/-800kV High-end Converter Transformer Phase A
3. Do not output generalized expressions such as "some high voltage/some converter station/some project".
4. Output only JSON, with no additional explanation.
""".strip("\n")
    return base + "\n" + constraints + "\n"


def _safe_format_prompt(template: str, **kwargs: Any) -> str:
    if not isinstance(template, str):
        template = str(template)

    escaped = template.replace("{", "{{").replace("}", "}}")
    for k in kwargs.keys():
        escaped = escaped.replace("{{" + k + "}}", "{" + k + "}")
    return escaped.format(**kwargs)


def _stage1_to_dimension_list(stage1: Dict[str, Any]) -> List[Dict[str, Any]]:
    dims: List[Dict[str, Any]] = []
    for key in ("Instantiation Dimensions", "Distinguishing Dimensions"):
        for d in stage1.get(key, []) or []:
            name = _compact_str(d.get("Dimension Name"))
            desc = _compact_str(d.get("Dimension Description"))
            vals = d.get("Recommended Values", [])
            if not name:
                continue
            if not isinstance(vals, list):
                vals = []
            vals = [_compact_str(v) for v in vals if _compact_str(v)]
            dims.append({"Dimension Name": name, "Dimension Description": desc, "Value Range": vals})

    # De-duplicate by Dimension Name, keep first
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for d in dims:
        if d["Dimension Name"] in seen:
            continue
        seen.add(d["Dimension Name"])
        uniq.append(d)
    return uniq


def call_stage1(client: OpenAICompatClient, device_type_path: str, max_retries: int = 3) -> Tuple[str, Dict[str, Any]]:
    prompt_text = _make_stage1_prompt(device_type_path)
    messages = [{"role": "user", "content": prompt_text}]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=0.2)
            js = _extract_first_json_object(raw)
            return raw, js
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Stage1 failed after {max_retries} retries: {last_err}")


def call_stage2(
    client: OpenAICompatClient,
    device_type: str,
    dimension_list: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    dimension_list_json = json.dumps(dimension_list, ensure_ascii=False, indent=2)
    prompt_text = _make_stage2_prompt(device_type=device_type, dimension_list_json=dimension_list_json)
    messages = [{"role": "user", "content": prompt_text}]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=0.2)
            js = _extract_first_json_object(raw)
            return raw, js
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Stage2 failed after {max_retries} retries: {last_err}")


def _records_to_dataframe(records: List[Dict[str, Any]], field_order: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        row: Dict[str, Any] = {}
        for f in field_order:
            row[f] = r.get(f, "")
        # include any extra fields at end (shouldn't happen per prompt constraints)
        for k, v in r.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def save_row_result(
    out_dir: Path,
    row_index_1based: int,
    device_type_path: str,
    stage1_raw: str,
    stage1_json: Dict[str, Any],
    stage2_raw: str,
    stage2_json: Dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device_type = _compact_str(stage2_json.get("Device Type")) or _compact_str(stage1_json.get("Device Type")) or "Device"
    fname = f"{row_index_1based:04d}_{_sanitize_filename(device_type)}.xlsx"
    out_path = out_dir / fname

    device_list = stage2_json.get("Equipment List", []) or []
    fields = stage2_json.get("Equipment List Fields", []) or []
    if not isinstance(fields, list) or not fields:
        # fallback: infer from union keys
        keyset = set()
        for r in device_list:
            if isinstance(r, dict):
                keyset.update(r.keys())
        fields = ["Device Name"] + [k for k in sorted(keyset) if k != "Device Name"]

    df = _records_to_dataframe(device_list, field_order=[_compact_str(f) for f in fields if _compact_str(f)])

    # Save only the final equipment list (single sheet)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Equipment List", index=False)

    return out_path


def process_one_row(
    client: OpenAICompatClient,
    out_dir: Path,
    row_index_1based: int,
    row_list: List[Any],
) -> Optional[Path]:
    device_type_path = build_device_type_path_from_row_first4(row_list)
    if not device_type_path:
        return None

    stage1_raw, stage1_json = call_stage1(client, device_type_path=device_type_path)
    device_type = _compact_str(stage1_json.get("Device Type")) or device_type_path.split("->")[-1].strip()
    dimension_list = _stage1_to_dimension_list(stage1_json)

    stage2_raw, stage2_json = call_stage2(client, device_type=device_type, dimension_list=dimension_list)

    out_path = save_row_result(
        out_dir=out_dir,
        row_index_1based=row_index_1based,
        device_type_path=device_type_path,
        stage1_raw=stage1_raw,
        stage1_json=stage1_json,
        stage2_raw=stage2_raw,
        stage2_json=stage2_json,
    )
    return out_path


def _parse_rows_spec(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a or not b:
                raise ValueError(f"Bad rows range: {p!r}")
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            for x in range(start, end + 1):
                out.add(x)
        else:
            out.add(int(p))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Expand each row of device_models.xlsx using two-stage prompts in prompt.py")
    ap.add_argument("--input", default=DEFAULT_INPUT_XLSX, help="input xlsx path")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="output directory for per-row xlsx files")
    ap.add_argument("--sheet", default=None, help="sheet name or index (default: first sheet)")
    ap.add_argument("--skip_rows", type=int, default=1, help="skip first N rows (default: 1, because first row is header)")
    ap.add_argument("--max_rows", type=int, default=0, help="process at most N rows (0 = all)")
    ap.add_argument("--start_row_1based", type=int, default=1, help="treat the first data row as this 1-based index")
    ap.add_argument(
        "--rows",
        type=str,
        default="14,15,82",
        help='Only process the specified Excel rows (1-based, including the header). Supports "14,15" or "10-20" or "2,5-7"; empty means process all.',
    )
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrency (default 4)")
    ap.add_argument("--base_url",type=str, default="https://api.your-provider.com/v1",help="Base URL of the LLM (passed to run_pipeline's base_url parameter)")
    ap.add_argument("--model", type=str, default="deepseek-v3",)
    ap.add_argument("--api-key",type=str,default="YOUR_API_KEY",help="API Key of the LLM (passed to run_pipeline's api_key parameter)",)
    args = ap.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Please pass --api-key before running.")

    client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # header=None: treat everything as raw cells; we only need first 4 columns
    sheet_arg = args.sheet
    if isinstance(sheet_arg, str) and sheet_arg.strip() == "":
        sheet_arg = None

    if sheet_arg is None:
        # pandas: sheet_name=None returns dict[name->DataFrame]; take first sheet
        book = pd.read_excel(in_path, sheet_name=None, header=None)
        if not isinstance(book, dict) or not book:
            raise SystemExit("No worksheets found in input Excel.")
        first_name = next(iter(book.keys()))
        df = book[first_name]
        print(f"[INFO] Using first worksheet: {first_name}")
    else:
        df = pd.read_excel(in_path, sheet_name=sheet_arg, header=None)
    if args.skip_rows:
        df = df.iloc[args.skip_rows :, :]
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows, :]

    target_rows = _parse_rows_spec(args.rows)
    if target_rows:
        print(f"[INFO] Only processing rows: {target_rows}")

    rows: List[Tuple[int, List[Any]]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        row_index_1based = args.start_row_1based + args.skip_rows + i
        if target_rows and row_index_1based not in target_rows:
            continue
        rows.append((row_index_1based, row.tolist()))

    concurrency = max(1, int(args.concurrency))
    ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_map = {
            ex.submit(process_one_row, client, out_dir, row_index_1based, row_list): row_index_1based
            for (row_index_1based, row_list) in rows
        }
        for fut in as_completed(fut_map):
            row_index_1based = fut_map[fut]
            try:
                out_path = fut.result()
                if out_path is None:
                    print(f"[SKIP] row {row_index_1based}: empty first-4-cols")
                    continue
                ok += 1
                print(f"[OK] row {row_index_1based} -> {out_path}")
            except Exception as e:
                failed += 1
                print(f"[FAIL] row {row_index_1based}: {e}")

    print("Done.")
    if failed:
        print(f"Summary: OK={ok}, FAIL={failed} (see logs above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
