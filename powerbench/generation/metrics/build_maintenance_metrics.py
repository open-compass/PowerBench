from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests import HTTPError

from powerbench.generation.metrics import rule_metric_prompt as prompt_module


from powerbench.paths import demo_models_dir

DEFAULT_MODELS_DIR = str(demo_models_dir())
DEFAULT_SHEET_NAME = "Device List"
DEFAULT_DIMENSIONS_JSON = "dimensions_summary.json"


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 40, max_cols: int = 30) -> str:
    """
    Avoid external deps (tabulate). Produce a simple pipe table.
    """
    if df is None or df.empty:
        return "|(empty)|\n|---|\n"

    # Limit size to keep prompts reasonable
    df2 = df.copy()
    if max_cols and df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    if max_rows and df2.shape[0] > max_rows:
        df2 = df2.iloc[:max_rows, :]

    cols = [str(c) for c in df2.columns.tolist()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df2.iterrows():
        vals = [_compact_str(v).replace("\n", " ") for v in r.tolist()]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("model output is not a string")
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
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
                return json.loads(t[start : i + 1])
    raise ValueError("could not find a complete JSON object in output")


def _safe_format_prompt(template: str, **kwargs: Any) -> str:
    """
    rule_metric_prompt.py contains many JSON example braces; safely handle str.format collisions.
    """
    if not isinstance(template, str):
        template = str(template)
    escaped = template.replace("{", "{{").replace("}", "}}")
    for k in kwargs.keys():
        escaped = escaped.replace("{{" + k + "}}", "{" + k + "}")
    return escaped.format(**kwargs)


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 1800

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            # Some gateways reject unknown fields; max_tokens is the most widely accepted.
            payload["max_tokens"] = int(max_new_tokens)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except HTTPError as e:
            # Print response body for 4xx/5xx diagnosis (e.g., context_length_exceeded)
            try:
                body = r.text  # type: ignore[name-defined]
            except Exception:
                body = "<no response body>"
            raise RuntimeError(f"HTTP error calling LLM: {e}; response_body={body[:2000]}") from e

        data = r.json()
        return data["choices"][0]["message"]["content"]


def call_stage1(
    client: OpenAICompatClient,
    device_type: str,
    device_table_markdown: str,
    indicator_summary_json: str,
    temperature: float,
    max_new_tokens: Optional[int],
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    prompt_text = _safe_format_prompt(
        prompt_module.stage_1_prompt,
        device_type=device_type,
        device_table_markdown=device_table_markdown,
        indicator_summary_json=indicator_summary_json,
    )
    messages = [{"role": "user", "content": prompt_text}]
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            js = _extract_first_json_object(raw)
            return raw, js
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Stage1 failed after {max_retries} retries: {last_err}")


def call_stage2(
    client: OpenAICompatClient,
    device_type: str,
    stage1_output_json: str,
    temperature: float,
    max_new_tokens: Optional[int],
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    prompt_text = _safe_format_prompt(
        prompt_module.stage_2_prompt,
        device_type=device_type,
        stage1_output_json=stage1_output_json,
    )
    messages = [{"role": "user", "content": prompt_text}]
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            js = _extract_first_json_object(raw)
            return raw, js
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Stage2 failed after {max_retries} retries: {last_err}")


def process_one_folder(
    client: OpenAICompatClient,
    sub: Path,
    sheet_name: str,
    dimensions_json_name: str,
    max_rows_md: int,
    max_cols_md: int,
    temperature: float,
    max_new_tokens: Optional[int],
    verbose: bool,
) -> str:
    prefix = f"[{sub.name}]"
    if verbose:
        print(f"{prefix} start")
    xlsx_files = sorted(sub.glob("*.xlsx"))
    if not xlsx_files:
        return f"[SKIP] {sub.name}: no xlsx"
    xlsx_path = xlsx_files[0]

    # device type: prefer folder name without leading index like "0077_xxx"
    device_type = sub.name
    if "_" in device_type and device_type.split("_", 1)[0].isdigit():
        device_type = device_type.split("_", 1)[1]

    if verbose:
        print(f"{prefix} read xlsx: {xlsx_path.name} (sheet={sheet_name})")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    device_md = dataframe_to_markdown_table(df, max_rows=max_rows_md, max_cols=max_cols_md)

    dim_path = sub / dimensions_json_name
    if dim_path.exists():
        if verbose:
            print(f"{prefix} read indicator summary: {dim_path.name}")
        indicator_summary_json = dim_path.read_text(encoding="utf-8")
    else:
        if verbose:
            print(f"{prefix} indicator summary not found: {dimensions_json_name}")
        indicator_summary_json = json.dumps({"note": "dimensions_summary.json not found"}, ensure_ascii=False)

    if verbose:
        print(f"{prefix} stage1 calling LLM...")
    s1_raw, s1_json = call_stage1(
        client,
        device_type=device_type,
        device_table_markdown=device_md,
        indicator_summary_json=indicator_summary_json,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    stage1_path = sub / "rule_metric_stage1.json"
    stage1_path.write_text(json.dumps(s1_json, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"{prefix} stage1 saved: {stage1_path.name}")

    if verbose:
        print(f"{prefix} stage2 calling LLM...")
    s2_raw, s2_json = call_stage2(
        client,
        device_type=device_type,
        stage1_output_json=json.dumps(s1_json, ensure_ascii=False, indent=2),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    stage2_path = sub / "rule_metric_stage2.json"
    stage2_path.write_text(json.dumps(s2_json, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"{prefix} done")
    return f"[OK] {sub.name}: saved {stage1_path.name}, {stage2_path.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build maintenance metrics (2-stage) for each models subfolder.")
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--sheet", type=str, default=DEFAULT_SHEET_NAME)
    ap.add_argument("--dimensions_json", type=str, default=DEFAULT_DIMENSIONS_JSON)
    ap.add_argument("--max_rows_md", type=int, default=40, help="Max rows included in markdown device table")
    ap.add_argument("--max_cols_md", type=int, default=30, help="Max cols included in markdown device table")
    ap.add_argument("--concurrency", type=int, default=1, help="Concurrency (default 4)")
    ap.add_argument(
        "--only_folder",
        type=str,
        default="",
        help="Only process a single specified subfolder (subfolder name or full path). Ignores --start_from when set.",
    )
    ap.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Which subfolder to start processing from (skip the first N; in sorted order, 0 means no skip)",
    )
    ap.add_argument("--verbose", action="store_true", help="Print staged progress logs for each subfolder")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0.2)")
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Limit model output length (default 16384); passes both max_tokens and max_completion_tokens",
    )
    ap.add_argument("--base_url",type=str, default="https://api.your-provider.com/v1",help="LLM Base URL (passed to run_pipeline's base_url parameter)")
    # ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    # ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    #ap.add_argument("--api-key",type=str,default="YOUR_API_KEY",help="LLM API Key (passed to run_pipeline's api_key parameter)",)
    ap.add_argument("--api-key",type=str,default="YOUR_API_KEY",help="LLM API Key (passed to run_pipeline's api_key parameter)",)

    args = ap.parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Please pass --api-key or set OPENAI_API_KEY.")

    client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")

    subdirs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[INFO] No subfolders under: {models_dir}")
        return 0

    subdirs_sorted = sorted(subdirs)
    only_folder = str(args.only_folder).strip()
    if only_folder:
        p = Path(only_folder)
        if p.is_absolute():
            target = p
        else:
            target = models_dir / only_folder
        if not target.exists() or not target.is_dir():
            matches = [d for d in subdirs_sorted if d.name == only_folder]
            if not matches:
                raise SystemExit(f"only_folder not found: {only_folder} (resolved={target})")
            target = matches[0]
        subdirs_sorted = [target]
        print(f"[INFO] only_folder set; processing single folder: {target.name}")
    else:
        start_from = max(0, int(args.start_from))
        if start_from:
            if start_from >= len(subdirs_sorted):
                print(f"[INFO] start_from={start_from} >= total_folders={len(subdirs_sorted)}; nothing to do.")
                return 0
            subdirs_sorted = subdirs_sorted[start_from:]
            print(f"[INFO] start_from={start_from}; processing {len(subdirs_sorted)} folders (skipped {start_from}).")

    concurrency = max(1, int(args.concurrency))
    total = len(subdirs_sorted)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_map = {
            ex.submit(
                process_one_folder,
                client,
                sub,
                args.sheet,
                args.dimensions_json,
                args.max_rows_md,
                args.max_cols_md,
                float(args.temperature),
                int(args.max_new_tokens) if args.max_new_tokens and int(args.max_new_tokens) > 0 else None,
                args.verbose,
            ): sub
            for sub in subdirs_sorted
        }
        for fut in as_completed(fut_map):
            sub = fut_map[fut]
            try:
                msg = fut.result()
                done += 1
                print(f"{msg}  ({done}/{total})")
            except Exception as e:
                done += 1
                print(f"[FAIL] {sub.name}: {e}  ({done}/{total})")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

