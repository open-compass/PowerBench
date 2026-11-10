#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:  # pragma: no cover
    openpyxl = None  # type: ignore


HEADER_ALIASES = {
    "Device Level": ("Device Level", "Level 1 Category", "Major Category"),
    "Device Type": ("Device Type", "Level 2 Category"),
    "Device Name": ("Device Name", "Level 3 Category"),
    "Spec/Model": ("Spec/Model", "Spec Model", "Model", "Device Model"),
    "Main Application": ("Main Application", "Application Scenario", "Purpose"),
}

VERDICT_ORDER = {"Truly Reasonable": 0, "Partially Reasonable": 1, "Unreasonable": 2, "Undeterminable": 3, "Script Error": 4}


def _compact(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("model output is not a string")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    start = t.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("could not find a complete JSON object in output")


def _parse_rows_spec(spec: Optional[str]) -> Optional[List[int]]:
    """Same as expand_device_models._parse_rows_spec: Excel 1-based row numbers (including the header)."""
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
            start, end = int(a.strip()), int(b.strip())
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(p))
    return sorted(x for x in out if x >= 2)


def _extract_responses_text(data: Dict[str, Any]) -> str:
    """Parse OpenAI/DeepSeek Responses API JSON (also tolerate chat.completions)."""
    t = data.get("output_text")
    if isinstance(t, str) and t.strip():
        return t
    chunks: List[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content)
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                chunks.append(str(part.get("text") or ""))
    if chunks:
        return "".join(chunks)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    raise ValueError(f"empty LLM output; keys={list(data)[:12]}")


def _count_web_search_calls(data: Dict[str, Any]) -> int:
    n = 0
    for item in data.get("output") or []:
        if isinstance(item, dict) and str(item.get("type") or "").startswith("web_search"):
            n += 1
    return n


@dataclass
class OpenAICompatClient:
    """DeepSeek-V4-Flash via your-provider: Responses API + server-side web_search."""

    api_key: str
    base_url: str
    model: str
    timeout_s: int = 300
    enable_search: bool = True
    forced_search: bool = True
    last_search_calls: int = 0

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _response_urls(self) -> List[str]:
        root = self.base_url.rstrip("/")
        urls = [root + "/responses"]
        # The official DeepSeek SDK commonly uses https://api.deepseek.com/responses (no /v1)
        if root.endswith("/v1"):
            alt = root[: -len("/v1")] + "/responses"
            if alt not in urls:
                urls.append(alt)
        return urls

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        instructions = ""
        user_parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = m.get("content") or ""
            if role == "system":
                instructions = (instructions + "\n" + content).strip() if instructions else content
            else:
                user_parts.append(content)
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": "\n\n".join(user_parts).strip(),
            "temperature": temperature,
        }
        if instructions:
            payload["instructions"] = instructions
        if self.enable_search:
            payload["tools"] = [{"type": "web_search"}]
            if self.forced_search:
                payload["tool_choice"] = {"type": "web_search"}
        last_err: Optional[Exception] = None
        data: Optional[Dict[str, Any]] = None
        for url in self._response_urls():
            try:
                r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
                if r.status_code == 404:
                    last_err = RuntimeError(f"404 {url}")
                    continue
                if r.status_code >= 400 and self.forced_search and "tool_choice" in payload:
                    slim = dict(payload)
                    slim.pop("tool_choice", None)
                    r2 = requests.post(url, headers=self._headers(), json=slim, timeout=self.timeout_s)
                    if r2.status_code < 400:
                        r = r2
                    else:
                        r.raise_for_status()
                else:
                    r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                last_err = e
                continue
        if data is None:
            raise RuntimeError(f"Responses API failed: {last_err}")
        self.last_search_calls = _count_web_search_calls(data)
        return _extract_responses_text(data)


def _map_header(cells: Sequence[Any]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, c in enumerate(cells):
        name = _compact(c)
        for canon, aliases in HEADER_ALIASES.items():
            if name in aliases and canon not in idx:
                idx[canon] = i
    if "Device Level" not in idx:
        # Fallback: the first 5 columns are the classification columns used by expand
        for i, canon in enumerate(["Device Level", "Device Type", "Device Name", "Spec/Model", "Main Application"]):
            idx.setdefault(canon, i)
    return idx


def _cell(row: Sequence[Any], idx: Dict[str, int], key: str) -> str:
    i = idx.get(key)
    if i is None or i >= len(row):
        return ""
    return _compact(row[i])


def load_catalog_rows(xlsx: Path) -> List[Dict[str, Any]]:
    if openpyxl is None:
        raise SystemExit("openpyxl is required: pip install openpyxl")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        header = None
        out: List[Dict[str, Any]] = []
        for excel_row, row in enumerate(ws.iter_rows(values_only=True), start=1):
            vals = list(row or ())
            if excel_row == 1:
                header = _map_header(vals)
                continue
            if not any(_compact(v) for v in vals[:8]):
                continue
            rec = {
                "excel_row": excel_row,
                "model_code": f"{excel_row:04d}",
                "Device Level": _cell(vals, header or {}, "Device Level"),
                "Device Type": _cell(vals, header or {}, "Device Type"),
                "Device Name": _cell(vals, header or {}, "Device Name"),
                "Spec/Model": _cell(vals, header or {}, "Spec/Model"),
                "Main Application": _cell(vals, header or {}, "Main Application"),
            }
            rec["Device Type Path"] = " -> ".join(
                p for p in (rec["Device Level"], rec["Device Type"], rec["Device Name"], rec["Spec/Model"]) if p
            )
            out.append(rec)
        return out
    finally:
        wb.close()


def resolve_model_dir(models_dir: Path, code: str) -> Optional[Path]:
    hits = sorted(
        p
        for p in models_dir.iterdir()
        if p.is_dir() and p.name.startswith(code + "_")
    )
    return hits[0] if hits else None


def resolve_expand_xlsx(model_dir: Path) -> Optional[Path]:
    named = model_dir / f"{model_dir.name}.xlsx"
    if named.exists():
        return named
    cands = [
        p
        for p in model_dir.glob("*.xlsx")
        if not p.name.lower().startswith("~$")
        and ".bak" not in p.name.lower()
        and not p.name.endswith("_with_metrics.xlsx")
    ]
    return sorted(cands)[0] if cands else None


def load_expansion(xlsx: Path, max_preview_rows: int = 100) -> Dict[str, Any]:
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        sn = "Equipment List" if "Equipment List" in wb.sheetnames else wb.sheetnames[0]
        ws = wb[sn]
        it = ws.iter_rows(values_only=True)
        try:
            header = [_compact(h) or f"col_{i+1}" for i, h in enumerate(next(it) or ())]
        except StopIteration:
            return {"sheet": sn, "n_rows": 0, "columns": [], "preview": [], "value_summary": {}}
        rows: List[Dict[str, Any]] = []
        for raw in it:
            if raw is None or all(v is None or str(v).strip() == "" for v in raw):
                continue
            rec = {}
            for i, h in enumerate(header):
                rec[h] = _compact(raw[i]) if i < len(raw) else ""
            rows.append(rec)
        summary: Dict[str, List[str]] = {}
        skip_cols = {"O&M Metrics_Shared", "O&M Metrics_Special (by trigger condition)", "O&M Metrics_All", "Anomaly Rule Number_All", "Anomaly Rule_Number and Object", "Derived Metric_key_All"}
        for h in header:
            if h in skip_cols:
                continue
            uniq = []
            seen = set()
            for r in rows:
                v = r.get(h) or ""
                if v and v not in seen:
                    seen.add(v)
                    uniq.append(v)
                if len(uniq) >= 100:
                    break
            summary[h] = uniq
        return {
            "sheet": sn,
            "n_rows": len(rows),
            "columns": header,
            "preview": rows[:max_preview_rows],
            "value_summary": summary,
        }
    finally:
        wb.close()


SYSTEM_PROMPT = """You are an expert reviewer of primary/secondary power equipment and converter station O&M ledgers.
You have the web_search tool enabled: before drawing a conclusion, you must first search the web to verify whether the model really exists, whether the classification hierarchy matches power industry conventions, and whether the expansion instances are engineering-reasonable.
Prefer citable verifiable sources: national standards GB/T, power industry standards DL/T, IEC, State Grid enterprise standards Q/GDW, public information on typical projects (Jinping-Sunan, Xiangjiaba-Shanghai, Xiluodu-Zhexi, Changji-Guquan, etc.), and mainstream manufacturer samples (Xidian, TBEA, Pinggao, NARI, XJ, Huawei, ZTE, etc.).
Do not treat "internal self-consistency of the dataset" as "real in the actual world". If a search cannot confirm something, explicitly write "search unconfirmed" and judge it as undeterminable or partially reasonable; do not fabricate standard numbers.
It is forbidden to review O&M metrics, anomaly judgment rules, or derived metrics (these columns have been removed from the input).
Output only a single JSON object, no Markdown."""


def build_user_prompt(cat: Dict[str, Any], expansion: Dict[str, Any], model_dir: str, xlsx_name: str) -> str:
    payload = {
        "Task": "Separately review (1) the device model classification row (2) the equipment list expanded by the LLM",
        "Classification Row_from_device_models.xlsx": {
            "Excel Row Number": cat["excel_row"],
            "Corresponding Directory Code": cat["model_code"],
            "Device Level": cat["Device Level"],
            "Device Type": cat["Device Type"],
            "Device Name": cat["Device Name"],
            "Spec/Model": cat["Spec/Model"],
            "Main Application": cat["Main Application"],
            "Device Type Path": cat["Device Type Path"],
        },
        "Expansion List_from_models": {
            "Model Directory": model_dir,
            "xlsx": xlsx_name,
            "sheet": expansion.get("sheet"),
            "Instance Count": expansion.get("n_rows"),
            "Ledger Fields_excluding O&M metrics and rules": expansion.get("columns"),
            "Value Sampling per Field": expansion.get("value_summary"),
            "Sample of First Rows": expansion.get("preview"),
            "Sampling Note": expansion.get("preview_note"),
        },
        "Review Points_Classification": [
            "Is the Spec/Model a real product model, voltage level, or project-common designation (project conventions such as '+/-800kV High-end Converter Transformer' are allowed)",
            "Is the four-level classification (Level -> Type -> Name -> Spec) in a correct hierarchical relationship, with nothing mislabeled",
            "Is the Main Application consistent with this category of equipment",
        ],
        "Review Points_Expansion": [
            "Only review device identity and technical configuration fields: name, project/site, system location, phase, primary/standby, voltage/capacity, cooling, wiring, bushing, etc.",
            "Do the fields match the ledger of this category of equipment (do not apply converter transformer fields to switchgear/communication equipment)",
            "Are the field values common in engineering and physically self-consistent",
            "Is the project/site name a real project; the dataset is allowed to use Jinping/Xiangjiaba/Xiluodu/Fufeng, etc. as demonstration stations, but the configuration must be like equipment of that voltage level",
            "Is the device count an absurd order of magnitude (use 'Instance Count' as the reference; a sample row count smaller than the device count is not an error)",
            "Does the device name look like ledger naming rather than a placeholder word (some, XX, test)",
        ],
        "Explicitly Do Not Review": [
            "O&M Metrics_*",
            "Anomaly Judgment Rules*",
            "Derived Metrics*",
            "Whether metrics can be monitored, whether rules are complete",
        ],
        "Verdict Enum": ["Truly Reasonable", "Partially Reasonable", "Unreasonable", "Undeterminable"],
        "Output JSON schema": {
            "excel_row": "int",
            "model_code": "str",
            "Classification Verdict": "Truly Reasonable|Partially Reasonable|Unreasonable|Undeterminable",
            "Classification Score_1to5": "int",
            "Classification Reasons": ["According to standard XX / public information: ..."],
            "Classification Issues": ["..."],
            "Expansion Verdict": "Truly Reasonable|Partially Reasonable|Unreasonable|Undeterminable",
            "Expansion Score_1to5": "int",
            "Expansion Reasons": ["..."],
            "Expansion Issues": ["Point out the specific field or instance ..."],
            "Referenced Standards or Sources": [{"Name": "GB/T or project name", "Description": "which conclusion it supports", "Found By Search": True}],
            "Search Summary": ["key facts found in search, one or two sentences"],
            "Overall Verdict": "Truly Reasonable|Partially Reasonable|Unreasonable|Undeterminable",
            "Overall Conclusion": "one sentence",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n\nPlease search the web and output only JSON."


def audit_one(
    client: OpenAICompatClient,
    cat: Dict[str, Any],
    models_dir: Path,
    max_retries: int,
) -> Dict[str, Any]:
    code = cat["model_code"]
    model_dir = resolve_model_dir(models_dir, code)
    base = {
        "excel_row": cat["excel_row"],
        "model_code": code,
        "Device Type Path": cat["Device Type Path"],
        "Spec/Model": cat["Spec/Model"],
        "Model Directory": model_dir.name if model_dir else "",
        "Expansion xlsx": "",
        "Expansion Device Count": 0,
    }
    if model_dir is None:
        return {
            **base,
            "Classification Verdict": "Undeterminable",
            "Expansion Verdict": "Unreasonable",
            "Overall Verdict": "Unreasonable",
            "Overall Conclusion": f"No models/{code}_* directory found",
            "Classification Reasons": [],
            "Expansion Reasons": [f"expand_device_models should generate a {code}_ prefixed directory based on the Excel row number"],
            "Classification Issues": [],
            "Expansion Issues": ["missing directory"],
            "Referenced Standards or Sources": [],
            "error": "missing_model_dir",
        }
    xlsx = resolve_expand_xlsx(model_dir)
    if xlsx is None:
        return {
            **base,
            "Model Directory": model_dir.name,
            "Classification Verdict": "Undeterminable",
            "Expansion Verdict": "Unreasonable",
            "Overall Verdict": "Unreasonable",
            "Overall Conclusion": f"{model_dir.name} is missing {{directory_name}}.xlsx",
            "Classification Reasons": [],
            "Expansion Reasons": [],
            "Classification Issues": [],
            "Expansion Issues": ["missing expansion xlsx"],
            "Referenced Standards or Sources": [],
            "error": "missing_xlsx",
        }
    expansion = load_expansion(xlsx)
    base["Expansion xlsx"] = xlsx.name
    base["Expansion Device Count"] = int(expansion.get("n_rows") or 0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(cat, expansion, model_dir.name, xlsx.name)},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=0.1)
            js = _extract_first_json_object(raw)
            if not isinstance(js, dict):
                raise ValueError("JSON root is not object")
            js.update(base)
            for k in ("Classification Verdict", "Expansion Verdict", "Overall Verdict"):
                v = _compact(js.get(k))
                if v not in VERDICT_ORDER:
                    js[k] = "Undeterminable"
            js["raw_preview"] = raw[:4000]
            js["web_search_calls"] = getattr(client, "last_search_calls", 0)
            js.pop("error", None)
            return js
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    return {
        **base,
        "Classification Verdict": "Script Error",
        "Expansion Verdict": "Script Error",
        "Overall Verdict": "Script Error",
        "Overall Conclusion": f"LLM call failed: {last_err}",
        "Classification Reasons": [],
        "Expansion Reasons": [],
        "Classification Issues": [],
        "Expansion Issues": [],
        "Referenced Standards or Sources": [],
        "error": str(last_err),
    }


def _join_list(x: Any) -> str:
    if isinstance(x, list):
        parts = []
        for it in x:
            if isinstance(it, dict):
                parts.append("; ".join(f"{k}:{v}" for k, v in it.items() if v not in (None, "")))
            else:
                parts.append(str(it))
        return " | ".join(parts)
    if x is None:
        return ""
    return str(x)


def write_outputs(rows: List[Dict[str, Any]], out_xlsx: Path, out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            slim = {k: v for k, v in r.items() if k != "raw_preview"}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    if openpyxl is None:
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Validation Results"
    cols = [
        "excel_row",
        "model_code",
        "Model Directory",
        "Spec/Model",
        "Device Type Path",
        "Expansion xlsx",
        "Expansion Device Count",
        "Classification Verdict",
        "Classification Score_1to5",
        "Classification Reasons",
        "Classification Issues",
        "Expansion Verdict",
        "Expansion Score_1to5",
        "Expansion Reasons",
        "Expansion Issues",
        "Overall Verdict",
        "Overall Conclusion",
        "Referenced Standards or Sources",
        "Search Summary",
        "web_search_calls",
        "error",
    ]
    fills = {
        "Truly Reasonable": PatternFill("solid", fgColor="C6EFCE"),
        "Partially Reasonable": PatternFill("solid", fgColor="FFEB9C"),
        "Unreasonable": PatternFill("solid", fgColor="FFC7CE"),
        "Undeterminable": PatternFill("solid", fgColor="D9D9D9"),
        "Script Error": PatternFill("solid", fgColor="F4B183"),
    }
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for r in rows:
        vals = []
        for k in cols:
            v = r.get(k)
            if k in ("Classification Reasons", "Classification Issues", "Expansion Reasons", "Expansion Issues", "Referenced Standards or Sources", "Search Summary"):
                vals.append(_join_list(v))
            else:
                vals.append(v if v is not None else "")
        ws.append(vals)
        row_idx = ws.max_row
        for name in ("Classification Verdict", "Expansion Verdict", "Overall Verdict"):
            col = cols.index(name) + 1
            cell = ws.cell(row_idx, col)
            fill = fills.get(_compact(cell.value))
            if fill:
                cell.fill = fill
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{max(ws.max_row, 1)}"
    for i in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 18
    ws.column_dimensions["E"].width = 42
    ws.column_dimensions["J"].width = 40
    ws.column_dimensions["N"].width = 40

    ws2 = wb.create_sheet("Summary")
    from collections import Counter

    def _cnt(key: str) -> Counter:
        return Counter(_compact(r.get(key)) or "(empty)" for r in rows)

    ws2.append(["Item", "Value", "Count"])
    for key in ("Overall Verdict", "Classification Verdict", "Expansion Verdict"):
        for val, n in sorted(_cnt(key).items(), key=lambda kv: VERDICT_ORDER.get(kv[0], 9)):
            ws2.append([key, val, n])
        ws2.append([])
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)


def main(argv: Optional[Iterable[str]] = None) -> int:
    from powerbench.paths import demo_models_dir, taxonomy_dir, repo_root, DEFAULT_DEVICE_MODELS_XLSX

    root = repo_root(Path(__file__))
    default_xlsx = taxonomy_dir() / DEFAULT_DEVICE_MODELS_XLSX

    ap = argparse.ArgumentParser(description="Web-search validation of device model classification and its models expansion list")
    ap.add_argument("--input", default=str(default_xlsx), help="device_models.xlsx")
    ap.add_argument("--models_dir", default=str(demo_models_dir()))
    ap.add_argument("--rows", type=str, default="", help='Excel rows (including header), e.g. "2,5-7"; empty = all data rows')
    ap.add_argument("--jobs", "-j", type=int, default=2)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-flash",
        help="default deepseek-v4-flash (Responses API + web_search)",
    )
    ap.add_argument("--api-key", type=str, default="", help="or environment variable OPENAI_API_KEY / LLM_API_KEY")
    ap.add_argument("--enable_search", action="store_true", default=True)
    ap.add_argument("--no_search", action="store_true", help="disable the web_search tool")
    ap.add_argument("--forced_search", action="store_true", default=True, help="tool_choice forces web_search")
    ap.add_argument("--dry_run", action="store_true", help="only print the Excel row <-> directory <-> xlsx mapping, do not call the LLM")
    ap.add_argument("--out_xlsx", type=str, default=str(root / "output" / "device_models_validation_results.xlsx"))
    ap.add_argument("--out_jsonl", type=str, default=str(root / "output" / "device_models_validation_results.jsonl"))
    args = ap.parse_args(list(argv) if argv is not None else None)

    catalog = load_catalog_rows(Path(args.input))
    want = _parse_rows_spec(args.rows)
    if want:
        catalog = [c for c in catalog if c["excel_row"] in want]

    models_dir = Path(args.models_dir)
    if args.dry_run:
        print("excel_row\tcode\tSpec/Model\tdirectory\txlsx\tdevice_count")
        n_ok = 0
        for c in catalog:
            d = resolve_model_dir(models_dir, c["model_code"])
            x = resolve_expand_xlsx(d) if d else None
            n = load_expansion(x)["n_rows"] if x else 0
            if d and x:
                n_ok += 1
            print(
                f"{c['excel_row']}\t{c['model_code']}\t{c['Spec/Model']}\t"
                f"{d.name if d else 'MISSING'}\t{x.name if x else 'MISSING'}\t{n}"
            )
        print(f"[DRY] rows={len(catalog)} mapped_ok={n_ok}")
        return 0

    api_key = (args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing API key: please pass --api-key or set OPENAI_API_KEY / LLM_API_KEY")

    client = OpenAICompatClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        enable_search=bool(args.enable_search) and not bool(args.no_search),
        forced_search=bool(args.forced_search) and not bool(args.no_search),
    )
    print(
        f"[INFO] model={client.model} api=responses "
        f"web_search={client.enable_search} forced={client.forced_search} "
        f"n={len(catalog)} jobs={args.jobs}"
    )

    results: List[Dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for c in catalog:
            rec = audit_one(client, c, models_dir, args.max_retries)
            print(
                f"[{rec.get('Overall Verdict')}] row {c['excel_row']} {c['Spec/Model']} "
                f"classification={rec.get('Classification Verdict')} expansion={rec.get('Expansion Verdict')}"
            )
            results.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(audit_one, client, c, models_dir, args.max_retries): c for c in catalog}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "excel_row": c["excel_row"],
                        "model_code": c["model_code"],
                        "Spec/Model": c["Spec/Model"],
                        "Overall Verdict": "Script Error",
                        "Classification Verdict": "Script Error",
                        "Expansion Verdict": "Script Error",
                        "error": str(e),
                    }
                print(
                    f"[{rec.get('Overall Verdict')}] row {c['excel_row']} {c['Spec/Model']} "
                    f"classification={rec.get('Classification Verdict')} expansion={rec.get('Expansion Verdict')}"
                )
                results.append(rec)
        results.sort(key=lambda r: int(r.get("excel_row") or 0))

    write_outputs(results, Path(args.out_xlsx), Path(args.out_jsonl))
    from collections import Counter

    cc = Counter(_compact(r.get("Overall Verdict")) for r in results)
    print("========== Summary ==========")
    for k, n in sorted(cc.items(), key=lambda kv: VERDICT_ORDER.get(kv[0], 9)):
        print(f"  {k}: {n}")
    print(f"[OK] wrote {args.out_xlsx}")
    print(f"[OK] wrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
