from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import HTTPError


from powerbench.paths import demo_models_dir

DEFAULT_MODELS_DIR = str(demo_models_dir())
DEFAULT_STAGE1_JSON = "rule_metric_stage1.json"
DEFAULT_STAGE2_JSON = "rule_metric_stage2.json"
DEFAULT_OUT_JSON = "metric_catalog.json"


def _compact_str(x: Any) -> str:
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
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    start = t.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(t[start : i + 1])
    raise ValueError("could not find a complete JSON object in output")


def _snake_ok(s: str) -> bool:
    if not bool(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", s)):
        return False
    # Disallow time-window suffixes in raw metric storage fields (should live in derived metrics/rules)
    # Examples to reject: duration_1h, avg_24h, slope_30d, consec_3h, count_365d
    if re.search(r"(^|_)\d+(h|d|w|m|y)($|_)", s):
        return False
    if re.search(r"(^|_)(avg|slope|consec|count)_\d+(h|d)($|_)", s):
        return False
    return True


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def normalize_unit(unit: str, *, target_pressure_unit: str = "MPa") -> Tuple[str, str]:
    """
    Normalize unit strings so the final catalog keeps a single canonical unit for the same dimension.
    Current policy:
    - Pressure: keep ONLY one of {"MPa","kPa"} as final display unit (default: MPa).
      If original is kPa/Pa, we normalize to MPa and append a note about the conversion factor.
    Returns: (normalized_unit, note_suffix_to_append)
    """
    u0 = _compact_str(unit)
    if not u0:
        return "", ""
    u = u0.replace(" ", "")
    u_l = u.lower()

    # common aliases
    aliases = {
        "MPa": "MPa",
        "kPa": "kPa",
        "Pa": "Pa",
        "pa": "Pa",
        "kpa": "kPa",
        "mpa": "MPa",
    }

    def _alias_one(x: str) -> str:
        x0 = _compact_str(x).replace(" ", "")
        xl = x0.lower()
        if x0 in aliases:
            return aliases[x0]
        if xl in aliases:
            return aliases[xl]
        return x0

    # If unit contains multi-options like "kPa or MPa", only keep ONE final unit.
    # We store original multi-unit info in note (caller appends into remarks).
    multi_note = ""
    if " or " in u.lower() or "/" in u:
        parts = [p for p in re.split(r"[\s/、,，]|\bor\b", u, flags=re.I) if _compact_str(p)]
        if parts:
            multi_note = f"(original unit: {u0})"
            # Prefer pressure canonical if present
            parts2 = [_alias_one(p) for p in parts]
            # If any pressure unit appears, we will normalize to target_pressure_unit later
            if any(p in {"Pa", "kPa", "MPa"} for p in parts2):
                u = target_pressure_unit
            else:
                # otherwise pick the first option as the canonical display unit
                u = parts2[0]
    else:
        u = _alias_one(u)

    tgt = _compact_str(target_pressure_unit) or "MPa"
    if tgt not in {"MPa", "kPa"}:
        tgt = "MPa"

    # pressure unification
    if u in {"Pa", "kPa", "MPa"}:
        if u == tgt:
            return u, multi_note
        # convert label to target; we don't touch data values here (catalog only), so keep factor info.
        factor_map = {
            ("Pa", "kPa"): 1e-3,
            ("Pa", "MPa"): 1e-6,
            ("kPa", "Pa"): 1e3,
            ("kPa", "MPa"): 1e-3,
            ("MPa", "Pa"): 1e6,
            ("MPa", "kPa"): 1e3,
        }
        f = factor_map.get((u, tgt))
        if f is None:
            note = f"(original unit: {u0})"
        else:
            note = f"(original unit: {u0}, conversion: {u0}×{f:g}={tgt})"
        # avoid duplicating multi_note
        if multi_note and multi_note not in note:
            note = multi_note + note
        return tgt, note

    return u, multi_note


def infer_var_map_from_stage2(stage2: Dict[str, Any]) -> Dict[str, str]:
    """
    Best-effort: if a rule has exactly 1 input_metrics entry, pick first identifier-like token
    from rule_expression as its var name.
    This reuses names like top_oil_temp, winding_hotspot_temp, etc.
    """
    rules = stage2.get("anomaly_rules")
    if not isinstance(rules, list):
        return {}

    ignore = {
        "true",
        "false",
        "and",
        "or",
        "consec_3h",
        "avg_24h",
        "avg_3h",
        "slope_30d",
        "slope_90d",
        "slope_365d",
        "count_30d",
        "count_365d",
    }

    out: Dict[str, str] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        inputs = r.get("input_metrics")
        expr = _compact_str(r.get("rule_expression"))
        if not isinstance(inputs, list) or len(inputs) != 1 or not expr:
            continue
        metric_name = _compact_str(inputs[0])
        if not metric_name:
            continue

        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        for tok in tokens:
            tl = tok.lower()
            if tl in ignore:
                continue
            # heuristic: prefer snake_case metrics, not threshold params
            if "_" in tok and not tok.startswith("T_") and not tok.startswith("P_") and not tok.startswith("S_") and not tok.startswith("N_"):
                out.setdefault(metric_name, tok)
                break
    # normalize to lower snake_case where possible
    out2: Dict[str, str] = {}
    for k, v in out.items():
        v2 = v.lower()
        if _snake_ok(v2):
            out2[k] = v2
    return out2


STAGE1_EXAMPLE = {
    "metric_name": "Oil DGA - Acetylene (C2H2)",
    "metric_definition": "Dissolved acetylene content in insulating oil, usually associated with arc discharge / severe overheating.",
    "data_type": "numeric",
    "unit": "μL/L",
    "value_range": "Attention/warning values per standards such as DL/T 722.",
    "discrete_values": [],
    "remarks": "",
}


def build_name_prompt(device_type: str, metric_item: Dict[str, Any], existing: List[str]) -> str:
    metric_in = json.dumps(metric_item, ensure_ascii=False, indent=2)
    example_in = json.dumps(STAGE1_EXAMPLE, ensure_ascii=False, indent=2)
    example_out = {"var_name": "oil_dga_c2h2", "db_field": "oil_dga_c2h2"}
    return (
        "You are a power-equipment maintenance data modeling engineer. Generate an English variable name (snake_case) for the maintenance metric, to be used as a code variable name and a database field name.\n"
        "Hard constraints:\n"
        "- Output strict JSON only: {\"var_name\":\"...\",\"db_field\":\"...\"}\n"
        "- var_name/db_field may only contain lowercase letters, digits and underscores; must start with a letter; length <= 64.\n"
        "- var_name/db_field must not contain any time-window suffix (e.g. 1h/3h/24h/30d/90d/365d); those belong to derived computation, not to raw storage field names.\n"
        "- Keep the name as short as possible while remaining readable and reflecting the physical meaning; it must be unique within the same device type.\n"
        "- If the metric name contains an \"X-subitem\", prefer reflecting parent + subitem, e.g. cooling_water_temp / cooling_water_pressure / cooling_water_flow.\n"
        f"- Must not conflict with existing names: {json.dumps(existing, ensure_ascii=False)}\n\n"
        f"Device type: {device_type}\n\n"
        "Example input:\n"
        f"{example_in}\n\n"
        "Example output:\n"
        f"{json.dumps(example_out, ensure_ascii=False)}\n\n"
        "Now process this metric entry:\n"
        f"{metric_in}\n"
    )


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 1800

    def chat(self, prompt: str, temperature: float = 0.0, max_new_tokens: Optional[int] = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
        }
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            payload["max_tokens"] = int(max_new_tokens)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except HTTPError as e:
            body = getattr(r, "text", "<no body>")
            raise RuntimeError(f"HTTP error calling LLM: {e}; response_body={body[:2000]}") from e
        data = r.json()
        return data["choices"][0]["message"]["content"]


def choose_var_name_via_llm(
    client: OpenAICompatClient,
    device_type: str,
    metric_item: Dict[str, Any],
    existing: List[str],
    max_new_tokens: int,
    max_retries: int = 3,
) -> Tuple[str, str]:
    prompt = build_name_prompt(device_type, metric_item, existing)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(prompt, temperature=0.0, max_new_tokens=max_new_tokens)
            js = _extract_first_json_object(raw)
            var_name = _compact_str(js.get("var_name"))
            db_field = _compact_str(js.get("db_field")) or var_name
            if not _snake_ok(var_name) or not _snake_ok(db_field):
                raise ValueError(f"invalid snake_case: var_name={var_name}, db_field={db_field}")
            if var_name in existing or db_field in existing:
                raise ValueError(f"name conflict: {var_name}/{db_field}")
            return var_name, db_field
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"LLM naming failed after {max_retries} retries: {last_err}")


def build_catalog_for_folder(
    client: OpenAICompatClient,
    folder: Path,
    stage1_name: str,
    stage2_name: str,
    out_name: str,
    max_new_tokens: int,
    verbose: bool,
) -> str:
    stage1_path = folder / stage1_name
    stage2_path = folder / stage2_name
    if not stage1_path.exists():
        return f"[SKIP] {folder.name}: missing {stage1_name}"
    if not stage2_path.exists():
        return f"[SKIP] {folder.name}: missing {stage2_name}"

    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    device_type = _compact_str(stage1.get("device_type") or folder.name)

    # gather metrics
    metrics: List[Dict[str, Any]] = []
    for k in ["common_maintenance_metrics", "special_maintenance_metrics"]:
        xs = stage1.get(k) or []
        if isinstance(xs, list):
            for it in xs:
                if isinstance(it, dict) and _compact_str(it.get("metric_name")):
                    metrics.append(it)
    # dedupe by name (keep first)
    by_name: Dict[str, Dict[str, Any]] = {}
    for it in metrics:
        nm = _compact_str(it.get("metric_name"))
        by_name.setdefault(nm, it)
    metric_items = [by_name[k] for k in sorted(by_name.keys())]

    inferred = infer_var_map_from_stage2(stage2)

    existing: List[str] = []
    catalog: List[Dict[str, Any]] = []

    for it in metric_items:
        name = _compact_str(it.get("metric_name"))
        dtype = _compact_str(it.get("data_type"))
        unit_raw = _compact_str(it.get("unit"))
        unit, unit_note = normalize_unit(unit_raw, target_pressure_unit="MPa")
        discrete_vals = it.get("discrete_values") if isinstance(it.get("discrete_values"), list) else []
        discrete_vals2 = [_compact_str(v) for v in discrete_vals if _compact_str(v)]
        # preserve unit normalization info without changing schema
        if unit_note:
            old_note = _compact_str(it.get("remarks"))
            it["remarks"] = (old_note + unit_note) if old_note else unit_note

        # prefer reuse
        var = inferred.get(name, "")
        if var and _snake_ok(var) and var not in existing:
            var_name, db_field = var, var
        else:
            var_name, db_field = choose_var_name_via_llm(
                client, device_type=device_type, metric_item=it, existing=existing, max_new_tokens=max_new_tokens
            )

        existing.extend([var_name, db_field])

        metric_type = "state_type" if dtype == "discrete" else "raw_collection"
        catalog.append(
            {
                "metric_name": name,
                "variable_name": var_name,
                "db_field_name": db_field,
                "data_type": "string" if dtype == "discrete" else "float",
                "unit": unit,
                "metric_type": metric_type,
                "discrete_values": discrete_vals2,
            }
        )
        if verbose:
            print(f"[{folder.name}] {name} -> {var_name}")

    out_path = folder / out_name
    out_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"[OK] {folder.name}: wrote {out_name} (metrics={len(catalog)})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build metric_catalog.json with English var/db field names for stage1 metrics.")
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--stage1_json", type=str, default=DEFAULT_STAGE1_JSON)
    ap.add_argument("--stage2_json", type=str, default=DEFAULT_STAGE2_JSON)
    ap.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    ap.add_argument("--only_index", type=int, default=-1, help="process only one subfolder by sorted index")
    ap.add_argument("--start_from", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--base_url",type=str, default="https://api.your-provider.com/v1",help="LLM Base URL (passed to run_pipeline's base_url parameter)")
    ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    ap.add_argument("--api-key",type=str,default="YOUR_API_KEY",help="LLM API Key (passed to run_pipeline's api_key parameter)",)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    subs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    if not subs:
        raise SystemExit(f"no subfolders under {models_dir}")

    if int(args.only_index) >= 0:
        idx = int(args.only_index)
        if idx >= len(subs):
            raise SystemExit(f"only_index out of range: {idx} >= {len(subs)}")
        targets = [subs[idx]]
    else:
        targets = subs[max(0, int(args.start_from)) :]

    client = OpenAICompatClient(api_key=str(args.api_key), base_url=str(args.base_url), model=str(args.model))

    conc = max(1, int(args.concurrency))
    total = len(targets)
    done = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        fut_map = {
            ex.submit(
                build_catalog_for_folder,
                client,
                folder,
                str(args.stage1_json),
                str(args.stage2_json),
                str(args.out_json),
                int(args.max_new_tokens),
                bool(args.verbose),
            ): folder
            for folder in targets
        }
        for fut in as_completed(fut_map):
            folder = fut_map[fut]
            try:
                msg = fut.result()
                done += 1
                print(f"{msg} ({done}/{total})")
            except Exception as e:
                done += 1
                print(f"[FAIL] {folder.name}: {e} ({done}/{total})")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
