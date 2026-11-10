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
DEFAULT_OUT_JSON = "rule_metric_stage1_atomic.json"
DEFAULT_STAGE2_JSON = "rule_metric_stage2.json"
DEFAULT_OUT_STAGE2_JSON = "rule_metric_stage2_atomic.json"
DEFAULT_SPLIT_MAP_JSON = "metric_split_map.json"


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _extract_first_json(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError("model output is not string")
    t = text.strip()
    # direct
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return json.loads(t)
    # find first { or [
    start_obj = t.find("{")
    start_arr = t.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("no JSON start found")
    start = start_obj if start_arr == -1 else start_arr if start_obj == -1 else min(start_obj, start_arr)

    stack = []
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
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    continue
                stack.pop()
                if not stack:
                    return json.loads(t[start : i + 1])
    raise ValueError("could not parse JSON from model output")


def _looks_composite_metric(name: str) -> bool:
    """
    Only pick high-confidence patterns of "multiple measurement points bundled in one name",
    to avoid harming single-state names (e.g. "heater/dehumidifier working status").
    The LLM still makes the final split decision; this is only to save tokens.
    """
    s = _compact_str(name)
    if not s:
        return False
    # Multiple items enumerated inside parentheses with separators/"etc.": oil DGA (total hydrocarbons, hydrogen, acetylene, etc.)
    m = re.search(r"[（(]([^）)]+)[）)]", s)
    if m:
        inner = m.group(1)
        if any(sep in inner for sep in (",", ";", " and ", " etc", "、", "，", "；")):
            return True
    # Double physical quantities bundled in the name, still ending at "pressure/density" (not further split into -body pressure etc.)
    if re.search(r"(?:pressure\s*[/／]\s*density|density\s*[/／]\s*pressure)\s*$", s, re.IGNORECASE):
        return True
    return False


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def normalize_unit(unit: str, *, target_pressure_unit: str = "MPa") -> Tuple[str, str]:
    u0 = _compact_str(unit)
    if not u0:
        return "", ""
    u = u0.replace(" ", "")
    u_l = u.lower()

    aliases = {
        "MPa": "MPa",
        "kPa": "kPa",
        "Pa": "Pa",
        "pa": "Pa",
        "kpa": "kPa",
        "mpa": "MPa",
    }
    if u in aliases:
        u = aliases[u]
    elif u_l in aliases:
        u = aliases[u_l]

    tgt = _compact_str(target_pressure_unit) or "MPa"
    if tgt not in {"MPa", "kPa"}:
        tgt = "MPa"

    if u in {"Pa", "kPa", "MPa"}:
        if u == tgt:
            return u, ""
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
            return tgt, f"(original unit: {u0})"
        return tgt, f"(original unit: {u0}, conversion: {u0}×{f:g}={tgt})"

    return u0, ""


def _apply_mapping_to_string_list(lst: Any, mapping: Dict[str, List[str]]) -> Any:
    if not isinstance(lst, list):
        return lst
    out: List[str] = []
    changed = False
    for x in lst:
        xs = _compact_str(x)
        if not xs:
            continue
        if xs in mapping and mapping[xs]:
            out.extend(mapping[xs])
            changed = True
        else:
            out.append(xs)
    out = _dedupe_keep_order(out)
    return out if changed else lst


def build_stage2_rewrite_prompt(device_type: str, rule_obj: Dict[str, Any], old_metric: str, atomic_metrics: List[str]) -> str:
    rule_in = json.dumps(rule_obj, ensure_ascii=False, indent=2)
    atomic_list = json.dumps(atomic_metrics, ensure_ascii=False, indent=2)
    return (
        "You are a power equipment maintenance rule engineer.\n"
        "A composite metric from stage1 has now been split into multiple atomic metrics. You need to rewrite a single \"anomaly rule\" from stage2, ensuring:\n"
        "- The rule's \"input_metrics\" must contain these atomic metrics (all of them, none may be missed).\n"
        "- The \"rule_expression\" must explicitly constrain/use all of these atomic metrics (not just one of them).\n"
        "- If you think some atomic metric should not participate in this rule, do not delete it; instead rewrite the rule to include it (e.g. add combined conditions / derived metrics).\n"
        "- Keep the original field structure unchanged (rule_id/object_name/object_type/applicable_conditions/pointwise_judgment/time_series_requirements/input_metrics/derived_metric_definitions/judgment_description/rule_expression/threshold_parameters).\n"
        "- Output strict JSON only, no explanation/Markdown.\n\n"
        f"Device type: {device_type}\n"
        f"Old metric name being split: {old_metric}\n"
        f"Atomic metric list that must all be involved: {atomic_list}\n\n"
        "Original rule input:\n"
        f"{rule_in}\n\n"
        "Please output:\n"
        "{\"rule\": { ...the rewritten rule object... }}\n"
    )


def _rule_involves_all_atoms(rule_obj: Dict[str, Any], atoms: List[str]) -> bool:
    inputs = rule_obj.get("input_metrics")
    if not isinstance(inputs, list):
        return False
    inputs_set = {_compact_str(x) for x in inputs if _compact_str(x)}
    if any(a not in inputs_set for a in atoms):
        return False
    text = _compact_str(rule_obj.get("judgment_description")) + " | " + _compact_str(rule_obj.get("rule_expression"))
    # if atoms are non-ASCII names, require substring presence
    for a in atoms:
        if a not in text:
            return False
    return True


def update_stage2_with_metric_mapping(
    client: Optional["OpenAICompatClient"],
    device_type: str,
    stage2_obj: Dict[str, Any],
    mapping: Dict[str, List[str]],
    rewrite_via_llm: bool,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], int]:
    if not isinstance(stage2_obj, dict):
        return stage2_obj, 0
    rules = stage2_obj.get("anomaly_rules")
    if not isinstance(rules, list):
        return stage2_obj, 0

    changed_rules = 0
    for r in rules:
        if not isinstance(r, dict):
            continue
        before = r.get("input_metrics")
        if not isinstance(before, list):
            continue

        out: List[str] = []
        changed = False
        # Track which composite metrics were expanded in this rule
        expanded: List[Tuple[str, List[str]]] = []
        for x in before:
            xs = _compact_str(x)
            if not xs:
                continue
            if xs in mapping and mapping[xs]:
                atoms_full = mapping[xs]
                out.extend(atoms_full)
                changed = True
                expanded.append((xs, atoms_full))
            else:
                out.append(xs)
        out = _dedupe_keep_order(out)
        if changed:
            if rewrite_via_llm:
                if client is None:
                    raise RuntimeError("rewrite_via_llm requires client")
                # If multiple composites expanded, rewrite sequentially (rule rewritten each time).
                rewritten_rule = dict(r)
                rewritten_rule["input_metrics"] = out
                ok = True
                for (old_metric, atoms_full) in expanded:
                    prompt = build_stage2_rewrite_prompt(device_type, rewritten_rule, old_metric, atoms_full)
                    last_err: Optional[Exception] = None
                    got: Optional[Dict[str, Any]] = None
                    for attempt in range(1, max_retries + 1):
                        try:
                            raw = client.chat(prompt, temperature=0.0, max_new_tokens=2048)
                            js = _extract_first_json(raw)
                            if not isinstance(js, dict) or "rule" not in js or not isinstance(js["rule"], dict):
                                raise ValueError("LLM output must be JSON object with key 'rule'")
                            cand = js["rule"]
                            # guardrail: must include all atoms
                            if not _rule_involves_all_atoms(cand, atoms_full):
                                raise ValueError("rewritten rule does not involve all atomic metrics")
                            got = cand
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(1.2 * attempt)
                    if got is None:
                        ok = False
                        # fallback: at least update input_metrics
                        if last_err and isinstance(last_err, Exception):
                            pass
                        break
                    rewritten_rule = got
                if ok:
                    r.clear()
                    r.update(rewritten_rule)
            else:
                r["input_metrics"] = out
            changed_rules += 1
    return stage2_obj, changed_rules


STAGE1_EXAMPLE_ITEM = {
    "metric_name": "Oil DGA (total hydrocarbons, hydrogen, acetylene, etc.)",
    "metric_definition": "Diagnose latent faults inside the transformer by analyzing the composition and content of dissolved gases in the insulating oil.",
    "data_type": "numeric",
    "unit": "μL/L",
    "value_range": "Attention/warning values per standards such as DL/T 722.",
    "discrete_values": [],
    "remarks": "Includes the content of key gases such as hydrogen (H2), methane (CH4), ethane (C2H6), ethylene (C2H4), acetylene (C2H2), carbon monoxide (CO), carbon dioxide (CO2), etc.",
}


def build_prompt(device_type: str, metric_item: Dict[str, Any]) -> str:
    example_in = json.dumps(STAGE1_EXAMPLE_ITEM, ensure_ascii=False, indent=2)
    example_out = {
        "atomic_metrics": [
            {
                "metric_name": "Oil DGA - Hydrogen (H2)",
                "metric_definition": "Dissolved hydrogen content in insulating oil, used to diagnose partial discharge / discharge-type faults, etc.",
                "data_type": "numeric",
                "unit": "μL/L",
                "value_range": "Attention/warning values per standards such as DL/T 722.",
                "discrete_values": [],
                "remarks": "",
            },
            {
                "metric_name": "Oil DGA - Acetylene (C2H2)",
                "metric_definition": "Dissolved acetylene content in insulating oil, usually associated with arc discharge / severe overheating.",
                "data_type": "numeric",
                "unit": "μL/L",
                "value_range": "Attention/warning values per standards such as DL/T 722.",
                "discrete_values": [],
                "remarks": "",
            },
            "... other atomic gas metrics ...",
        ]
    }
    example_out_s = json.dumps(example_out, ensure_ascii=False, indent=2)

    metric_in = json.dumps(metric_item, ensure_ascii=False, indent=2)

    return (
        "You are a power equipment maintenance metric engineer. Your task is to split one \"composite metric entry\" into multiple \"atomic metric entries\".\n"
        "Requirements:\n"
        "- The input is a stage1 metric entry (JSON object); you must output strict JSON.\n"
        "- The output format is fixed: {\"atomic_metrics\": [ ... ]}\n"
        "- Each atomic metric must keep the same fields: metric_name/metric_definition/data_type/unit/value_range/discrete_values/remarks.\n"
        "- The atomic metric's \"metric_name\" must be prefixed with the \"stem of the parent metric name\", format: parent_stem-subitem.\n"
        "  For example \"Cooling water system parameters (water temperature, water pressure, flow)\" -> \"Cooling water system parameters - water temperature\", etc.\n"
        "- If the input is already an atomic metric (no multiple splittable subitems), return atomic_metrics containing only the original entry (metric_name may stay unchanged).\n"
        "- Do not output any explanatory text, no Markdown, JSON only.\n\n"
        f"Device type: {device_type}\n\n"
        "Example input (one composite metric from stage1.json):\n"
        f"{example_in}\n\n"
        "Example output (for brevity, omitted parts use a string placeholder):\n"
        f"{example_out_s}\n\n"
        "Now process this metric input:\n"
        f"{metric_in}\n"
    )


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 1800

    def chat(self, prompt: str, temperature: float = 0.1, max_new_tokens: Optional[int] = None) -> str:
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


def split_one_metric(
    client: OpenAICompatClient,
    device_type: str,
    metric_item: Dict[str, Any],
    temperature: float,
    max_new_tokens: Optional[int],
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    prompt = build_prompt(device_type=device_type, metric_item=metric_item)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            js = _extract_first_json(raw)
            if not isinstance(js, dict) or "atomic_metrics" not in js:
                raise ValueError("JSON must be an object with key 'atomic_metrics'")
            am = js.get("atomic_metrics")
            if isinstance(am, list):
                out = []
                for x in am:
                    if isinstance(x, dict):
                        out.append(x)
                if out:
                    return out
            # fallback to original
            return [metric_item]
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"split metric failed after {max_retries} retries: {last_err}")


def process_stage1_file(
    client: OpenAICompatClient,
    stage1_path: Path,
    out_path: Path,
    split_map_path: Path,
    stage2_path: Optional[Path],
    out_stage2_path: Optional[Path],
    rewrite_stage2_via_llm: bool,
    temperature: float,
    max_new_tokens: Optional[int],
    verbose: bool,
    *,
    dry_run: bool = False,
    write_only_if_changed: bool = True,
) -> str:
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    device_type = _compact_str(stage1.get("device_type") or stage1.get("device_type") or stage1_path.parent.name)

    if dry_run:
        candidates: List[str] = []
        for key in ["common_maintenance_metrics", "special_maintenance_metrics"]:
            metrics = stage1.get(key)
            if not isinstance(metrics, list):
                continue
            for m in metrics:
                if not isinstance(m, dict):
                    continue
                name = _compact_str(m.get("metric_name"))
                if _looks_composite_metric(name):
                    candidates.append(f"{key}:{name}")
        preview = "; ".join(candidates) if candidates else "(none)"
        return (
            f"[DRY-RUN] {stage1_path.parent.name}: heuristic_candidates={len(candidates)} → {preview}; "
            f"no LLM / no write"
        )

    changed = 0
    split_map: Dict[str, List[str]] = {}
    for key in ["common_maintenance_metrics", "special_maintenance_metrics"]:
        metrics = stage1.get(key)
        if not isinstance(metrics, list):
            continue
        new_list: List[Dict[str, Any]] = []
        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = _compact_str(m.get("metric_name"))
            # Normalize unit early so both original and atomic metrics are consistent.
            u_raw = _compact_str(m.get("unit"))
            u_norm, u_note = normalize_unit(u_raw, target_pressure_unit="MPa")
            if u_norm:
                m["unit"] = u_norm
            if u_note:
                old_note = _compact_str(m.get("remarks"))
                m["remarks"] = (old_note + u_note) if old_note else u_note
            if _looks_composite_metric(name):
                if verbose:
                    print(f"[{stage1_path.parent.name}] split {key}: {name}")
                atoms = split_one_metric(
                    client, device_type=device_type, metric_item=m, temperature=temperature, max_new_tokens=max_new_tokens
                )
                # LLM may output inconsistent unit spellings; normalize for each atom too.
                for a in atoms:
                    if not isinstance(a, dict):
                        continue
                    au_raw = _compact_str(a.get("unit"))
                    au_norm, au_note = normalize_unit(au_raw, target_pressure_unit="MPa")
                    if au_norm:
                        a["unit"] = au_norm
                    if au_note:
                        old_note = _compact_str(a.get("remarks"))
                        a["remarks"] = (old_note + au_note) if old_note else au_note
                # If actually split into multiple distinct names, count as changed
                atom_names = [_compact_str(a.get("metric_name")) for a in atoms if isinstance(a, dict)]
                atom_names2 = [n for n in atom_names if n]
                if len(atom_names2) >= 2 and name:
                    changed += 1
                    split_map[name] = _dedupe_keep_order(atom_names2)
                new_list.extend(atoms)
            else:
                new_list.append(m)
        stage1[key] = new_list

    should_write = (not write_only_if_changed) or bool(split_map)
    stage2_msg = ""

    if should_write:
        out_path.write_text(json.dumps(stage1, ensure_ascii=False, indent=2), encoding="utf-8")
        split_map_path.write_text(json.dumps(split_map, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # No split: do not overwrite stage1/stage2; still write an empty map to record "already checked"
        split_map_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
        return (
            f"[SKIP] {stage1_path.parent.name}: no composite split needed; "
            f"left {stage1_path.name}/{DEFAULT_STAGE2_JSON} unchanged, wrote empty {split_map_path.name}"
        )

    if stage2_path and out_stage2_path and stage2_path.exists() and split_map:
        try:
            stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
            stage2_updated, changed_rules = update_stage2_with_metric_mapping(
                client=client,
                device_type=device_type,
                stage2_obj=stage2,
                mapping=split_map,
                rewrite_via_llm=rewrite_stage2_via_llm,
            )
            out_stage2_path.write_text(json.dumps(stage2_updated, ensure_ascii=False, indent=2), encoding="utf-8")
            stage2_msg = f", stage2_updated_rules={changed_rules}, wrote {out_stage2_path.name}"
        except Exception as e:
            stage2_msg = f", stage2_update_failed={e}"

    return (
        f"[OK] {stage1_path.parent.name}: wrote {out_path.name} "
        f"(changed_metrics={changed}), wrote {split_map_path.name}{stage2_msg}"
    )


def resolve_targets(models_dir: Path, only_folder: str, start_from: int) -> List[Path]:
    subdirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    only = (only_folder or "").strip()
    if only:
        p = Path(only)
        if p.is_absolute():
            return [p]
        return [models_dir / only]
    if start_from:
        return subdirs[max(0, int(start_from)) :]
    return subdirs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Optional remediation: split composite metric names in stage1 into atomic metrics. "
            "By default only writes back in place to rule_metric_stage1/stage2.json when a split occurs; "
            "leaves the originals untouched when there is no split."
        )
    )
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--stage1_json", type=str, default=DEFAULT_STAGE1_JSON)
    ap.add_argument(
        "--sidecar",
        action="store_true",
        help="Sidecar mode: writes rule_metric_stage1_atomic.json / stage2_atomic.json without overwriting the original files",
    )
    ap.add_argument("--out_json", type=str, default="", help="Override the output stage1 filename (default follows --sidecar / in-place write-back)")
    ap.add_argument("--split_map_json", type=str, default=DEFAULT_SPLIT_MAP_JSON, help="old->new mapping (for auditing)")
    ap.add_argument("--stage2_json", type=str, default=DEFAULT_STAGE2_JSON, help="stage2 filename to update")
    ap.add_argument("--out_stage2_json", type=str, default="", help="Override the output stage2 filename")
    ap.add_argument(
        "--no-update-stage2",
        action="store_true",
        help="Do not update stage2 even when a split occurs (by default it is updated to keep input_metrics consistent with stage1)",
    )
    ap.add_argument(
        "--rewrite_stage2_via_llm",
        action="store_true",
        help="When updating stage2, use the LLM to rewrite affected rules so that every atomic metric enters the expression",
    )
    ap.add_argument(
        "--force-write",
        action="store_true",
        help="Write back stage1 even when there is no split (generally unnecessary; by default only writes back when there is a split)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be split, do not write files")
    ap.add_argument("--only_folder", type=str, default="", help="only process one subfolder (name or full path)")
    ap.add_argument(
        "--only_index",
        type=int,
        default=-1,
        help="only process one subfolder by sorted index under models_dir (useful when folder name has special chars)",
    )
    ap.add_argument("--list_folders", action="store_true", help="print sorted subfolder indices under models_dir and exit")
    ap.add_argument("--start_from", type=int, default=0, help="skip first N folders (sorted order)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--base_url",type=str, default="https://api.your-provider.com/v1",help="LLM Base URL (passed to run_pipeline's base_url parameter)")
    ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    ap.add_argument("--api-key",type=str,default="YOUR_API_KEY",help="LLM API Key (passed to run_pipeline's api_key parameter)",)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")

    if bool(args.list_folders):
        subs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
        print(f"total_folders={len(subs)}")
        for i, p in enumerate(subs):
            print(f"{i}\t{p.name}")
        return 0

    client = OpenAICompatClient(api_key=str(args.api_key), base_url=str(args.base_url), model=str(args.model))
    if int(args.only_index) >= 0:
        subdirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
        idx = int(args.only_index)
        if idx >= len(subdirs):
            raise SystemExit(f"only_index out of range: {idx} >= {len(subdirs)}")
        targets = [subdirs[idx]]
    else:
        targets = resolve_targets(models_dir, args.only_folder, int(args.start_from))
    targets = [t for t in targets if t.exists() and t.is_dir()]
    if not targets:
        print("[INFO] no folders to process")
        return 0

    sidecar = bool(args.sidecar)
    update_stage2 = not bool(args.no_update_stage2)
    write_only_if_changed = not bool(args.force_write)

    conc = max(1, int(args.concurrency))
    total = len(targets)
    done = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        fut_map = {}
        for folder in targets:
            stage1_path = folder / str(args.stage1_json)
            if not stage1_path.exists():
                continue
            if args.out_json:
                out_path = folder / str(args.out_json)
            elif sidecar:
                out_path = folder / DEFAULT_OUT_JSON
            else:
                out_path = stage1_path

            split_map_path = folder / str(args.split_map_json)
            stage2_path = (folder / str(args.stage2_json)) if update_stage2 else None
            if update_stage2:
                if args.out_stage2_json:
                    out_stage2_path = folder / str(args.out_stage2_json)
                elif sidecar:
                    out_stage2_path = folder / DEFAULT_OUT_STAGE2_JSON
                else:
                    out_stage2_path = folder / str(args.stage2_json)
            else:
                out_stage2_path = None
            fut = ex.submit(
                process_stage1_file,
                client,
                stage1_path,
                out_path,
                split_map_path,
                stage2_path,
                out_stage2_path,
                bool(args.rewrite_stage2_via_llm),
                float(args.temperature),
                int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else None,
                bool(args.verbose),
                dry_run=bool(args.dry_run),
                write_only_if_changed=write_only_if_changed,
            )
            fut_map[fut] = folder

        if not fut_map:
            print("[INFO] no stage1 json found under target folders")
            return 0

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
