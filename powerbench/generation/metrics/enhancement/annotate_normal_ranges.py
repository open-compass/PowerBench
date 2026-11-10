from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


def _compact_str(x: Any) -> str:
    if x is None: return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _extract_first_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON found")
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i+1])
    raise ValueError("incomplete JSON")


# ── LLM client ──

def _call_llm(prompt: str, *, api_key: str, base_url: str, model: str,
              temperature: float = 0.1, max_tokens: int = 512,
              stream: bool = False, verbose: bool = False, stream_label: str = "") -> str:
    import requests
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if not stream:
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # --- streaming mode ---
    payload["stream"] = True
    r = requests.post(url, json=payload, headers=headers, timeout=120, stream=True)
    r.raise_for_status()
    r.encoding = "utf-8"

    if verbose and stream_label:
        print(f"  [LLM stream] {stream_label}", flush=True)
    chunks: list[str] = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            choices = chunk.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                if verbose:
                    sys.stdout.buffer.write(content.encode("utf-8"))
                    sys.stdout.buffer.flush()
                chunks.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    if verbose and stream_label:
        print()
    return "".join(chunks)


# ── Stage2 rule parsing: extract thresholds associated with a metric ──

def _parse_stage2_thresholds(stage2_path: Path) -> list[dict]:
    """
    Extract from stage2 the "non-time-series" threshold comparisons associated with each metric.

    Steps:
      1. Remove time-series functions such as consec_* / slope_* / std_* / avg_* and their parenthesized content
      2. Clean up leftover = true / = True
      3. For each input_metrics entry, find its comparison operation in the cleaned expression
         (supports both non-ASCII metric names and English variable names)
    """
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    results: list[dict] = []

    for r in stage2.get("anomaly_rules") or []:
        if not isinstance(r, dict):
            continue
        rid = _compact_str(r.get("rule_id"))
        expr_raw = _compact_str(r.get("rule_expression"))
        if not expr_raw:
            continue
        inputs = r.get("input_metrics") or []
        params = {_compact_str(p.get("parameter_name")): p.get("parameter_value") for p in (r.get("threshold_parameters") or []) if isinstance(p, dict)}

        # determine top-level logic connective
        logic = ""
        if " OR " in expr_raw.upper() or " or " in expr_raw:
            logic = "OR"
        elif " AND " in expr_raw.upper() or " and " in expr_raw:
            logic = "AND"

        # ── (1) remove time-series functions and their arguments ──
        _TIME_FNS = ("consec_", "slope_", "std_", "avg_", "max_", "min_",
                     "duration_", "count_", "corr_", "rolling_")
        expr_clean = expr_raw
        for fn_prefix in _TIME_FNS:
            # match fn_pattern( ... ) — supports nested parentheses and non-ASCII characters
            fn_pat = re.compile(rf"{fn_prefix}\w*\s*\([^()]*(?:\([^()]*\)[^()]*)*\)")
            expr_clean = fn_pat.sub("", expr_clean)
        # clean up leftover = true / tru
        expr_clean = re.sub(r"=\s*[Tt]rue", "", expr_clean)

        # ── (2) collect input_metrics -> possible variable-name mappings ──
        # Some expressions use English variable names (e.g. top_oil_temp), others use non-ASCII names
        # For each input metric we construct the list of possible names used for matching
        def _possible_names(cn: str) -> list[str]:
            names = [cn]  # the original name
            # Also try matching contiguous segments (for "AB > thr OR CD < thr" scenarios)
            # No extra transformation; directly use regex to search name > / < patterns
            return names

        # ── (3) extract bare comparisons from the cleaned expression ──
        # Pattern: (name or English variable name) immediately followed by > or <
        for cn in inputs:
            cn = _compact_str(cn)
            if not cn:
                continue

            # Strategy A: directly search name + comparison operator
            for m in re.finditer(re.escape(cn) + r"\s*(>|<)", expr_clean):
                op = m.group(1)
                tail = expr_clean[m.end():]
                tok = re.match(r"\s*(\w+)", tail)
                if not tok:
                    continue
                thr_name = tok.group(1)
                thr_val = params.get(thr_name)
                if not _is_number(thr_val):
                    continue
                results.append({
                    "metric_name": cn, "rule_id": rid,
                    "variable_name": cn, "direction": op,
                    "threshold": float(thr_val), "threshold_param_name": thr_name,
                    "logic_connective": logic,
                })

            # Strategy B: also search English variable-name form
            for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*(>|<)", expr_clean):
                var_name = m.group(1)
                if var_name.lower() in {"or", "and", "not", "true", "false", "in", "abs"}:
                    continue
                op = m.group(2)
                tail = expr_clean[m.end():]
                tok = re.match(r"\s*(\w+)", tail)
                if not tok:
                    continue
                thr_name = tok.group(1)
                thr_val = params.get(thr_name)
                if not _is_number(thr_val):
                    continue
                results.append({
                    "metric_name": cn, "rule_id": rid,
                    "variable_name": var_name, "direction": op,
                    "threshold": float(thr_val), "threshold_param_name": thr_name,
                    "logic_connective": logic,
                })

    return results


# ── Check whether a range conflicts with the rules ──

def _check_conflict(normal_range: dict, thresholds: list[dict]) -> list[str]:
    """
    Returns a list of conflict descriptions (empty = no conflict).
    Considers AND/OR logic connectives:
    - OR / single condition: any one comparison conflict -> overall conflict
    - AND: all comparisons conflict -> overall conflict; otherwise no conflict
    """
    if not thresholds:
        return []

    nr_min = normal_range.get("min")
    nr_max = normal_range.get("max")
    nr_mean = normal_range.get("mean")

    # group by rule
    by_rule: dict[str, list[dict]] = {}
    for t in thresholds:
        by_rule.setdefault(t["rule_id"], []).append(t)

    conflicts: list[str] = []
    for rid, group in by_rule.items():
        logic = group[0].get("logic_connective", "") if group else ""
        individual: list[str] = []

        for t in group:
            thr = t["threshold"]
            op = t["direction"]
            issue = ""
            if op == ">" and nr_max is not None and nr_max >= thr:
                issue = f"{rid}: normal upper bound {nr_max} >= alarm threshold {thr} (>{thr} triggers), conflict"
            elif op == "<" and nr_min is not None and nr_min <= thr:
                issue = f"{rid}: normal lower bound {nr_min} <= alarm threshold {thr} (<{thr} triggers), conflict"
            elif op == ">" and nr_mean is not None and nr_mean >= thr:
                issue = f"{rid}: normal mean {nr_mean} >= alarm threshold {thr} (>{thr} triggers), suspected conflict"
            elif op == "<" and nr_mean is not None and nr_mean <= thr:
                issue = f"{rid}: normal mean {nr_mean} <= alarm threshold {thr} (<{thr} triggers), suspected conflict"
            if issue:
                individual.append(issue)

        if not individual:
            continue

        if logic == "AND":
            # AND: only a real conflict if all conditions conflict
            if len(individual) == len(group):
                conflicts.extend(individual)
        else:
            # OR or no connective: report if any one conflicts
            conflicts.extend(individual)

    return conflicts


# ── Build the LLM prompt ──

def _build_prompt(metric: dict, thresholds: list[dict],
                  constraints: str = "") -> str:
    metric_s = json.dumps(metric, ensure_ascii=False, indent=2)
    thr_s = json.dumps([{
        "rule_id": t["rule_id"],
        "direction": t["direction"],
        "threshold": t["threshold"],
    } for t in thresholds], ensure_ascii=False, indent=2) if thresholds else "No related anomaly rules"

    constraint_s = f"\n\n## Hard constraints (must be satisfied, otherwise the range is invalid)\n{constraints}" if constraints else ""

    return (
        "You are a power equipment maintenance data modeling expert. Give a reasonable numeric range for the following maintenance metric during normal operation (no fault).\n\n"
        "Output format (strict JSON):\n"
        '{"mean": mean, "std": std, "min": normal minimum, "max": normal maximum, "unit": "unit", "note": "brief justification"}\n\n'
        "Notes:\n"
        "- Continuous numeric metrics must have mean/std/min/max\n"
        "- Values must respect physical common sense (e.g. temperature cannot be below absolute zero, current cannot be negative, etc.)\n"
        "- mean ± 3*std should be within [min, max]\n\n"
        f"## Metric information\n{metric_s}\n\n"
        f"## Anomaly rules (thresholds) associated with this metric\n{thr_s}"
        f"{constraint_s}\n\n"
        "Output JSON only, nothing else."
    )


def _suggest_range(metric: dict, thresholds: list[dict], *,
                   api_key: str, base_url: str, model: str,
                   constraints: str = "", max_retries: int = 3,
                   verbose: bool = False, metric_label: str = "",
                   no_stream: bool = False) -> dict | None:
    prompt = _build_prompt(metric, thresholds, constraints)
    for attempt in range(max_retries):
        try:
            if verbose:
                label = f"{metric_label}" + (f" retry#{attempt+1}" if attempt > 0 else "")
                print(f"      -> LLM call ...", end="", flush=True)
                t0 = time.time()
            raw = _call_llm(prompt, api_key=api_key, base_url=base_url, model=model,
                            stream=not no_stream, verbose=verbose, stream_label=label if verbose else "")
            if verbose:
                print(f"      <- done, elapsed {time.time() - t0:.1f}s")
            js = _extract_first_json(raw)
            mean = float(js.get("mean", 0))
            std = float(js.get("std", 1))
            vmin = float(js.get("min", mean - 3*std))
            vmax = float(js.get("max", mean + 3*std))
            return {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(vmin, 4),
                "max": round(vmax, 4),
                "unit": _compact_str(js.get("unit", "")),
                "note": _compact_str(js.get("note", "")),
            }
        except Exception as e:
            if verbose:
                print(f"\n      x LLM call failed: {e}")
            time.sleep(1.2 * attempt)
    return None


def _fill_stage2_placeholders_with_llm(stage2_path: Path, *,
                                       api_key: str, base_url: str, model: str,
                                       verbose: bool = True) -> int:
    """
    Scan placeholder parameter values in rule_metric_stage2.json, call the LLM to fill them in, and write back.
    Returns the number of parameters filled.
    """
    if not stage2_path.exists():
        return 0
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    device_type = stage2.get("device_type", "Unknown device")

    # collect placeholders
    placeholders: list[dict] = []
    for rule in stage2.get("anomaly_rules") or []:
        rid = _compact_str(rule.get("rule_id"))
        for p in rule.get("threshold_parameters") or []:
            name = _compact_str(p.get("parameter_name"))
            val = p.get("parameter_value")
            if not name:
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                placeholders.append({
                    "rule_id": rid,
                    "object_name": _compact_str(rule.get("object_name", "")),
                    "parameter_name": name,
                    "meaning": _compact_str(p.get("meaning", "")),
                    "current_placeholder": str(val)[:60] if val else "(empty)",
                })

    if not placeholders:
        if verbose:
            print(f"  [stage2] No placeholders to fill")
        return 0

    if verbose:
        print(f"  [stage2] Found {len(placeholders)} placeholders:")
        for ph in placeholders:
            print(f"    {ph['rule_id']}/{ph['parameter_name']}: {ph['meaning'][:30]}  <- current='{ph['current_placeholder']}'")

    prompt = (
        f"You are a power equipment maintenance expert. The following are threshold parameters that need to be filled in for \"{device_type}\" anomaly rules.\n"
        "Based on each parameter's meaning, the device type, and industry common sense, give a reasonable numeric value for each parameter.\n\n"
        "Output format (strict JSON):\n"
        '{"parameter_suggestions":[{"rule_id":"K1","parameter_name":"T_alarm","suggested_value":85}, ...]}\n\n'
        f"Parameters to fill:\n{json.dumps(placeholders, ensure_ascii=False, indent=2)}\n\n"
        "Note: the value must be a pure number, without units. Temperature in °C, current in mA. Output JSON only."
    )
    raw = _call_llm(prompt, api_key=api_key, base_url=base_url, model=model, temperature=0.1, max_tokens=2048)
    js = _extract_first_json(raw)
    suggestions = js.get("parameter_suggestions", js.get("parameters", []))

    filled = 0
    unfilled: list[str] = []
    if isinstance(suggestions, list):
        lookup: dict[tuple[str, str], float] = {}
        for s in suggestions:
            rid = _compact_str(s.get("rule_id", ""))
            pname = _compact_str(s.get("parameter_name", ""))
            try:
                lookup[(rid, pname)] = float(s.get("suggested_value"))
            except (ValueError, TypeError):
                pass

        for rule in stage2.get("anomaly_rules") or []:
            rid = _compact_str(rule.get("rule_id"))
            for p in rule.get("threshold_parameters") or []:
                name = _compact_str(p.get("parameter_name"))
                key = (rid, name)
                if key in lookup:
                    old = p.get("parameter_value", "?")
                    p["parameter_value"] = str(lookup[key])
                    if verbose:
                        print(f"    [OK] {rid}/{name}: '{old}' -> {lookup[key]}")
                    filled += 1

        # check which placeholders the LLM did not return
        for ph in placeholders:
            key = (ph["rule_id"], ph["parameter_name"])
            if key not in lookup:
                unfilled.append(f"{ph['rule_id']}/{ph['parameter_name']}")

        stage2_path.write_text(json.dumps(stage2, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        if unfilled:
            print(f"    [WARN] The LLM did not return suggested values for these {len(unfilled)} parameters: {', '.join(unfilled)}")
        print(f"  [stage2] Filled {filled}/{len(placeholders)} parameters and wrote back to rule_metric_stage2.json\n")
    return filled


# ── Main flow ──

def process_one_model(model_dir: Path, *, api_key: str, base_url: str, model: str,
                      max_iter: int = 10, verbose: bool = True, no_stream: bool = False) -> str:
    catalog_path = model_dir / "metric_catalog.json"
    stage2_path = model_dir / "rule_metric_stage2.json"

    if not catalog_path.exists():
        return f"[SKIP] {model_dir.name}: no metric_catalog.json"
    if not stage2_path.exists():
        return f"[SKIP] {model_dir.name}: no rule_metric_stage2.json"

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    # ── First fill in the placeholder parameters in stage2 (LLM write-back) ──
    _fill_stage2_placeholders_with_llm(stage2_path, api_key=api_key, base_url=base_url, model=model, verbose=verbose)

    thresholds = _parse_stage2_thresholds(stage2_path)

    # count metrics to process
    pending: list[tuple[int, dict]] = []
    for i, item in enumerate(catalog):
        if not isinstance(item, dict):
            continue
        if _compact_str(item.get("data_type")) != "float":
            continue
        if item.get("discrete_values"):
            continue
        if item.get("normal_range"):
            continue
        pending.append((i, item))

    if verbose:
        t0_total = time.time()
        print(f"\n{'='*60}")
        print(f"[Model directory] {model_dir.name}")
        print(f"[Total metrics] {len(catalog)}")
        print(f"[To fill] {len(pending)} continuous numeric metrics")
        print(f"[Anomaly rule thresholds] {len(thresholds)} entries")
        print(f"{'='*60}")

    updated = 0
    conflicted: list[dict] = []  # metrics that still fail the conflict check
    for idx, (orig_idx, item) in enumerate(pending):
        name = _compact_str(item.get("metric_name"))
        if verbose:
            print(f"\n  [{idx + 1}/{len(pending)}] Processing metric: {name}")

        related = [t for t in thresholds if t["metric_name"] == name]

        constraints = ""
        best_range = None
        suggestion = None
        final_conflicts: list[str] = []
        for iteration in range(max_iter):
            suggestion = _suggest_range(item, related,
                                        api_key=api_key, base_url=base_url, model=model,
                                        constraints=constraints, verbose=verbose,
                                        metric_label=f"{model_dir.name}/{name}",
                                        no_stream=no_stream)
            if suggestion is None:
                if verbose:
                    print(f"      x LLM returned invalid result; skipping without retrying")
                break

            conflicts = _check_conflict(suggestion, related)
            final_conflicts = conflicts
            if not conflicts:
                best_range = suggestion
                if verbose:
                    print(f"      OK passed: mean={suggestion['mean']}, std={suggestion['std']}, range=[{suggestion['min']}, {suggestion['max']}]")
                break

            if verbose:
                print(f"      ! round {iteration+1} conflict: {conflicts[0][:80]}... retrying with constraints")

            constraints = (
                "## The previous suggestion was rejected for the following reason. Please adjust the range to ensure a clear margin between the normal value and the anomaly threshold.\n"
                + "\n".join(f"- {c}" for c in conflicts)
                + "\n\nRequirement: the normal range upper/lower bound must keep at least a 10-20% safety margin from the anomaly threshold."
            )

        if best_range:
            item["normal_range"] = best_range
            updated += 1
        else:
            if verbose:
                print(f"      ! still not passed after {max_iter} rounds; keeping the last suggestion and marking as to-confirm")
            if suggestion:
                suggestion["note"] = (suggestion.get("note", "") + " [to confirm: conflicts with anomaly rules]").strip()
                item["normal_range"] = suggestion
            else:
                item["normal_range"] = {"note": "not set yet", "mean": 0, "std": 1, "min": 0, "max": 1}
            # record conflict info for the summary
            conflicted.append({
                "metric_name": name,
                "suggested_range": suggestion if suggestion else None,
                "conflicting_rules": [
                    {
                        "rule_id": t["rule_id"],
                        "direction": t["direction"],
                        "threshold": t["threshold"],
                        "threshold_param_name": t["threshold_param_name"],
                    }
                    for t in related
                ],
                "final_conflicts": final_conflicts,
            })

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Summary output: metrics that still fail the conflict check ──
    if conflicted and verbose:
        print(f"\n{'-'*60}")
        print(f"[Manual review] {model_dir.name}: {len(conflicted)} metrics still fail the conflict check")
        print(f"{'-'*60}")
        for c in conflicted:
            print(f"\n  * metric: {c['metric_name']}")
            sg = c["suggested_range"]
            if sg:
                print(f"     suggested range: mean={sg.get('mean')}, std={sg.get('std')}, min={sg.get('min')}, max={sg.get('max')}")
            print(f"     related anomaly rules:")
            for t in c["conflicting_rules"]:
                print(f"       - {t['rule_id']}: metric {t['direction']} {t['threshold']} (param: {t['threshold_param_name']})")
            if c["final_conflicts"]:
                print(f"     conflict details:")
                for cf in c["final_conflicts"]:
                    print(f"       x {cf}")
        print()

    if verbose:
        print(f"[Result] {model_dir.name}: {updated}/{len(pending)} metrics filled with normal range, total elapsed {time.time() - t0_total:.1f}s")
    return f"[OK] {model_dir.name}: {updated}/{len(pending)} metrics filled with normal range" + (f", {len(conflicted)} pending manual review" if conflicted else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill in the normal_range of continuous metrics for metric_catalog.json")
    ap.add_argument("--models_dir", type=str, default=str(__import__("powerbench.paths", fromlist=["demo_models_dir"]).demo_models_dir()))
    ap.add_argument("--model_dir", type=str, default="", help="single model directory (takes priority over --models_dir)")
    ap.add_argument("--api-key", type=str, default="YOUR_API_KEY")
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument("--model", type=str, default="deepseek-v3.2")
    ap.add_argument("--max-iter", type=int, default=5, help="maximum iterations per metric")
    ap.add_argument("--force", action="store_true", help="overwrite existing normal ranges")
    ap.add_argument("--quiet", action="store_true", help="quiet mode, do not output detailed logs")
    ap.add_argument("--no-stream", action="store_true", help="disable LLM streaming output")
    args = ap.parse_args()

    if args.model_dir:
        targets = [Path(args.model_dir).resolve()]
    else:
        models_dir = Path(args.models_dir).resolve()
        targets = sorted([p for p in models_dir.iterdir() if p.is_dir() and (p / "metric_catalog.json").exists()])

    ok = 0
    for t in targets:
        try:
            msg = process_one_model(t, api_key=args.api_key, base_url=args.base_url,
                                    model=args.model, max_iter=args.max_iter,
                                    verbose=not args.quiet, no_stream=args.no_stream)
            ok += 1
            print(msg)
        except Exception as e:
            print(f"[FAIL] {t.name}: {e}")

    print(f"\nDone. {ok}/{len(targets)} succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
