from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from powerbench.generation.metrics import rule_metric_prompt as prompt_module


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_first_json(text: str) -> Any:
    """
    Extract first JSON object/array from possibly wrapped LLM output.
    """
    if not isinstance(text, str):
        raise ValueError("LLM output is not a string")
    t = text.strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return json.loads(t)
    start_obj = t.find("{")
    start_arr = t.find("[")
    starts = [x for x in (start_obj, start_arr) if x != -1]
    if not starts:
        raise ValueError("no JSON start found in output")
    start = min(starts)
    open_ch = t[start]
    close_ch = "}" if open_ch == "{" else "]"
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
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("could not find a complete JSON payload in output")


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
        stream: bool = False,
        verbose: bool = False,
        stream_label: str = "",
    ) -> str:
        import requests
        from requests import HTTPError

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            payload["max_tokens"] = int(max_new_tokens)

        if not stream:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
            except HTTPError as e:
                body = getattr(r, "text", "<no body>")
                raise RuntimeError(f"HTTP error calling LLM: {e}; response_body={body[:2000]}") from e
            data = r.json()
            return data["choices"][0]["message"]["content"]

        # --- streaming mode ---
        payload["stream"] = True
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s, stream=True)
            r.raise_for_status()
        except HTTPError as e:
            body = getattr(r, "text", "<no body>")
            raise RuntimeError(f"HTTP error calling LLM (stream): {e}; response_body={body[:2000]}") from e

        # Force UTF-8 decoding for SSE lines
        r.encoding = "utf-8"

        if verbose and stream_label:
            print(f"  [LLM stream] {stream_label}", flush=True)
        chunks: List[str] = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
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
                        # Write UTF-8 bytes directly to bypass terminal encoding issues
                        sys.stdout.buffer.write(content.encode("utf-8"))
                        sys.stdout.buffer.flush()
                    chunks.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        if verbose and stream_label:
            print()  # newline after streaming
        return "".join(chunks)


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


def _load_metric_catalog(metric_catalog_path: Path) -> List[Dict[str, Any]]:
    try:
        items = _read_json(metric_catalog_path)
    except Exception:
        return []
    if isinstance(items, list):
        return [it for it in items if isinstance(it, dict)]
    return []


def _load_metric_catalog_names(metric_catalog_path: Path) -> List[str]:
    """
    Return list of metric_name from metric_catalog.json (if present).
    """
    try:
        items = _read_json(metric_catalog_path)
    except Exception:
        return []
    out: List[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                nm = _compact_str(it.get("metric_name"))
                if nm:
                    out.append(nm)
    # keep order but dedupe
    seen = set()
    out2: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        out2.append(x)
    return out2


    return out2


def _atomic_catalog_entries_for_derived(
    catalog: List[Dict[str, Any]],
    derived_metric: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Return metric_catalog entries matching a derived metric's atomic_metrics.
    """
    atoms = derived_metric.get("atomic_metrics") or []
    if not isinstance(atoms, list):
        return []

    want_db = {_compact_str(a.get("db_field")) for a in atoms if isinstance(a, dict)}
    want_db.discard("")
    want_names = {_compact_str(a.get("indicator_name")) for a in atoms if isinstance(a, dict)}
    want_names.discard("")

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for it in catalog:
        nm = _compact_str(it.get("metric_name"))
        db = _compact_str(it.get("db_field_name") or it.get("variable_name"))
        if nm not in want_names and db not in want_db:
            continue
        key = nm or db
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _pick_style_example_rules(stage2: Dict[str, Any], *, max_examples: int = 3) -> List[Dict[str, Any]]:
    """
    Pick a few existing high-quality rules as style references for the LLM.
    Prefer rules with derived_metric_definitions or window/consec expressions.
    """
    rules = stage2.get("anomaly_rules") or []
    if not isinstance(rules, list):
        return []

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        score = 0
        expr = _compact_str(r.get("rule_expression"))
        if r.get("derived_metric_definitions"):
            score += 3
        if "consec_" in expr or "_24h(" in expr or "_6h(" in expr or "duration_" in expr:
            score += 2
        if expr and "ratio" not in expr.lower():
            score += 1
        if _compact_str(r.get("conclusion")):
            score -= 2
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    examples: List[Dict[str, Any]] = []
    for _, r in scored[:max_examples]:
        ex = {k: v for k, v in r.items() if k != "conclusion"}
        examples.append(ex)
    return examples


def _existing_rule_keys(stage2: Dict[str, Any]) -> Tuple[set, set]:
    """
    Returns:
    - existing_ids: set of rule_id
    - existing_signatures: set of "object_name||rule_expression" for basic dedupe
    """
    existing_ids: set = set()
    sigs: set = set()
    rules = stage2.get("anomaly_rules") or []
    if isinstance(rules, list):
        for r in rules:
            if not isinstance(r, dict):
                continue
            rid = _compact_str(r.get("rule_id"))
            if rid:
                existing_ids.add(rid)
            obj = _compact_str(r.get("object_name"))
            expr = _compact_str(r.get("rule_expression"))
            if obj and expr:
                sigs.add(f"{obj}||{expr}")
    return existing_ids, sigs


def _next_rule_id(existing_ids: set, *, prefix: str = "K") -> str:
    """
    Allocate next numeric id with prefix (default 'K') based on existing 'K\\d+' ids.
    """
    mx = 0
    for rid in existing_ids:
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(rid))
        if not m:
            continue
        mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1}"


def _build_prompt(
    *,
    device_type: str,
    derived_metric: Dict[str, Any],
    atomic_catalog_entries: List[Dict[str, Any]],
    existing_rules_brief: List[Dict[str, str]],
    existing_rules_examples: List[Dict[str, Any]],
) -> str:
    return _safe_format_prompt(
        prompt_module.derived_metric_stage2_prompt,
        device_type=device_type,
        derived_metric_json=json.dumps(derived_metric, ensure_ascii=False, indent=2),
        atomic_metrics_catalog_json=json.dumps(atomic_catalog_entries, ensure_ascii=False, indent=2),
        existing_rules_brief_json=json.dumps(existing_rules_brief[:50], ensure_ascii=False, indent=2),
        existing_rules_examples_json=json.dumps(existing_rules_examples, ensure_ascii=False, indent=2),
    )


def _brief_existing_rules(stage2: Dict[str, Any]) -> List[Dict[str, str]]:
    rules = stage2.get("anomaly_rules") or []
    out: List[Dict[str, str]] = []
    if isinstance(rules, list):
        for r in rules:
            if not isinstance(r, dict):
                continue
            out.append(
                {
                    "rule_id": _compact_str(r.get("rule_id")),
                    "object_name": _compact_str(r.get("object_name")),
                    "rule_expression": _compact_str(r.get("rule_expression")),
                }
            )
    return out


# ── Rule expression variable validation ──

_EXPR_KEYWORDS = frozenset({"true", "false", "and", "or", "not", "in", "True", "False", "None"})
_EXPR_FUNCTIONS = frozenset({
    "consec_3h", "consec_2h", "consec_6h", "consec_12h", "consec_24h",
    "duration_1h", "duration_6h", "duration_12h", "duration_24h", "duration_48h", "duration_1095d",
    "avg_24h", "avg_48h", "avg_72h", "avg_7d", "avg_6h", "avg_12h", "avg_1h",
    "std_24h", "std_12h", "std_48h", "std_72h", "std_90d", "std_6h", "std_1h", "std_30d",
    "slope_12h", "slope_14d", "slope_24h", "slope_6h", "slope_1h", "slope_7d",
    "slope_30d", "slope_90d", "slope_180d", "slope_365d",
    "count_365d", "count_1h", "count_6h", "count_12h", "count_24h", "count_48h", "count_14d",
    "max_duration_24h", "max_7d", "min_7d", "max_1h", "max_6h", "max_24h", "max_12h", "max_30d", "max_365d",
    "min_1h", "min_6h", "min_24h", "min_12h", "min_30d", "min_365d",
    "max_1p", "min_1p", "max_abs_delta_1h_over_24h", "rate_12h", "unchanged", "prev_1h",
    "max", "min", "abs", "corr_6h", "delta_24h",
})
_LIT_PLACEHOLDER_RE = re.compile(r"^__LIT_\d+__$")


@dataclass
class MetricVariableContext:
    """Set of referenceable variables registered by metric_catalog + derived_metrics (all lowercase)."""
    atomic_ids: frozenset[str]
    derived_ids: frozenset[str]
    catalog_cn_to_vars: Dict[str, frozenset[str]]
    derived_cn_to_vars: Dict[str, frozenset[str]]


def _normalize_expr_for_identifiers(expr: str) -> str:
    s = _compact_str(expr)
    s = re.sub(r"\bAND\b", "and", s)
    s = re.sub(r"\bOR\b", "or", s)
    s = re.sub(r"\bNOT\b", "not", s)
    s = re.sub(r"\bIN\b", "in", s)
    s = s.replace("= true", "== True").replace("= false", "== False")
    s = re.sub(r"(?<![<>=!])=(?!=)", "==", s)
    return s


def _mask_string_literals(expr: str) -> str:
    lit_pat = re.compile(
        r"(?:'[^'\\]*(?:\\.[^'\\]*)*')|(?:\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
    )
    literals: List[str] = []

    def _mask(m: re.Match) -> str:
        literals.append(m.group(0))
        return f" __LIT_{len(literals) - 1}__ "

    return lit_pat.sub(_mask, expr)


def _extract_identifiers_from_text(text: str) -> set[str]:
    masked = _mask_string_literals(_normalize_expr_for_identifiers(text))
    ids = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", masked))
    ids -= _EXPR_KEYWORDS
    ids -= _EXPR_FUNCTIONS
    ids = {i for i in ids if not _LIT_PLACEHOLDER_RE.fullmatch(i)}
    return ids


def _extract_identifiers_from_expr(expr: str) -> set[str]:
    return _extract_identifiers_from_text(expr)


def _build_metric_variable_context(
    catalog: List[Dict[str, Any]],
    derived_list: List[Dict[str, Any]],
) -> MetricVariableContext:
    atomic_ids: set[str] = set()
    catalog_cn_to_vars: Dict[str, set[str]] = {}

    for item in catalog:
        if not isinstance(item, dict):
            continue
        cn = _compact_str(item.get("metric_name"))
        vars_for_cn: set[str] = set()
        for key in ("variable_name", "db_field_name"):
            v = _compact_str(item.get(key))
            if v:
                atomic_ids.add(v.lower())
                vars_for_cn.add(v)
        if cn:
            catalog_cn_to_vars[cn] = vars_for_cn

    derived_ids: set[str] = set()
    derived_cn_to_vars: Dict[str, set[str]] = {}

    for dm in derived_list:
        if not isinstance(dm, dict):
            continue
        cn = _compact_str(dm.get("derived_name"))
        vars_for_cn: set[str] = set()
        for key in ("derived_name", "derived_key"):
            v = _compact_str(dm.get(key))
            if v:
                derived_ids.add(v.lower())
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
                    vars_for_cn.add(v)
        for am in dm.get("atomic_metrics") or []:
            if not isinstance(am, dict):
                continue
            db = _compact_str(am.get("db_field"))
            if db:
                derived_ids.add(db.lower())
                vars_for_cn.add(db)
        formula = _compact_str(dm.get("formula"))
        for fid in _extract_identifiers_from_text(formula):
            derived_ids.add(fid.lower())
            vars_for_cn.add(fid)
        if cn:
            derived_cn_to_vars[cn] = vars_for_cn

    return MetricVariableContext(
        atomic_ids=frozenset(atomic_ids),
        derived_ids=frozenset(derived_ids),
        catalog_cn_to_vars={k: frozenset(v) for k, v in catalog_cn_to_vars.items()},
        derived_cn_to_vars={k: frozenset(v) for k, v in derived_cn_to_vars.items()},
    )


def _threshold_param_names(rule: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for p in rule.get("threshold_parameters") or []:
        if not isinstance(p, dict):
            continue
        name = _compact_str(p.get("parameter_name"))
        if name:
            out.add(name.lower())
    return out


def _defined_intermediate_names(rule: Dict[str, Any]) -> set[str]:
    """Intermediate variables (lowercase) already defined with a non-empty formula in this rule's derived_metric_definitions."""
    out: set[str] = set()
    for d in rule.get("derived_metric_definitions") or []:
        if not isinstance(d, dict):
            continue
        name = _compact_str(d.get("derived_metric_name"))
        formula = _compact_str(d.get("formula"))
        if name and formula:
            out.add(name.lower())
    return out


def _missing_intermediate_vars(rule: Dict[str, Any], ctx: MetricVariableContext) -> List[str]:
    """
    Variables in the rule expression that belong neither to catalog / derived_metrics / threshold parameters,
    nor are defined by the same name in derived_metric_definitions -> all treated as intermediate variables to be filled.
    """
    expr = _compact_str(rule.get("rule_expression"))
    if not expr:
        return []
    expr_ids = _extract_identifiers_from_expr(expr)
    if not expr_ids:
        return []

    allowed = (
        set(ctx.atomic_ids)
        | set(ctx.derived_ids)
        | _threshold_param_names(rule)
        | _defined_intermediate_names(rule)
    )
    return sorted(i for i in expr_ids if i.lower() not in allowed)


def _validate_rule_expression_semantics(
    rule: Dict[str, Any],
    ctx: MetricVariableContext,
) -> Optional[str]:
    """
    Validate rule expression variables:
    - Allowed: metric_catalog / derived_metrics / threshold parameters / intermediate variable names defined in this rule's derived_metric_definitions
    - Everything else is treated as an intermediate variable; error if there is no non-empty same-name formula definition
    """
    missing = _missing_intermediate_vars(rule, ctx)
    if missing:
        return (
            "Rule expression contains non-unified variable names (must be changed to metric_catalog/derived_metrics "
            "canonical field names, or written into derived_metric_definitions as a true intermediate quantity): "
            + ", ".join(missing)
        )

    # The right-hand side of derived formulas may also only reference catalog / derived / threshold / other defined intermediate variable names
    defined = _defined_intermediate_names(rule)
    base = set(ctx.atomic_ids) | set(ctx.derived_ids) | _threshold_param_names(rule) | defined
    for d in rule.get("derived_metric_definitions") or []:
        if not isinstance(d, dict):
            continue
        name = _compact_str(d.get("derived_metric_name"))
        formula = _compact_str(d.get("formula"))
        if not formula:
            if name:
                return f"the formula of derived_metric_definition {name!r} is empty"
            continue
        bad = sorted(i for i in _extract_identifiers_from_text(formula) if i.lower() not in base)
        if bad:
            return f"the formula of derived_metric_definition {name!r} contains unregistered variables: {', '.join(bad)}"
    return None


@dataclass
class RuleValidationIssue:
    model_name: str
    rule_index: int
    rule_id: str
    error: str
    rule: Dict[str, Any]

    def format(self) -> str:
        return f"{self.model_name} [{self.rule_id}]: {self.error}"


def collect_model_stage2_issues(model_dir: Path) -> Tuple[List[RuleValidationIssue], MetricVariableContext, Dict[str, Any]]:
    """Collect validation issues of all stage2 rules under the model directory. Returns (issues, var_ctx, stage2_obj)."""
    stage2_path = model_dir / "rule_metric_stage2.json"
    catalog_path = model_dir / "metric_catalog.json"
    derived_path = model_dir / "derived_metrics.json"
    if not stage2_path.exists():
        empty_ctx = MetricVariableContext(
            atomic_ids=frozenset(),
            derived_ids=frozenset(),
            catalog_cn_to_vars={},
            derived_cn_to_vars={},
        )
        return (
            [RuleValidationIssue(model_dir.name, -1, "-", "missing rule_metric_stage2.json", {})],
            empty_ctx,
            {},
        )

    stage2 = _read_json(stage2_path)
    catalog = _load_metric_catalog(catalog_path) if catalog_path.exists() else []
    derived_raw = _read_json(derived_path) if derived_path.exists() else {}
    derived_list = derived_raw.get("derived_metrics") if isinstance(derived_raw, dict) else []
    if not isinstance(derived_list, list):
        derived_list = []

    ctx = _build_metric_variable_context(catalog, derived_list)
    issues: List[RuleValidationIssue] = []
    rules = stage2.get("anomaly_rules") or []
    if not isinstance(rules, list):
        issues.append(RuleValidationIssue(model_dir.name, -1, "-", "anomaly_rules is not a list", {}))
        return issues, ctx, stage2

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rid = _compact_str(rule.get("rule_id")) or f"#{i + 1}"
        err = _validate_rule_expression_semantics(rule, ctx)
        if err:
            issues.append(
                RuleValidationIssue(
                    model_name=model_dir.name,
                    rule_index=i,
                    rule_id=rid,
                    error=err,
                    rule=rule,
                )
            )
    return issues, ctx, stage2


def validate_model_stage2_rules(model_dir: Path) -> List[str]:
    """Validate all stage2 rules under the model directory and return a list of issues."""
    issues, _, _ = collect_model_stage2_issues(model_dir)
    return [it.format() for it in issues]


_RULE_FIX_PROMPT = """
You are a power maintenance rule modeling expert. Below is one anomaly rule that failed variable-name validation.
Please fix it and return the complete rule object.

[Validation Error]
{validation_error}

[Unregistered/Pending Symbols]
{missing_vars_json}

[Rule to Fix]
{rule_json}

[Available Atomic Metrics (metric_catalog: must use its "variable_name" or "db_field_name")]
{catalog_json}

[Available Derived Metrics (derived_metrics)]
{derived_json}

[Fix Strategy (in priority order)]
1. **Unify variable names (preferred)**: if an unregistered symbol is an atomic-metric engineering alias (e.g. P_sf6->sf6_gas_pressure, T_action->circuit_breaker_operating_time, PD->pd_level),
   rewrite it in "rule_expression" and "derived_metric_definitions.formula" to the catalog's variable_name/db_field_name.
2. **Do not invent nonexistent fields**: do not write symbols not in the catalog such as pd_level_A / h2o_sf6_B / PD_A.
3. **Delete the rule if it depends on a missing measurement point**: if the rule semantics depend on a measurement point not present in the catalog (e.g. per-phase PD_A/B/C, H2O_A/B/C while the catalog has only single-channel pd_level/h2o_sf6),
   **do not** approximate the semantics with a single channel; instead output:
   {{"_delete": true, "rule_id": "Kxx", "reason": "catalog lacks per-phase measurement points; the original three-phase difference rule cannot be implemented"}}
4. **True intermediate computed quantity**: only when a symbol is the result of a multi-metric computation (e.g. ratio=a/b), add a same-name entry in "derived_metric_definitions";
   the entry must use the fields: derived_metric_name / formula / note (do not reuse the derived_name/formula/atomic_metrics structure of derived_metrics.json)
5. Do not modify metric_catalog; do not output "conclusion"; keep "rule_id" (except when deleting)
6. Output a single JSON object only, no Markdown, no explanation
""".strip()


def _normalize_derived_defs_schema(rule_obj: Dict[str, Any]) -> bool:
    """
    Normalize the derived_metrics-style entries the LLM may return into stage2's derived_metric_name/formula/note.
    Returns whether a modification occurred.
    """
    defs = rule_obj.get("derived_metric_definitions")
    if not isinstance(defs, list):
        rule_obj["derived_metric_definitions"] = []
        return True
    before = json.dumps(defs, ensure_ascii=False, sort_keys=True)
    out: List[Dict[str, str]] = []
    for d in defs:
        if not isinstance(d, dict):
            continue
        name = _compact_str(
            d.get("derived_metric_name") or d.get("derived_name") or d.get("derived_key")
        )
        formula = _compact_str(d.get("formula") or d.get("formula"))
        note = _compact_str(d.get("note") or d.get("description"))
        if not name and not formula:
            continue
        out.append({"derived_metric_name": name, "formula": formula, "note": note})
    rule_obj["derived_metric_definitions"] = out
    after = json.dumps(out, ensure_ascii=False, sort_keys=True)
    return before != after


def normalize_model_stage2_derived_defs(model_dir: Path, *, dry_run: bool = False) -> Tuple[int, int]:
    """
    Normalize the derived_metric_definitions schema for all rules in the model directory's rule_metric_stage2.json (no LLM call).
    Returns (number of modified rules, total rules).
    """
    stage2_path = model_dir / "rule_metric_stage2.json"
    if not stage2_path.exists():
        return 0, 0
    stage2 = _read_json(stage2_path)
    rules = stage2.get("anomaly_rules")
    if not isinstance(rules, list):
        return 0, 0
    n_changed = 0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        touched = _normalize_derived_defs_schema(rule)
        had_conclusion = "conclusion" in rule
        _ensure_threshold_params_have_value(rule)
        rule.pop("conclusion", None)
        if touched or had_conclusion:
            n_changed += 1
    if n_changed and not dry_run:
        stage2["anomaly_rules"] = rules
        _write_json(stage2_path, stage2)
    return n_changed, len(rules)


def _build_fix_prompt(
    *,
    rule: Dict[str, Any],
    validation_error: str,
    missing_vars: List[str],
    catalog: List[Dict[str, Any]],
    derived_list: List[Dict[str, Any]],
) -> str:
    cat_brief = []
    for it in catalog:
        if not isinstance(it, dict):
            continue
        cat_brief.append(
            {
                "metric_name": it.get("metric_name"),
                "variable_name": it.get("variable_name"),
                "db_field_name": it.get("db_field_name"),
                "unit": it.get("unit"),
            }
        )
    dm_brief = []
    for dm in derived_list:
        if not isinstance(dm, dict):
            continue
        dm_brief.append(
            {
                "derived_name": dm.get("derived_name"),
                "derived_key": dm.get("derived_key"),
                "formula": dm.get("formula"),
                "atomic_metrics": dm.get("atomic_metrics"),
                "output_unit": dm.get("output_unit"),
            }
        )
    return _RULE_FIX_PROMPT.format(
        validation_error=validation_error,
        missing_vars_json=json.dumps(missing_vars, ensure_ascii=False),
        rule_json=json.dumps(rule, ensure_ascii=False, indent=2),
        catalog_json=json.dumps(cat_brief, ensure_ascii=False, indent=2),
        derived_json=json.dumps(dm_brief, ensure_ascii=False, indent=2),
    )


def _merge_fixed_rule(original: Dict[str, Any], fixed: Dict[str, Any]) -> Dict[str, Any]:
    """Overwrite the original rule with the LLM fix result, keeping the rule_id."""
    out = dict(original)
    for k, v in fixed.items():
        if k == "rule_id":
            continue
        out[k] = v
    if _compact_str(original.get("rule_id")):
        out["rule_id"] = original["rule_id"]
    elif _compact_str(fixed.get("rule_id")):
        out["rule_id"] = fixed["rule_id"]
    _normalize_rule_obj(out)
    return out


def repair_rule_with_llm(
    *,
    rule: Dict[str, Any],
    validation_error: str,
    catalog: List[Dict[str, Any]],
    derived_list: List[Dict[str, Any]],
    ctx: MetricVariableContext,
    client: OpenAICompatClient,
    temperature: float,
    max_new_tokens: int,
    max_attempts: int = 2,
    verbose: bool = False,
    no_stream: bool = False,
    stream_label: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call the LLM to fill in derived_metric_definitions for intermediate variables until validation passes or attempts are exhausted."""
    last_err: Optional[str] = validation_error
    current = rule
    for attempt in range(1, max_attempts + 1):
        missing = _missing_intermediate_vars(current, ctx)
        prompt = _build_fix_prompt(
            rule=current,
            validation_error=last_err or validation_error,
            missing_vars=missing,
            catalog=catalog,
            derived_list=derived_list,
        )
        if verbose:
            print(f"      -> LLM fill derived defs attempt {attempt}/{max_attempts}: missing={missing}", flush=True)
        raw = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=not no_stream,
            verbose=verbose,
            stream_label=stream_label or "fix_rule",
        )
        try:
            fixed_obj = _extract_first_json(raw)
        except Exception as e:
            last_err = f"LLM output is not JSON: {e}"
            continue
        if not isinstance(fixed_obj, dict):
            last_err = f"LLM output is not an object: {type(fixed_obj)}"
            continue
        if fixed_obj.get("_delete") in (True, "true", "True", 1, "1"):
            return {
                "_delete": True,
                "rule_id": _compact_str(fixed_obj.get("rule_id") or current.get("rule_id")),
                "reason": _compact_str(fixed_obj.get("reason")) or "catalog lacks necessary measurement points; the rule cannot be implemented",
            }, None
        merged = _merge_fixed_rule(current, fixed_obj)
        schema_err = _validate_rule_obj(merged, ctx=ctx)
        if schema_err:
            last_err = schema_err
            current = merged
            continue
        return merged, None
    return None, last_err


def repair_invalid_stage2_rules(
    *,
    model_dir: Path,
    client: OpenAICompatClient,
    temperature: float,
    max_new_tokens: int,
    max_attempts: int = 2,
    dry_run: bool = False,
    verbose: bool = False,
    no_stream: bool = False,
) -> str:
    """Validate the entire stage2: rewrite/fill non-unified variables via LLM; rules that cannot be implemented may be deleted."""
    stage2_path = model_dir / "rule_metric_stage2.json"
    catalog_path = model_dir / "metric_catalog.json"
    derived_path = model_dir / "derived_metrics.json"
    if not stage2_path.exists():
        return f"[SKIP] {model_dir.name}: missing rule_metric_stage2.json"

    # First do deterministic schema normalization (no LLM): derived_name/formula -> derived_metric_name/formula
    n_schema, n_rules = normalize_model_stage2_derived_defs(model_dir, dry_run=dry_run)
    if verbose and n_schema:
        prefix = "[DRY normalize]" if dry_run else "[normalize]"
        print(f"{prefix} {model_dir.name}: corrected derived_metric_definitions field names for {n_schema}/{n_rules} rules", flush=True)

    issues, ctx, stage2 = collect_model_stage2_issues(model_dir)
    if not issues:
        if n_schema:
            return f"[OK] {model_dir.name}: no invalid rules (schema_normalized={n_schema})"
        return f"[OK] {model_dir.name}: no invalid rules"

    catalog = _load_metric_catalog(catalog_path) if catalog_path.exists() else []
    derived_raw = _read_json(derived_path) if derived_path.exists() else {}
    derived_list = derived_raw.get("derived_metrics") if isinstance(derived_raw, dict) else []
    if not isinstance(derived_list, list):
        derived_list = []

    rules = stage2.get("anomaly_rules")
    if not isinstance(rules, list):
        return f"[SKIP] {model_dir.name}: anomaly_rules is not a list"

    if verbose:
        print(f"\n[repair] {model_dir.name}: found {len(issues)} rules to fill", flush=True)

    fixed_n = 0
    deleted_n = 0
    failed_n = 0
    delete_indices: List[int] = []
    for issue in issues:
        if issue.rule_index < 0 or issue.rule_index >= len(rules):
            failed_n += 1
            continue
        rid = issue.rule_id
        if verbose:
            print(f"\n  [{rid}] {issue.error}", flush=True)
        if dry_run:
            missing = _missing_intermediate_vars(rules[issue.rule_index], ctx)
            print(f"  [DRY] would LLM-normalize or delete; missing={missing}")
            continue
        fixed, err = repair_rule_with_llm(
            rule=rules[issue.rule_index],
            validation_error=issue.error,
            catalog=catalog,
            derived_list=derived_list,
            ctx=ctx,
            client=client,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_attempts=max_attempts,
            verbose=verbose,
            no_stream=no_stream,
            stream_label=f"{model_dir.name}/{rid}/fix",
        )
        if fixed is None:
            failed_n += 1
            print(f"  [FAIL] {rid}: {err}")
            continue
        if fixed.get("_delete"):
            delete_indices.append(issue.rule_index)
            deleted_n += 1
            if verbose:
                print(f"  OK deleted rule [{rid}]: {_compact_str(fixed.get('reason'))}", flush=True)
            continue
        rules[issue.rule_index] = fixed
        fixed_n += 1
        if verbose:
            print(f"  OK filled [{rid}]", flush=True)

    if dry_run:
        return f"[DRY] {model_dir.name}: would repair {len(issues)} rules"

    for idx in sorted(set(delete_indices), reverse=True):
        if 0 <= idx < len(rules):
            del rules[idx]

    if fixed_n > 0 or deleted_n > 0:
        stage2["anomaly_rules"] = rules
        _write_json(stage2_path, stage2)

    return (
        f"[OK] {model_dir.name}: fixed={fixed_n}, deleted={deleted_n}, "
        f"failed={failed_n}, total_issues={len(issues)}"
    )


def _validate_rule_obj(r: Any, *, ctx: Optional[MetricVariableContext] = None) -> Optional[str]:
    required_keys = {
        "object_name",
        "object_type",
        "applicable_conditions",
        "pointwise_judgment",
        "time_series_requirements",
        "input_metrics",
        "derived_metric_definitions",
        "judgment_description",
        "rule_expression",
        "threshold_parameters",
    }
    if not isinstance(r, dict):
        return "rule is not object"
    missing = [k for k in required_keys if k not in r]
    if missing:
        return f"missing keys: {missing}"
    # light checks
    if not _compact_str(r.get("object_name")):
        return "object_name empty"
    if not isinstance(r.get("input_metrics"), list) or not r.get("input_metrics"):
        return "input_metrics empty"
    if not _compact_str(r.get("judgment_description")):
        return "judgment_description empty"
    if not _compact_str(r.get("rule_expression")):
        return "rule_expression empty"
    if not isinstance(r.get("derived_metric_definitions"), list):
        return "derived_metric_definitions not list"
    if not isinstance(r.get("threshold_parameters"), list):
        return "threshold_parameters not list"
    # Each param should contain parameter_name/meaning/parameter_value (parameter_value may be a placeholder, but the field must exist)
    for p in (r.get("threshold_parameters") or []):
        if not isinstance(p, dict):
            return "threshold_parameters contains non-object"
        if not _compact_str(p.get("parameter_name")):
            return "threshold_parameters missing parameter_name"
        if not _compact_str(p.get("meaning")):
            return "threshold_parameters missing meaning"
        if "parameter_value" not in p:
            return "threshold_parameters missing parameter_value"
    if ctx is not None:
        err = _validate_rule_expression_semantics(r, ctx)
        if err:
            return err
    return None


def _ensure_threshold_params_have_value(rule_obj: Dict[str, Any]) -> None:
    """
    Ensure each threshold_parameters item contains a parameter_value field.
    """
    params = rule_obj.get("threshold_parameters")
    if not isinstance(params, list):
        rule_obj["threshold_parameters"] = []
        return
    for p in params:
        if not isinstance(p, dict):
            continue
        if "parameter_value" not in p:
            p["parameter_value"] = "to be determined"
        else:
            v = p.get("parameter_value")
            p["parameter_value"] = _compact_str(v) if v is not None else ""


def _normalize_rule_obj(rule_obj: Dict[str, Any]) -> None:
    """Align generated rule with stage2 schema used by build_maintenance_metrics."""
    rule_obj.pop("conclusion", None)
    _ensure_threshold_params_have_value(rule_obj)
    _normalize_derived_defs_schema(rule_obj)


def generate_and_append_for_model(
    *,
    model_dir: Path,
    client: OpenAICompatClient,
    max_new_tokens: int,
    temperature: float,
    dry_run: bool,
    rule_id_prefix: str,
    verbose: bool = False,
    no_stream: bool = False,
    auto_fix: bool = True,
    fix_max_attempts: int = 2,
) -> str:
    derived_path = model_dir / "derived_metrics.json"
    stage2_path = model_dir / "rule_metric_stage2.json"
    metric_catalog_path = model_dir / "metric_catalog.json"
    if not derived_path.exists():
        return f"[SKIP] {model_dir.name}: missing derived_metrics.json"
    if not stage2_path.exists():
        return f"[SKIP] {model_dir.name}: missing rule_metric_stage2.json"

    stage2 = _read_json(stage2_path)
    device_type = _compact_str(stage2.get("device_type") or stage2.get("device_type") or model_dir.name)
    existing_ids, existing_sigs = _existing_rule_keys(stage2)
    existing_brief = _brief_existing_rules(stage2)
    style_examples = _pick_style_example_rules(stage2)
    catalog = _load_metric_catalog(metric_catalog_path) if metric_catalog_path.exists() else []

    derived = _read_json(derived_path)
    dm_list = derived.get("derived_metrics") if isinstance(derived, dict) else None
    if not isinstance(dm_list, list) or not dm_list:
        return f"[SKIP] {model_dir.name}: derived_metrics.json has no derived_metrics"

    var_ctx = _build_metric_variable_context(catalog, dm_list)

    if verbose:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[Model directory] {model_dir.name}")
        print(f"[Device type] {device_type}")
        print(f"[Derived metrics] {len(dm_list)}")
        print(f"[Existing rules] {len(existing_ids)}")
        print(f"{'='*60}")

    new_rules: List[Dict[str, Any]] = []
    for idx, dm in enumerate(dm_list):
        if not isinstance(dm, dict):
            continue
        dm_name = _compact_str(dm.get("derived_name") or dm.get("name") or f"#{idx + 1}")
        if verbose:
            print(f"\n  [{idx + 1}/{len(dm_list)}] Processing derived metric: {dm_name}")

        prompt = _build_prompt(
            device_type=device_type,
            derived_metric=dm,
            atomic_catalog_entries=_atomic_catalog_entries_for_derived(catalog, dm),
            existing_rules_brief=existing_brief,
            existing_rules_examples=style_examples,
        )

        if verbose:
            t_call = time.time()
            print(f"    -> calling LLM ({client.model}) ...", flush=True)

        raw = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=not no_stream,
            verbose=verbose,
            stream_label=f"{model_dir.name}/{dm_name}",
        )

        if verbose:
            elapsed = time.time() - t_call
            print(f"\n    <- LLM response done, elapsed {elapsed:.1f}s, response length {len(raw)} chars", flush=True)

        js = _extract_first_json(raw)
        if not isinstance(js, dict) or "rules" not in js or not isinstance(js["rules"], list):
            raise RuntimeError(f"{model_dir.name}: LLM output must be {{'rules':[...]}}, got={type(js)}")

        generated = js["rules"]
        if verbose:
            print(f"    -> generated rule candidates: {len(generated)}", flush=True)

        accepted = 0
        for cand in generated:
            err = _validate_rule_obj(cand, ctx=var_ctx)
            if err and auto_fix:
                if verbose:
                    print(f"      x validation failed, LLM filling derived defs: {err}", flush=True)
                fixed, fix_err = repair_rule_with_llm(
                    rule=cand,
                    validation_error=err,
                    catalog=catalog,
                    derived_list=dm_list,
                    ctx=var_ctx,
                    client=client,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_attempts=fix_max_attempts,
                    verbose=verbose,
                    no_stream=no_stream,
                    stream_label=f"{model_dir.name}/{dm_name}/fix",
                )
                if fixed is None:
                    if verbose:
                        print(f"      x repair failed: {fix_err}", flush=True)
                    continue
                cand = fixed
                err = None
            if err:
                if verbose:
                    print(f"      x skipping invalid rule: {err}", flush=True)
                continue
            # assign rule id
            rid = _next_rule_id(existing_ids, prefix=rule_id_prefix)
            existing_ids.add(rid)
            cand["rule_id"] = rid
            _normalize_rule_obj(cand)
            # dedupe by signature
            obj = _compact_str(cand.get("object_name"))
            expr = _compact_str(cand.get("rule_expression"))
            sig = f"{obj}||{expr}" if obj and expr else ""
            if sig and sig in existing_sigs:
                if verbose:
                    print(f"      x skipping duplicate rule: {obj}", flush=True)
                continue
            if sig:
                existing_sigs.add(sig)
            new_rules.append(cand)
            accepted += 1
            if verbose:
                print(f"      OK accepted rule [{rid}]: {obj}", flush=True)

        if verbose and accepted == 0 and len(generated) > 0:
            print(f"      (all candidate rules were filtered out)", flush=True)

    # No more accidental placeholder empties — already filtered inline
    if not new_rules:
        if verbose:
            print(f"\n[Result] {model_dir.name}: no new rules", flush=True)
        return f"[OK] {model_dir.name}: no new rules generated"

    if verbose:
        print(f"\n[Result] {model_dir.name}: generated {len(new_rules)} new rules in total", flush=True)

    if dry_run:
        return f"[DRY] {model_dir.name}: would append rules={len(new_rules)} to rule_metric_stage2.json"

    rules_list = stage2.get("anomaly_rules")
    if not isinstance(rules_list, list):
        stage2["anomaly_rules"] = []
        rules_list = stage2["anomaly_rules"]
    rules_list.extend(new_rules)
    _write_json(stage2_path, stage2)

    # After generation: validate the entire stage2, and use the LLM to fill in rules missing intermediate variable definitions
    if auto_fix and not dry_run:
        if verbose:
            print(f"\n[post-validation] {model_dir.name}: validating all anomaly rules and filling derived definitions...", flush=True)
        post = repair_invalid_stage2_rules(
            model_dir=model_dir,
            client=client,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_attempts=fix_max_attempts,
            dry_run=False,
            verbose=verbose,
            no_stream=no_stream,
        )
        if verbose:
            print(f"[post-validation] {post}", flush=True)

    if verbose:
        t_total = time.time() - t0
        print(f"[write] {model_dir.name}: appended {len(new_rules)} rules to rule_metric_stage2.json, total elapsed {t_total:.1f}s", flush=True)
    return f"[OK] {model_dir.name}: appended rules={len(new_rules)}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate stage2 anomaly rules from derived_metrics.json via LLM and append into rule_metric_stage2.json")
    ap.add_argument("--models_dir", type=str, default=str(__import__("powerbench.paths", fromlist=["demo_models_dir"]).demo_models_dir()))
    ap.add_argument("--only_folder", type=str, default="", help="only process one subfolder (name or full path)")
    ap.add_argument("--start_from", type=int, default=0, help="skip first N folders (sorted order)")
    ap.add_argument("--concurrency", type=int, default=4, help="number of concurrent folders to process (default: 4)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rule_id_prefix", type=str, default="K", help="rule id prefix for newly appended rules (default: K)")
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument("--model", type=str, default="deepseek-v3.2")
    ap.add_argument("--api-key", type=str, default="YOUR_API_KEY", help="used to generate conclusions of the \"special concern\" type; skipped if missing")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=8096)
    ap.add_argument("--sleep_s", type=float, default=0.0, help="sleep between LLM calls (seconds)")
    ap.add_argument("--verbose", action="store_true", default=False, help="print detailed progress logs (default: False)")
    ap.add_argument("--no-stream", action="store_true", default=False, help="disable LLM streaming output (default: stream enabled)")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="only validate the rule expression variables of existing rule_metric_stage2.json, no LLM call",
    )
    ap.add_argument(
        "--fix-invalid",
        action="store_true",
        help="validate the entire stage2: use the LLM to fill in derived_metric_definitions for undefined intermediate variables and write back",
    )
    ap.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="do no validation/repair during generation; and skip the LLM fill of the entire stage2 after generation",
    )
    ap.add_argument(
        "--fix-max-attempts",
        type=int,
        default=2,
        help="max LLM fill attempts per rule (default 2)",
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")

    only = _compact_str(args.only_folder)
    if only:
        p = Path(only)
        targets = [p if p.is_absolute() else (models_dir / only)]
    else:
        subs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
        targets = subs[max(0, int(args.start_from)) :]

    if args.validate_only and args.fix_invalid:
        raise SystemExit("cannot specify both --validate-only and --fix-invalid")

    if args.validate_only:
        total_issues = 0
        for folder in targets:
            if not folder.exists() or not folder.is_dir():
                print(f"[SKIP] {folder}: not a directory")
                continue
            issues = validate_model_stage2_rules(folder)
            if issues:
                total_issues += len(issues)
                for msg in issues:
                    print(f"[INVALID] {msg}")
            else:
                print(f"[OK] {folder.name}: all rules passed expression variable checks")
        print(f"[DONE validate-only] issues={total_issues}")
        return 0 if total_issues == 0 else 2

    api_key = _compact_str(getattr(args, "api_key", ""))
    if not api_key:
        raise SystemExit("--api-key is required")

    client = OpenAICompatClient(api_key=api_key, base_url=str(args.base_url), model=str(args.model))

    if args.fix_invalid:
        ok = 0
        fail = 0
        for folder in targets:
            if not folder.exists() or not folder.is_dir():
                print(f"[SKIP] {folder}: not a directory")
                continue
            try:
                msg = repair_invalid_stage2_rules(
                    model_dir=folder,
                    client=client,
                    temperature=float(args.temperature),
                    max_new_tokens=int(args.max_new_tokens),
                    max_attempts=max(1, int(args.fix_max_attempts)),
                    dry_run=bool(args.dry_run),
                    verbose=bool(args.verbose),
                    no_stream=bool(args.no_stream),
                )
                print(msg)
                ok += 1
            except Exception as e:
                print(f"[FAIL] {folder.name}: {e}")
                fail += 1
        print(f"[DONE fix-invalid] ok={ok}, fail={fail}, dry_run={bool(args.dry_run)}")
        return 0 if fail == 0 else 2

    ok = 0
    fail = 0
    conc = max(1, int(args.concurrency))

    def _one(folder: Path) -> str:
        if not folder.exists() or not folder.is_dir():
            return f"[SKIP] {folder}: not a directory"
        msg = generate_and_append_for_model(
            model_dir=folder,
            client=client,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            dry_run=bool(args.dry_run),
            rule_id_prefix=_compact_str(args.rule_id_prefix) or "K",
            verbose=bool(args.verbose),
            no_stream=bool(args.no_stream),
            auto_fix=not bool(args.no_auto_fix),
            fix_max_attempts=max(1, int(args.fix_max_attempts)),
        )
        if float(args.sleep_s) > 0:
            time.sleep(float(args.sleep_s))
        return msg

    with ThreadPoolExecutor(max_workers=conc) as ex:
        fut_map = {ex.submit(_one, folder): folder for folder in targets if folder.exists() and folder.is_dir()}
        for fut in as_completed(fut_map):
            folder = fut_map[fut]
            try:
                msg = fut.result()
                print(msg)
                ok += 1
            except Exception as e:
                print(f"[FAIL] {folder.name}: {e}")
                fail += 1

    print(f"[DONE] ok={ok}, fail={fail}, dry_run={bool(args.dry_run)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
