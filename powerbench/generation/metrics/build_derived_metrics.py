from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import HTTPError

from powerbench.generation.metrics import derived_metric_prompt as prompt_module


from powerbench.paths import demo_models_dir

DEFAULT_MODELS_DIR = str(demo_models_dir())


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
    Prompt contains lots of JSON braces; avoid str.format collisions.
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
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            payload["max_tokens"] = int(max_new_tokens)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except HTTPError as e:
            try:
                body = r.text  # type: ignore[name-defined]
            except Exception:
                body = "<no response body>"
            raise RuntimeError(f"HTTP error calling LLM: {e}; response_body={body[:2000]}") from e
        data = r.json()
        return data["choices"][0]["message"]["content"]


def call_stage1(
    client: OpenAICompatClient,
    *,
    device_type: str,
    metric_catalog_json: str,
    temperature: float,
    max_new_tokens: Optional[int],
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    prompt_text = _safe_format_prompt(
        prompt_module.stage_1_prompt,
        device_type=device_type,
        metric_catalog_json=metric_catalog_json,
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
    *,
    device_type: str,
    metric_catalog_json: str,
    derived_metric_json: str,
    temperature: float,
    max_new_tokens: Optional[int],
    max_retries: int = 3,
) -> Tuple[str, str]:
    prompt_text = _safe_format_prompt(
        prompt_module.stage_2_prompt,
        device_type=device_type,
        metric_catalog_json=metric_catalog_json,
        derived_metric_json=derived_metric_json,
    )
    messages = [{"role": "user", "content": prompt_text}]
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            verdict = str(raw).strip()
            # must be exactly yes/no (ignore surrounding whitespace)
            if verdict not in {"yes", "no"}:
                # tolerate JSON like {"verdict":"yes"} (some models still do)
                try:
                    js = _extract_first_json_object(raw)
                    v = str(js.get("verdict") or js.get("is_valid") or "").strip()
                    verdict = v if v in {"yes", "no"} else verdict
                except Exception:
                    pass
            if verdict not in {"yes", "no"}:
                raise ValueError(f"Stage2 output not valid verdict: {raw[:200]!r}")
            return raw, verdict
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Stage2 failed after {max_retries} retries: {last_err}")


def _device_type_from_folder_name(folder_name: str) -> str:
    name = str(folder_name).strip()
    if "_" in name and name.split("_", 1)[0].isdigit():
        return name.split("_", 1)[1].strip()
    return name


def _filter_continuous_metrics(metric_catalog_json: str) -> str:
    """
    Only keep continuous-valued metrics for derived-metric generation.

        We drop items whose "discrete_values" is non-empty (list/dict/string/any truthy value),
    because the derived-metric pipeline only supports continuous numeric computation.
    """
    try:
        items = json.loads(metric_catalog_json)
    except Exception:
        # If it's not valid JSON, keep original to avoid breaking the pipeline.
        return metric_catalog_json

    if not isinstance(items, list):
        return metric_catalog_json

    kept: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        dv = it.get("discrete_values")
        # treat any non-empty as discrete
        if dv is None:
            kept.append(it)
            continue
        if isinstance(dv, str):
            if dv.strip() == "":
                kept.append(it)
            continue
        if isinstance(dv, (list, tuple, set, dict)):
            if len(dv) == 0:
                kept.append(it)
            continue
        # fallback: bool(dv) indicates discrete
        if not bool(dv):
            kept.append(it)

    return json.dumps(kept, ensure_ascii=False, indent=2)


def process_one_folder(
    client: OpenAICompatClient,
    sub: Path,
    *,
    temperature: float,
    max_new_tokens: Optional[int],
    verbose: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    prefix = f"[{sub.name}]"
    metric_path = sub / "metric_catalog.json"
    if not metric_path.exists():
        return f"[SKIP] {sub.name}: metric_catalog.json not found", None

    device_type = _device_type_from_folder_name(sub.name)
    metric_catalog_json = metric_path.read_text(encoding="utf-8")
    metric_catalog_json = _filter_continuous_metrics(metric_catalog_json)

    if verbose:
        print(f"{prefix} stage1 calling LLM...")
    _s1_raw, s1_json = call_stage1(
        client,
        device_type=device_type,
        metric_catalog_json=metric_catalog_json,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    derived = s1_json.get("derived_metrics", []) if isinstance(s1_json, dict) else []
    if not isinstance(derived, list):
        derived = []

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, str]] = []

    # Stage2: validate EACH metric; output must be just yes/no; filter out no
    for m in derived:
        if not isinstance(m, dict):
            continue
        dm_json = json.dumps(m, ensure_ascii=False, indent=2)
        if verbose:
            print(f"{prefix} stage2 check: {m.get('derived_key', '') or m.get('derived_name', '')}")
        _raw2, verdict = call_stage2(
            client,
            device_type=device_type,
            metric_catalog_json=metric_catalog_json,
            derived_metric_json=dm_json,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if verdict == "yes":
            kept.append(m)
        else:
            removed.append({"derived_key": str(m.get("derived_key", "")), "reason": "LLM review determined does not meet constraints"})

    final_obj: Dict[str, Any] = {"device_type": device_type, "derived_metrics": kept, "removed": removed}

    # only save ONE final json
    final_path = sub / "derived_metrics.json"
    final_path.write_text(json.dumps(final_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"{prefix} saved: {final_path.name}")

    return f"[OK] {sub.name}: saved derived_metrics.json", final_obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Build derived metrics (2-stage) for each models subfolder.")
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--only_folder", type=str, default="", help="Only process a single subfolder (name or path).")
    ap.add_argument("--start_from", type=int, default=0, help="Skip first N subfolders (sorted).")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrency (default 1).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument("--model", type=str, default="deepseek-v3.2")
    ap.add_argument(
        "--api-key",
        type=str,
        default="YOUR_API_KEY",
        help="LLM API Key (recommended). If empty, will try OPENAI_API_KEY env var.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="derived_metrics_all.json",
        help="Aggregate output json filename under repo root (default: derived_metrics_all.json)",
    )
    args = ap.parse_args()

    api_key = _compact_str(args.api_key) or _compact_str(__import__("os").environ.get("OPENAI_API_KEY"))

    client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"models_dir not found: {models_dir}")

    subdirs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[INFO] No subfolders under: {models_dir}")
        return 0

    subdirs_sorted = sorted(subdirs)
    only_folder = _compact_str(args.only_folder)
    if only_folder:
        p = Path(only_folder)
        target = p if p.is_absolute() else (models_dir / only_folder)
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

    aggregate: Dict[str, Any] = {"models_dir": str(models_dir), "generated": []}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_map = {
            ex.submit(
                process_one_folder,
                client,
                sub,
                temperature=float(args.temperature),
                max_new_tokens=int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else None,
                verbose=bool(args.verbose),
            ): sub
            for sub in subdirs_sorted
        }
        for fut in as_completed(fut_map):
            sub = fut_map[fut]
            try:
                msg, js = fut.result()
                done += 1
                print(f"{msg}  ({done}/{total})")
                if js is not None:
                    aggregate["generated"].append(
                        {
                            "folder": sub.name,
                            "device_type": _device_type_from_folder_name(sub.name),
                            "derived_metrics": js.get("derived_metrics", []),
                            "removed": js.get("removed", []),
                        }
                    )
            except Exception as e:
                done += 1
                print(f"[FAIL] {sub.name}: {e}  ({done}/{total})")

    # save aggregate under repo root (same dir as this script)
    repo_root = Path(__file__).resolve().parent
    out_path = repo_root / str(args.out)
    out_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved aggregate: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

