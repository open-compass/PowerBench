from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jsonschema

from ..bundle import load_bundle
from ..config import PilotConfig
from ..generation.metric_eligibility import (
    audit_metric_eligibility,
    clarify_affected_records,
)
from ..generation.question_generation import (
    STYLE_BRIEFS,
    validate_generation_record,
    validate_generation_records,
)
from ..inventory import build_inventory
from ..models import (
    CorpusInventory,
    PolishProvenance,
    QuestionGenerationRecord,
    QuestionTextParts,
)
from ..records import semantic_tasks_from_records
from ..rules import hydrate_supported_rules
from ..util import canonical_json, read_jsonl, write_jsonl
from .publication import publish_question_records

POLISH_POLICY_VERSION = "autonomous-goal-v2"

POLISH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": ["question"],
}

POLISH_INSTRUCTIONS = """You edit questions for a power-system O&M retrieval-and-reasoning benchmark. Rewrite the supplied draft into one natural, concise question that states only the business goal.

Rules:
1. Use only facts from the input draft; never add, guess, or alter attribute conditions, times, thresholds, units, or result requirements.
2. Do not output or infer any answer.
3. Fully preserve the conjunctive business attributes that identify devices and events, the counterfactual change, evaluation time range, window boundaries, whether consecutive state carries across the evaluation boundary, and the final conclusions requested.
4. Do not mention stable device identifiers, file paths, report numbers, rule numbers, CSV field names, tool names, retrieval order, or calculation code.
5. Do not teach formulas, retrieval order, JSON fields, or solution steps; window boundaries and continuity across the evaluation boundary are part of the task definition and must remain.
6. For L3, preserve the mapping between D1, D2, D3 (and any additional aliases) and their respective conditions.
7. Do not use generation-process language such as "fault injection succeeded", "injection modified field", "discovery-time strategy", "near_miss", or "per_device annotation".
8. Keep one compact natural-language paragraph; do not expand into an operations manual, and do not exceed the draft length by much.
9. Return only a JSON object of the form {\"question\": \"...\"}; no explanation or Markdown code fences.
"""


def _polish_prompt(
    record: QuestionGenerationRecord, *, prior_errors: list[str] | None = None
) -> str:
    style_brief = dict(STYLE_BRIEFS)[record.style_id]
    payload: dict[str, Any] = {
        "policy_version": POLISH_POLICY_VERSION,
        "task_level": record.task_level,
        "style_brief": style_brief,
        "draft": record.draft.model_dump(mode="json"),
    }
    if prior_errors:
        payload["previous_validation_errors"] = prior_errors
        payload["revision_instruction"] = "Fix these errors and still return only the required JSON."
    return canonical_json(payload)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _chat_completion_text(
    completion: Any,
    *,
    requested_model: str,
    stream_output: Callable[[str], None] | None,
) -> tuple[str, str, str]:
    parts: list[str] = []
    response_id = ""
    resolved_model = requested_model
    for chunk in completion:
        response_id = str(_value(chunk, "id", response_id) or response_id)
        resolved_model = str(_value(chunk, "model", resolved_model) or resolved_model)
        choices = _value(chunk, "choices", []) or []
        if not choices:
            continue
        delta = _value(choices[0], "delta", {}) or {}
        content = _value(delta, "content", "") or ""
        if not isinstance(content, str):
            continue
        parts.append(content)
        if stream_output is not None and content:
            stream_output(content)
    if stream_output is not None and parts:
        stream_output("\n")
    return "".join(parts), response_id, resolved_model


def polish_generation_record(
    raw_record: Mapping[str, Any],
    task: Mapping[str, Any],
    inventory: CorpusInventory,
    *,
    client: Any,
    model: str,
    max_attempts: int = 3,
    progress: Callable[[str], None] | None = None,
    stream_output: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Polish and locally validate one question without supplying its answer."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    source = QuestionGenerationRecord.model_validate(raw_record)
    prior_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        if progress is not None:
            progress(f"request attempt {attempt}/{max_attempts}")
        prompt = _polish_prompt(source, prior_errors=prior_errors)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": POLISH_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
            stream=True,
        )
        response_text, response_id, resolved_model = _chat_completion_text(
            completion,
            requested_model=model,
            stream_output=stream_output,
        )
        try:
            value = json.loads(response_text)
            jsonschema.validate(value, POLISH_RESPONSE_SCHEMA)
        except json.JSONDecodeError as exc:
            prior_errors = [f"invalid JSON response: {exc.msg}"]
        except jsonschema.ValidationError as exc:
            prior_errors = [f"invalid structured response: {exc.message}"]
        else:
            candidate = source.model_copy(
                update={
                    "final": QuestionTextParts(question=value["question"].strip()),
                    "status": "polished",
                    "provenance": PolishProvenance(
                        policy_version=POLISH_POLICY_VERSION,
                        requested_model=model,
                        resolved_model=resolved_model,
                        response_id=response_id,
                        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        attempt_count=attempt,
                    ),
                    "validation": {},
                }
            ).model_dump(mode="json")
            report = validate_generation_record(candidate, task, inventory)
            if report["valid"]:
                candidate["validation"] = report
                return QuestionGenerationRecord.model_validate(candidate).model_dump(mode="json")
            prior_errors = list(report["errors"])
        if progress is not None:
            progress(f"deterministic validation failed: {'; '.join(prior_errors)}")
    raise ValueError(
        f"Question polishing failed deterministic validation for {source.pair_id}: {prior_errors}"
    )


def _select_batch(records: list[dict[str, Any]], per_level: int) -> list[dict[str, Any]]:
    """Select up to N not-yet-published questions from each level."""

    if per_level < 1:
        raise ValueError("per_level must be positive")
    chosen: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        if record.get("status") == "polished":
            continue
        level = str(record["task_level"])
        if counts[level] < per_level:
            chosen.append(record)
            counts[level] += 1
    return chosen


def polish_benchmark_questions(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    output_root: str | Path,
    client: Any,
    model: str,
    per_level: int | None = None,
    max_attempts: int = 3,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
    stream_output: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Polish a resumable batch and publish it into the official bundle."""

    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_root).resolve()
    bundle = load_bundle(output / "bundle.json")
    bundle.verify_inputs(config, raw_root)
    inventory = build_inventory(raw_root)
    hydrate_supported_rules(inventory)
    task_records = read_jsonl(output / "private" / "tasks.jsonl")
    tasks = semantic_tasks_from_records(task_records)
    records = read_jsonl(output / "private" / "question_generation.jsonl")
    ledger_validation = validate_generation_records(records, tasks, inventory)
    if not ledger_validation["valid"]:
        raise ValueError(f"Cannot polish an invalid question ledger: {ledger_validation['errors']}")

    tasks_by_pair = {str(task["pair_id"]): task for task in tasks}
    selected = _select_batch(records, per_level) if per_level is not None else records
    checkpoint_path = output / "private" / ".question_polish_checkpoint.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    cache_candidates = list(records)
    if checkpoint_path.is_file():
        cache_candidates.extend(read_jsonl(checkpoint_path))
    for candidate in cache_candidates:
        pair_id = str(candidate.get("pair_id", ""))
        task = tasks_by_pair.get(pair_id)
        provenance = candidate.get("provenance") or {}
        if (
            task is not None
            and provenance.get("policy_version") == POLISH_POLICY_VERSION
            and provenance.get("requested_model") == model
            and validate_generation_records([candidate], [task], inventory, require_polished=True)[
                "valid"
            ]
        ):
            cached[pair_id] = candidate

    polished_by_pair: dict[str, dict[str, Any]] = {}
    total = len(selected)

    def save_checkpoint() -> None:
        write_jsonl(
            checkpoint_path,
            [cached[row["pair_id"]] for row in records if row["pair_id"] in cached],
        )

    def polish_one(record: dict[str, Any], *, show_stream: bool) -> dict[str, Any]:
        pair_id = str(record["pair_id"])
        return polish_generation_record(
            record,
            tasks_by_pair[pair_id],
            inventory,
            client=client,
            model=model,
            max_attempts=max_attempts,
            progress=(
                (lambda message, current=pair_id: progress(f"  {current}: {message}"))
                if progress is not None
                else None
            ),
            stream_output=stream_output if show_stream else None,
        )

    pending: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(selected, start=1):
        pair_id = str(record["pair_id"])
        result = cached.get(pair_id)
        if result is None:
            pending.append((index, record))
            continue
        polished_by_pair[pair_id] = result
        if progress is not None:
            progress(
                f"[{index}/{total}] {pair_id} ({record['task_level']}, {record['style_id']}): cache"
            )
            progress(f"[{index}/{total}] {pair_id}: validated")

    if workers == 1:
        for index, record in pending:
            pair_id = str(record["pair_id"])
            if progress is not None:
                progress(
                    f"[{index}/{total}] {pair_id} "
                    f"({record['task_level']}, {record['style_id']}): model"
                )
            try:
                result = polish_one(record, show_stream=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Question polishing stopped at {pair_id}. Previously validated pairs "
                    f"are saved in {checkpoint_path} and will be reused on rerun. Cause: {exc}"
                ) from exc
            cached[pair_id] = result
            polished_by_pair[pair_id] = result
            save_checkpoint()
            if progress is not None:
                progress(f"[{index}/{total}] {pair_id}: validated")
    elif pending:
        actual_workers = min(workers, len(pending))
        if progress is not None:
            progress(
                f"Starting {len(pending)} model requests with {actual_workers} workers; "
                "streamed JSON is buffered to prevent interleaved output."
            )
            for index, record in pending:
                progress(
                    f"[{index}/{total}] {record['pair_id']} "
                    f"({record['task_level']}, {record['style_id']}): model"
                )
        failures: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(polish_one, record, show_stream=False): (index, record)
                for index, record in pending
            }
            for future in as_completed(futures):
                index, record = futures[future]
                pair_id = str(record["pair_id"])
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append((pair_id, exc))
                    if progress is not None:
                        progress(
                            f"[{index}/{total}] {pair_id}: failed: {type(exc).__name__}: {exc}"
                        )
                    continue
                cached[pair_id] = result
                polished_by_pair[pair_id] = result
                save_checkpoint()
                if progress is not None:
                    progress(f"[{index}/{total}] {pair_id}: validated")
        if failures:
            details = "; ".join(
                f"{pair_id}: {type(exc).__name__}: {exc}" for pair_id, exc in failures
            )
            raise RuntimeError(
                f"Question polishing failed for {len(failures)} pair(s). Successful pairs are "
                f"saved in {checkpoint_path} and will be reused on rerun. {details}"
            )

    committed_records = [polished_by_pair.get(str(record["pair_id"]), record) for record in records]
    committed_records, _ = clarify_affected_records(
        committed_records,
        audit_metric_eligibility(tasks, inventory),
    )
    publication = publish_question_records(
        config,
        output_root=output,
        source_bundle=bundle,
        inventory=inventory,
        tasks=tasks,
        task_records=task_records,
        records=committed_records,
    )
    if publication["status"] == "polished":
        checkpoint_path.unlink(missing_ok=True)
    return {
        **publication,
        "model": model,
        "committed_pair_count": len(polished_by_pair),
    }


__all__ = [
    "POLISH_INSTRUCTIONS",
    "POLISH_POLICY_VERSION",
    "POLISH_RESPONSE_SCHEMA",
    "polish_benchmark_questions",
    "polish_generation_record",
]