from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import GoldRecord
from .task_specs import public_answer
from .util import read_jsonl


def pack_task_records(
    semantic_tasks: Iterable[dict[str, Any]],
    gold_records: Iterable[dict[str, Any]],
    *,
    schema_version: str,
) -> list[dict[str, Any]]:
    """Store one private record per autonomous semantic task."""

    tasks = list(semantic_tasks)
    by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in gold_records:
        gold = GoldRecord.model_validate(raw).model_dump(mode="json")
        by_pair[gold["pair_id"]].append(gold)

    packed: list[dict[str, Any]] = []
    for task in tasks:
        pair_id = str(task["pair_id"])
        variants = by_pair.get(pair_id, [])
        if len(variants) == 2 and {row["anchor_type"] for row in variants} == {
            "Direct",
            "Indirect",
        }:
            record = dict(task)
            record["schema_version"] = schema_version
            record["variants"] = {
                row["anchor_type"]: {
                    "qa_id": row["qa_id"],
                    "evidence_atoms": row["evidence_atoms"],
                    "complexity": row["complexity"],
                    "diagnostics": row.get("diagnostics", {}),
                }
                for row in variants
            }
            packed.append(record)
            continue
        if len(variants) != 1:
            raise ValueError(f"{pair_id} has an invalid number of Gold records")
        gold = variants[0]
        record = dict(task)
        record["schema_version"] = schema_version
        record["evaluation"] = {
            "qa_id": gold["qa_id"],
            "evidence_atoms": gold["evidence_atoms"],
            "complexity": gold["complexity"],
            "diagnostics": gold.get("diagnostics", {}),
        }
        packed.append(record)
    if set(by_pair) != {str(task["pair_id"]) for task in tasks}:
        raise ValueError("Gold and semantic-task pair IDs differ")
    return packed


def expand_gold_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand packed task records; also accept legacy Gold and paired records."""

    source = list(rows)
    if not source:
        return []
    if all("qa_id" in row and "variants" not in row and "evaluation" not in row for row in source):
        return source

    expanded: list[dict[str, Any]] = []
    for task in source:
        evaluation = task.get("evaluation")
        if isinstance(evaluation, dict):
            level = task["task_level"]
            answer = public_answer(level, task["answer"], task.get("task_family"))
            record = GoldRecord(
                schema_version=str(task["schema_version"]),
                qa_id=str(evaluation["qa_id"]),
                pair_id=str(task["pair_id"]),
                task_level=level,
                task_family=task.get("task_family"),
                anchor_type="Indirect",
                target_device_uids=list(task["target_device_uids"]),
                answer=answer,
                tolerances={
                    key: value for key, value in task.get("tolerances", {}).items() if key in answer
                },
                evidence_atoms=evaluation["evidence_atoms"],
                gold_trace=list(task["gold_trace"]),
                complexity=evaluation["complexity"],
                diagnostics=dict(evaluation.get("diagnostics", {})),
            )
            expanded.append(record.model_dump(mode="json"))
            continue
        variants = task.get("variants")
        if not isinstance(variants, dict):
            raise ValueError(f"Task record {task.get('pair_id')} has no variants")
        for anchor_type in ("Direct", "Indirect"):
            variant = variants.get(anchor_type)
            if not isinstance(variant, dict):
                raise ValueError(f"Task record {task.get('pair_id')} has no {anchor_type} variant")
            record = GoldRecord(
                schema_version=str(task["schema_version"]),
                qa_id=str(variant["qa_id"]),
                pair_id=str(task["pair_id"]),
                task_level=task["task_level"],
                task_family=task.get("task_family"),
                anchor_type=anchor_type,
                target_device_uids=list(task["target_device_uids"]),
                answer=dict(task["answer"]),
                tolerances=dict(task.get("tolerances", {})),
                evidence_atoms=variant["evidence_atoms"],
                gold_trace=list(task["gold_trace"]),
                complexity=variant["complexity"],
                diagnostics=dict(variant.get("diagnostics", {})),
            )
            expanded.append(record.model_dump(mode="json"))
    return expanded


def read_gold_records(path: str | Path) -> list[dict[str, Any]]:
    return expand_gold_records(read_jsonl(Path(path)))


def semantic_tasks_from_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return verifier-ready task payloads from packed private records."""

    result: list[dict[str, Any]] = []
    for row in rows:
        task = {
            key: value
            for key, value in row.items()
            if key not in {"schema_version", "variants", "evaluation"}
        }
        result.append(task)
    return result