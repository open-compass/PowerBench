from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..config import PilotConfig
from ..corpus import INDEXED_SUFFIXES
from ..models import CorpusInventory
from ..util import (
    directory_content_sha256,
    directory_manifest_sha256,
)
from .validation import summarize_validation


def build_generation_report(
    config: PilotConfig,
    *,
    raw_root: Path,
    output_root: Path,
    inventory: CorpusInventory,
    locator_count: int,
    supported_rule_count: int,
    semantic_tasks: list[dict[str, Any]],
    l1_rejections: dict[str, Any],
    replay_stats: dict[str, Any],
    validation: dict[str, Any],
    family_pool_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe an already-committed deterministic draft dataset."""

    raw_manifest_sha256, indexed_raw_file_count = directory_manifest_sha256(
        raw_root, suffixes=INDEXED_SUFFIXES
    )
    raw_content_sha256, content_file_count = directory_content_sha256(
        raw_root, suffixes=INDEXED_SUFFIXES
    )
    if content_file_count != indexed_raw_file_count:
        raise RuntimeError("Raw manifest/content hash file counts disagree")
    semantic_counts = Counter(task["task_level"] for task in semantic_tasks)
    family_counts = Counter(
        task.get("task_family")
        or {"L1": "post_window", "L2": "threshold_replay", "L3": "multi_device_replay"}[
            task["task_level"]
        ]
        for task in semantic_tasks
    )
    answer_counts = {
        "L2_triggered_true": sum(
            task["task_level"] == "L2" and task["answer"]["triggered"] for task in semantic_tasks
        ),
        "L2_triggered_false": sum(
            task["task_level"] == "L2" and not task["answer"]["triggered"]
            for task in semantic_tasks
        ),
        "L3_triggered_device_count": dict(
            sorted(
                Counter(
                    len(task["answer"]["triggered_devices"])
                    for task in semantic_tasks
                    if task["task_level"] == "L3" and "triggered_devices" in task["answer"]
                ).items()
            )
        ),
        "L3_target_device_count": dict(
            sorted(
                Counter(
                    len(task["target_device_uids"])
                    for task in semantic_tasks
                    if task["task_level"] == "L3"
                ).items()
            )
        ),
        "L2_counterfactual_changed_output": sum(
            task["task_level"] == "L2"
            and bool(task.get("diagnostics", {}).get("counterfactual_changed_output"))
            for task in semantic_tasks
        ),
        "L2_relaxed_threshold": sum(
            task["task_level"] == "L2"
            and task.get("diagnostics", {}).get("intervention_direction") == "relax"
            for task in semantic_tasks
        ),
    }
    diversity = {
        level: {
            "unique_models": len(
                {
                    uid.split("/", 1)[0]
                    for task in semantic_tasks
                    if task["task_level"] == level
                    for uid in task["target_device_uids"]
                }
            ),
            "unique_devices": len(
                {
                    uid
                    for task in semantic_tasks
                    if task["task_level"] == level
                    for uid in task["target_device_uids"]
                }
            ),
        }
        for level in ("L1", "L2", "L3")
    }
    return {
        "schema_version": config.data["schema_version"],
        "config_path": str(config.source),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "frozen_device_count": len(inventory.devices),
        "device_model_count": len(inventory.metrics_by_model),
        "inventory_artifact_count": len(inventory.artifacts),
        "indexed_raw_file_count": indexed_raw_file_count,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_content_sha256": raw_content_sha256,
        "strong_indirect_locator_count": locator_count,
        "supported_rule_count": supported_rule_count,
        "semantic_task_counts": dict(sorted(semantic_counts.items())),
        "task_family_counts": dict(sorted(family_counts.items())),
        "public_question_count": len(semantic_tasks),
        "question_mode": "autonomous_single_prompt",
        "question_generation_status": "draft",
        "scope_sizes": list(config.scope_sizes),
        "scope_seeds": list(config.scope_seeds),
        "l1_rejections": l1_rejections,
        "replay_pool": replay_stats,
        "family_candidate_pool_stats": family_pool_stats,
        "answer_balance": answer_counts,
        "task_diversity": diversity,
        "validation": summarize_validation(validation),
    }


__all__ = ["build_generation_report"]