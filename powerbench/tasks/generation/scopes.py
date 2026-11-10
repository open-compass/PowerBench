from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

from ..models import CorpusInventory, Device

_CALIBRATION_FAMILIES = {
    "document_routed_window",
    "context_conditioned_replay",
    "heterogeneous_metric_comparison",
}
_HARD_NEGATIVE_COUNT = 16


def _stable_tiebreak(pair_id: str, seed: int, uid: str) -> str:
    return hashlib.sha256(f"{pair_id}\x1f{seed}\x1f{uid}".encode()).hexdigest()


def _metadata_values(device: Device) -> Counter[str]:
    ignored = {"Archive ID", "Device ID", "Device name", "Tag number", "Device tag"}
    return Counter(
        str(value).strip()
        for key, value in device.metadata.items()
        if key not in ignored and str(value).strip()
    )


def distractor_score(candidate: Device, targets: list[Device]) -> int:
    """Higher means a harder, more target-like distractor."""

    candidate_values = _metadata_values(candidate)
    best = 0
    for target in targets:
        target_values = _metadata_values(target)
        score = 0
        if candidate.model_key == target.model_key:
            score += 1000
        score += 35 * len(set(candidate.csv_columns).intersection(target.csv_columns))
        score += 80 * sum((candidate_values & target_values).values())
        if candidate.name[:4] == target.name[:4]:
            score += 20
        best = max(best, score)
    return best


def select_hard_negatives(
    target_uids: Iterable[str],
    inventory: CorpusInventory,
    *,
    count: int = _HARD_NEGATIVE_COUNT,
) -> list[str]:
    targets = [inventory.device(uid) for uid in sorted(set(target_uids))]
    target_set = {device.uid for device in targets}
    ranked = sorted(
        (device for device in inventory.devices.values() if device.uid not in target_set),
        key=lambda device: (-distractor_score(device, targets), device.uid),
    )
    return [device.uid for device in ranked[:count] if distractor_score(device, targets) > 0]


def make_nested_scope(
    pair_id: str,
    target_uids: Iterable[str],
    required_confusers: Iterable[str],
    inventory: CorpusInventory,
    seed: int,
) -> list[str]:
    targets = [inventory.device(uid) for uid in sorted(set(target_uids))]
    target_set = {device.uid for device in targets}
    required = sorted(set(required_confusers).difference(target_set))
    all_candidates = [
        device for device in inventory.devices.values() if device.uid not in target_set
    ]
    required_set = set(required)
    remainder = [device for device in all_candidates if device.uid not in required_set]
    remainder.sort(
        key=lambda device: (
            -distractor_score(device, targets),
            _stable_tiebreak(pair_id, seed, device.uid),
            device.uid,
        )
    )
    ordered = [device.uid for device in targets]
    ordered.extend(required)
    ordered.extend(device.uid for device in remainder)
    if len(ordered) != len(inventory.devices) or len(set(ordered)) != len(ordered):
        raise AssertionError(f"Invalid scope ordering for {pair_id}")
    return ordered


def build_scope_rows(
    semantic_tasks: Iterable[dict[str, Any]],
    inventory: CorpusInventory,
    scope_sizes: Iterable[int],
    scope_seeds: Iterable[int],
) -> list[dict[str, Any]]:
    sizes = tuple(scope_sizes)
    seeds = tuple(scope_seeds)
    corpus_size = len(inventory.devices)
    if not sizes or sizes[-1] != corpus_size:
        raise ValueError(f"Largest scope must equal frozen corpus size ({corpus_size})")
    rows: list[dict[str, Any]] = []
    for task in sorted(semantic_tasks, key=lambda item: item["pair_id"]):
        targets = list(task["target_device_uids"])
        confusers: set[str] = set()
        for locator in task.get("locators", {}).values():
            confusers.update(locator.get("ablation_confusers", []))
        hard_negatives = (
            select_hard_negatives(targets, inventory)
            if task.get("task_family") in _CALIBRATION_FAMILIES
            else []
        )
        required = set(confusers) | set(hard_negatives)
        for seed in seeds:
            ordering = make_nested_scope(task["pair_id"], targets, required, inventory, seed)
            for size in sizes:
                if size < len(set(targets) | required):
                    raise ValueError(
                        f"N={size} cannot include targets and required hard negatives for "
                        f"{task['pair_id']}"
                    )
                # S761 is identical for every seed, so only materialize it once.
                if size == corpus_size and seed != seeds[0]:
                    continue
                members = ordering[:size] if size < corpus_size else sorted(inventory.devices)
                rows.append(
                    {
                        "pair_id": task["pair_id"],
                        "scope_seed": seed,
                        "N": size,
                        "device_uids": members,
                        "target_device_uids": sorted(targets),
                        "required_locator_confusers": sorted(confusers),
                        "required_hard_negatives": sorted(hard_negatives),
                    }
                )
    return rows


def validate_nested_scopes(rows: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["pair_id"], int(row["scope_seed"])), []).append(row)
    for (pair_id, seed), group in grouped.items():
        previous: set[str] = set()
        for row in sorted(group, key=lambda item: int(item["N"])):
            current = set(row["device_uids"])
            if len(current) != int(row["N"]):
                errors.append(f"{pair_id}/seed={seed}/N={row['N']}: wrong cardinality")
            if not previous.issubset(current):
                errors.append(f"{pair_id}/seed={seed}: scopes are not nested")
            required = (
                set(row["target_device_uids"])
                | set(row["required_locator_confusers"])
                | set(row.get("required_hard_negatives", []))
            )
            if not required.issubset(current):
                errors.append(f"{pair_id}/seed={seed}/N={row['N']}: missing target/confuser")
            previous = current
    return errors


__all__ = [
    "build_scope_rows",
    "distractor_score",
    "make_nested_scope",
    "select_hard_negatives",
    "validate_nested_scopes",
]