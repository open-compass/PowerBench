from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..bundle import create_bundle
from ..config import PilotConfig
from ..corpus import build_corpus_index
from ..inventory import build_inventory
from ..records import pack_task_records
from ..rules import hydrate_supported_rules
from ..util import write_json, write_jsonl
from .calibration_tasks import (
    build_context_conditioned_pool,
    build_document_routed_pool,
    build_heterogeneous_metric_pool,
)
from .family_tasks import (
    compact_candidate_manifest,
    l1_family_pools,
    l2_family_pools,
    l3_family_pools,
    select_family_tasks,
)
from .gold import attach_gold_execution
from .l1_tasks import build_l1_candidate_pool, generate_l1_tasks
from .locators import build_strong_locators
from .question_generation import (
    build_generation_record,
    question_texts,
    validate_generation_records,
)
from .render import render_task_pair
from .replay_tasks import build_replay_pool, generate_l2_tasks, generate_l3_tasks
from .report import build_generation_report
from .scopes import build_scope_rows
from .validation import assert_valid, validate_generated_records


def generate_pilot(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    raw = Path(raw_root).resolve()
    output = Path(output_root).resolve()
    inventory = build_inventory(raw)
    if len(inventory.devices) != 761:
        raise ValueError(
            f"Pilot is defined on the frozen 761-device corpus, found {len(inventory.devices)}"
        )
    supported_rule_count = hydrate_supported_rules(inventory)
    candidate_search = config.section("candidate_search")
    locators = build_strong_locators(
        inventory.devices.values(),
        max_fields=int(candidate_search["locator_max_fields"]),
    )

    l1_config = config.section("l1")
    l2_config = config.section("l2")
    l3_config = config.section("l3")
    replay_windows = sorted(
        set(int(value) for value in l2_config["window_hours"]) | {int(l3_config["window_hours"])}
    )
    replay_factors = sorted(
        set(float(value) for value in l2_config["threshold_factors"])
        | set(float(value) for value in l3_config["threshold_factors"])
    )
    replay_pool, replay_stats = build_replay_pool(
        inventory,
        locators,
        window_hours=replay_windows,
        threshold_factors=replay_factors,
        seed=config.seed,
    )
    private_root = output / "private"
    family_counts = config.task_family_counts
    candidate_manifest: list[dict[str, Any]] = []
    family_pool_stats: dict[str, Any] = {}
    if family_counts is None:
        l1_tasks, l1_rejections = generate_l1_tasks(
            inventory,
            locators,
            count=config.task_counts["L1"],
            window_hours=[int(value) for value in l1_config["window_hours"]],
            digits=int(l1_config["answer_round_digits"]),
            seed=config.seed,
            max_reports_per_device=int(candidate_search["l1_max_reports_per_device"]),
            max_metrics_per_device=int(candidate_search["l1_max_metrics_per_device"]),
        )
        l2_tasks = generate_l2_tasks(
            replay_pool,
            count=config.task_counts["L2"],
            window_hours=[int(value) for value in l2_config["window_hours"]],
            seed=config.seed,
        )
        l3_tasks = generate_l3_tasks(
            replay_pool,
            count=config.task_counts["L3"],
            window_hours=int(l3_config["window_hours"]),
            seed=config.seed,
            device_counts=config.l3_device_counts,
            max_local_options_per_device=int(candidate_search["l3_max_local_options_per_device"]),
            max_candidates_per_group_pattern=int(
                candidate_search["l3_max_candidates_per_group_pattern"]
            ),
        )
        semantic_tasks = attach_gold_execution([*l1_tasks, *l2_tasks, *l3_tasks], inventory)
    else:
        old_l1_requested = any(
            count
            for family, count in family_counts["L1"].items()
            if family != "document_routed_window"
        )
        old_l3_window_requested = any(
            family_counts["L3"].get(family, 0)
            for family in ("aligned_window_comparison", "filter_then_compute")
        )
        if old_l1_requested or old_l3_window_requested:
            l1_pool, l1_rejections = build_l1_candidate_pool(
                inventory,
                locators,
                window_hours=[int(value) for value in l1_config["window_hours"]],
                digits=int(l1_config["answer_round_digits"]),
                seed=config.seed,
                max_reports_per_device=int(candidate_search["l1_max_reports_per_device"]),
                max_metrics_per_device=int(candidate_search["l1_max_metrics_per_device"]),
            )
        else:
            l1_pool, l1_rejections = [], {"pool_size": 0, "skipped_unused_pool": True}
        requested_l3_replay = int(family_counts["L3"].get("multi_device_replay", 0))
        l3_strata = len(config.l3_device_counts)
        base_l3_count = max(30, requested_l3_replay)
        base_l3_count += (-base_l3_count) % l3_strata
        base_l3 = (
            generate_l3_tasks(
                replay_pool,
                count=base_l3_count,
                window_hours=int(l3_config["window_hours"]),
                seed=config.seed,
                device_counts=config.l3_device_counts,
                max_local_options_per_device=int(
                    candidate_search["l3_max_local_options_per_device"]
                ),
                max_candidates_per_group_pattern=int(
                    candidate_search["l3_max_candidates_per_group_pattern"]
                ),
            )
            if requested_l3_replay
            else []
        )
        l1_families = (
            l1_family_pools(
                inventory,
                locators,
                l1_pool,
                replay_pool,
                digits=int(l1_config["answer_round_digits"]),
                seed=config.seed,
            )
            if old_l1_requested
            else {}
        )
        old_l2_requested = any(
            count
            for family, count in family_counts["L2"].items()
            if family != "context_conditioned_replay"
        )
        l2_families = (
            l2_family_pools(inventory, replay_pool, locators, seed=config.seed)
            if old_l2_requested
            else {}
        )
        routed_pool: list[dict[str, Any]] = []
        if family_counts["L1"].get("document_routed_window", 0) or family_counts["L3"].get(
            "heterogeneous_metric_comparison", 0
        ):
            routed_pool = build_document_routed_pool(
                inventory,
                locators,
                window_hours=[int(value) for value in l1_config["window_hours"]],
                digits=int(l1_config["answer_round_digits"]),
                seed=config.seed,
                max_reports_per_device=int(candidate_search["l1_max_reports_per_device"]),
                max_candidates=max(
                    60,
                    2
                    * (
                        int(family_counts["L1"].get("document_routed_window", 0))
                        + max(config.l3_device_counts)
                        * int(
                            family_counts["L3"].get("heterogeneous_metric_comparison", 0)
                        )
                    ),
                ),
            )
        l1_families["document_routed_window"] = routed_pool
        l2_families["context_conditioned_replay"] = (
            build_context_conditioned_pool(
                inventory,
                replay_pool,
                threshold_factors=[
                    float(value) for value in l2_config["threshold_factors"]
                ],
            )
            if family_counts["L2"].get("context_conditioned_replay", 0)
            else []
        )
        old_l3_requested = any(
            count
            for family, count in family_counts["L3"].items()
            if family != "heterogeneous_metric_comparison"
        )
        l3_families = (
            l3_family_pools(inventory, base_l3, l1_pool, seed=config.seed)
            if old_l3_requested
            else {}
        )
        l3_families["heterogeneous_metric_comparison"] = (
            build_heterogeneous_metric_pool(
                inventory,
                routed_pool,
                device_counts=config.l3_device_counts,
                digits=int(l1_config["answer_round_digits"]),
                seed=config.seed,
            )
            if family_counts["L3"].get("heterogeneous_metric_comparison", 0)
            else []
        )
        pools = {"L1": l1_families, "L2": l2_families, "L3": l3_families}
        family_pool_stats = {
            level: {
                family: {
                    "candidate_count": len(candidates),
                    "device_count": len(
                        {uid for task in candidates for uid in task["target_device_uids"]}
                    ),
                    "model_count": len(
                        {
                            inventory.device(uid).model_key
                            for task in candidates
                            for uid in task["target_device_uids"]
                        }
                    ),
                }
                for family, candidates in families.items()
            }
            for level, families in pools.items()
        }
        context_candidates = pools["L2"].get("context_conditioned_replay", [])
        if context_candidates:
            distribution = Counter(
                (
                    str(task["answer"]["load_category"]),
                    bool(task["answer"]["triggered"]),
                )
                for task in context_candidates
            )
            family_pool_stats["L2"]["context_conditioned_replay"][
                "load_outcome_distribution"
            ] = {
                f"{category}/{str(triggered).lower()}": distribution[(category, triggered)]
                for category in ("high", "normal", "low")
                for triggered in (True, False)
            }
        candidate_manifest = compact_candidate_manifest(pools)
        semantic_tasks = attach_gold_execution(
            select_family_tasks(pools, family_counts, seed=config.seed), inventory
        )

    question_generation = [
        build_generation_record(
            task,
            inventory,
            schema_version=str(config.data["schema_version"]),
        )
        for task in semantic_tasks
    ]
    generation_by_pair = {row["pair_id"]: row for row in question_generation}

    questions: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for task in semantic_tasks:
        public_pair, gold_pair = render_task_pair(
            task,
            inventory,
            schema_version=str(config.data["schema_version"]),
            question_texts=question_texts(generation_by_pair[task["pair_id"]]),
        )
        questions.extend(public_pair)
        gold.extend(gold_pair)
    scopes = build_scope_rows(
        semantic_tasks,
        inventory,
        config.scope_sizes,
        config.scope_seeds,
    )
    validation = validate_generated_records(
        questions,
        gold,
        semantic_tasks,
        scopes,
        inventory,
        expected_scope_sizes=config.scope_sizes,
        expected_scope_seeds=config.scope_seeds,
        question_generation=question_generation,
    )
    assert_valid(validation)
    task_records = pack_task_records(
        semantic_tasks,
        gold,
        schema_version=str(config.data["schema_version"]),
    )

    # Commit generated artifacts only after the in-memory build passes every
    # validator. A failed generation cannot partially overwrite a valid build.
    public_root = output / "public"
    write_jsonl(public_root / "questions.jsonl", questions)
    write_jsonl(public_root / "scopes.jsonl", scopes)
    write_jsonl(private_root / "tasks.jsonl", task_records)
    write_jsonl(private_root / "question_generation.jsonl", question_generation)
    if candidate_manifest:
        write_jsonl(private_root / "candidate_pool.jsonl", candidate_manifest)
    # The full replay pool is deterministically reproducible from raw + config;
    # storing it added a large duplicate artifact without improving auditability.
    write_json(
        private_root / "audit.json",
        {
            "l1_rejections": l1_rejections,
            "replay_candidate_stats": replay_stats,
            "family_candidate_pool_stats": family_pool_stats,
            "validation": validation,
            "question_generation": validate_generation_records(
                question_generation, semantic_tasks, inventory
            ),
        },
    )

    return build_generation_report(
        config,
        raw_root=raw,
        output_root=output,
        inventory=inventory,
        locator_count=len(locators),
        supported_rule_count=supported_rule_count,
        semantic_tasks=semantic_tasks,
        l1_rejections=l1_rejections,
        replay_stats=replay_stats,
        validation=validation,
        family_pool_stats=family_pool_stats,
    )


def build_benchmark(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Generate, index, and bind one model-independent benchmark bundle."""

    output = Path(output_root).resolve()
    generation = generate_pilot(config, raw_root=raw_root, output_root=output)
    index_path = output / "public" / "corpus.sqlite"
    index = build_corpus_index(raw_root=raw_root, database_path=index_path)
    bundle = create_bundle(
        config,
        output_root=output,
        generation_report=generation,
        index_path=index_path,
    )
    return {
        "bundle_id": bundle.bundle_id,
        "bundle_path": str(bundle.path),
        "index": index,
        "generation": generation,
    }


__all__ = ["build_benchmark", "generate_pilot"]