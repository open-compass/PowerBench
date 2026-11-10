from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..bundle import BenchmarkBundle, create_bundle, load_bundle
from ..config import PilotConfig
from ..generation.metric_eligibility import (
    UNIT_ELIGIBILITY_CLAUSE,
    audit_metric_eligibility,
    clarify_affected_records,
)
from ..generation.question_generation import (
    polished_public_texts,
    validate_generation_records,
)
from ..generation.render import render_task_pair
from ..generation.validation import (
    assert_valid,
    summarize_validation,
    validate_generated_records,
)
from ..inventory import build_inventory
from ..models import CorpusInventory
from ..records import expand_gold_records, semantic_tasks_from_records
from ..rules import hydrate_supported_rules
from ..util import read_jsonl, write_json, write_jsonl


def publish_question_records(
    config: PilotConfig,
    *,
    output_root: str | Path,
    source_bundle: BenchmarkBundle,
    inventory: CorpusInventory,
    tasks: list[dict[str, Any]],
    task_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate, render and bind one complete mixed draft/polished ledger."""

    output = Path(output_root).resolve()
    eligibility_findings = audit_metric_eligibility(tasks, inventory)
    eligibility_pair_ids = [str(item["pair_id"]) for item in eligibility_findings]
    eligibility_pair_set = set(eligibility_pair_ids)
    missing_clarifications = sorted(
        str(record["pair_id"])
        for record in records
        if str(record["pair_id"]) in eligibility_pair_set
        and UNIT_ELIGIBILITY_CLAUSE not in str(record["final"]["question"])
    )
    unexpected_clarifications = sorted(
        str(record["pair_id"])
        for record in records
        if str(record["pair_id"]) not in eligibility_pair_set
        and UNIT_ELIGIBILITY_CLAUSE in str(record["final"]["question"])
    )
    if missing_clarifications or unexpected_clarifications:
        raise ValueError(
            "Metric-eligibility clarification scope is invalid: "
            f"missing={missing_clarifications}, unexpected={unexpected_clarifications}"
        )
    polished_pair_ids = sorted(
        str(record["pair_id"]) for record in records if record.get("status") == "polished"
    )
    all_polished = len(polished_pair_ids) == len(records)
    language_validation = validate_generation_records(
        records,
        tasks,
        inventory,
        require_polished=all_polished,
    )
    if not language_validation["valid"]:
        raise ValueError(f"Polished question ledger is invalid: {language_validation['errors']}")

    texts_by_pair = polished_public_texts(records)
    questions: list[dict[str, Any]] = []
    regenerated_gold: list[dict[str, Any]] = []
    for task in tasks:
        public_pair, gold_pair = render_task_pair(
            task,
            inventory,
            schema_version=str(config.data["schema_version"]),
            question_texts=texts_by_pair[task["pair_id"]],
        )
        questions.extend(public_pair)
        regenerated_gold.extend(gold_pair)

    stored_gold = expand_gold_records(task_records)
    if regenerated_gold != stored_gold:
        raise ValueError("Question polishing unexpectedly changed regenerated Gold records")
    scopes = read_jsonl(output / "public" / "scopes.jsonl")
    validation = validate_generated_records(
        questions,
        stored_gold,
        tasks,
        scopes,
        inventory,
        expected_scope_sizes=config.scope_sizes,
        expected_scope_seeds=config.scope_seeds,
        question_generation=records,
        require_polished_questions=all_polished,
    )
    assert_valid(validation)

    audit_path = output / "private" / "audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read existing audit record: {exc}") from exc
    audit["question_generation"] = language_validation
    audit["validation"] = validation
    audit["metric_eligibility_clarification"] = {
        "policy": (
            "catalog unit must be non-empty and exactly equal the inspection-report unit"
        ),
        "clause": UNIT_ELIGIBILITY_CLAUSE,
        "pair_count": len(eligibility_pair_ids),
        "pair_ids": eligibility_pair_ids,
        "findings": eligibility_findings,
        "gold_reexecution_matches_stored": True,
    }

    # Publish only after the complete in-memory representation passes all
    # validators.  The private ledger/audit and public questions are distinct
    # artifacts even though one bundle binds both for reproducibility.
    write_jsonl(output / "private" / "question_generation.jsonl", records)
    write_jsonl(output / "public" / "questions.jsonl", questions)
    write_json(audit_path, audit)

    generation_report = dict(source_bundle.data)
    generation_report.pop("polish_model", None)
    polish_models = sorted(
        {
            str(record["provenance"]["requested_model"])
            for record in records
            if record.get("provenance")
        }
    )
    status = "polished" if all_polished else "partial"
    generation_report.update(
        {
            "question_generation_status": status,
            "polished_pair_ids": polished_pair_ids,
            "polished_pair_count": len(polished_pair_ids),
            "polish_models": polish_models,
            "metric_eligibility_clarification_count": len(eligibility_pair_ids),
            "metric_eligibility_clarification_pair_ids": eligibility_pair_ids,
            "validation": summarize_validation(validation),
        }
    )
    if all_polished and len(polish_models) == 1:
        generation_report["polish_model"] = polish_models[0]
    final_bundle = create_bundle(
        config,
        output_root=output,
        generation_report=generation_report,
        index_path=output / "public" / "corpus.sqlite",
    )
    final_bundle.verify()
    return {
        "status": status,
        "polished_pair_count": len(polished_pair_ids),
        "total_pair_count": len(records),
        "question_count": len(questions),
        "questions_path": str(output / "public" / "questions.jsonl"),
        "bundle_id": final_bundle.bundle_id,
        "bundle_path": str(final_bundle.path),
        "validation": summarize_validation(validation),
        "metric_eligibility_clarification_count": len(eligibility_pair_ids),
        "metric_eligibility_clarification_pair_ids": eligibility_pair_ids,
    }


def apply_metric_eligibility_clarifications(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    output_root: str | Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Apply the audited wording clarification without calling a language model."""

    output = Path(output_root).resolve()
    bundle = load_bundle(output / "bundle.json")
    bundle.verify_inputs(config, raw_root)
    inventory = build_inventory(raw_root)
    hydrate_supported_rules(inventory)
    task_records = read_jsonl(output / "private" / "tasks.jsonl")
    tasks = semantic_tasks_from_records(task_records)
    records = read_jsonl(output / "private" / "question_generation.jsonl")
    findings = audit_metric_eligibility(tasks, inventory)
    if expected_count is not None and len(findings) != expected_count:
        raise ValueError(
            f"Expected {expected_count} affected questions, found {len(findings)}; "
            "refusing to modify the frozen bundle"
        )
    clarified, changed_pair_ids = clarify_affected_records(records, findings)
    result = publish_question_records(
        config,
        output_root=output,
        source_bundle=bundle,
        inventory=inventory,
        tasks=tasks,
        task_records=task_records,
        records=clarified,
    )
    return {
        **result,
        "changed_pair_count": len(changed_pair_ids),
        "changed_pair_ids": changed_pair_ids,
        "llm_calls": 0,
        "gold_changed": False,
    }


__all__ = ["apply_metric_eligibility_clarifications", "publish_question_records"]