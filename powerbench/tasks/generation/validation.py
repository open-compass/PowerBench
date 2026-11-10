from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import jsonschema

from ..models import CorpusInventory, GoldRecord, Locator, PublicQuestion, answer_schema
from ..task_specs import normalize_task_family, public_answer
from ..util import format_timestamp
from .anchors import check_anchor, fault_anchor
from .gold_verification import verify_semantic_tasks
from .leakage import audit_shortcuts
from .locators import matching_devices, metadata_key_group
from .question_generation import polished_public_texts, validate_generation_records
from .render import render_task_pair
from .scopes import select_hard_negatives, validate_nested_scopes


def summarize_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Remove per-question audit witnesses while retaining all verdicts and counts."""

    shortcut = validation["shortcut_audit"]
    cover_distribution = Counter(
        item["minimum_distinct_source_files"] for item in shortcut["minimum_source_cover"]
    )
    language = validation.get("question_generation")
    language_summary = language
    if isinstance(language, dict):
        duplicate_groups = language.get("duplicate_surface_groups", {})
        groups = (
            [pairs for level_groups in duplicate_groups.values() for pairs in level_groups]
            if isinstance(duplicate_groups, dict)
            else []
        )
        language_summary = {
            key: value for key, value in language.items() if key != "duplicate_surface_groups"
        } | {
            "duplicate_surface_group_count": len(groups),
            "max_repeated_surface_group_size": max(map(len, groups), default=1),
        }
    return {
        key: value
        for key, value in validation.items()
        if key not in {"shortcut_audit", "question_generation"}
    } | {
        "question_generation": language_summary,
        "shortcut_audit": {
            "valid": shortcut["valid"],
            "answer_scan_complete": shortcut["answer_scan_complete"],
            "gold_record_count": shortcut["gold_record_count"],
            "invalid_alternative_count": shortcut["invalid_alternative_count"],
            "single_file_sufficient_count": shortcut["single_file_sufficient_count"],
            "exact_answer_hit_count": shortcut["exact_answer_hit_count"],
            "warning_count": len(shortcut["warnings"]),
            "minimum_source_cover_distribution": dict(sorted(cover_distribution.items())),
        },
    }


def _validate_report_selectors(
    tasks: list[dict[str, Any]], inventory: CorpusInventory
) -> list[str]:
    """Ensure every Indirect report clue resolves to exactly one raw report."""

    errors: list[str] = []
    check_cache: dict[tuple[str, str], Any] = {}
    fault_cache: dict[tuple[str, str], Any] = {}

    def cached_check(uid: str, path: str) -> Any:
        key = (uid, path)
        if key not in check_cache:
            check_cache[key] = check_anchor(inventory, uid, path)
        return check_cache[key]

    def cached_fault(uid: str, path: str) -> Any:
        key = (uid, path)
        if key not in fault_cache:
            fault_cache[key] = fault_anchor(inventory, uid, path)
        return fault_cache[key]

    for task in tasks:
        family = normalize_task_family(task["task_level"], task.get("task_family"))
        for alias, uid in task.get("device_aliases", {}).items():
            device = inventory.device(uid)
            expected_path = task["report_paths"][alias]
            if family in {
                "post_window",
                "threshold_episode",
                "metadata_normalized",
                "document_routed_window",
                "aligned_window_comparison",
                "filter_then_compute",
                "heterogeneous_metric_comparison",
            }:
                anchor_record = (
                    task["anchors"][alias]
                    if family
                    in {
                        "aligned_window_comparison",
                        "filter_then_compute",
                        "heterogeneous_metric_comparison",
                    }
                    else task["anchor"]
                )
                start_date = str(anchor_record["inspection_start"]).split(" ", 1)[0]
                matches: list[str] = []
                for path in device.check_reports:
                    try:
                        anchor = cached_check(uid, path)
                    except (OSError, ValueError):
                        continue
                    if format_timestamp(anchor.start).split(" ", 1)[0] == start_date:
                        matches.append(path)
            else:
                anchor_record = (
                    task["anchor"] if family == "pre_post_comparison" else task["anchors"][alias]
                )
                discovery_date = str(
                    anchor_record["discovery_time"] or anchor_record["fault_end"]
                ).split(" ", 1)[0]
                report_number = str(anchor_record.get("report_number") or "").strip()
                matches = []
                for path in device.fault_reports:
                    try:
                        anchor = cached_fault(uid, path)
                    except (OSError, ValueError):
                        continue
                    anchor_date = anchor.metadata.get("Discovery time", "").split(" ", 1)[0]
                    if anchor_date == discovery_date and anchor.report_number == report_number:
                        matches.append(path)
                try:
                    selected = cached_fault(uid, expected_path)
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"{task['pair_id']}/{alias}: selected fault report is not a valid "
                        f"single-rule anchor ({exc})"
                    )
                else:
                    if task.get("rule") is not None and selected.rule_id != task["rule"]["rule_id"]:
                        errors.append(
                            f"{task['pair_id']}/{alias}: report rule {selected.rule_id} "
                            f"does not match task rule {task['rule']['rule_id']}"
                        )
            if matches != [expected_path]:
                errors.append(
                    f"{task['pair_id']}/{alias}: report clue matches {matches}, "
                    f"expected only {expected_path}"
                )
    return errors


def validate_generated_records(
    questions: Iterable[dict[str, Any]],
    gold: Iterable[dict[str, Any]],
    semantic_tasks: Iterable[dict[str, Any]],
    scopes: Iterable[dict[str, Any]],
    inventory: CorpusInventory,
    *,
    expected_scope_sizes: Iterable[int] | None = None,
    expected_scope_seeds: Iterable[int] | None = None,
    question_generation: Iterable[dict[str, Any]] | None = None,
    require_polished_questions: bool = False,
) -> dict[str, Any]:
    question_rows = list(questions)
    gold_rows = list(gold)
    tasks = list(semantic_tasks)
    scope_rows = list(scopes)
    question_generation_rows = (
        list(question_generation) if question_generation is not None else None
    )
    errors: list[str] = []

    language_validation: dict[str, Any] | None = None
    generated_texts: dict[str, str] = {}
    if question_generation_rows is not None:
        language_validation = validate_generation_records(
            question_generation_rows,
            tasks,
            inventory,
            require_polished=require_polished_questions,
        )
        errors.extend(f"question generation: {error}" for error in language_validation["errors"])
        try:
            generated_texts = polished_public_texts(question_generation_rows)
        except ValueError as exc:
            errors.append(f"question generation text map: {exc}")

    try:
        validated_questions = [PublicQuestion.model_validate(row) for row in question_rows]
    except ValueError as exc:
        errors.append(f"public contract: {exc}")
        validated_questions = []
    try:
        validated_gold = [GoldRecord.model_validate(row) for row in gold_rows]
    except ValueError as exc:
        errors.append(f"gold contract: {exc}")
        validated_gold = []

    question_ids = [row.get("qa_id") for row in question_rows]
    gold_ids = [row.get("qa_id") for row in gold_rows]
    if len(question_ids) != len(set(question_ids)):
        errors.append("duplicate public qa_id")
    if len(gold_ids) != len(set(gold_ids)):
        errors.append("duplicate gold qa_id")
    if set(question_ids) != set(gold_ids):
        errors.append("public and gold qa_id sets differ")

    task_pair_ids = [task.get("pair_id") for task in tasks]
    if len(task_pair_ids) != len(set(task_pair_ids)):
        errors.append("duplicate semantic task pair_id")

    by_pair_gold: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        by_pair_gold[row["pair_id"]].append(row)
    for row in question_rows:
        by_pair_question[row["pair_id"]].append(row)
    task_by_pair = {task["pair_id"]: task for task in tasks}
    if set(task_by_pair) != set(by_pair_gold) or set(task_by_pair) != set(by_pair_question):
        errors.append("semantic task, public question, and gold pair_id sets differ")
    for pair_id in sorted(set(by_pair_gold) | set(by_pair_question)):
        gold_pair = by_pair_gold[pair_id]
        question_pair = by_pair_question[pair_id]
        if len(gold_pair) != 1:
            errors.append(f"{pair_id}: expected exactly one Gold record")
        if len(question_pair) != 1:
            errors.append(f"{pair_id}: expected exactly one public question")
        task = task_by_pair.get(pair_id)
        if task is None:
            continue
        family = normalize_task_family(task["task_level"], task.get("task_family"))
        projected_answer = public_answer(task["task_level"], task["answer"], family)
        projected_tolerances = {
            key: value
            for key, value in task.get("tolerances", {}).items()
            if key in projected_answer
        }
        for public in question_pair:
            if public.get("qa_id") != pair_id:
                errors.append(f"{pair_id}: public qa_id must equal the semantic task ID")
            if public.get("task_level") != task.get("task_level"):
                errors.append(f"{pair_id}: public task_level disagrees with semantic task")
            if public.get("task_family") != task.get("task_family"):
                errors.append(f"{pair_id}: public task_family disagrees with semantic task")
            if public.get("scope_key") != pair_id:
                errors.append(f"{pair_id}: public scope_key disagrees with pair_id")
            expected_schema = answer_schema(task["task_level"], family)
            if public.get("answer_schema") != expected_schema:
                errors.append(f"{pair_id}: public answer schema disagrees with task level")
            else:
                try:
                    jsonschema.Draft202012Validator.check_schema(expected_schema)
                    jsonschema.validate(
                        {"answer": projected_answer, "evidence": []}, expected_schema
                    )
                except jsonschema.SchemaError as exc:
                    errors.append(f"{pair_id}: invalid public answer schema: {exc.message}")
                except jsonschema.ValidationError as exc:
                    errors.append(
                        f"{pair_id}: semantic answer violates public schema: {exc.message}"
                    )
        for record in gold_pair:
            if record.get("qa_id") != pair_id:
                errors.append(f"{pair_id}: gold qa_id must equal the semantic task ID")
            expected_fields = {
                "task_level": task.get("task_level"),
                "task_family": task.get("task_family"),
                "target_device_uids": task.get("target_device_uids"),
                "answer": projected_answer,
                "tolerances": projected_tolerances,
                "gold_trace": task.get("gold_trace"),
            }
            for field, expected in expected_fields.items():
                if record.get(field) != expected:
                    errors.append(f"{pair_id}: gold {field} disagrees with semantic task")
            diagnostics = record.get("diagnostics")
            if not isinstance(diagnostics, dict) or any(
                diagnostics.get(key) != value for key, value in task.get("diagnostics", {}).items()
            ):
                errors.append(f"{pair_id}: gold diagnostics disagree with semantic task")
        versions = {str(row.get("schema_version")) for row in [*question_pair, *gold_pair]}
        if len(versions) != 1 or "None" in versions:
            errors.append(f"{pair_id}: public/gold schema versions are inconsistent")
        else:
            try:
                expected_public, expected_gold = render_task_pair(
                    task,
                    inventory,
                    schema_version=next(iter(versions)),
                    question_texts=generated_texts.get(pair_id),
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{pair_id}: deterministic re-render failed: {exc}")
            else:
                actual_public = {row.get("qa_id"): row for row in question_pair}
                actual_gold = {row.get("qa_id"): row for row in gold_pair}
                for expected in expected_public:
                    if actual_public.get(expected["qa_id"]) != expected:
                        errors.append(
                            f"{expected['qa_id']}: public question differs from deterministic render"
                        )
                for expected in expected_gold:
                    if actual_gold.get(expected["qa_id"]) != expected:
                        errors.append(
                            f"{expected['qa_id']}: gold record differs from deterministic render"
                        )

    for task in tasks:
        family = normalize_task_family(task["task_level"], task.get("task_family"))
        if family == "filter_then_compute":
            filter_field = str(task["filter"]["field"])
            filter_value = str(task["filter"]["value"])
            filter_group = metadata_key_group(filter_field)
            for alias, raw_locator in task.get("locators", {}).items():
                exposed = [
                    item
                    for item in raw_locator.get("fields", [])
                    if metadata_key_group(str(item["field"])) == filter_group
                    or str(item["value"]) == filter_value
                ]
                if exposed:
                    errors.append(
                        f"{task['pair_id']}/{alias}: locator exposes filter dimension "
                        f"{filter_field!r}: {exposed}"
                    )
        for alias, raw_locator in task.get("locators", {}).items():
            locator = Locator(
                fields=tuple(
                    (str(item["field"]), str(item["value"]))
                    for item in raw_locator.get("fields", [])
                ),
                ablation_confusers=tuple(raw_locator.get("ablation_confusers", [])),
            )
            matches = matching_devices(locator, inventory.devices.values())
            expected = task["device_aliases"][alias]
            if matches != [expected]:
                errors.append(
                    f"{task['pair_id']}/{alias}: indirect locator matches {matches}, expected {expected}"
                )
            for omitted in range(len(locator.fields)):
                remainder = Locator(
                    fields=tuple(
                        item for index, item in enumerate(locator.fields) if index != omitted
                    ),
                    ablation_confusers=(),
                )
                if len(matching_devices(remainder, inventory.devices.values())) < 2:
                    errors.append(
                        f"{task['pair_id']}/{alias}: locator becomes unique after clue ablation"
                    )

    errors.extend(_validate_report_selectors(tasks, inventory))

    errors.extend(validate_nested_scopes(scope_rows))
    scope_keys = [
        (row.get("pair_id"), int(row.get("N", -1)), int(row.get("scope_seed", -1)))
        for row in scope_rows
    ]
    if len(scope_keys) != len(set(scope_keys)):
        errors.append("duplicate scope pair_id/N/scope_seed key")
    if expected_scope_sizes is not None and expected_scope_seeds is not None:
        sizes = tuple(int(value) for value in expected_scope_sizes)
        seeds = tuple(int(value) for value in expected_scope_seeds)
        expected_keys = {
            (pair_id, size, seed)
            for pair_id in task_by_pair
            for seed in seeds
            for size in sizes
            if size != len(inventory.devices) or seed == seeds[0]
        }
        missing_scope_keys = sorted(expected_keys.difference(scope_keys))
        unexpected_scope_keys = sorted(set(scope_keys).difference(expected_keys))
        if missing_scope_keys:
            errors.append(f"missing expected scope keys: {missing_scope_keys[:10]}")
        if unexpected_scope_keys:
            errors.append(f"unexpected scope keys: {unexpected_scope_keys[:10]}")
    for row in scope_rows:
        pair_id = row.get("pair_id")
        if pair_id not in task_by_pair:
            errors.append(f"scope references unknown pair_id: {pair_id}")
            continue
        task = task_by_pair[pair_id]
        members = list(row.get("device_uids", []))
        unknown = sorted(set(members).difference(inventory.devices))
        if unknown:
            errors.append(f"{pair_id}/N={row.get('N')}: unknown devices {unknown[:10]}")
        if sorted(row.get("target_device_uids", [])) != sorted(task["target_device_uids"]):
            errors.append(f"{pair_id}/N={row.get('N')}: scope target devices disagree")
        expected_confusers = sorted(
            {
                uid
                for locator in task.get("locators", {}).values()
                for uid in locator.get("ablation_confusers", [])
            }
        )
        if sorted(row.get("required_locator_confusers", [])) != expected_confusers:
            errors.append(f"{pair_id}/N={row.get('N')}: scope confusers disagree")
        expected_hard_negatives = (
            select_hard_negatives(task["target_device_uids"], inventory)
            if task.get("task_family")
            in {
                "document_routed_window",
                "context_conditioned_replay",
                "heterogeneous_metric_comparison",
            }
            else []
        )
        if sorted(row.get("required_hard_negatives", [])) != sorted(expected_hard_negatives):
            errors.append(f"{pair_id}/N={row.get('N')}: hard negatives disagree")
    for record in gold_rows:
        for atom in record.get("evidence_atoms", []):
            if not atom.get("acceptable_sources"):
                errors.append(f"{record['qa_id']}/{atom.get('atom_id')}: no evidence source")
            for source in atom.get("acceptable_sources", []):
                path = source.get("path", "")
                if path not in inventory.artifacts and not (inventory.raw_root / path).is_file():
                    errors.append(f"{record['qa_id']}: missing evidence path {path}")

    executable_verification = verify_semantic_tasks(tasks, inventory)
    errors.extend(executable_verification["errors"])
    shortcut_audit = audit_shortcuts(tasks, gold_rows, inventory)
    if not shortcut_audit["valid"]:
        errors.append(
            "shortcut audit failed: "
            f"errors={len(shortcut_audit['errors'])}, "
            f"invalid_alternatives={shortcut_audit['invalid_alternative_count']}, "
            f"single_file_sufficient={shortcut_audit['single_file_sufficient_count']}, "
            f"exact_answer_hits={shortcut_audit['exact_answer_hit_count']}; "
            f"examples={shortcut_audit['invalid_alternatives'][:3]}"
        )
    counts = Counter(row.get("task_level") for row in question_rows)
    replicate_groups: defaultdict[tuple[str, int], set[tuple[str, ...]]] = defaultdict(set)
    for row in scope_rows:
        replicate_groups[(str(row.get("pair_id")), int(row.get("N", -1)))].add(
            tuple(sorted(row.get("device_uids", [])))
        )
    effective_scope_replicates = Counter(
        len(unique_scopes) for unique_scopes in replicate_groups.values()
    )
    return {
        "valid": not errors,
        "errors": errors,
        "question_count": len(question_rows),
        "semantic_task_count": len(tasks),
        "questions_by_level": dict(sorted(counts.items())),
        "validated_public_records": len(validated_questions),
        "validated_gold_records": len(validated_gold),
        "single_file_sufficient_count": shortcut_audit["single_file_sufficient_count"],
        "scope_row_count": len(scope_rows),
        "effective_unique_scope_replicate_distribution": dict(
            sorted(effective_scope_replicates.items())
        ),
        "executable_gold_verification": executable_verification,
        "shortcut_audit": shortcut_audit,
        "question_generation": language_validation,
    }


def assert_valid(report: dict[str, Any]) -> None:
    if not report.get("valid"):
        details = "\n".join(f"- {error}" for error in report.get("errors", [])[:30])
        raise ValueError(f"Generated pilot failed validation:\n{details}")