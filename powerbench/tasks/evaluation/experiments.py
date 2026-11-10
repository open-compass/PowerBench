from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from ..bundle import BenchmarkBundle, load_bundle
from ..config import PilotConfig
from ..generation.scopes import validate_nested_scopes
from ..models import PublicQuestion, answer_schema
from ..util import canonical_json, file_sha256, read_jsonl, stable_id, write_json, write_jsonl

REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "unspecified")
MATRIX_DESIGN = "autonomous_retrieval_scaling"
DEFAULT_MAX_JOB_SECONDS = 300


def _validated_questions(
    rows: list[dict[str, Any]], config: PilotConfig
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Question file is empty")
    required = ("qa_id", "pair_id", "task_level")
    qa_ids: list[str] = []
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        try:
            PublicQuestion.model_validate(row)
        except ValueError as exc:
            raise ValueError(f"Question row {index} violates the public contract: {exc}") from exc
        for field in required:
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Question row {index} has an invalid {field}")
        if row["task_level"] not in {"L1", "L2", "L3"}:
            raise ValueError(f"Question {row['qa_id']} has an invalid task_level")
        legacy_anchor = row.get("anchor_type")
        if legacy_anchor is None and row["qa_id"] != row["pair_id"]:
            raise ValueError(f"Question {row['qa_id']} must use its semantic task ID as qa_id")
        if legacy_anchor is not None and legacy_anchor not in {"Direct", "Indirect"}:
            raise ValueError(f"Question {row['qa_id']} has an invalid legacy anchor_type")
        if row.get("answer_schema") != answer_schema(row["task_level"], row.get("task_family")):
            raise ValueError(
                f"Question {row['qa_id']} answer schema disagrees with its task family"
            )
        qa_ids.append(row["qa_id"])
        by_pair.setdefault(row["pair_id"], []).append(row)
    if len(qa_ids) != len(set(qa_ids)):
        raise ValueError("Question file contains duplicate qa_id values")
    levels = {row["task_level"] for row in rows}
    if levels != {"L1", "L2", "L3"}:
        raise ValueError("Question file must contain L1, L2, and L3 tasks")

    for pair_id, questions in by_pair.items():
        anchors = {row.get("anchor_type") for row in questions}
        if len(questions) == 1 and anchors == {None}:
            continue
        if len(questions) == 2 and anchors == {"Direct", "Indirect"}:
            continue
        raise ValueError(f"Semantic task {pair_id} has an invalid question representation")

    expected_counts = config.data.get("task_counts")
    if isinstance(expected_counts, dict):
        actual = Counter(question[0]["task_level"] for question in by_pair.values())
        expected = {str(level): int(count) for level, count in expected_counts.items()}
        if dict(actual) != expected:
            raise ValueError(
                f"Question matrix has semantic task counts {dict(actual)}, expected {expected}"
            )
        if len(by_pair) != sum(expected.values()):
            raise ValueError("Question matrix has an unexpected number of semantic tasks")
    return by_pair


def _validated_scope_members(
    rows: list[dict[str, Any]],
    *,
    pair_ids: set[str],
    scope_sizes: tuple[int, ...],
    scope_seeds: tuple[int, ...],
) -> dict[tuple[str, int, int], tuple[str, ...]]:
    if not rows:
        raise ValueError("Scope file is empty")
    scope_members: dict[tuple[str, int, int], tuple[str, ...]] = {}
    invariants: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for index, row in enumerate(rows):
        pair_id = row.get("pair_id")
        size = row.get("N")
        seed = row.get("scope_seed")
        if not isinstance(pair_id, str) or pair_id not in pair_ids:
            raise ValueError(f"Scope row {index} references an unknown pair_id: {pair_id!r}")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"Scope row {index} has a non-integer N")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Scope row {index} has a non-integer scope_seed")
        key = (pair_id, size, seed)
        if key in scope_members:
            raise ValueError(f"Duplicate scope key: {key}")

        members = row.get("device_uids")
        targets = row.get("target_device_uids")
        confusers = row.get("required_locator_confusers")
        if not isinstance(members, list) or any(not isinstance(uid, str) for uid in members):
            raise ValueError(f"Scope {key} has invalid device_uids")
        if len(members) != size or len(members) != len(set(members)):
            raise ValueError(f"Scope {key} must contain exactly N unique devices")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(uid, str) for uid in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError(f"Scope {key} has invalid target_device_uids")
        if (
            not isinstance(confusers, list)
            or any(not isinstance(uid, str) for uid in confusers)
            or len(confusers) != len(set(confusers))
        ):
            raise ValueError(f"Scope {key} has invalid required_locator_confusers")
        required = set(targets) | set(confusers)
        if not required.issubset(members):
            raise ValueError(f"Scope {key} omits a target device or required confuser")
        invariant = (tuple(sorted(targets)), tuple(sorted(confusers)))
        if pair_id in invariants and invariants[pair_id] != invariant:
            raise ValueError(f"Scope rows for {pair_id} disagree on targets or confusers")
        invariants[pair_id] = invariant
        scope_members[key] = tuple(sorted(members))

    expected_keys = {
        (pair_id, size, seed)
        for pair_id in pair_ids
        for seed in scope_seeds
        for size in scope_sizes
        if size != scope_sizes[-1] or seed == scope_seeds[0]
    }
    actual_keys = set(scope_members)
    missing = sorted(expected_keys.difference(actual_keys))
    unexpected = sorted(actual_keys.difference(expected_keys))
    if missing:
        raise ValueError(f"Scope file is missing expected keys: {missing[:10]}")
    if unexpected:
        raise ValueError(f"Scope file contains unexpected keys: {unexpected[:10]}")
    nested_errors = validate_nested_scopes(rows)
    if nested_errors:
        raise ValueError(f"Scope nesting validation failed: {nested_errors[:10]}")
    return scope_members


def build_run_manifest(
    config: PilotConfig,
    *,
    model: str,
    reasoning_effort: str,
    questions_path: str | Path,
    scopes_path: str | Path,
    gold_path: str | Path,
    bundle_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    task_levels: tuple[str, ...] | None = None,
    scope_sizes: tuple[int, ...] | None = None,
    qa_ids: tuple[str, ...] | None = None,
    max_job_seconds: int = DEFAULT_MAX_JOB_SECONDS,
    api_format: str = "responses",
) -> dict[str, Any]:
    """Create a reproducible subset of the evaluation matrix without API calls."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if api_format not in {"responses", "chat-completions"}:
        raise ValueError("api_format must be responses or chat-completions")
    if not isinstance(reasoning_effort, str) or reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    if (
        isinstance(max_job_seconds, bool)
        or not isinstance(max_job_seconds, int)
        or max_job_seconds < 60
    ):
        raise ValueError("max_job_seconds must be an integer of at least 60")
    selected_levels = tuple(dict.fromkeys(task_levels or ("L1", "L2", "L3")))
    if not selected_levels or any(level not in {"L1", "L2", "L3"} for level in selected_levels):
        raise ValueError("task_levels must be a non-empty subset of L1, L2, and L3")
    selected_sizes = tuple(dict.fromkeys(scope_sizes or config.scope_sizes))
    unknown_sizes = sorted(set(selected_sizes).difference(config.scope_sizes))
    if not selected_sizes or unknown_sizes:
        raise ValueError(
            "scope_sizes must be a non-empty subset of configured scope sizes; "
            f"unknown values: {unknown_sizes}"
        )
    matrix_selection = {
        "task_levels": list(selected_levels),
        "scope_sizes": list(selected_sizes),
        "max_job_seconds": max_job_seconds,
    }
    if api_format != "responses":
        matrix_selection["api_format"] = api_format
    selected_qa_ids = tuple(dict.fromkeys(qa_ids or ()))
    if qa_ids is not None:
        if not selected_qa_ids or any(
            not isinstance(value, str) or not value.strip() for value in selected_qa_ids
        ):
            raise ValueError("qa_ids must contain one or more non-empty question IDs")
        matrix_selection["qa_ids"] = list(selected_qa_ids)
    bundle = load_bundle(bundle_path, verify=True)
    generation_status = bundle.data.get("question_generation_status")
    question_source = Path(questions_path).resolve()
    scope_source = Path(scopes_path).resolve()
    gold_source = Path(gold_path).resolve()
    index_source = Path(index_path).resolve()
    questions_digest = _require_bundle_artifact(bundle, "questions", question_source)
    scope_digest = _require_bundle_artifact(bundle, "scopes", scope_source)
    gold_digest = _require_bundle_artifact(bundle, "tasks", gold_source)
    index_digest = _require_bundle_artifact(bundle, "corpus_index", index_source)

    config_semantic_digest = hashlib.sha256(canonical_json(config.data).encode("utf-8")).hexdigest()
    if config_semantic_digest != bundle.data.get("config_semantic_sha256"):
        raise ValueError("Config semantics disagree with the benchmark bundle")
    config_digest = (
        file_sha256(config.source) if config.source.is_file() else config_semantic_digest
    )
    artifact_hashes = bundle.data["artifact_hashes"]
    if artifact_hashes.get("config_sha256") != config_digest:
        raise ValueError("Config file disagrees with the benchmark bundle")

    all_questions = read_jsonl(question_source)
    questions_by_pair = _validated_questions(all_questions, config)
    questions_semantic_digest = hashlib.sha256(
        canonical_json(all_questions).encode("utf-8")
    ).hexdigest()
    scope_rows = read_jsonl(scope_source)
    scope_members = _validated_scope_members(
        scope_rows,
        pair_ids=set(questions_by_pair),
        scope_sizes=config.scope_sizes,
        scope_seeds=config.scope_seeds,
    )
    bundle_source = bundle.path
    bundle_digest = file_sha256(bundle_source)
    raw_content_digest = str(bundle.data["raw_content_sha256"])
    experiment_id = stable_id(
        questions_digest,
        scope_digest or "scopes-unbound",
        gold_digest or "gold-unbound",
        raw_content_digest or "raw-unbound",
        index_digest or "index-unbound",
        config_digest,
        MATRIX_DESIGN,
        canonical_json(matrix_selection),
        model,
        reasoning_effort,
        length=24,
    )
    if generation_status == "draft":
        raise ValueError(
            "Benchmark questions are deterministic drafts. Run 'powerbench-qa polish' first."
        )
    questions = all_questions
    if generation_status == "partial":
        polished_pair_ids = {str(pair_id) for pair_id in bundle.data.get("polished_pair_ids", [])}
        questions = [row for row in all_questions if str(row["pair_id"]) in polished_pair_ids]
        if not questions:
            raise ValueError(
                "The benchmark has no polished question pairs available for evaluation."
            )
    elif generation_status != "polished":
        raise ValueError(f"Unsupported question generation status: {generation_status!r}")

    # Read compatibility for pre-autonomous paired bundles. New bundles have no
    # anchor_type field and already contain exactly one question per task.
    if any(row.get("anchor_type") is not None for row in questions):
        questions = [row for row in questions if row.get("anchor_type") == "Indirect"]
    if selected_qa_ids:
        available_qa_ids = {str(row["qa_id"]) for row in questions}
        unknown_qa_ids = sorted(set(selected_qa_ids).difference(available_qa_ids))
        if unknown_qa_ids:
            raise ValueError(f"Unknown published qa_ids: {unknown_qa_ids}")
        questions = [row for row in questions if row["qa_id"] in selected_qa_ids]
    questions = [row for row in questions if row["task_level"] in selected_levels]
    if selected_qa_ids and len(questions) != len(selected_qa_ids):
        excluded = sorted(set(selected_qa_ids).difference(row["qa_id"] for row in questions))
        raise ValueError(f"qa_ids excluded by task_levels: {excluded}")
    if not questions:
        raise ValueError("Evaluation matrix selection contains no published questions")

    base_seed = config.scope_seeds[0]

    jobs: list[dict[str, Any]] = []
    for question in questions:
        for size in selected_sizes:
            scope_key = (question["pair_id"], size, base_seed)
            if scope_key not in scope_members:
                raise ValueError(f"Missing scope key: {scope_key}")
            jobs.append(
                _job(
                    question,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    condition="full_retrieval",
                    N=size,
                    scope_seed=base_seed,
                    experiment_id=experiment_id,
                    max_job_seconds=max_job_seconds,
                    api_format=api_format,
                )
            )

    schedule_seed = int(config.data.get("seed", 0))
    jobs.sort(key=lambda job: stable_id("schedule", schedule_seed, job["run_id"], length=64))

    output = Path(output_path)
    write_jsonl(output, jobs)
    manifest_digest = file_sha256(output)
    counts = Counter((job["model"], job["condition"], job["anchor_type"], job["N"]) for job in jobs)
    report = {
        "job_count": len(jobs),
        "selected_question_count": len(questions),
        "selected_semantic_task_count": len({row["pair_id"] for row in questions}),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "api_format": api_format,
        "max_job_seconds": max_job_seconds,
        "experiment_id": experiment_id,
        "questions_path": str(question_source),
        "questions_sha256": questions_digest,
        "questions_semantic_sha256": questions_semantic_digest,
        "scopes_path": str(scope_source),
        "scopes_sha256": scope_digest,
        "gold_path": str(gold_source),
        "gold_sha256": gold_digest,
        "bundle_path": str(bundle_source),
        "bundle_sha256": bundle_digest,
        "raw_content_sha256": raw_content_digest,
        "index_path": str(index_source),
        "index_sha256": index_digest,
        "config_sha256": config_digest,
        "config_semantic_sha256": config_semantic_digest,
        "manifest_path": str(output.resolve()),
        "manifest_sha256": manifest_digest,
        "matrix_design": MATRIX_DESIGN,
        "matrix_selection": matrix_selection,
        "paid_requests_started": 0,
        "counts": [
            {
                "model": key[0],
                "condition": key[1],
                "anchor_type": key[2],
                "N": key[3],
                "count": value,
            }
            for key, value in sorted(counts.items())
        ],
    }
    write_json(output.with_suffix(".summary.json"), report)
    return report


def _require_bundle_artifact(bundle: BenchmarkBundle, name: str, supplied_path: Path) -> str:
    try:
        return bundle.verify_artifact(name, supplied_path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Run manifest input disagrees with bundle artifact {name!r}: {exc}"
        ) from exc


def _job(
    question: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    condition: str,
    N: int,
    scope_seed: int,
    replicate: str = "main",
    experiment_id: str,
    max_job_seconds: int,
    api_format: str = "responses",
) -> dict[str, Any]:
    run_id = stable_id(
        experiment_id,
        canonical_json(question),
        model,
        reasoning_effort,
        condition,
        N,
        scope_seed,
        replicate,
        max_job_seconds,
        *([api_format] if api_format != "responses" else []),
        length=24,
    )
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "qa_id": question["qa_id"],
        "pair_id": question["pair_id"],
        "task_level": question["task_level"],
        # Kept only as a private compatibility marker for older scoring files;
        # autonomous public questions do not expose an anchor variant.
        "anchor_type": "Indirect",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "condition": condition,
        "N": int(N),
        "scope_seed": int(scope_seed),
        "replicate": replicate,
        "max_job_seconds": max_job_seconds,
        **({"api_format": api_format} if api_format != "responses" else {}),
    }