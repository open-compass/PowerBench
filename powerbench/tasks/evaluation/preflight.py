from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from ..bundle import load_bundle
from ..config import PilotConfig
from ..corpus import INDEXED_SUFFIXES, CorpusIndex, CorpusTools
from ..models import GoldRecord, PublicQuestion, answer_schema
from ..util import canonical_json, directory_content_sha256, file_sha256
from .manifest import load_manifest_binding, validate_reproducible_run_ids
from .oracle import build_oracle_bundle


def retrieval_settings(config: PilotConfig) -> dict[str, Any]:
    retrieval = config.section("retrieval")
    positive_integer_keys = (
        "max_tool_rounds",
        "max_read_lines",
        "max_timeseries_rows",
        "max_total_read_bytes",
        "python_max_output_chars",
    )
    for key in positive_integer_keys:
        value = retrieval.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"retrieval.{key} must be a positive integer")
    retries = retrieval.get("format_retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retrieval.format_retries must be a non-negative integer")
    timeout = retrieval.get("python_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("retrieval.python_timeout_seconds must be positive")
    return retrieval


def verify_manifest_binding(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    index_path: str | Path,
    questions_path: str | Path,
    scopes_path: str | Path,
    jobs_path: str | Path,
    gold_path: str | Path | None,
    question_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind every paid run to the exact generated artifacts and run IDs."""

    if gold_path is None:
        raise ValueError("Paid benchmark execution requires a bound private Gold file")
    config_semantic = hashlib.sha256(canonical_json(config.data).encode("utf-8")).hexdigest()
    config_digest = file_sha256(config.source) if config.source.is_file() else config_semantic
    raw_content_digest, _ = directory_content_sha256(raw_root, suffixes=set(INDEXED_SUFFIXES))
    summary_path = Path(jobs_path).resolve().with_suffix(".summary.json")
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load run manifest binding summary {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Run manifest binding summary must be an object: {summary_path}")
    bundle_value = summary.get("bundle_path")
    if not isinstance(bundle_value, str) or not bundle_value:
        raise ValueError("Run manifest binding summary has no bundle_path")
    bundle = load_bundle(bundle_value, verify=True)
    for artifact_name, supplied_path in (
        ("questions", questions_path),
        ("scopes", scopes_path),
        ("tasks", gold_path),
        ("corpus_index", index_path),
    ):
        bundle.verify_artifact(artifact_name, supplied_path)
    if bundle.data.get("config_semantic_sha256") != config_semantic:
        raise ValueError("Current config semantics disagree with the manifest-bound bundle")
    if bundle.data["artifact_hashes"].get("config_sha256") != config_digest:
        raise ValueError("Current config file disagrees with the manifest-bound bundle")
    if bundle.data.get("raw_content_sha256") != raw_content_digest:
        raise ValueError("Current raw corpus disagrees with the manifest-bound bundle")

    binding = load_manifest_binding(
        jobs_path,
        artifact_paths={
            "questions": questions_path,
            "scopes": scopes_path,
            "gold": gold_path,
            "index": index_path,
            "bundle": bundle.path,
        },
        extra_hashes={
            "questions_semantic_sha256": hashlib.sha256(
                canonical_json(question_rows).encode("utf-8")
            ).hexdigest(),
            "config_sha256": config_digest,
            "config_semantic_sha256": config_semantic,
            "raw_content_sha256": raw_content_digest,
        },
        required_path_identities=(
            "manifest",
            "questions",
            "scopes",
            "gold",
            "index",
            "bundle",
        ),
        manifest_rows=job_rows,
        require_manifest_file=False,
    )
    validate_reproducible_run_ids(binding, question_rows)
    return binding.summary


def _unique_rows(rows: list[dict[str, Any]], key: str, *, label: str) -> dict[Any, dict[str, Any]]:
    values = [row.get(key) for row in rows]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {label} {key}: {duplicates[:10]}")
    if any(value is None for value in values):
        raise ValueError(f"Missing {label} {key}")
    return {row[key]: row for row in rows}


def _manifest_int(row: dict[str, Any], key: str, *, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} {key} must be an integer, got {value!r}")
    return value


def validate_run_inputs(
    *,
    question_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, int, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Fail before the first paid call if public/private inputs disagree."""

    questions = _unique_rows(question_rows, "qa_id", label="question")
    jobs = _unique_rows(job_rows, "run_id", label="job")
    gold = _unique_rows(gold_rows, "qa_id", label="Gold") if gold_rows else {}
    scope_keys = [
        (
            row.get("pair_id"),
            _manifest_int(row, "N", label="scope"),
            _manifest_int(row, "scope_seed", label="scope"),
        )
        for row in scope_rows
    ]
    if len(scope_keys) != len(set(scope_keys)):
        raise ValueError("Duplicate scope pair_id/N/scope_seed")
    scopes = {key: row for key, row in zip(scope_keys, scope_rows, strict=True)}

    for question in question_rows:
        PublicQuestion.model_validate(question)
        jsonschema.Draft202012Validator.check_schema(question["answer_schema"])
        if question["answer_schema"] != answer_schema(
            question["task_level"], question.get("task_family")
        ):
            raise ValueError(
                f"Question {question['qa_id']} answer schema disagrees with its task family"
            )
        if question["scope_key"] != question["pair_id"]:
            raise ValueError(f"Question {question['qa_id']} scope_key must equal its pair_id")
    for record in gold_rows:
        GoldRecord.model_validate(record)
        if record["qa_id"] not in questions:
            raise ValueError(f"Gold references unknown question: {record['qa_id']}")
        question = questions[record["qa_id"]]
        for field in ("pair_id", "task_level", "task_family"):
            if record.get(field) != question.get(field):
                raise ValueError(
                    f"Gold {record['qa_id']} {field}={record.get(field)!r} disagrees with "
                    f"question value {question.get(field)!r}"
                )
        targets = list(record.get("target_device_uids", []))
        if not targets or len(targets) != len(set(targets)):
            raise ValueError(f"Gold {record['qa_id']} must name unique target devices")
        atom_ids = [atom.get("atom_id") for atom in record.get("evidence_atoms", [])]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError(f"Gold {record['qa_id']} contains duplicate evidence atom IDs")
        if any(not atom.get("acceptable_sources") for atom in record.get("evidence_atoms", [])):
            raise ValueError(f"Gold {record['qa_id']} contains an atom with no evidence source")
        try:
            jsonschema.validate(
                {"answer": record["answer"], "evidence": []}, question["answer_schema"]
            )
        except jsonschema.ValidationError as exc:
            raise ValueError(
                f"Gold answer does not satisfy question schema: {record['qa_id']}: {exc.message}"
            ) from exc

    for job in jobs.values():
        if job.get("api_format", "responses") not in {"responses", "chat-completions"}:
            raise ValueError(f"Job has unsupported api_format: {job.get('api_format')!r}")
        for field in ("run_id", "qa_id", "pair_id", "model", "reasoning_effort"):
            if not isinstance(job.get(field), str) or not job[field]:
                raise ValueError(f"Job has an invalid {field}: {job.get(field)!r}")
        condition = job.get("condition")
        if condition not in {"full_retrieval", "oracle_evidence"}:
            raise ValueError(f"Job {job['run_id']} has unsupported condition: {condition!r}")
        size = _manifest_int(job, "N", label=f"job {job['run_id']}")
        seed = _manifest_int(job, "scope_seed", label=f"job {job['run_id']}")
        if size <= 0:
            raise ValueError(f"Job {job['run_id']} N must be positive")
        qa_id = job.get("qa_id")
        if qa_id not in questions:
            raise ValueError(f"Job references unknown qa_id: {qa_id}")
        question = questions[qa_id]
        for field in ("pair_id", "task_level"):
            if job.get(field) != question.get(field):
                raise ValueError(
                    f"Job {job['run_id']} {field}={job.get(field)!r} disagrees with "
                    f"question value {question.get(field)!r}"
                )
        scope_key = (job["pair_id"], size, seed)
        if scope_key not in scopes:
            raise ValueError(f"Job {job['run_id']} has no scope {scope_key}")
        scope = scopes[scope_key]
        members = list(scope.get("device_uids", []))
        if len(members) != size or len(members) != len(set(members)):
            raise ValueError(f"Scope {scope_key} does not contain exactly N unique devices")
        scope_targets = set(scope.get("target_device_uids", []))
        required = scope_targets | set(scope.get("required_locator_confusers", []))
        if not required.issubset(members):
            raise ValueError(f"Scope {scope_key} omits a target or required confuser")
        if qa_id not in gold:
            raise ValueError(f"Scheduled job {job['run_id']} has no private Gold record")
        if set(gold[qa_id]["target_device_uids"]) != scope_targets:
            raise ValueError(f"Scope {scope_key} target devices disagree with Gold {qa_id}")
    return questions, scopes, gold


def validate_corpus_inputs(
    *,
    inventory: Any,
    jobs: list[dict[str, Any]],
    scopes: dict[tuple[str, int, int], dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> None:
    """Verify scheduled scopes and Gold sources against the frozen inventory."""

    checked_scopes: set[tuple[str, int, int]] = set()
    for job in jobs:
        scope_key = (job["pair_id"], int(job["N"]), int(job["scope_seed"]))
        if scope_key in checked_scopes:
            continue
        checked_scopes.add(scope_key)
        unknown = sorted(set(scopes[scope_key]["device_uids"]).difference(inventory.devices))
        if unknown:
            raise ValueError(f"Scope {scope_key} contains unknown devices: {unknown[:10]}")

    checked_questions: set[tuple[str, tuple[str, int, int]]] = set()
    for job in jobs:
        qa_id = job["qa_id"]
        if qa_id not in gold:
            continue
        scope_key = (job["pair_id"], int(job["N"]), int(job["scope_seed"]))
        check_key = (qa_id, scope_key)
        if check_key in checked_questions:
            continue
        checked_questions.add(check_key)
        members = set(scopes[scope_key]["device_uids"])
        model_keys = {inventory.devices[uid].model_key for uid in members}
        for atom in gold[qa_id]["evidence_atoms"]:
            visible_source = False
            for source in atom["acceptable_sources"]:
                artifact = inventory.artifacts.get(source["path"])
                if artifact is None or not artifact.get("public", True):
                    continue
                device_uid = artifact.get("device_uid")
                if device_uid in members or (
                    device_uid is None and artifact.get("model_key") in model_keys
                ):
                    visible_source = True
                    break
            if not visible_source:
                raise ValueError(
                    f"No visible public source for Gold atom {qa_id}/{atom['atom_id']} "
                    f"in scope {scope_key}"
                )


def preflight_oracle_evidence(
    *,
    inventory: Any,
    index: CorpusIndex,
    config: PilotConfig,
    jobs: list[dict[str, Any]],
    scopes: dict[tuple[str, int, int], dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> None:
    """Materialize each distinct pending Oracle bundle before any model call."""

    retrieval = retrieval_settings(config)
    checked: set[tuple[str, str, int, int]] = set()
    for job in jobs:
        if job["condition"] != "oracle_evidence":
            continue
        key = (job["qa_id"], job["pair_id"], int(job["N"]), int(job["scope_seed"]))
        if key in checked:
            continue
        checked.add(key)
        scope_key = (job["pair_id"], int(job["N"]), int(job["scope_seed"]))
        try:
            with CorpusTools(
                inventory,
                scopes[scope_key],
                index=index,
                max_timeseries_rows=int(retrieval["max_timeseries_rows"]),
                max_file_lines=int(retrieval["max_read_lines"]),
                max_total_read_bytes=int(retrieval["max_total_read_bytes"]),
            ) as corpus:
                bundle = build_oracle_bundle(inventory, corpus, gold[job["qa_id"]])
        except Exception as exc:
            raise ValueError(
                f"Oracle evidence preflight failed for {job['run_id']}: {type(exc).__name__}: {exc}"
            ) from exc
        expected_atoms = [atom["atom_id"] for atom in gold[job["qa_id"]]["evidence_atoms"]]
        audited_atoms = [entry.get("atom_id") for entry in bundle.get("_audit", [])]
        if audited_atoms != expected_atoms:
            raise ValueError(
                f"Oracle evidence preflight did not cover every atom for {job['qa_id']}"
            )


__all__ = [
    "preflight_oracle_evidence",
    "retrieval_settings",
    "validate_corpus_inputs",
    "validate_run_inputs",
    "verify_manifest_binding",
]