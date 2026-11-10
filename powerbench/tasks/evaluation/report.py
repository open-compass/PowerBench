from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from ..util import file_sha256, read_jsonl, write_json
from .scoring import ERROR_TAXONOMY, INFRASTRUCTURE_STATUSES, classify_error

EVALUABLE_TERMINAL_STATUSES = {
    "completed",
    "format_error",
    "tool_budget_exhausted",
    "response_incomplete",
    "refusal",
    "time_budget_exhausted",
}
LEVELS = ("L1", "L2", "L3")
MIN_ERROR_EXAMPLES_PER_CATEGORY = 3
COMPLEXITY_FIELDS = (
    "target_device_count",
    "evidence_atom_count",
    "modality_count",
    "operator_count",
    "critical_path_depth",
)
CURVE_METRICS = (
    "joint_accuracy",
    "field_accuracy",
    "format_valid",
    "triggered_device_f1",
    "l3_local_joint_accuracy",
    "l3_reduction_joint_accuracy",
    "l3_reduction_joint_given_local_correct",
    "l1_numeric_joint_accuracy",
    "l2_triggered_accuracy",
    "l2_first_trigger_time_accuracy",
    "l2_trigger_hours_accuracy",
    "l2_margin_accuracy",
    "l3_triggered_device_exact_match",
    "l3_earliest_trigger_device_accuracy",
    "l3_margin_order_accuracy",
    "read_evidence_recall",
    "available_evidence_recall",
    "cited_evidence_recall",
    "complete_evidence_rate",
    "read_complete_evidence_rate",
    "search_call_count",
    "read_call_count",
    "distinct_read_files",
    "over_read_file_ratio",
    "irrelevant_read_file_ratio",
    "read_bytes",
    "steps_to_complete_read",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "tool_budget_reached",
    "budget_finalization_used",
    "budget_finalization_succeeded",
    "time_budget_reached",
    "time_finalization_used",
    "time_finalization_succeeded",
)


class _SourceSnapshots:
    """Read report inputs once and retain the digest of those exact bytes.

    Provenance checks must not parse one version of a file and then hash a
    newer version reached through the same path. Parsed inputs are therefore
    cached as bytes, while hash-only inputs retain their first observed digest.
    A final verification detects any source that changed while the report was
    being assembled.
    """

    def __init__(self) -> None:
        self._payloads: dict[Path, bytes] = {}
        self._digests: dict[Path, str] = {}

    @staticmethod
    def _key(path: str | Path) -> Path:
        return Path(path).resolve()

    def read_bytes(self, path: str | Path) -> bytes:
        key = self._key(path)
        if key in self._payloads:
            return self._payloads[key]
        payload = key.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        previous = self._digests.get(key)
        if previous is not None and previous != digest:
            raise RuntimeError(f"source changed while it was being captured: {key}")
        self._payloads[key] = payload
        self._digests[key] = digest
        return payload

    def load_json(self, path: str | Path) -> dict[str, Any]:
        key = self._key(path)
        value = json.loads(self.read_bytes(key).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {key}")
        return value

    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        payload = self.read_bytes(path).decode("utf-8")
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    def digest(self, path: str | Path) -> str:
        key = self._key(path)
        if key not in self._digests:
            self._digests[key] = file_sha256(key)
        return self._digests[key]

    def bindings(self) -> dict[str, str]:
        return {str(path): digest for path, digest in sorted(self._digests.items())}

    def changed_sources(self) -> list[str]:
        errors: list[str] = []
        for path, expected in sorted(self._digests.items()):
            try:
                actual = file_sha256(path)
            except OSError as exc:
                errors.append(
                    f"source became unavailable during pilot report generation: {path}: {exc}"
                )
                continue
            if actual != expected:
                errors.append(f"source changed during pilot report generation: {path}")
        return errors


def _bootstrap_ci(values: list[float], *, seed: int = 20260902) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(2000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return [estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates))]]


def _key_label(value: Any) -> str:
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return str(value)


def _paired(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], Any],
    metric: str,
) -> dict[str, Any]:
    """Pair rows without silently overwriting duplicate experimental units."""

    left_groups: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    right_groups: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in left:
        left_groups[key(row)].append(row)
    for row in right:
        right_groups[key(row)].append(row)
    duplicate_left = sorted(
        (_key_label(value) for value, rows in left_groups.items() if len(rows) > 1), key=str
    )
    duplicate_right = sorted(
        (_key_label(value) for value, rows in right_groups.items() if len(rows) > 1), key=str
    )
    left_keys = {value for value, rows in left_groups.items() if len(rows) == 1}
    right_keys = {value for value, rows in right_groups.items() if len(rows) == 1}
    shared = sorted(left_keys & right_keys, key=str)
    differences = [
        float(left_groups[value][0][metric]) - float(right_groups[value][0][metric])
        for value in shared
    ]
    valid = not duplicate_left and not duplicate_right
    return {
        "valid": valid,
        "left_count": len(left),
        "right_count": len(right),
        "paired_count": len(shared),
        "unpaired_left": sorted((_key_label(value) for value in left_keys - right_keys), key=str),
        "unpaired_right": sorted((_key_label(value) for value in right_keys - left_keys), key=str),
        "duplicate_left_keys": duplicate_left,
        "duplicate_right_keys": duplicate_right,
        "mean_difference": mean(differences) if differences and valid else None,
        "bootstrap_95_ci": _bootstrap_ci(differences) if valid else None,
    }


def _gate(status: str, rationale: str, evidence: Any) -> dict[str, Any]:
    return {"status": status, "rationale": rationale, "evidence": evidence}


def _load_json(path: Path, snapshots: _SourceSnapshots) -> dict[str, Any]:
    return snapshots.load_json(path)


def _duplicate_values(rows: Iterable[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(str(row.get(field)) for row in rows if row.get(field) is not None)
    return sorted(value for value, count in counts.items() if count > 1)


def _logical_job_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("qa_id"),
        row.get("model"),
        row.get("reasoning_effort"),
        row.get("condition"),
        row.get("N"),
        row.get("scope_seed"),
        row.get("replicate", "main"),
    )


def _duplicate_logical_jobs(rows: Iterable[dict[str, Any]]) -> list[str]:
    counts = Counter(_logical_job_key(row) for row in rows)
    return sorted(_key_label(value) for value, count in counts.items() if count > 1)


def _manifest_provenance(
    manifest_path: Path,
    manifest: list[dict[str, Any]],
    bundle_path: Path,
    bundle: dict[str, Any],
    snapshots: _SourceSnapshots,
) -> tuple[list[str], dict[tuple[str, int, int], tuple[str, ...]]]:
    """Verify the persisted manifest summary and load its bound scope memberships."""

    errors: list[str] = []
    summary_path = manifest_path.with_suffix(".summary.json")
    if not summary_path.is_file():
        return [f"manifest binding summary is missing: {summary_path}"], {}
    try:
        summary = _load_json(summary_path, snapshots)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"manifest binding summary is invalid: {exc}"], {}
    if summary.get("manifest_sha256") != snapshots.digest(manifest_path):
        errors.append("manifest SHA disagrees with its binding summary")
    if summary.get("job_count") != len(manifest):
        errors.append("manifest job_count disagrees with its binding summary")
    experiment_ids = {row.get("experiment_id") for row in manifest}
    if experiment_ids != {summary.get("experiment_id")}:
        errors.append("manifest experiment_id disagrees with its binding summary")
    if summary.get("bundle_sha256") != snapshots.digest(bundle_path):
        errors.append("bundle SHA disagrees with the manifest binding summary")

    bundle_hashes = bundle.get("artifact_hashes")
    if not isinstance(bundle_hashes, dict):
        errors.append("bundle has no artifact hash bindings")
        bundle_hashes = {}
    for name in (
        "config_sha256",
        "questions_sha256",
        "scopes_sha256",
        "gold_sha256",
        "index_sha256",
    ):
        if summary.get(name) != bundle_hashes.get(name) or not summary.get(name):
            errors.append(f"manifest/bundle binding disagrees for {name}")
    if summary.get("raw_content_sha256") != bundle.get("raw_content_sha256") or not summary.get(
        "raw_content_sha256"
    ):
        errors.append("manifest/bundle raw-content binding disagrees")
    index_value = summary.get("index_path")
    if not isinstance(index_value, str) or not Path(index_value).is_file():
        errors.append("bound corpus index is missing")
    elif summary.get("index_sha256") != snapshots.digest(index_value):
        errors.append("bound corpus index SHA disagrees with the manifest summary")

    scope_members: dict[tuple[str, int, int], tuple[str, ...]] = {}
    scopes_value = summary.get("scopes_path")
    if not isinstance(scopes_value, str):
        errors.append("manifest summary has no scopes_path")
        return errors, scope_members
    scopes_path = Path(scopes_value)
    if not scopes_path.is_file():
        errors.append(f"bound scope file is missing: {scopes_path}")
        return errors, scope_members
    scope_rows = snapshots.read_jsonl(scopes_path)
    if snapshots.digest(scopes_path) != summary.get("scopes_sha256"):
        errors.append("bound scope file SHA disagrees with the manifest summary")
        return errors, scope_members
    for index, row in enumerate(scope_rows, start=1):
        try:
            key = (str(row["pair_id"]), int(row["N"]), int(row["scope_seed"]))
            members = tuple(sorted(str(uid) for uid in row["device_uids"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid bound scope row {index}: {exc}")
            continue
        if key in scope_members:
            errors.append(f"duplicate bound scope key: {key}")
        scope_members[key] = members
    return errors, scope_members


def _evaluation_provenance(
    scores_path: Path,
    scores: list[dict[str, Any]],
    manifest_path: Path,
    snapshots: _SourceSnapshots,
) -> list[str]:
    """Verify the evaluator's complete input/output chain against the run manifest."""

    errors: list[str] = []
    evaluation_summary_path = scores_path.parent / "evaluation_summary.json"
    manifest_summary_path = manifest_path.with_suffix(".summary.json")
    if not evaluation_summary_path.is_file():
        return [f"evaluation binding summary is missing: {evaluation_summary_path}"]
    try:
        evaluation = _load_json(evaluation_summary_path, snapshots)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"evaluation binding summary is invalid: {exc}"]
    if not manifest_summary_path.is_file():
        return [f"manifest binding summary is missing: {manifest_summary_path}"]
    try:
        manifest_summary = _load_json(manifest_summary_path, snapshots)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"manifest binding summary is invalid: {exc}"]

    expected_paths = {
        "per_run_scores_path": scores_path,
        "manifest_path": manifest_path,
        "manifest_summary_path": manifest_summary_path,
    }
    for name, expected in expected_paths.items():
        value = evaluation.get(name)
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            errors.append(f"evaluation {name} disagrees with the report input")

    actual_scores_sha = snapshots.digest(scores_path)
    if evaluation.get("per_run_scores_sha256") != actual_scores_sha:
        errors.append("per-run score SHA disagrees with the evaluation summary")
    actual_manifest_sha = snapshots.digest(manifest_path)
    if evaluation.get("manifest_sha256") != actual_manifest_sha or evaluation.get(
        "manifest_sha256"
    ) != manifest_summary.get("manifest_sha256"):
        errors.append("evaluation/manifest SHA bindings disagree")
    actual_manifest_summary_sha = snapshots.digest(manifest_summary_path)
    if evaluation.get("manifest_summary_sha256") != actual_manifest_summary_sha:
        errors.append("manifest-summary SHA disagrees with the evaluation summary")

    if evaluation.get("experiment_id") != manifest_summary.get("experiment_id"):
        errors.append("evaluation experiment_id disagrees with the manifest summary")
    for artifact in ("questions", "gold"):
        path_name = f"{artifact}_path"
        sha_name = f"{artifact}_sha256"
        value = evaluation.get(path_name)
        manifest_value = manifest_summary.get(path_name)
        if (
            not isinstance(value, str)
            or not isinstance(manifest_value, str)
            or Path(value).resolve() != Path(manifest_value).resolve()
        ):
            errors.append(f"evaluation/manifest {artifact} identities disagree")
            continue
        source = Path(value)
        if not source.is_file():
            errors.append(f"evaluation-bound {artifact} file is missing: {source}")
            continue
        actual_sha = snapshots.digest(source)
        if evaluation.get(sha_name) != actual_sha or actual_sha != manifest_summary.get(sha_name):
            errors.append(f"evaluation/manifest {artifact} SHA bindings disagree")

    results_value = evaluation.get("results_path")
    if not isinstance(results_value, str) or not Path(results_value).is_file():
        errors.append("evaluation-bound results file is missing")
    elif evaluation.get("results_sha256") != snapshots.digest(results_value):
        errors.append("results SHA disagrees with the evaluation summary")

    if evaluation.get("scored_count") != len(scores):
        errors.append("evaluation scored_count disagrees with the score file")
    unique_score_ids = {row.get("run_id") for row in scores if isinstance(row.get("run_id"), str)}
    if evaluation.get("unique_run_count") != len(unique_score_ids):
        errors.append("evaluation unique_run_count disagrees with the score file")
    return errors


def _manifest_matrix_errors(
    manifest: list[dict[str, Any]],
    bundle: dict[str, Any],
    scope_members: dict[tuple[str, int, int], tuple[str, ...]],
) -> tuple[list[str], dict[str, Any]]:
    if manifest and all(
        row.get("anchor_type") == "Indirect"
        and row.get("condition") == "full_retrieval"
        and row.get("replicate", "main") == "main"
        for row in manifest
    ):
        return _autonomous_matrix_errors(manifest, bundle, scope_members)

    errors: list[str] = []
    main = [
        row
        for row in manifest
        if row.get("condition") == "full_retrieval" and row.get("replicate", "main") == "main"
    ]
    models = sorted({str(row.get("model")) for row in manifest if row.get("model")})
    main_seeds = sorted(
        {int(row["scope_seed"]) for row in main if row.get("scope_seed") is not None}
    )
    base_seed = main_seeds[0] if len(main_seeds) == 1 else None
    if len(main_seeds) != 1:
        errors.append(f"main full-retrieval jobs must use exactly one base seed, got {main_seeds}")

    expected_n = [int(value) for value in bundle.get("scope_sizes", [])]
    observed_n = sorted({int(row["N"]) for row in main if row.get("N") is not None})
    n_values = expected_n or observed_n
    if not n_values:
        errors.append("no main scope sizes are present")
    elif observed_n != n_values:
        errors.append(f"main scope sizes {observed_n} do not match bundle {n_values}")

    question_count = int(bundle.get("public_question_count", 0))
    pair_count = int(bundle.get("direct_indirect_pairs", 0))
    if question_count <= 0 or pair_count <= 0:
        errors.append("bundle has no positive public question/pair count")
    if not models:
        errors.append("manifest contains no models")

    configured_seeds = [int(value) for value in bundle.get("scope_seeds", [])]
    allowed_replicate_seeds = set(configured_seeds[1:]) if configured_seeds else set()
    high_n = n_values[-1] if n_values else None
    for row in manifest:
        condition = row.get("condition")
        replicate = row.get("replicate", "main")
        try:
            size = int(row["N"])
            seed = int(row["scope_seed"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"run {row.get('run_id')} has an invalid N or scope_seed")
            continue
        if condition == "full_retrieval" and replicate != "main":
            valid_stability_job = (
                replicate == "distractor_stability"
                and row.get("anchor_type") == "Indirect"
                and high_n is not None
                and size < high_n
                and seed != base_seed
                and (not allowed_replicate_seeds or seed in allowed_replicate_seeds)
            )
            if not valid_stability_job:
                errors.append(
                    f"run {row.get('run_id')} is not a valid Indirect distractor-stability replicate"
                )
        elif condition == "oracle_evidence" and (replicate != "main" or seed != base_seed):
            errors.append(f"run {row.get('run_id')} has an invalid Oracle replicate or seed")

    reference_qids: set[str] | None = None
    expected_level_counts = {
        str(level): int(count)
        for level, count in bundle.get("validation", {}).get("questions_by_level", {}).items()
    }
    for model in models:
        efforts = {
            str(row.get("reasoning_effort")) for row in manifest if row.get("model") == model
        }
        if len(efforts) != 1:
            errors.append(f"model {model} has multiple reasoning efforts: {sorted(efforts)}")
        for n in n_values:
            cell = [row for row in main if row.get("model") == model and int(row.get("N", -1)) == n]
            qids = {str(row.get("qa_id")) for row in cell}
            if len(cell) != question_count or len(qids) != question_count:
                errors.append(
                    f"model={model}/N={n} has {len(cell)} rows and {len(qids)} unique questions; "
                    f"expected {question_count}"
                )
            if reference_qids is None:
                reference_qids = qids
            elif qids != reference_qids:
                errors.append(f"model={model}/N={n} question set differs from the main matrix")
            anchor_counts = Counter(str(row.get("anchor_type")) for row in cell)
            if anchor_counts != Counter({"Direct": pair_count, "Indirect": pair_count}):
                errors.append(f"model={model}/N={n} anchor counts are {dict(anchor_counts)}")
            pair_anchors: defaultdict[str, list[str]] = defaultdict(list)
            for row in cell:
                pair_anchors[str(row.get("pair_id"))].append(str(row.get("anchor_type")))
            invalid_pairs = sorted(
                pair_id
                for pair_id, anchors in pair_anchors.items()
                if Counter(anchors) != Counter({"Direct": 1, "Indirect": 1})
            )
            if len(pair_anchors) != pair_count or invalid_pairs:
                errors.append(
                    f"model={model}/N={n} has invalid Direct/Indirect pairs: {invalid_pairs}"
                )
            if expected_level_counts:
                level_counts = Counter(str(row.get("task_level")) for row in cell)
                if dict(level_counts) != expected_level_counts:
                    errors.append(f"model={model}/N={n} level counts are {dict(level_counts)}")

        oracle = [
            row
            for row in manifest
            if row.get("model") == model and row.get("condition") == "oracle_evidence"
        ]
        oracle_qids = {str(row.get("qa_id")) for row in oracle}
        main_indirect_qids = {
            str(row.get("qa_id"))
            for row in main
            if row.get("model") == model
            and row.get("anchor_type") == "Indirect"
            and int(row.get("N", -1)) == high_n
        }
        if (
            len(oracle) != pair_count
            or oracle_qids != main_indirect_qids
            or any(row.get("anchor_type") != "Indirect" for row in oracle)
            or any(int(row.get("N", -1)) != high_n for row in oracle)
            or any(row.get("replicate", "main") != "main" for row in oracle)
        ):
            errors.append(f"model={model} Oracle jobs do not match all high-N Indirect questions")

        reference_indirect = [
            row
            for row in main
            if row.get("model") == model
            and row.get("anchor_type") == "Indirect"
            and int(row.get("N", -1)) == high_n
        ]
        expected_stability: set[tuple[str, int, int]] = set()
        for reference in reference_indirect:
            pair_id = str(reference.get("pair_id"))
            qa_id = str(reference.get("qa_id"))
            for seed_index, seed in enumerate(configured_seeds[1:], start=1):
                for size in n_values[:-1]:
                    scope_key = (pair_id, size, seed)
                    if scope_key not in scope_members:
                        errors.append(f"bound scopes omit stability key {scope_key}")
                        continue
                    earlier_scopes = {
                        scope_members[(pair_id, size, earlier)]
                        for earlier in configured_seeds[:seed_index]
                        if (pair_id, size, earlier) in scope_members
                    }
                    if scope_members[scope_key] not in earlier_scopes:
                        expected_stability.add((qa_id, size, seed))
        actual_stability_rows = [
            row
            for row in manifest
            if row.get("model") == model
            and row.get("condition") == "full_retrieval"
            and row.get("replicate") == "distractor_stability"
        ]
        actual_stability = {
            (str(row.get("qa_id")), int(row.get("N", -1)), int(row.get("scope_seed", -1)))
            for row in actual_stability_rows
        }
        if len(actual_stability) != len(actual_stability_rows):
            errors.append(f"model={model} has duplicate distractor-stability logical jobs")
        missing_stability = sorted(expected_stability - actual_stability)
        unexpected_stability = sorted(actual_stability - expected_stability)
        if missing_stability:
            errors.append(
                f"model={model} is missing distractor-stability jobs: {missing_stability[:10]}"
            )
        if unexpected_stability:
            errors.append(
                f"model={model} has unexpected distractor-stability jobs: "
                f"{unexpected_stability[:10]}"
            )

    known_conditions = {"full_retrieval", "oracle_evidence"}
    unknown_conditions = sorted(
        {
            str(row.get("condition"))
            for row in manifest
            if row.get("condition") not in known_conditions
        }
    )
    if unknown_conditions:
        errors.append(f"unknown experimental conditions: {unknown_conditions}")
    return errors, {
        "design": "paired_diagnostics",
        "models": models,
        "base_seed": base_seed,
        "n_values": n_values,
        "question_count": question_count,
        "pair_count": pair_count,
    }


def _autonomous_matrix_errors(
    manifest: list[dict[str, Any]],
    bundle: dict[str, Any],
    scope_members: dict[tuple[str, int, int], tuple[str, ...]],
) -> tuple[list[str], dict[str, Any]]:
    """Validate the lean autonomous-question-by-N main-experiment matrix."""

    errors: list[str] = []
    models = sorted({str(row.get("model")) for row in manifest if row.get("model")})
    expected_n = [int(value) for value in bundle.get("scope_sizes", [])]
    observed_n = sorted({int(row["N"]) for row in manifest if row.get("N") is not None})
    n_values = expected_n or observed_n
    if not models:
        errors.append("manifest contains no models")
    if not n_values:
        errors.append("no scope sizes are present")
    elif observed_n != n_values:
        errors.append(f"scope sizes {observed_n} do not match bundle {n_values}")

    seeds = sorted(
        {int(row["scope_seed"]) for row in manifest if row.get("scope_seed") is not None}
    )
    base_seed = seeds[0] if len(seeds) == 1 else None
    if len(seeds) != 1:
        errors.append(f"all jobs must use exactly one base seed, got {seeds}")

    reference_qids: set[str] | None = None
    reference_levels: Counter[str] | None = None
    for model in models:
        efforts = {
            str(row.get("reasoning_effort")) for row in manifest if row.get("model") == model
        }
        if len(efforts) != 1:
            errors.append(f"model {model} has multiple reasoning efforts: {sorted(efforts)}")
        for n in n_values:
            cell = [
                row for row in manifest if row.get("model") == model and int(row.get("N", -1)) == n
            ]
            qids = [str(row.get("qa_id")) for row in cell]
            if len(qids) != len(set(qids)):
                errors.append(f"model={model}/N={n} has duplicate questions")
            qid_set = set(qids)
            if reference_qids is None:
                reference_qids = qid_set
            elif qid_set != reference_qids:
                errors.append(f"model={model}/N={n} question set differs from the main matrix")
            levels = Counter(str(row.get("task_level")) for row in cell)
            if reference_levels is None:
                reference_levels = levels
            elif levels != reference_levels:
                errors.append(f"model={model}/N={n} level counts are {dict(levels)}")
            for row in cell:
                scope_key = (str(row.get("pair_id")), n, int(row.get("scope_seed", -1)))
                if scope_key not in scope_members:
                    errors.append(f"bound scopes omit main key {scope_key}")

    if not reference_qids:
        errors.append("matrix contains no questions")
    if not reference_levels or set(reference_levels) != set(LEVELS):
        errors.append(f"matrix must contain L1, L2, and L3, got {dict(reference_levels or {})}")
    return errors, {
        "design": "autonomous_retrieval_scaling",
        "models": models,
        "base_seed": base_seed,
        "n_values": n_values,
        "question_count": len(reference_qids or ()),
        "pair_count": len(reference_qids or ()),
    }


def _audit_runs(
    scores: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    provenance_errors: list[str],
    scope_members: dict[tuple[str, int, int], tuple[str, ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    duplicate_manifest_ids = _duplicate_values(manifest, "run_id")
    duplicate_score_ids = _duplicate_values(scores, "run_id")
    duplicate_manifest_jobs = _duplicate_logical_jobs(manifest)
    missing_manifest_ids = sum(row.get("run_id") is None for row in manifest)
    missing_score_ids = sum(row.get("run_id") is None for row in scores)
    expected_ids = {str(row["run_id"]) for row in manifest if row.get("run_id") is not None}
    actual_ids = {str(row["run_id"]) for row in scores if row.get("run_id") is not None}
    missing_run_ids = sorted(expected_ids - actual_ids)
    unexpected_run_ids = sorted(actual_ids - expected_ids)

    manifest_by_id = {
        str(row["run_id"]): row
        for row in manifest
        if row.get("run_id") is not None and str(row["run_id"]) not in duplicate_manifest_ids
    }
    score_by_id = {
        str(row["run_id"]): row
        for row in scores
        if row.get("run_id") is not None and str(row["run_id"]) not in duplicate_score_ids
    }
    mismatch_fields = (
        "experiment_id",
        "qa_id",
        "pair_id",
        "task_level",
        "anchor_type",
        "model",
        "reasoning_effort",
        "condition",
        "N",
        "scope_seed",
        "replicate",
    )
    mismatches: list[dict[str, Any]] = []
    for run_id in sorted(expected_ids & actual_ids):
        if run_id not in manifest_by_id or run_id not in score_by_id:
            continue
        expected = manifest_by_id[run_id]
        actual = score_by_id[run_id]
        fields = [
            field
            for field in mismatch_fields
            if actual.get(field, "main" if field == "replicate" else None)
            != expected.get(field, "main" if field == "replicate" else None)
        ]
        if fields:
            mismatches.append({"run_id": run_id, "fields": fields})

    classification_mismatches = [
        {
            "run_id": row.get("run_id"),
            "reported": row.get("error_category"),
            "derived": classify_error(row),
        }
        for row in scores
        if row.get("error_category") != classify_error(row)
    ]

    infrastructure = [
        {
            "run_id": row.get("run_id"),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for row in scores
        if row.get("status") in INFRASTRUCTURE_STATUSES
        or row.get("error_category") == "infrastructure"
    ]
    non_evaluable = [
        {
            "run_id": row.get("run_id"),
            "status": row.get("status"),
            "error_category": row.get("error_category"),
        }
        for row in scores
        if row.get("status") not in EVALUABLE_TERMINAL_STATUSES
        or row.get("error_category") == "infrastructure"
    ]
    matrix_errors, matrix = _manifest_matrix_errors(manifest, bundle, scope_members)
    matrix_errors = [*provenance_errors, *matrix_errors]
    complete = not any(
        (
            duplicate_manifest_ids,
            duplicate_score_ids,
            duplicate_manifest_jobs,
            missing_manifest_ids,
            missing_score_ids,
            missing_run_ids,
            unexpected_run_ids,
            mismatches,
            classification_mismatches,
            non_evaluable,
            matrix_errors,
        )
    )
    audit = {
        "complete": complete,
        "expected_run_count": len(manifest),
        "score_row_count": len(scores),
        "evaluable_terminal_count": len(scores) - len(non_evaluable),
        "missing_run_ids": missing_run_ids,
        "unexpected_run_ids": unexpected_run_ids,
        "duplicate_manifest_run_ids": duplicate_manifest_ids,
        "duplicate_score_run_ids": duplicate_score_ids,
        "duplicate_manifest_logical_jobs": duplicate_manifest_jobs,
        "rows_without_manifest_run_id": missing_manifest_ids,
        "rows_without_score_run_id": missing_score_ids,
        "field_mismatches": mismatches,
        "error_classification_mismatches": classification_mismatches,
        "non_evaluable_runs": non_evaluable,
        "matrix_errors": matrix_errors,
        "infrastructure_failure_count": len(infrastructure),
        "infrastructure_failures": infrastructure,
    }
    usable = [
        score_by_id[run_id]
        for run_id in sorted(expected_ids)
        if run_id in score_by_id
        and score_by_id[run_id].get("status") in EVALUABLE_TERMINAL_STATUSES
        and score_by_id[run_id].get("error_category") != "infrastructure"
    ]
    return audit, usable, matrix


def _metric_mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return mean(values) if values else None


def _curve_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    curves: list[dict[str, Any]] = []
    for values, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = {key: value for key, value in zip(keys, values, strict=True)}
        result["count"] = len(group)
        for metric in CURVE_METRICS:
            result[metric] = _metric_mean(group, metric)
        budget_reached = [row for row in group if row.get("tool_budget_reached") is True]
        result["tool_budget_reached_rate"] = len(budget_reached) / len(group)
        result["budget_finalization_success_given_reached"] = (
            sum(row.get("budget_finalization_succeeded") is True for row in budget_reached)
            / len(budget_reached)
            if budget_reached
            else None
        )
        time_reached = [row for row in group if row.get("time_budget_reached") is True]
        result["time_budget_reached_rate"] = len(time_reached) / len(group)
        result["time_finalization_success_given_reached"] = (
            sum(row.get("time_finalization_succeeded") is True for row in time_reached)
            / len(time_reached)
            if time_reached
            else None
        )
        result["l3_reduction_given_local_correct_count"] = sum(
            row.get("l3_reduction_joint_given_local_correct") is not None for row in group
        )
        result["search_read_call_count"] = mean(
            float(row.get("search_call_count") or 0) + float(row.get("read_call_count") or 0)
            for row in group
        )
        result["error_counts"] = dict(
            sorted(Counter(str(row.get("error_category")) for row in group).items())
        )
        result["run_failure_mode_counts"] = dict(
            sorted(Counter(str(row.get("run_failure_mode", "none")) for row in group).items())
        )
        curves.append(result)
    return curves


def _complexity_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe Gold complexity once per question without implying an independent curve."""

    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        qa_id = row.get("qa_id")
        if isinstance(qa_id, str):
            by_question.setdefault(qa_id, row)
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_question.values():
        grouped[(str(row.get("task_level")), str(row.get("anchor_type")))].append(row)

    result: list[dict[str, Any]] = []
    for (task_level, anchor_type), group in sorted(grouped.items()):
        profiles = Counter(tuple(row.get(field) for field in COMPLEXITY_FIELDS) for row in group)
        result.append(
            {
                "task_level": task_level,
                "anchor_type": anchor_type,
                "question_count": len(group),
                "profiles": [
                    {
                        **dict(zip(COMPLEXITY_FIELDS, values, strict=True)),
                        "question_count": count,
                    }
                    for values, count in sorted(profiles.items(), key=lambda item: str(item[0]))
                ],
            }
        )
    return result


def _indirect_metrics(
    rows: list[dict[str, Any]], n_values: list[int]
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for n in n_values:
        group = [row for row in rows if row.get("anchor_type") == "Indirect" and int(row["N"]) == n]
        if not group:
            continue
        result[n] = {
            "joint_accuracy": float(_metric_mean(group, "joint_accuracy") or 0.0),
            "read_evidence_recall": float(_metric_mean(group, "read_evidence_recall") or 0.0),
            "search_read_call_count": mean(
                float(row.get("search_call_count") or 0) + float(row.get("read_call_count") or 0)
                for row in group
            ),
            "read_bytes": float(_metric_mean(group, "read_bytes") or 0.0),
        }
    return result


def _relative_growth(low: float, high: float) -> float:
    return (high - low) / max(low, 1.0)


def _not_evaluable(
    output_path: Path,
    *,
    reason: str,
    run_audit: dict[str, Any],
    manifest_path: Path | None,
) -> dict[str, Any]:
    report = {
        "decision": "not_evaluable",
        "reason": reason,
        "primary_metric": "Macro Joint Accuracy",
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "run_audit": run_audit,
        "infrastructure": {
            "failure_count": int(run_audit.get("infrastructure_failure_count", 0)),
            "failures": run_audit.get("infrastructure_failures", []),
        },
        "curves": [],
        "complexity_profiles": [],
        "replicate_curves": [],
        "gates": {},
    }
    write_json(output_path, report)
    return report


def build_pilot_report(
    *,
    scores_path: str | Path,
    bundle_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    scores_source = Path(scores_path)
    output = Path(output_path)
    bundle_source = Path(bundle_path)
    snapshots = _SourceSnapshots()
    bundle = _load_json(bundle_source, snapshots)
    resolved_manifest = Path(manifest_path) if manifest_path is not None else None
    if resolved_manifest is None or not resolved_manifest.is_file():
        return _not_evaluable(
            output,
            reason="The planned run manifest is unavailable; task-list completeness cannot be verified.",
            run_audit={
                "complete": False,
                "manifest_missing": True,
                "score_row_count": len(read_jsonl(scores_source)),
                "infrastructure_failure_count": 0,
                "infrastructure_failures": [],
            },
            manifest_path=resolved_manifest,
        )

    scores = snapshots.read_jsonl(scores_source)
    manifest = snapshots.read_jsonl(resolved_manifest)
    provenance_errors, scope_members = _manifest_provenance(
        resolved_manifest,
        manifest,
        bundle_source,
        bundle,
        snapshots,
    )
    provenance_errors.extend(
        _evaluation_provenance(scores_source, scores, resolved_manifest, snapshots)
    )
    audit, usable, matrix = _audit_runs(
        scores,
        manifest,
        bundle,
        provenance_errors=provenance_errors,
        scope_members=scope_members,
    )
    if not audit["complete"]:
        return _not_evaluable(
            output,
            reason="The score file does not contain exactly one terminal evaluable result for every planned task.",
            run_audit=audit,
            manifest_path=resolved_manifest,
        )

    models = matrix["models"]
    base_seed = matrix["base_seed"]
    n_values = matrix["n_values"]
    indirect_only = matrix.get("design") == "autonomous_retrieval_scaling"
    if len(n_values) < 2:
        audit["matrix_errors"].append("at least two scope sizes are required for retrieval scaling")
        audit["complete"] = False
        return _not_evaluable(
            output,
            reason="The complete task list has no evaluable retrieval-size contrast.",
            run_audit=audit,
            manifest_path=resolved_manifest,
        )
    low_n, high_n = n_values[0], n_values[-1]
    main = [
        row
        for row in usable
        if row.get("condition") == "full_retrieval"
        and row.get("replicate", "main") == "main"
        and int(row.get("scope_seed")) == base_seed
    ]
    curve_keys = (
        ("model", "task_level", "N")
        if indirect_only
        else ("model", "anchor_type", "task_level", "N")
    )
    curves = _curve_rows(main, curve_keys)
    complexity_profiles = _complexity_profiles(main)
    replicate_curves = (
        []
        if indirect_only
        else _curve_rows(
            [row for row in usable if row.get("condition") == "full_retrieval"],
            ("model", "anchor_type", "task_level", "N", "scope_seed", "replicate"),
        )
    )
    oracle_accuracy_curves = (
        []
        if indirect_only
        else _curve_rows(
            [row for row in usable if row.get("condition") == "oracle_evidence"],
            ("model", "anchor_type", "task_level"),
        )
    )
    # Oracle is an accuracy-only control. Do not expose evaluator-supplied
    # evidence as if it were model retrieval cost.
    oracle_accuracy_curves = [
        {
            "model": row["model"],
            "anchor_type": row["anchor_type"],
            "task_level": row["task_level"],
            "count": row["count"],
            "joint_accuracy": row["joint_accuracy"],
            "field_accuracy": row["field_accuracy"],
        }
        for row in oracle_accuracy_curves
    ]

    per_model: dict[str, Any] = {}
    gate1_support: list[bool] = []
    gate2_support: list[bool] = []
    gate3_support: list[bool] = []
    gate4_support: list[bool] = []
    gate5_support: list[bool] = []
    for model in models:
        rows = [row for row in usable if row.get("model") == model]
        model_main = [row for row in main if row.get("model") == model]
        direct_high = [
            row
            for row in model_main
            if row.get("anchor_type") == "Direct" and int(row["N"]) == high_n
        ]
        indirect_high = [
            row
            for row in model_main
            if row.get("anchor_type") == "Indirect" and int(row["N"]) == high_n
        ]
        if indirect_only:
            direct_high = indirect_high
        direct_gap = _paired(
            direct_high,
            indirect_high,
            key=lambda row: row["pair_id"],
            metric="joint_accuracy",
        )
        direct_gap_by_level = {
            level: _paired(
                [row for row in direct_high if row.get("task_level") == level],
                [row for row in indirect_high if row.get("task_level") == level],
                key=lambda row: row["pair_id"],
                metric="joint_accuracy",
            )
            for level in LEVELS
        }
        ci = direct_gap["bootstrap_95_ci"]
        gate1 = bool(
            direct_gap["valid"]
            and not direct_gap["unpaired_left"]
            and not direct_gap["unpaired_right"]
            and direct_gap["mean_difference"] is not None
            and direct_gap["mean_difference"] >= 0.05
            and ci is not None
            and ci[0] > 0
        )

        direct_by_n = {
            n: float(
                _metric_mean(
                    [
                        row
                        for row in model_main
                        if row.get("anchor_type") == "Direct" and int(row["N"]) == n
                    ],
                    "joint_accuracy",
                )
                or 0.0
            )
            for n in n_values
        }
        direct_by_level_n = {
            level: {
                n: float(
                    _metric_mean(
                        [
                            row
                            for row in model_main
                            if row.get("anchor_type") == "Direct"
                            and row.get("task_level") == level
                            and int(row["N"]) == n
                        ],
                        "joint_accuracy",
                    )
                    or 0.0
                )
                for n in n_values
            }
            for level in LEVELS
        }
        direct_range = max(direct_by_n.values()) - min(direct_by_n.values())
        direct_range_by_level = {
            level: max(values.values()) - min(values.values())
            for level, values in direct_by_level_n.items()
        }
        gate2 = all(value <= 0.05 for value in direct_range_by_level.values())

        indirect_by_n = _indirect_metrics(model_main, n_values)
        indirect_by_level_n = {
            level: _indirect_metrics(
                [row for row in model_main if row.get("task_level") == level], n_values
            )
            for level in LEVELS
        }
        low = indirect_by_n[low_n]
        high = indirect_by_n[high_n]
        evidence_delta = abs(high["read_evidence_recall"] - low["read_evidence_recall"])
        retrieval_call_growth = _relative_growth(
            low["search_read_call_count"], high["search_read_call_count"]
        )
        byte_growth = _relative_growth(low["read_bytes"], high["read_bytes"])
        gate3 = bool(evidence_delta >= 0.05 or retrieval_call_growth >= 0.10 or byte_growth >= 0.10)

        oracle = [
            row
            for row in rows
            if row.get("condition") == "oracle_evidence" and row.get("anchor_type") == "Indirect"
        ]
        if indirect_only:
            oracle = indirect_high
        oracle_gap = _paired(
            oracle,
            indirect_high,
            key=lambda row: row["qa_id"],
            metric="joint_accuracy",
        )
        oracle_gap_by_level = {
            level: _paired(
                [row for row in oracle if row.get("task_level") == level],
                [row for row in indirect_high if row.get("task_level") == level],
                key=lambda row: row["qa_id"],
                metric="joint_accuracy",
            )
            for level in LEVELS
        }
        oracle_ci = oracle_gap["bootstrap_95_ci"]
        gate4 = bool(
            oracle_gap["valid"]
            and not oracle_gap["unpaired_left"]
            and not oracle_gap["unpaired_right"]
            and oracle_gap["mean_difference"] is not None
            and oracle_gap["mean_difference"] >= 0.05
            and oracle_ci is not None
            and oracle_ci[0] > 0
        )

        level_accuracy = {
            level: float(
                _metric_mean(
                    [row for row in indirect_high if row.get("task_level") == level],
                    "joint_accuracy",
                )
                or 0.0
            )
            for level in LEVELS
        }
        represented_levels = [
            level
            for level in LEVELS
            if any(row.get("task_level") == level for row in indirect_high)
        ]
        macro_joint_accuracy = mean(level_accuracy[level] for level in represented_levels)
        micro_joint_accuracy = float(_metric_mean(indirect_high, "joint_accuracy") or 0.0)
        all_saturated = all(value >= 0.95 for value in level_accuracy.values())
        gate5 = bool(not all_saturated and level_accuracy["L3"] >= 0.05)

        stable_accuracy_statement = None
        accuracy_delta = high["joint_accuracy"] - low["joint_accuracy"]
        cost_up = retrieval_call_growth >= 0.10 or byte_growth >= 0.10
        if abs(accuracy_delta) <= 0.02 and cost_up:
            stable_accuracy_statement = (
                "At the 761-device scale this model kept retrieval success relatively stable, but sustaining that success required more search and read operations."
            )

        indirect_summary = {
            "indirect_metrics_by_N": indirect_by_n,
            "indirect_metrics_by_level_N": indirect_by_level_n,
            "indirect_evidence_recall_absolute_change": evidence_delta,
            "indirect_search_read_call_relative_growth": retrieval_call_growth,
            # Compatibility alias; this now excludes Python calls.
            "indirect_tool_call_relative_growth": retrieval_call_growth,
            "indirect_read_byte_relative_growth": byte_growth,
            "L_accuracy_at_full_corpus_indirect": level_accuracy,
            "macro_joint_accuracy_at_full_corpus_indirect": macro_joint_accuracy,
            "micro_joint_accuracy_at_full_corpus_indirect": micro_joint_accuracy,
            "stable_accuracy_higher_cost_interpretation": stable_accuracy_statement,
        }
        per_model[model] = (
            indirect_summary
            if indirect_only
            else {
                "direct_minus_indirect_at_full_corpus": direct_gap,
                "direct_minus_indirect_by_level_at_full_corpus": direct_gap_by_level,
                "direct_joint_accuracy_by_N": direct_by_n,
                "direct_joint_accuracy_by_level_N": direct_by_level_n,
                "direct_range": direct_range,
                "direct_range_by_level": direct_range_by_level,
                **indirect_summary,
                "oracle_minus_full_retrieval": oracle_gap,
                "oracle_minus_full_retrieval_by_level": oracle_gap_by_level,
            }
        )
        gate1_support.append(gate1)
        gate2_support.append(gate2)
        gate3_support.append(gate3)
        gate4_support.append(gate4)
        gate5_support.append(gate5)

    allowed_errors = {*ERROR_TAXONOMY, "none", "unclassified"}
    observed_errors = Counter(str(row.get("error_category")) for row in usable)
    unexpected_errors = sorted(set(observed_errors) - allowed_errors)
    unclassified_count = observed_errors.get("unclassified", 0)
    taxonomy_pairs = {
        category: sorted(
            {str(row.get("pair_id")) for row in usable if row.get("error_category") == category}
        )
        for category in ERROR_TAXONOMY
    }
    taxonomy_coverage = {category: len(pair_ids) for category, pair_ids in taxonomy_pairs.items()}
    taxonomy_samples: dict[str, list[dict[str, str]]] = {}
    for category in ERROR_TAXONOMY:
        seen_pairs: set[str] = set()
        samples: list[dict[str, str]] = []
        for row in usable:
            if row.get("error_category") != category:
                continue
            pair_id = str(row.get("pair_id"))
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            samples.append({"pair_id": pair_id, "run_id": str(row.get("run_id"))})
            if len(samples) == 10:
                break
        taxonomy_samples[category] = samples
    observed_failure_categories = [
        category for category in ERROR_TAXONOMY if observed_errors.get(category, 0) > 0
    ]
    if unexpected_errors or unclassified_count:
        gate6_status = "not_supported"
    elif len(observed_failure_categories) >= 2 and all(
        taxonomy_coverage[category] >= MIN_ERROR_EXAMPLES_PER_CATEGORY
        for category in observed_failure_categories
    ):
        gate6_status = "supported"
    else:
        gate6_status = "inconclusive"
    shortcut_audit = bundle.get("validation", {}).get("shortcut_audit", {})
    shortcuts = int(shortcut_audit.get("single_file_sufficient_count", -1))
    exact_answer_hits = int(shortcut_audit.get("exact_answer_hit_count", -1))
    invalid_alternatives = int(shortcut_audit.get("invalid_alternative_count", -1))
    question_count = int(bundle.get("public_question_count", 0))
    gate7 = (
        shortcuts == 0
        and exact_answer_hits == 0
        and invalid_alternatives == 0
        and question_count > 0
    )

    retrieval_gate = _gate(
        "supported" if models and all(gate3_support) else "not_supported",
        "Requires >=0.05 absolute Evidence Recall change or >=10% search/read-call or "
        "read-byte growth for each model; Python calls are excluded and accuracy need not "
        "decline.",
        {
            model: {
                "evidence_change": per_model[model]["indirect_evidence_recall_absolute_change"],
                "search_read_call_growth": per_model[model][
                    "indirect_search_read_call_relative_growth"
                ],
                "byte_growth": per_model[model]["indirect_read_byte_relative_growth"],
            }
            for model in models
        },
    )
    levels_gate = _gate(
        "supported" if models and all(gate5_support) else "not_supported",
        "Requires levels not all >=0.95 and L3 >=0.05 for each model.",
        {model: per_model[model]["L_accuracy_at_full_corpus_indirect"] for model in models},
    )
    error_gate = _gate(
        gate6_status,
        "Requires no unclassified label, at least two observed failure classes, and at "
        "least three distinct semantic tasks in each; repeated experimental cells cannot "
        "inflate coverage. Emitted samples still require manual audit before publication.",
        {
            "counts": dict(sorted(observed_errors.items())),
            "unexpected_labels": unexpected_errors,
            "unclassified_count": unclassified_count,
            "minimum_distinct_pairs_per_observed_failure_class": (MIN_ERROR_EXAMPLES_PER_CATEGORY),
            "distinct_pair_coverage": taxonomy_coverage,
            "observed_failure_classes": observed_failure_categories,
            "manual_review_samples": taxonomy_samples,
        },
    )
    shortcut_gate = _gate(
        "supported" if gate7 else "not_supported",
        "The shortcut audit must find no one-file solution, exact-answer corpus hit, or "
        "invalid evidence alternative.",
        {
            "single_file_sufficient_count": shortcuts,
            "exact_answer_hit_count": exact_answer_hits,
            "invalid_alternative_count": invalid_alternatives,
            "public_question_count": question_count,
        },
    )
    if indirect_only:
        gates = {
            "1_retrieval_changes_with_N": retrieval_gate,
            "2_levels_not_degenerate": levels_gate,
            "3_error_taxonomy_is_closed": error_gate,
            "4_no_single_file_shortcuts": shortcut_gate,
        }
    else:
        gates = {
            "1_indirect_harder_than_direct": _gate(
                "supported" if models and all(gate1_support) else "not_supported",
                "Requires paired Direct-Indirect Joint Accuracy gap >= 0.05 and bootstrap CI "
                "above zero for each model.",
                {
                    model: per_model[model]["direct_minus_indirect_at_full_corpus"]
                    for model in models
                },
            ),
            "2_direct_insensitive_to_N": _gate(
                "supported" if models and all(gate2_support) else "not_supported",
                "Requires the Direct Joint Accuracy range across N to be <= 0.05 within every "
                "level for each model.",
                {model: per_model[model]["direct_range_by_level"] for model in models},
            ),
            "3_indirect_retrieval_changes_with_N": retrieval_gate,
            "4_oracle_better_than_full_retrieval": _gate(
                "supported" if models and all(gate4_support) else "not_supported",
                "Requires paired Oracle-Full Joint Accuracy gap >= 0.05 and bootstrap CI above "
                "zero for each model; Oracle retrieval cost is not compared.",
                {model: per_model[model]["oracle_minus_full_retrieval"] for model in models},
            ),
            "5_levels_not_degenerate": levels_gate,
            "6_error_taxonomy_is_closed": error_gate,
            "7_no_single_file_shortcuts": shortcut_gate,
        }
    expand = all(gate["status"] == "supported" for gate in gates.values())
    report = {
        "decision": "expand_to_300" if expand else "revise_or_collect_more_pilot_evidence",
        "primary_metric": "Macro Joint Accuracy",
        "thresholds_are_pilot_decision_rules_not_claims": True,
        "manifest_path": str(resolved_manifest.resolve()),
        "source_snapshot_sha256": snapshots.bindings(),
        "run_audit": audit,
        "infrastructure": {"failure_count": 0, "failures": []},
        "matrix_design": matrix["design"],
        "curve_dimensions": list(curve_keys),
        "curves": curves,
        "complexity_profiles": complexity_profiles,
        "complexity_interpretation": (
            "Gold complexity descriptors define the joint L1/L2/L3 task strata in this pilot. "
            "They do not vary independently within a task-level/anchor cell, so this report does "
            "not estimate a separate causal effect of operator count or critical-path depth."
        ),
        "models": per_model,
        "gates": gates,
    }
    if not indirect_only:
        report.update(
            {
                "replicate_curve_dimensions": [
                    "model",
                    "anchor_type",
                    "task_level",
                    "N",
                    "scope_seed",
                    "replicate",
                ],
                "replicate_curves": replicate_curves,
                "oracle_accuracy_curves": oracle_accuracy_curves,
            }
        )
    changed_sources = snapshots.changed_sources()
    if changed_sources:
        audit["complete"] = False
        audit["matrix_errors"].extend(changed_sources)
        return _not_evaluable(
            output,
            reason="One or more source artifacts changed while the pilot report was generated.",
            run_audit=audit,
            manifest_path=resolved_manifest,
        )
    write_json(output, report)
    return report