from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import canonical_json, file_sha256, read_jsonl, stable_id

ATTEMPT_IDENTITY_FIELDS = (
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
    "max_job_seconds",
    "api_format",
)

JsonObjectLoader = Callable[[Path], dict[str, Any]]
JsonlLoader = Callable[[Path], list[dict[str, Any]]]
DigestLoader = Callable[[Path], str]


@dataclass(frozen=True)
class ManifestBinding:
    """Validated identity binding between a manifest, its summary, and artifacts."""

    manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]
    jobs: list[dict[str, Any]]
    experiment_id: Any
    artifact_hashes: dict[str, str]
    jobs_by_run_id: dict[str, dict[str, Any]] | None

    def provenance(self, artifact_paths: Mapping[str, Path]) -> dict[str, Any]:
        values: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.artifact_hashes["manifest_sha256"],
            "manifest_summary_path": str(self.summary_path),
            "manifest_summary_sha256": self.artifact_hashes["manifest_summary_sha256"],
        }
        for name, path in artifact_paths.items():
            values[f"{name}_path"] = str(path)
            values[f"{name}_sha256"] = self.artifact_hashes[f"{name}_sha256"]
        return values


def _default_json_object_loader(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Run manifest binding summary must be an object: {path}")
    return value


def load_manifest_binding(
    manifest_path: str | Path,
    *,
    artifact_paths: Mapping[str, str | Path],
    extra_hashes: Mapping[str, str] | None = None,
    required_path_identities: Sequence[str] = (),
    manifest_rows: list[dict[str, Any]] | None = None,
    json_object_loader: JsonObjectLoader = _default_json_object_loader,
    jsonl_loader: JsonlLoader = read_jsonl,
    digest_loader: DigestLoader = file_sha256,
    require_manifest_file: bool = True,
    require_valid_experiment_id: bool = False,
    index_jobs: bool = False,
) -> ManifestBinding:
    """Load and validate the common immutable run-manifest bindings.

    Callers supply any domain-specific hashes (for example raw-corpus and
    semantic-config digests) through ``extra_hashes``. This function owns the
    shared summary, artifact, job-count, experiment, and optional run-id index
    checks.
    """

    manifest_source = Path(manifest_path).resolve()
    if require_manifest_file and not manifest_source.is_file():
        raise ValueError(f"Run manifest is missing: {manifest_source}")
    summary_source = manifest_source.with_suffix(".summary.json")
    if not summary_source.is_file():
        raise ValueError(f"Run manifest binding summary is missing: {summary_source}")

    summary = json_object_loader(summary_source)
    if not isinstance(summary, dict):
        raise ValueError(f"Run manifest binding summary must be an object: {summary_source}")
    jobs = manifest_rows if manifest_rows is not None else jsonl_loader(manifest_source)

    resolved_artifacts = {name: Path(path).resolve() for name, path in artifact_paths.items()}
    artifact_hashes = {
        "manifest_sha256": digest_loader(manifest_source),
        "manifest_summary_sha256": digest_loader(summary_source),
        **{f"{name}_sha256": digest_loader(path) for name, path in resolved_artifacts.items()},
        **dict(extra_hashes or {}),
    }
    summary_hashes = {
        name: digest
        for name, digest in artifact_hashes.items()
        if name != "manifest_summary_sha256"
    }
    hash_mismatches = sorted(
        name for name, actual in summary_hashes.items() if summary.get(name) != actual
    )
    if hash_mismatches:
        raise ValueError(f"Run manifest artifact bindings disagree for: {hash_mismatches}")

    identity_paths = {"manifest": manifest_source, **resolved_artifacts}
    path_mismatches = sorted(
        name
        for name in required_path_identities
        if (
            name not in identity_paths
            or not isinstance(summary.get(f"{name}_path"), str)
            or Path(summary[f"{name}_path"]).resolve() != identity_paths[name]
        )
    )
    if path_mismatches:
        raise ValueError(f"Run manifest artifact identities disagree for: {path_mismatches}")

    if summary.get("job_count") != len(jobs):
        raise ValueError("Run manifest summary job_count disagrees with the manifest")

    experiment_id = summary.get("experiment_id")
    if require_valid_experiment_id and (not isinstance(experiment_id, str) or not experiment_id):
        raise ValueError("Run manifest summary has no valid experiment_id")
    manifest_experiments = {row.get("experiment_id") for row in jobs}
    if manifest_experiments != {experiment_id}:
        raise ValueError("Run manifest experiment_id disagrees with its binding summary")

    jobs_by_run_id: dict[str, dict[str, Any]] | None = None
    if index_jobs:
        jobs_by_run_id = {}
        for index, job in enumerate(jobs, start=1):
            run_id = job.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"Run manifest row {index} has no valid run_id")
            if run_id in jobs_by_run_id:
                raise ValueError(f"Duplicate run manifest run_id: {run_id}")
            jobs_by_run_id[run_id] = job

    return ManifestBinding(
        manifest_path=manifest_source,
        summary_path=summary_source,
        summary=summary,
        jobs=jobs,
        experiment_id=experiment_id,
        artifact_hashes=artifact_hashes,
        jobs_by_run_id=jobs_by_run_id,
    )


def validate_attempt_identities(runs: Sequence[dict[str, Any]], binding: ManifestBinding) -> None:
    """Ensure every result attempt belongs to, and agrees with, its manifest job."""

    if binding.jobs_by_run_id is None:
        raise ValueError("Manifest jobs must be indexed before validating result attempts")
    for index, run in enumerate(runs, start=1):
        run_id = run.get("run_id")
        expected = binding.jobs_by_run_id.get(run_id) if isinstance(run_id, str) else None
        if expected is None:
            raise ValueError(f"Result attempt at line {index} is not present in the run manifest")
        mismatches = [
            field
            for field in ATTEMPT_IDENTITY_FIELDS
            if run.get(field, "main" if field == "replicate" else None)
            != expected.get(field, "main" if field == "replicate" else None)
        ]
        if mismatches:
            raise ValueError(
                f"Result attempt for run_id {run_id} disagrees with the run manifest on: "
                f"{sorted(mismatches)}"
            )


def validate_reproducible_run_ids(
    binding: ManifestBinding, question_rows: Sequence[dict[str, Any]]
) -> None:
    """Recompute every scheduled run ID from its bound public question and job."""

    questions = {row.get("qa_id"): row for row in question_rows}
    if len(questions) != len(question_rows) or None in questions:
        raise ValueError("Cannot verify run IDs against duplicate or missing question IDs")
    for job in binding.jobs:
        question = questions.get(job.get("qa_id"))
        if question is None:
            raise ValueError(f"Cannot bind run to unknown question: {job.get('qa_id')!r}")
        identity = [
            binding.experiment_id,
            canonical_json(question),
            job.get("model"),
            job.get("reasoning_effort"),
            job.get("condition"),
            job.get("N"),
            job.get("scope_seed"),
            job.get("replicate", "main"),
        ]
        # Historical manifests predate the wall-clock budget. Preserve their
        # original reproducible IDs while binding all new runs to the limit.
        if "max_job_seconds" in job:
            identity.append(job["max_job_seconds"])
        if job.get("api_format", "responses") != "responses":
            identity.append(job["api_format"])
        expected_run_id = stable_id(*identity, length=24)
        if job.get("run_id") != expected_run_id:
            raise ValueError(f"Run ID is not reproducible for job {job.get('run_id')!r}")