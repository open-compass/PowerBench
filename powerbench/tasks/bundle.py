from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .corpus import INDEXED_SUFFIXES
from .util import (
    canonical_json,
    directory_content_sha256,
    file_sha256,
    stable_id,
    write_json,
)

BUNDLE_VERSION = "2"
REQUIRED_BUNDLE_ARTIFACTS = frozenset(
    {"questions", "scopes", "corpus_index", "tasks", "question_generation", "audit"}
)
ARTIFACT_HASH_MIRRORS = {
    "questions_sha256": "questions",
    "scopes_sha256": "scopes",
    "index_sha256": "corpus_index",
    "gold_sha256": "tasks",
    "tasks_sha256": "tasks",
    "question_generation_sha256": "question_generation",
    "audit_sha256": "audit",
}


def _computed_bundle_id(
    *,
    config_semantic_sha256: str,
    raw_content_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    return stable_id(
        BUNDLE_VERSION,
        config_semantic_sha256,
        raw_content_sha256,
        *(str(artifacts[name]["sha256"]) for name in sorted(REQUIRED_BUNDLE_ARTIFACTS)),
        length=24,
    )


@dataclass(frozen=True)
class BenchmarkBundle:
    """One authoritative binding for all immutable benchmark artifacts."""

    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.resolve()

    @property
    def bundle_id(self) -> str:
        return str(self.data["bundle_id"])

    def artifact_path(self, name: str) -> Path:
        artifacts = self.data.get("artifacts")
        record = artifacts.get(name) if isinstance(artifacts, Mapping) else None
        relative = record.get("path") if isinstance(record, Mapping) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Bundle has no artifact named {name!r}")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Bundle artifact escapes its root: {relative}") from exc
        return candidate

    def verify(self) -> None:
        artifacts = self.data.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Bundle has no artifacts")
        artifact_names = set(artifacts)
        if artifact_names != REQUIRED_BUNDLE_ARTIFACTS:
            missing = sorted(REQUIRED_BUNDLE_ARTIFACTS - artifact_names)
            unexpected = sorted(artifact_names - REQUIRED_BUNDLE_ARTIFACTS)
            raise ValueError(
                f"Bundle artifact set is invalid: missing={missing}, unexpected={unexpected}"
            )
        for name, raw in artifacts.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"Bundle artifact {name!r} is invalid")
            expected = raw.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Bundle artifact {name!r} has no SHA-256")
            path = self.artifact_path(str(name))
            if not path.is_file():
                raise ValueError(f"Bundle artifact is missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"Bundle artifact {name!r} changed: expected {expected}, found {actual}"
                )

        artifact_hashes = self.data.get("artifact_hashes")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError("Bundle has no artifact hash mirrors")
        mirror_mismatches = sorted(
            mirror
            for mirror, artifact_name in ARTIFACT_HASH_MIRRORS.items()
            if artifact_hashes.get(mirror) != artifacts[artifact_name].get("sha256")
        )
        if mirror_mismatches:
            raise ValueError(f"Bundle artifact hash mirrors disagree for: {mirror_mismatches}")

        config_semantic_sha256 = self.data.get("config_semantic_sha256")
        raw_content_sha256 = self.data.get("raw_content_sha256")
        for name, value in (
            ("config_semantic_sha256", config_semantic_sha256),
            ("raw_content_sha256", raw_content_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Bundle has no valid {name}")
        if artifact_hashes.get("config_semantic_sha256") != config_semantic_sha256:
            raise ValueError("Bundle config semantic hash mirror disagrees")
        expected_bundle_id = _computed_bundle_id(
            config_semantic_sha256=config_semantic_sha256,
            raw_content_sha256=raw_content_sha256,
            artifacts=artifacts,
        )
        if self.data.get("bundle_id") != expected_bundle_id:
            raise ValueError(
                "Bundle ID disagrees with its immutable artifact bindings: "
                f"expected {expected_bundle_id}, found {self.data.get('bundle_id')!r}"
            )

    def verify_artifact(self, name: str, path: str | Path) -> str:
        """Require an explicit caller input to be the bundle-bound artifact."""

        expected_path = self.artifact_path(name)
        actual_path = Path(path).resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"Bundle artifact {name!r} path disagrees: "
                f"expected {expected_path}, found {actual_path}"
            )
        expected = str(self.data["artifacts"][name]["sha256"])
        actual = file_sha256(actual_path)
        if actual != expected:
            raise ValueError(
                f"Bundle artifact {name!r} changed: expected {expected}, found {actual}"
            )
        return actual

    def verify_inputs(self, config: PilotConfig, raw_root: str | Path) -> None:
        """Verify that config and raw corpus still match this immutable bundle."""

        semantic = hashlib.sha256(canonical_json(config.data).encode("utf-8")).hexdigest()
        if semantic != self.data.get("config_semantic_sha256"):
            raise ValueError("Current config semantics disagree with the benchmark bundle")
        bound_hashes = self.data.get("artifact_hashes")
        expected_config = (
            bound_hashes.get("config_sha256") if isinstance(bound_hashes, Mapping) else None
        )
        if config.source.is_file() and expected_config != file_sha256(config.source):
            raise ValueError("Current config file disagrees with the benchmark bundle")
        raw_digest, _ = directory_content_sha256(raw_root, suffixes=INDEXED_SUFFIXES)
        if raw_digest != self.data.get("raw_content_sha256"):
            raise ValueError("Current raw corpus disagrees with the benchmark bundle")


def _artifact(root: Path, path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    try:
        relative = source.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Bundle artifact must be under {root}: {source}") from exc
    return {"path": relative, "sha256": file_sha256(source)}


def create_bundle(
    config: PilotConfig,
    *,
    output_root: str | Path,
    generation_report: Mapping[str, Any],
    index_path: str | Path,
) -> BenchmarkBundle:
    output = Path(output_root).resolve()
    path = output / "bundle.json"
    artifacts = {
        "questions": _artifact(output, output / "public" / "questions.jsonl"),
        "scopes": _artifact(output, output / "public" / "scopes.jsonl"),
        "corpus_index": _artifact(output, index_path),
        "tasks": _artifact(output, output / "private" / "tasks.jsonl"),
        "question_generation": _artifact(output, output / "private" / "question_generation.jsonl"),
        "audit": _artifact(output, output / "private" / "audit.json"),
    }
    config_semantic_sha256 = hashlib.sha256(canonical_json(config.data).encode("utf-8")).hexdigest()
    raw_content_sha256 = generation_report.get("raw_content_sha256")
    if not isinstance(raw_content_sha256, str) or len(raw_content_sha256) != 64:
        raise ValueError("Generation report has no valid raw_content_sha256")
    bundle_id = _computed_bundle_id(
        config_semantic_sha256=config_semantic_sha256,
        raw_content_sha256=raw_content_sha256,
        artifacts=artifacts,
    )
    # Retain the build-statistic field names consumed by the report while
    # replacing its former web of sidecar summaries with this single record.
    payload = dict(generation_report)
    payload.update(
        {
            "bundle_version": BUNDLE_VERSION,
            "bundle_id": bundle_id,
            "config": config.data,
            "config_semantic_sha256": config_semantic_sha256,
            "artifacts": artifacts,
            "artifact_hashes": {
                "config_sha256": file_sha256(config.source),
                "config_semantic_sha256": config_semantic_sha256,
                "questions_sha256": artifacts["questions"]["sha256"],
                "scopes_sha256": artifacts["scopes"]["sha256"],
                # Compatibility key for the existing run/evaluation API. In
                # schema v0.2 this digest binds the packed private task file.
                "gold_sha256": artifacts["tasks"]["sha256"],
                "tasks_sha256": artifacts["tasks"]["sha256"],
                "question_generation_sha256": artifacts["question_generation"]["sha256"],
                "audit_sha256": artifacts["audit"]["sha256"],
                "index_sha256": artifacts["corpus_index"]["sha256"],
            },
        }
    )
    write_json(path, payload)
    return BenchmarkBundle(path=path, data=payload)


def load_bundle(path: str | Path, *, verify: bool = True) -> BenchmarkBundle:
    source = Path(path).resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load benchmark bundle {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Benchmark bundle must be a JSON object")
    if value.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Unsupported benchmark bundle version: {value.get('bundle_version')!r}")
    bundle = BenchmarkBundle(path=source, data=value)
    if verify:
        bundle.verify()
    return bundle