from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .task_specs import AnchorType, TaskLevel, answer_schema, normalize_task_family, task_spec


@dataclass(frozen=True)
class Device:
    uid: str
    model_key: str
    prefix: str
    name: str
    csv_path: str
    annotation_path: str | None
    archive_path: str
    check_reports: tuple[str, ...]
    fault_reports: tuple[str, ...]
    metadata: dict[str, str] = field(compare=False, hash=False)
    csv_columns: tuple[str, ...] = field(compare=False, hash=False)


@dataclass(frozen=True)
class Metric:
    field: str
    name: str
    unit: str
    data_type: str
    metric_type: str
    normal_mean: float | None
    normal_std: float | None
    normal_min: float | None
    normal_max: float | None


@dataclass(frozen=True)
class Rule:
    model_key: str
    rule_id: str
    expression: str
    field: str
    metric_name: str
    unit: str
    operator: str
    consecutive_hours: int
    parameter_name: str
    parameter_value: float
    parameter_description: str
    rule_path: str
    metric_catalog_path: str
    derived_metrics_path: str | None
    description: str


@dataclass
class CorpusInventory:
    raw_root: Path
    devices: dict[str, Device]
    metrics_by_model: dict[str, dict[str, Metric]]
    rules_by_model: dict[str, dict[str, Rule]]
    artifacts: dict[str, dict[str, Any]]

    def device(self, uid: str) -> Device:
        try:
            return self.devices[uid]
        except KeyError as exc:
            raise KeyError(f"Unknown device_uid: {uid}") from exc


@dataclass(frozen=True)
class DocumentAnchor:
    path: str
    device_uid: str
    kind: str
    start: datetime
    end: datetime
    event_id: str
    rule_id: str | None = None
    metric_field: str | None = None
    metric_name: str | None = None
    report_number: str | None = None
    metadata: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Locator:
    fields: tuple[tuple[str, str], ...]
    ablation_confusers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [{"field": key, "value": value} for key, value in self.fields],
            "ablation_confusers": list(self.ablation_confusers),
        }


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceSource(StrictModel):
    path: str
    selector: dict[str, Any] = Field(default_factory=dict)


class EvidenceAtom(StrictModel):
    atom_id: str
    fact: str
    modality: Literal["document", "metadata", "rule", "timeseries", "derived_metric"]
    acceptable_sources: list[EvidenceSource]


class Complexity(StrictModel):
    target_device_count: int = Field(ge=1)
    evidence_atom_count: int = Field(ge=1)
    modality_count: int = Field(ge=1)
    operator_count: int = Field(ge=1)
    critical_path_depth: int = Field(ge=1)


class PublicQuestion(StrictModel):
    schema_version: str
    qa_id: str
    pair_id: str
    task_level: TaskLevel
    task_family: str | None = None
    # Read-only compatibility for old Guided fixtures. New serializations omit it.
    anchor_type: AnchorType | None = Field(default=None, exclude=True)
    question: str
    answer_schema: dict[str, Any]
    scope_key: str


class QuestionTextParts(StrictModel):
    question: str = Field(min_length=1)


class PolishProvenance(StrictModel):
    policy_version: str
    requested_model: str
    resolved_model: str
    response_id: str
    prompt_sha256: str
    attempt_count: int = Field(ge=1)


class QuestionGenerationRecord(StrictModel):
    """Private, immutable account of how one autonomous question was worded."""

    schema_version: str
    pair_id: str
    task_level: TaskLevel
    task_family: str | None = None
    style_id: str
    facts: dict[str, Any]
    facts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    draft: QuestionTextParts
    final: QuestionTextParts
    status: Literal["draft", "polished"]
    provenance: PolishProvenance | None = None
    validation: dict[str, Any] = Field(default_factory=dict)


class GoldRecord(StrictModel):
    schema_version: str
    qa_id: str
    pair_id: str
    task_level: TaskLevel
    task_family: str | None = None
    anchor_type: AnchorType
    target_device_uids: list[str]
    answer: dict[str, Any]
    tolerances: dict[str, float] = Field(default_factory=dict)
    evidence_atoms: list[EvidenceAtom]
    gold_trace: list[dict[str, Any]]
    complexity: Complexity
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def evidence_count_matches(self) -> GoldRecord:
        if self.complexity.evidence_atom_count != len(self.evidence_atoms):
            raise ValueError("complexity.evidence_atom_count does not match evidence_atoms")

        # Gold execution depends on the inventory dataclasses above, so importing
        # its trace helper at module load time would create models -> gold ->
        # models.  The local import preserves one trace implementation while
        # keeping model imports acyclic.
        from .generation.gold import trace_complexity

        operator_count, depth = trace_complexity(self.gold_trace)
        if self.complexity.operator_count != operator_count:
            raise ValueError("complexity.operator_count does not match gold_trace")
        if self.complexity.critical_path_depth != depth:
            raise ValueError("complexity.critical_path_depth does not match gold_trace")
        if self.complexity.target_device_count != len(self.target_device_uids):
            raise ValueError("complexity.target_device_count does not match targets")
        task_spec(self.task_level).validate_target_count(len(self.target_device_uids))
        family = normalize_task_family(self.task_level, self.task_family)
        if family == "multi_device_replay":
            expected = {f"D{index}" for index in range(1, len(self.target_device_uids) + 1)}
            if not set(self.answer.get("triggered_devices", [])).issubset(expected):
                raise ValueError("L3 triggered_devices contains an unknown alias")
            earliest = self.answer.get("earliest_trigger_device")
            if earliest is not None and earliest not in expected:
                raise ValueError("L3 earliest_trigger_device is unknown")
        return self


class PredictionEnvelope(StrictModel):
    answer: dict[str, Any]
    evidence: list[str] = Field(default_factory=list)


__all__ = [
    "AnchorType",
    "Complexity",
    "CorpusInventory",
    "Device",
    "DocumentAnchor",
    "EvidenceAtom",
    "EvidenceSource",
    "GoldRecord",
    "Locator",
    "Metric",
    "PolishProvenance",
    "PredictionEnvelope",
    "PublicQuestion",
    "QuestionGenerationRecord",
    "QuestionTextParts",
    "Rule",
    "StrictModel",
    "TaskLevel",
    "answer_schema",
]