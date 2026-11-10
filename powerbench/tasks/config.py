from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .task_specs import TASK_FAMILIES, TASK_LEVELS
from powerbench.paths import configs_dir, demo_raw_dir, repo_root

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = repo_root(Path(__file__))
DEFAULT_CONFIG = configs_dir() / "pilot.json"
DEFAULT_RAW_ROOT = demo_raw_dir()
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "output" / "pilot"

SUPPORTED_SCHEMA_VERSION = "0.2.0"


@dataclass(frozen=True)
class PilotConfig:
    data: dict[str, Any]
    source: Path

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def task_counts(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.data["task_counts"].items()}

    @property
    def scope_sizes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["scope_sizes"])

    @property
    def scope_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["scope_seeds"])

    @property
    def l3_device_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["l3"]["device_counts"])

    @property
    def task_family_counts(self) -> dict[str, dict[str, int]] | None:
        raw = self.data.get("task_families")
        if raw is None:
            return None
        return {
            level: {family: int(count) for family, count in raw[level].items()}
            for level in TASK_LEVELS
        }

    def section(self, name: str) -> dict[str, Any]:
        # Keep the configuration hashed into experiment IDs immutable to callers.
        return deepcopy(self.data[name])


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Config contains duplicate key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {unknown}")


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        if maximum is not None:
            qualifier = f"between {minimum} and {maximum}"
        raise ValueError(f"{label} must be an integer {qualifier}")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _integer_list(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    exact_length: int | None = None,
    sorted_values: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    values = tuple(
        _integer(item, f"{label}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    )
    if exact_length is not None and len(values) != exact_length:
        raise ValueError(f"{label} must contain exactly {exact_length} values")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")
    if sorted_values and tuple(sorted(values)) != values:
        raise ValueError(f"{label} values must be sorted in ascending order")
    return values


def _threshold_factors(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    factors = tuple(_positive_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if any(factor == 1.0 for factor in factors):
        raise ValueError(f"{label} cannot contain 1.0 because the threshold must change")
    if len(factors) != len(set(factors)):
        raise ValueError(f"{label} values must be unique")
    if tuple(sorted(factors)) != factors:
        raise ValueError(f"{label} must be sorted")
    if not any(factor < 1 for factor in factors) or not any(factor > 1 for factor in factors):
        raise ValueError(f"{label} must contain factors both below and above 1.0")
    return factors


def _validate_task_sections(
    data: dict[str, Any],
    *,
    require_stratum_divisibility: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    task_counts = data["task_counts"]
    l1 = _mapping(data["l1"], "l1")
    _exact_keys(l1, {"window_hours", "answer_round_digits"}, "l1")
    l1_windows = _integer_list(l1["window_hours"], "l1.window_hours", minimum=1)
    _integer(l1["answer_round_digits"], "l1.answer_round_digits", minimum=0)
    if require_stratum_divisibility and task_counts["L1"] % len(l1_windows):
        raise ValueError("L1 task count must be divisible by the number of window sizes")

    l2 = _mapping(data["l2"], "l2")
    _exact_keys(l2, {"window_hours", "threshold_factors"}, "l2")
    l2_windows = _integer_list(l2["window_hours"], "l2.window_hours", minimum=1)
    _threshold_factors(l2["threshold_factors"], "l2.threshold_factors")
    if require_stratum_divisibility and task_counts["L2"] % len(l2_windows):
        raise ValueError("L2 task count must be divisible by the number of window sizes")

    l3 = _mapping(data["l3"], "l3")
    _exact_keys(l3, {"device_counts", "window_hours", "threshold_factors"}, "l3")
    l3_device_counts = _integer_list(
        l3["device_counts"],
        "l3.device_counts",
        minimum=3,
        maximum=5,
        sorted_values=True,
    )
    if require_stratum_divisibility and task_counts["L3"] % len(l3_device_counts):
        raise ValueError("L3 task count must be divisible by the number of device counts")
    l3_window = _integer(l3["window_hours"], "l3.window_hours", minimum=1)
    _threshold_factors(l3["threshold_factors"], "l3.threshold_factors")
    return l1_windows, l2_windows, l3_window, l3_device_counts


def _validate_candidate_search(data: dict[str, Any]) -> None:
    candidate_search = _mapping(data["candidate_search"], "candidate_search")
    keys = {
        "locator_max_fields",
        "l1_max_reports_per_device",
        "l1_max_metrics_per_device",
        "l3_max_local_options_per_device",
        "l3_max_candidates_per_group_pattern",
    }
    _exact_keys(candidate_search, keys, "candidate_search")
    _integer(
        candidate_search["locator_max_fields"],
        "candidate_search.locator_max_fields",
        minimum=4,
    )
    for key in sorted(keys - {"locator_max_fields"}):
        _integer(candidate_search[key], f"candidate_search.{key}", minimum=1)


def _validate_retrieval(data: dict[str, Any], *, largest_window: int) -> None:
    retrieval = _mapping(data["retrieval"], "retrieval")
    positive_integer_keys = {
        "max_tool_rounds",
        "max_read_lines",
        "max_timeseries_rows",
        "max_total_read_bytes",
        "python_max_output_chars",
    }
    _exact_keys(
        retrieval,
        positive_integer_keys | {"python_timeout_seconds", "format_retries"},
        "retrieval",
    )
    for key in sorted(positive_integer_keys):
        _integer(retrieval[key], f"retrieval.{key}", minimum=1)
    if int(retrieval["max_timeseries_rows"]) < largest_window:
        raise ValueError(
            "retrieval.max_timeseries_rows must cover the largest configured task window"
        )
    timeout = _positive_number(
        retrieval["python_timeout_seconds"], "retrieval.python_timeout_seconds"
    )
    if timeout > 60:
        raise ValueError("retrieval.python_timeout_seconds must not exceed 60")
    retries = _integer(retrieval["format_retries"], "retrieval.format_retries", minimum=0)
    if retries > 5:
        raise ValueError("retrieval.format_retries must not exceed 5")


def _validate_config(data: dict[str, Any]) -> None:
    expected_top_level = {
        "schema_version",
        "seed",
        "task_counts",
        "scope_sizes",
        "scope_seeds",
        "l1",
        "l2",
        "l3",
        "candidate_search",
        "retrieval",
    }
    if "task_families" in data:
        expected_top_level.add("task_families")
    _exact_keys(data, expected_top_level, "config")
    if data["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    _integer(data["seed"], "seed", minimum=0)

    task_counts = _mapping(data["task_counts"], "task_counts")
    _exact_keys(task_counts, set(TASK_LEVELS), "task_counts")
    for level in TASK_LEVELS:
        _integer(task_counts[level], f"task_counts.{level}", minimum=1)

    if "task_families" in data:
        families = _mapping(data["task_families"], "task_families")
        _exact_keys(families, set(TASK_LEVELS), "task_families")
        for level in TASK_LEVELS:
            level_counts = _mapping(families[level], f"task_families.{level}")
            if not level_counts:
                raise ValueError(f"task_families.{level} must not be empty")
            unknown = sorted(set(level_counts).difference(TASK_FAMILIES[level]))
            if unknown:
                raise ValueError(f"task_families.{level} contains unknown families: {unknown}")
            total = sum(
                _integer(count, f"task_families.{level}.{family}", minimum=0)
                for family, count in level_counts.items()
            )
            if total != task_counts[level]:
                raise ValueError(
                    f"task_families.{level} sums to {total}, expected {task_counts[level]}"
                )

    scope_sizes = _integer_list(data["scope_sizes"], "scope_sizes", minimum=1, sorted_values=True)
    _integer_list(data["scope_seeds"], "scope_seeds")

    l1_windows, l2_windows, l3_window, l3_device_counts = _validate_task_sections(
        data,
        require_stratum_divisibility="task_families" not in data,
    )
    if scope_sizes[0] < max(l3_device_counts):
        raise ValueError("every scope size must cover the largest configured L3 device count")
    _validate_candidate_search(data)
    _validate_retrieval(
        data,
        largest_window=max(*l1_windows, *l2_windows, l3_window),
    )


def load_config(path: str | Path | None = None) -> PilotConfig:
    source = Path(path).resolve() if path else DEFAULT_CONFIG
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    root = _mapping(data, "config")
    _validate_config(root)
    return PilotConfig(data=deepcopy(root), source=source)
