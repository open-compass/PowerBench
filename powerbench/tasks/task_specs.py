from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

TaskLevel = Literal["L1", "L2", "L3"]
AnchorType = Literal["Direct", "Indirect"]
TASK_LEVELS: tuple[TaskLevel, ...] = ("L1", "L2", "L3")

DEFAULT_TASK_FAMILY: Mapping[TaskLevel, str] = MappingProxyType(
    {
        "L1": "post_window",
        "L2": "threshold_replay",
        "L3": "multi_device_replay",
    }
)

TASK_FAMILIES: Mapping[TaskLevel, tuple[str, ...]] = MappingProxyType(
    {
        "L1": (
            "post_window",
            "threshold_episode",
            "pre_post_comparison",
            "metadata_normalized",
            "document_routed_window",
        ),
        "L2": (
            "threshold_replay",
            "duration_replay",
            "shifted_window_replay",
            "metadata_bound_replay",
            "context_conditioned_replay",
        ),
        "L3": (
            "multi_device_replay",
            "aligned_window_comparison",
            "filter_then_compute",
            "heterogeneous_metric_comparison",
        ),
    }
)

QuestionRenderer = Callable[..., str]
SchemaBuilder = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class QuestionParts:
    """One autonomous, business-goal-oriented question."""

    question: str

    def render(self) -> str:
        return self.question.strip()

    @property
    def shared_task(self) -> str:
        """Compatibility alias for code inspecting the old split representation."""

        return self.question


def _locator_text(locator: dict[str, Any]) -> str:
    return ", ".join(
        f'{item["field"]} is "{item["value"]}"' for item in locator.get("fields", [])
    )


def _date(value: str) -> str:
    return value.split(" ", 1)[0] if value else "the report date"


def _report_number(anchor: dict[str, Any]) -> str:
    value = str(anchor.get("report_number") or "").strip()
    return f', report number "{value}"' if value else ""


def _l1_parts(task: dict[str, Any]) -> QuestionParts:
    metric = task["metric"]
    hours = task["window"]["hours"]
    digits = int(task.get("answer_round_digits", 4))
    question = (
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, within the {hours} consecutive hourly "
        f"samples strictly after (excluding) the inspection end time from the inspection "
        f"starting on {_date(task['anchor']['inspection_start'])}, what are the mean and "
        f'maximum of "{metric["name"]}", and when does the maximum first occur? How many '
        f"hours are strictly above the reading at inspection end? Round numeric results to "
        f"{digits} decimal places."
    )
    return QuestionParts(question)


def _l1_threshold_episode_parts(task: dict[str, Any]) -> QuestionParts:
    metric = task["metric"]
    hours = task["window"]["hours"]
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, locate the inspection report starting on "
        f"{_date(task['anchor']['inspection_start'])}. Using the "
        f'"{metric["name"]}" reading at inspection end as the threshold, count the total '
        f"hours strictly above this threshold, the longest consecutive run, and the first "
        f"exceedance time within the {hours} consecutive hourly samples strictly after "
        f"(excluding) inspection end."
    )


def _l1_pre_post_parts(task: dict[str, Any]) -> QuestionParts:
    metric = task["metric"]
    hours = task["window"]["hours"]
    discovery_date = _date(task["anchor"].get("discovery_time") or task["anchor"]["fault_end"])
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, locate the "
        f'"{metric["name"]}" anomaly report discovered on {discovery_date}'
        f"{_report_number(task['anchor'])}. Compute the mean over {hours} consecutive hourly "
        f"samples before and after fault-window end, excluding the end instant in both windows; "
        f"also give the post-window change rate relative to the pre-window mean, the post-window "
        f"maximum, and when it first occurs. Round numeric results to {digits} decimal places."
    )


def _l1_metadata_normalized_parts(task: dict[str, Any]) -> QuestionParts:
    metric = task["metric"]
    hours = task["window"]["hours"]
    start_date = _date(task["anchor"]["inspection_start"])
    threshold = task["utilization_threshold_percent"]
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, determine the rated uplink bandwidth from the "
        f"device archive. Within the {hours} consecutive hourly samples strictly after "
        f"(excluding) inspection end from the inspection starting on {start_date}, compute the "
        f'mean utilization of "{metric["name"]}" relative to rated bandwidth, the longest '
        f"consecutive run with utilization strictly above {threshold}%, and when that run first "
        f"occurs. Round numeric results to {digits} decimal places."
    )


def _l1_document_routed_parts(task: dict[str, Any]) -> QuestionParts:
    hours = task["window"]["hours"]
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, locate the inspection report starting on "
        f"{_date(task['anchor']['inspection_start'])}. Consider only numeric raw metrics from "
        f'the report section "Key metrics at inspection end" whose catalog entries provide both '
        f"a normal mean and standard deviation; select the metric with the largest "
        f"|inspection-end reading − normal mean| / standard deviation. For that metric, compute "
        f"the post-inspection-end mean, maximum, and first maximum time over {hours} consecutive "
        f"hourly samples strictly after (excluding) inspection end, and report the metric name, "
        f"standardized deviation at inspection end, and unit. Round numeric results to "
        f"{digits} decimal places."
    )


def _l1_question(task: dict[str, Any], _legacy_anchor: str | None = None) -> str:
    return _l1_parts(task).render()


def _l2_parts(task: dict[str, Any]) -> QuestionParts:
    rule = task["rule"]
    discovery_date = _date(
        task["anchors"]["A"]["discovery_time"] or task["anchors"]["A"]["fault_end"]
    )
    hours = task["window"]["hours"]
    threshold = rule["counterfactual_threshold_text"]
    unit = f" {rule['unit']}" if rule["unit"] else ""
    question = (
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, for the "
        f'"{rule["metric_name"]}" anomaly discovered on {discovery_date}'
        f"{_report_number(task['anchors']['A'])} and its associated judgment rule, if only the "
        f"threshold is adjusted to {threshold}{unit} while all other conditions remain unchanged, "
        f"will the rule trigger again within {hours} hours starting 24 hours after that "
        f"fault-window end? The evaluation window boundary does not reset any consecutive state "
        f"already accumulated by the rule. If it triggers, give the first trigger time and the "
        f"total trigger hours in that period."
    )
    return QuestionParts(question)


def _l2_duration_parts(task: dict[str, Any]) -> QuestionParts:
    rule = task["rule"]
    discovery_date = _date(
        task["anchors"]["A"]["discovery_time"] or task["anchors"]["A"]["fault_end"]
    )
    hours = task["window"]["hours"]
    duration = rule["counterfactual_consecutive_hours"]
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, for the "
        f'"{rule["metric_name"]}" anomaly discovered on {discovery_date}'
        f"{_report_number(task['anchors']['A'])} and its associated judgment rule, if only the "
        f"duration requirement is changed to {duration} consecutive hours while all other "
        f"parameters remain unchanged, will the rule trigger again within {hours} hours starting "
        f"24 hours after that fault-window end? The evaluation window boundary does not reset any "
        f"consecutive state already accumulated by the rule. If it triggers, give the first "
        f"trigger time and the total trigger hours in that period."
    )


def _l2_shifted_parts(task: dict[str, Any]) -> QuestionParts:
    rule = task["rule"]
    anchor = task["anchors"]["A"]
    discovery_date = _date(anchor["discovery_time"] or anchor["fault_end"])
    hours = task["window"]["hours"]
    offset = task["window"]["offset_hours"]
    threshold = rule["counterfactual_threshold_text"]
    unit = f" {rule['unit']}" if rule["unit"] else ""
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, for the "
        f'"{rule["metric_name"]}" anomaly discovered on {discovery_date}'
        f"{_report_number(anchor)} and its associated judgment rule, if the threshold is adjusted "
        f"to {threshold}{unit} while all other conditions remain unchanged, and observation "
        f"begins {offset} hours after fault-window end instead of 24 hours later, will the rule "
        f"trigger within the following {hours} hours? The evaluation window boundary does not "
        f"reset any consecutive state already accumulated by the rule. Give the first trigger "
        f"time and the total trigger hours in that period."
    )


def _l2_metadata_bound_parts(task: dict[str, Any]) -> QuestionParts:
    rule = task["rule"]
    anchor = task["anchors"]["A"]
    discovery_date = _date(anchor["discovery_time"] or anchor["fault_end"])
    hours = task["window"]["hours"]
    threshold = float(rule["counterfactual_threshold"]) * 100
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, for the uplink traffic anomaly discovered on "
        f"{discovery_date}{_report_number(anchor)} and its associated judgment rule, if rated "
        f"bandwidth is determined from the device archive uplink specification and the sustained "
        f"high-utilization threshold is adjusted to {threshold:g}% while all other conditions "
        f"remain unchanged, will the rule trigger again within {hours} hours starting 24 hours "
        f"after fault-window end? The evaluation window boundary does not reset any consecutive "
        f"state already accumulated by the rule. Give the first trigger time and the total "
        f"trigger hours in that period."
    )


def _l2_context_conditioned_parts(task: dict[str, Any]) -> QuestionParts:
    anchor = task["anchors"]["A"]
    policy = task["context_policy"]
    hours = task["window"]["hours"]
    offset = int(task["window"].get("offset_hours", 24))
    discovery_date = _date(anchor["discovery_time"] or anchor["fault_end"])
    start_text = (
        "starting at fault-window end"
        if offset == 0
        else f"starting {offset} hours after fault-window end"
    )
    return QuestionParts(
        f"For devices in the archive that simultaneously satisfy "
        f"{_locator_text(task['locators']['A'])}, for the anomaly discovered on "
        f"{discovery_date}{_report_number(anchor)} and its associated judgment rule, adjust the "
        f'original threshold using the "load level" recorded in that fault report: multiply by '
        f"{policy['factors']['high']:g} for high-load cases, by {policy['factors']['low']:g} for "
        f"low-load cases, and by {policy['factors']['normal']:g} for all other regular-load "
        f"cases. High-load includes descriptions containing {policy['high_description']}; "
        f"low-load includes descriptions containing {policy['low_description']}; if both apply, "
        f"classify as low-load. All other rule conditions remain unchanged. Re-evaluate triggering "
        f"over the {hours} hours {start_text}; the evaluation window boundary does not reset "
        f"consecutive state already accumulated. Report the load category, effective threshold "
        f"and unit, whether triggering occurs, the first trigger time, and total trigger hours."
    )


def _l2_question(task: dict[str, Any], _legacy_anchor: str | None = None) -> str:
    return _l2_parts(task).render()


def _l3_parts(task: dict[str, Any]) -> QuestionParts:
    rule = task["rule"]
    aliases = list(task["device_aliases"])
    descriptions: list[str] = []
    for alias in aliases:
        discovery_date = _date(
            task["anchors"][alias]["discovery_time"] or task["anchors"][alias]["fault_end"]
        )
        descriptions.append(
            f"{alias} ({_locator_text(task['locators'][alias])}; anomaly discovery date "
            f"{discovery_date}{_report_number(task['anchors'][alias])})"
        )
    hours = task["window"]["hours"]
    threshold = rule["counterfactual_threshold_text"]
    unit = f" {rule['unit']}" if rule["unit"] else ""
    question = (
        f"The following {len(aliases)} devices all show a "
        f'"{rule["metric_name"]}" anomaly: {"; ".join(descriptions)}. If the thresholds of the '
        f"associated judgment rules are uniformly adjusted to {threshold}{unit} while all other "
        f"conditions remain unchanged, which devices trigger again within {hours} hours starting "
        f"24 hours after each fault-window end? The evaluation window boundary does not reset "
        f"consecutive state already accumulated by the rule. Which device triggers earliest? "
        f"Compute first-trigger offsets from each device's evaluation start and give the span "
        f"between the earliest and latest first-trigger offsets."
    )
    return QuestionParts(question)


def _l3_aligned_window_parts(task: dict[str, Any]) -> QuestionParts:
    metric = task["metric"]
    aliases = list(task["device_aliases"])
    descriptions = [
        f"{alias} ({_locator_text(task['locators'][alias])}; inspection start date "
        f"{_date(task['anchors'][alias]['inspection_start'])})"
        for alias in aliases
    ]
    hours = task["window"]["hours"]
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"Compare the following {len(aliases)} devices: {'; '.join(descriptions)}. Using each "
        f'device\'s "{metric["name"]}" reading at inspection end as the baseline, compute the '
        f"change in mean over {hours} consecutive hourly samples strictly after (excluding) "
        f"inspection end relative to that baseline. Which device shows the largest increase? By "
        f"how many percentage points does that increase exceed the group median? Also report the "
        f"largest increase and the group median increase. Round numeric results to {digits} "
        f"decimal places."
    )


def _l3_filter_compute_parts(task: dict[str, Any]) -> QuestionParts:
    aliases = list(task["device_aliases"])
    descriptions = [
        f"{alias} ({_locator_text(task['locators'][alias])}; inspection start date "
        f"{_date(task['anchors'][alias]['inspection_start'])})"
        for alias in aliases
    ]
    criterion = task["filter"]
    metric = task["metric"]
    hours = task["window"]["hours"]
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"Given the candidate devices: {'; '.join(descriptions)}. From the device archive, "
        f'filter to devices whose "{criterion["field"]}" is "{criterion["value"]}"; then compute '
        f'the mean "{metric["name"]}" over {hours} consecutive hourly samples strictly after '
        f"(excluding) each device's inspection end. Return the filtered devices, identify the "
        f"highest mean, and give the difference between the highest and lowest means among filtered "
        f"devices. Round numeric results to {digits} decimal places."
    )


def _l3_heterogeneous_metric_parts(task: dict[str, Any]) -> QuestionParts:
    aliases = list(task["device_aliases"])
    descriptions = [
        f"{alias} ({_locator_text(task['locators'][alias])}; inspection start date "
        f"{_date(task['anchors'][alias]['inspection_start'])})"
        for alias in aliases
    ]
    hours = task["window"]["hours"]
    digits = int(task.get("answer_round_digits", 4))
    return QuestionParts(
        f"Compare the following {len(aliases)} devices: {'; '.join(descriptions)}. For each "
        f'device, consider only numeric raw metrics from the report section "Key metrics at '
        f'inspection end" whose catalog entries provide both a normal mean and standard '
        f"deviation; independently select the metric with the largest "
        f"|inspection-end reading − normal mean| / standard deviation. Over {hours} consecutive "
        f"hourly samples strictly after (excluding) inspection end, compute the maximum absolute "
        f"standardized deviation from normal mean for the selected metric and when it first "
        f"occurs. Report each device's selected metric and peak deviation, identify the device "
        f"with the highest peak deviation, and give the span between the highest and lowest peak "
        f"deviations in the group. Round numeric results to {digits} decimal places."
    )


def _l3_question(task: dict[str, Any], _legacy_anchor: str | None = None) -> str:
    return _l3_parts(task).render()


def _l1_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "window_mean": {"type": "number"},
            "window_max": {"type": "number"},
            "first_max_time": {"type": "string"},
            "hours_above_anchor_value": {"type": "integer"},
            "unit": {"type": "string"},
        },
        "required": [
            "window_mean",
            "window_max",
            "first_max_time",
            "hours_above_anchor_value",
            "unit",
        ],
    }


def _l2_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "triggered": {"type": "boolean"},
            "first_trigger_time": {"type": ["string", "null"]},
            "trigger_hours": {"type": "integer"},
        },
        "required": [
            "triggered",
            "first_trigger_time",
            "trigger_hours",
        ],
    }


def _l3_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "triggered_devices": {
                "type": "array",
                "items": {"type": "string", "pattern": "^D[1-5]$"},
                "uniqueItems": True,
            },
            "earliest_trigger_device": {
                "type": ["string", "null"],
                "pattern": "^D[1-5]$",
            },
            "trigger_offset_span_hours": {"type": ["integer", "null"]},
        },
        "required": [
            "triggered_devices",
            "earliest_trigger_device",
            "trigger_offset_span_hours",
        ],
    }


def _l1_threshold_episode_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "exceedance_hours": {"type": "integer"},
            "longest_exceedance_run_hours": {"type": "integer"},
            "first_exceedance_time": {"type": ["string", "null"]},
        },
        "required": [
            "exceedance_hours",
            "longest_exceedance_run_hours",
            "first_exceedance_time",
        ],
    }


def _l1_pre_post_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pre_mean": {"type": "number"},
            "post_mean": {"type": "number"},
            "change_rate_percent": {"type": "number"},
            "post_max": {"type": "number"},
            "first_post_max_time": {"type": "string"},
            "unit": {"type": "string"},
        },
        "required": [
            "pre_mean",
            "post_mean",
            "change_rate_percent",
            "post_max",
            "first_post_max_time",
            "unit",
        ],
    }


def _l1_metadata_normalized_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "capacity_mbps": {"type": "number"},
            "mean_utilization_percent": {"type": "number"},
            "longest_above_utilization_run_hours": {"type": "integer"},
            "first_above_utilization_time": {"type": ["string", "null"]},
        },
        "required": [
            "capacity_mbps",
            "mean_utilization_percent",
            "longest_above_utilization_run_hours",
            "first_above_utilization_time",
        ],
    }


def _l3_aligned_window_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "largest_increase_device": {"type": "string", "pattern": "^D[1-5]$"},
            "largest_increase_percentage_points": {"type": "number"},
            "group_median_increase_percentage_points": {"type": "number"},
            "lead_over_median_percentage_points": {"type": "number"},
        },
        "required": [
            "largest_increase_device",
            "largest_increase_percentage_points",
            "group_median_increase_percentage_points",
            "lead_over_median_percentage_points",
        ],
    }


def _l3_filter_compute_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_devices": {
                "type": "array",
                "items": {"type": "string", "pattern": "^D[1-5]$"},
                "uniqueItems": True,
            },
            "highest_mean_device": {"type": "string", "pattern": "^D[1-5]$"},
            "mean_span": {"type": "number"},
            "unit": {"type": "string"},
        },
        "required": ["selected_devices", "highest_mean_device", "mean_span", "unit"],
    }


def _l1_document_routed_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_metric": {"type": "string"},
            "anchor_deviation_z": {"type": "number"},
            "window_mean": {"type": "number"},
            "window_max": {"type": "number"},
            "first_max_time": {"type": "string"},
            "unit": {"type": "string"},
        },
        "required": [
            "selected_metric",
            "anchor_deviation_z",
            "window_mean",
            "window_max",
            "first_max_time",
            "unit",
        ],
    }


def _l2_context_conditioned_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "load_category": {"type": "string", "enum": ["high", "normal", "low"]},
            "effective_threshold": {"type": "number"},
            "unit": {"type": "string"},
            "triggered": {"type": "boolean"},
            "first_trigger_time": {"type": ["string", "null"]},
            "trigger_hours": {"type": "integer"},
        },
        "required": [
            "load_category",
            "effective_threshold",
            "unit",
            "triggered",
            "first_trigger_time",
            "trigger_hours",
        ],
    }


def _l3_heterogeneous_metric_answer() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "per_device": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "alias": {"type": "string", "pattern": "^D[1-5]$"},
                        "selected_metric": {"type": "string"},
                        "peak_deviation_z": {"type": "number"},
                        "first_peak_time": {"type": "string"},
                    },
                    "required": [
                        "alias",
                        "selected_metric",
                        "peak_deviation_z",
                        "first_peak_time",
                    ],
                },
            },
            "highest_risk_device": {"type": "string", "pattern": "^D[1-5]$"},
            "risk_score_span": {"type": "number"},
        },
        "required": ["per_device", "highest_risk_device", "risk_score_span"],
    }


_FAMILY_ANSWER_BUILDERS: Mapping[str, SchemaBuilder] = MappingProxyType(
    {
        "post_window": _l1_answer,
        "threshold_episode": _l1_threshold_episode_answer,
        "pre_post_comparison": _l1_pre_post_answer,
        "metadata_normalized": _l1_metadata_normalized_answer,
        "document_routed_window": _l1_document_routed_answer,
        "threshold_replay": _l2_answer,
        "duration_replay": _l2_answer,
        "shifted_window_replay": _l2_answer,
        "metadata_bound_replay": _l2_answer,
        "context_conditioned_replay": _l2_context_conditioned_answer,
        "multi_device_replay": _l3_answer,
        "aligned_window_comparison": _l3_aligned_window_answer,
        "filter_then_compute": _l3_filter_compute_answer,
        "heterogeneous_metric_comparison": _l3_heterogeneous_metric_answer,
    }
)

_FAMILY_QUESTION_BUILDERS: Mapping[str, Callable[[dict[str, Any]], QuestionParts]] = (
    MappingProxyType(
        {
            "post_window": _l1_parts,
            "threshold_episode": _l1_threshold_episode_parts,
            "pre_post_comparison": _l1_pre_post_parts,
            "metadata_normalized": _l1_metadata_normalized_parts,
            "document_routed_window": _l1_document_routed_parts,
            "threshold_replay": _l2_parts,
            "duration_replay": _l2_duration_parts,
            "shifted_window_replay": _l2_shifted_parts,
            "metadata_bound_replay": _l2_metadata_bound_parts,
            "context_conditioned_replay": _l2_context_conditioned_parts,
            "multi_device_replay": _l3_parts,
            "aligned_window_comparison": _l3_aligned_window_parts,
            "filter_then_compute": _l3_filter_compute_parts,
            "heterogeneous_metric_comparison": _l3_heterogeneous_metric_parts,
        }
    )
)


@dataclass(frozen=True)
class TaskSpec:
    level: TaskLevel
    min_target_devices: int
    max_target_devices: int
    render_question: QuestionRenderer
    build_answer_schema: SchemaBuilder

    def validate_target_count(self, count: int) -> None:
        if not self.min_target_devices <= count <= self.max_target_devices:
            expected = (
                str(self.min_target_devices)
                if self.min_target_devices == self.max_target_devices
                else f"{self.min_target_devices}..{self.max_target_devices}"
            )
            raise ValueError(f"{self.level} requires {expected} target devices, found {count}")

    def envelope_schema(self, family: str | None = None) -> dict[str, Any]:
        builder = (
            self.build_answer_schema if family is None else _family_builder(self.level, family)
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": builder(),
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "evidence"],
        }


TASK_REGISTRY: Mapping[TaskLevel, TaskSpec] = MappingProxyType(
    {
        "L1": TaskSpec("L1", 1, 1, _l1_question, _l1_answer),
        "L2": TaskSpec("L2", 1, 1, _l2_question, _l2_answer),
        "L3": TaskSpec("L3", 3, 5, _l3_question, _l3_answer),
    }
)

_QUESTION_PART_BUILDERS: Mapping[TaskLevel, Callable[[dict[str, Any]], QuestionParts]] = (
    MappingProxyType({"L1": _l1_parts, "L2": _l2_parts, "L3": _l3_parts})
)


def task_spec(level: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[level]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported task level: {level!r}") from exc


def normalize_task_family(level: TaskLevel | str, family: str | None) -> str:
    if level not in TASK_LEVELS:
        raise ValueError(f"Unsupported task level: {level!r}")
    value = family or DEFAULT_TASK_FAMILY[level]  # type: ignore[index]
    if value not in TASK_FAMILIES[level]:  # type: ignore[index]
        raise ValueError(f"Unsupported {level} task family: {value!r}")
    return value


def _family_builder(level: TaskLevel | str, family: str | None) -> SchemaBuilder:
    normalized = normalize_task_family(level, family)
    return _FAMILY_ANSWER_BUILDERS[normalized]


def answer_schema(level: TaskLevel, family: str | None = None) -> dict[str, Any]:
    normalized = normalize_task_family(level, family)
    return TASK_REGISTRY[level].envelope_schema(normalized)


def public_answer(
    level: TaskLevel, answer: Mapping[str, Any], family: str | None = None
) -> dict[str, Any]:
    """Project the executable internal result onto the model-facing contract."""

    schema = _family_builder(level, family)()
    fields = tuple(schema["required"])
    return {field: answer[field] for field in fields}


def question_parts(task: dict[str, Any]) -> QuestionParts:
    """Return the deterministic autonomous draft used as the polishing source."""

    level = task.get("task_level")
    family = normalize_task_family(str(level), task.get("task_family"))
    try:
        builder = _FAMILY_QUESTION_BUILDERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported task level: {level!r}") from exc
    return builder(task)


__all__ = [
    "DEFAULT_TASK_FAMILY",
    "TASK_FAMILIES",
    "TASK_LEVELS",
    "TASK_REGISTRY",
    "AnchorType",
    "QuestionParts",
    "TaskLevel",
    "TaskSpec",
    "answer_schema",
    "normalize_task_family",
    "public_answer",
    "question_parts",
    "task_spec",
]