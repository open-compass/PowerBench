from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .documents import parse_document
from .models import CorpusInventory, Device, Metric, Rule
from .util import finite_float, relative_posix

DEVICE_PREFIX_RE = re.compile(r"^(?P<prefix>\d{4})(?:_|$)")
REPORT_SUMMARY_CHARS = 65_536


def _device_prefix(path: Path) -> str | None:
    match = DEVICE_PREFIX_RE.match(path.name)
    return match.group("prefix") if match else None


def _single_file_index(paths: Iterable[Path], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(paths, key=lambda item: item.name):
        prefix = _device_prefix(path)
        if prefix is None:
            continue
        if prefix in result:
            raise ValueError(
                f"Duplicate {label} for device prefix {prefix}: {result[prefix]} and {path}"
            )
        result[prefix] = path
    return result


def _multi_file_index(paths: Iterable[Path]) -> dict[str, tuple[Path, ...]]:
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths, key=lambda item: item.name):
        prefix = _device_prefix(path)
        if prefix is not None:
            grouped[prefix].append(path)
    return {prefix: tuple(items) for prefix, items in grouped.items()}


def _assert_known_prefixes(
    csv_index: dict[str, Path],
    other_index: dict[str, Any],
    *,
    model_key: str,
    label: str,
) -> None:
    unknown = sorted(set(other_index).difference(csv_index))
    if unknown:
        raise ValueError(f"{model_key} has {label} files for unknown device prefixes: {unknown}")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _metric_rows(payload: Any, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("metrics") or payload.get("metric_catalog") or payload.get("metrics")
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported metric catalog structure: {source}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Metric catalog contains a non-object row: {source}")
    return rows


def load_metrics(path: str | Path) -> dict[str, Metric]:
    source = Path(path)
    metrics: dict[str, Metric] = {}
    for row in _metric_rows(_read_json(source), source=source):
        field = str(row.get("database_field") or row.get("variable_name") or "").strip()
        if not field:
            raise ValueError(f"Metric without a field name in {source}")
        if field in metrics:
            raise ValueError(f"Duplicate metric field {field!r} in {source}")
        normal = row.get("normal_range")
        if not isinstance(normal, dict):
            normal = {}
        metrics[field] = Metric(
            field=field,
            name=str(row.get("metric_name") or field).strip(),
            unit=str(row.get("unit") or normal.get("unit") or "").strip(),
            data_type=str(row.get("data_type") or "").strip(),
            metric_type=str(row.get("metric_type") or "").strip(),
            normal_mean=finite_float(normal.get("mean")),
            normal_std=finite_float(normal.get("std")),
            normal_min=finite_float(normal.get("min")),
            normal_max=finite_float(normal.get("max")),
        )
    return metrics


def _basic_archive_metadata(parsed: dict[str, Any]) -> dict[str, str]:
    for table in parsed.get("tables", []):
        section = str(table.get("section", ""))
        headers = list(table.get("headers", []))
        if "Basic Information" not in section or len(headers) != 2:
            continue
        left, right = headers
        metadata: dict[str, str] = {}
        for row in table.get("rows", []):
            key = str(row.get(left, "")).strip()
            value = str(row.get(right, "")).strip()
            if key:
                metadata[key] = value
        if metadata:
            return metadata
    return {
        str(key): str(value)
        for key, value in parsed.get("key_values", {}).items()
        if str(key).strip()
    }


def _csv_header_and_name(path: Path) -> tuple[tuple[str, ...], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc
        columns = tuple(value.strip() for value in raw_header)
        if not columns or any(not value for value in columns):
            raise ValueError(f"CSV has an invalid header: {path}")
        try:
            first_row = next(reader)
        except StopIteration:
            first_row = []

    name = ""
    if "device_name" in columns:
        index = columns.index("device_name")
        if index < len(first_row):
            name = first_row[index].strip()
    return columns, name


def _relative_paths(paths: Iterable[Path], raw_root: Path) -> tuple[str, ...]:
    return tuple(relative_posix(path, raw_root) for path in paths)


def _artifact(
    path: Path,
    raw_root: Path,
    *,
    kind: str,
    model_key: str,
    device_uid: str | None = None,
    shortcut_risk: bool = False,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relative = relative_posix(path, raw_root)
    result: dict[str, Any] = {
        "artifact_id": relative,
        "path": relative,
        "kind": kind,
        "model_key": model_key,
        "device_uid": device_uid,
        "public": True,
        "shortcut_risk": shortcut_risk,
        "size_bytes": path.stat().st_size,
    }
    if summary:
        result.update(summary)
    return result


def _document_artifact(
    path: Path,
    raw_root: Path,
    *,
    kind: str,
    model_key: str,
    device_uid: str,
) -> dict[str, Any]:
    parsed = parse_document(path, max_chars=REPORT_SUMMARY_CHARS, include_tables=False)
    summary = {
        "title": parsed["title"],
        "rule_ids": parsed["rule_ids"],
        "inspection_windows": parsed["inspection_windows"],
        "fault_windows": parsed["fault_windows"],
    }
    return _artifact(
        path,
        raw_root,
        kind=kind,
        model_key=model_key,
        device_uid=device_uid,
        shortcut_risk=True,
        summary=summary,
    )


def _add_artifact(artifacts: dict[str, dict[str, Any]], value: dict[str, Any]) -> None:
    path = str(value["path"])
    if path in artifacts:
        raise ValueError(f"Duplicate corpus artifact: {path}")
    artifacts[path] = value


def build_inventory(raw_root: str | Path) -> CorpusInventory:
    """Index the frozen raw corpus without reading full time-series bodies.

    Device linkage is based exclusively on the four-digit prefix found in the
    actual filenames. Human-readable device names are descriptive fields and are
    never used to construct or associate paths.
    """

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {root}")

    devices: dict[str, Device] = {}
    metrics_by_model: dict[str, dict[str, Metric]] = {}
    rules_by_model: dict[str, dict[str, Rule]] = {}
    artifacts: dict[str, dict[str, Any]] = {}

    model_directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
    )
    for model_directory in model_directories:
        model_key = model_directory.name
        per_device = model_directory / "per_device"
        if not per_device.is_dir():
            continue

        csv_index = _single_file_index(per_device.glob("*.csv"), label="CSV")
        annotation_index = _single_file_index(per_device.glob("*.json"), label="annotation")
        archive_index = _single_file_index(
            (model_directory / "device_archive").glob("*.md")
            if (model_directory / "device_archive").is_dir()
            else (),
            label="device archive",
        )
        check_index = _multi_file_index(
            (model_directory / "check_report").glob("*.md")
            if (model_directory / "check_report").is_dir()
            else ()
        )
        fault_index = _multi_file_index(
            (model_directory / "fault_report").glob("*.md")
            if (model_directory / "fault_report").is_dir()
            else ()
        )

        for label, index in (
            ("annotation", annotation_index),
            ("device archive", archive_index),
            ("check report", check_index),
            ("fault report", fault_index),
        ):
            _assert_known_prefixes(csv_index, index, model_key=model_key, label=label)

        metric_catalog = model_directory / "metric_catalog.json"
        rule_catalog = model_directory / "rule_metric_stage2.json"
        derived_metrics = model_directory / "derived_metrics.json"
        if not metric_catalog.is_file():
            raise FileNotFoundError(f"Missing metric catalog: {metric_catalog}")
        if not rule_catalog.is_file():
            raise FileNotFoundError(f"Missing stage-2 rule catalog: {rule_catalog}")

        metrics_by_model[model_key] = load_metrics(metric_catalog)
        # Rule semantics and executable parsing deliberately belong to rules.py.
        rules_by_model[model_key] = {}
        _add_artifact(
            artifacts,
            _artifact(metric_catalog, root, kind="metric_catalog", model_key=model_key),
        )
        _add_artifact(
            artifacts,
            _artifact(rule_catalog, root, kind="rule_catalog", model_key=model_key),
        )
        if derived_metrics.is_file():
            _add_artifact(
                artifacts,
                _artifact(derived_metrics, root, kind="derived_metrics", model_key=model_key),
            )

        for prefix, csv_path in csv_index.items():
            uid = f"{model_key}/{prefix}"
            if uid in devices:
                raise ValueError(f"Duplicate device UID: {uid}")

            archive_path = archive_index.get(prefix)
            archive_parsed = (
                parse_document(archive_path, max_chars=REPORT_SUMMARY_CHARS)
                if archive_path is not None
                else None
            )
            metadata = _basic_archive_metadata(archive_parsed) if archive_parsed else {}
            csv_columns, csv_name = _csv_header_and_name(csv_path)
            name = (
                csv_name
                or metadata.get("Installation location")
                or metadata.get("Device tag")
                or metadata.get("Device name")
                or uid
            )
            annotation_path = annotation_index.get(prefix)
            checks = check_index.get(prefix, ())
            faults = fault_index.get(prefix, ())

            device = Device(
                uid=uid,
                model_key=model_key,
                prefix=prefix,
                name=name,
                csv_path=relative_posix(csv_path, root),
                annotation_path=(
                    relative_posix(annotation_path, root) if annotation_path is not None else None
                ),
                archive_path=(
                    relative_posix(archive_path, root) if archive_path is not None else ""
                ),
                check_reports=_relative_paths(checks, root),
                fault_reports=_relative_paths(faults, root),
                metadata=metadata,
                csv_columns=csv_columns,
            )
            devices[uid] = device

            _add_artifact(
                artifacts,
                _artifact(
                    csv_path,
                    root,
                    kind="timeseries",
                    model_key=model_key,
                    device_uid=uid,
                    summary={"csv_columns": list(csv_columns)},
                ),
            )
            if annotation_path is not None:
                _add_artifact(
                    artifacts,
                    _artifact(
                        annotation_path,
                        root,
                        kind="annotation",
                        model_key=model_key,
                        device_uid=uid,
                        shortcut_risk=True,
                    ),
                )
            if archive_path is not None:
                _add_artifact(
                    artifacts,
                    _artifact(
                        archive_path,
                        root,
                        kind="device_archive",
                        model_key=model_key,
                        device_uid=uid,
                        shortcut_risk=True,
                        summary={
                            "title": archive_parsed["title"] if archive_parsed else "",
                            "metadata": metadata,
                            "rule_ids": archive_parsed["rule_ids"] if archive_parsed else [],
                        },
                    ),
                )
            for report_path in checks:
                _add_artifact(
                    artifacts,
                    _document_artifact(
                        report_path,
                        root,
                        kind="check_report",
                        model_key=model_key,
                        device_uid=uid,
                    ),
                )
            for report_path in faults:
                _add_artifact(
                    artifacts,
                    _document_artifact(
                        report_path,
                        root,
                        kind="fault_report",
                        model_key=model_key,
                        device_uid=uid,
                    ),
                )

    return CorpusInventory(
        raw_root=root,
        devices=devices,
        metrics_by_model=metrics_by_model,
        rules_by_model=rules_by_model,
        artifacts=artifacts,
    )