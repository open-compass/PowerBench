from __future__ import annotations

import csv
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .util import parse_timestamp

ONE_HOUR = timedelta(hours=1)


class TimeSeriesError(ValueError):
    """Base class for deterministic time-series validation errors."""


class InvalidWindowError(TimeSeriesError):
    """Raised when a requested evaluation window is not well formed."""


class MissingColumnError(TimeSeriesError):
    """Raised when a required CSV column is absent."""


class InvalidNumericValueError(TimeSeriesError):
    """Raised when a required metric value is not a finite decimal number."""


class NonMonotonicTimeError(TimeSeriesError):
    """Raised when CSV timestamps are duplicated or not strictly increasing."""


class IncompleteHourlySeriesError(TimeSeriesError):
    """Raised when an evaluation window or its lookback is not hourly-complete."""


@dataclass(frozen=True, order=True)
class NumericPoint:
    timestamp: datetime
    value: Decimal


@dataclass(frozen=True)
class NumericWindow:
    """A numeric CSV slice plus the rows needed for temporal lookback.

    ``start`` is inclusive and ``end`` is exclusive. ``points`` may therefore
    contain rows before ``start``, but never rows at or after ``end``.
    """

    field: str
    start: datetime
    end: datetime
    lookback_hours: int
    points: tuple[NumericPoint, ...]
    source_path: str | None = None

    @property
    def output_points(self) -> tuple[NumericPoint, ...]:
        return tuple(point for point in self.points if self.start <= point.timestamp < self.end)

    @property
    def lookback_points(self) -> tuple[NumericPoint, ...]:
        return tuple(point for point in self.points if point.timestamp < self.start)

    def values_by_timestamp(self) -> dict[datetime, Decimal]:
        return {point.timestamp: point.value for point in self.points}


def decimal_value(value: object, *, context: str = "value") -> Decimal:
    """Convert a CSV-like scalar to a finite ``Decimal`` without float noise."""

    text = str(value).strip()
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise InvalidNumericValueError(f"Invalid decimal {context}: {value!r}") from exc
    if not number.is_finite():
        raise InvalidNumericValueError(f"Non-finite decimal {context}: {value!r}")
    return number


def window_hours(start: datetime, end: datetime) -> int:
    """Return the integer number of hours in a left-closed/right-open window."""

    if end <= start:
        raise InvalidWindowError("Window end must be later than window start")
    seconds = (end - start).total_seconds()
    if seconds % 3600:
        raise InvalidWindowError("Window duration must be a whole number of hours")
    return int(seconds // 3600)


def read_numeric_window(
    csv_path: str | Path,
    *,
    field: str,
    start: datetime,
    end: datetime,
    lookback_hours: int = 0,
    timestamp_field: str = "timestamp",
) -> NumericWindow:
    """Read one numeric field for ``[start, end)`` and its hourly lookback.

    The complete CSV is checked for strictly increasing timestamps. Values are
    parsed only for rows in the requested slice. Completeness of the hourly
    grid is checked separately by :func:`validate_complete_hourly_window` so a
    caller can load a slice before choosing a temporal rule.
    """

    window_hours(start, end)
    if isinstance(lookback_hours, bool) or not isinstance(lookback_hours, int):
        raise InvalidWindowError("lookback_hours must be an integer")
    if lookback_hours < 0:
        raise InvalidWindowError("lookback_hours must be non-negative")
    if not field or not field.strip():
        raise MissingColumnError("Metric field must be non-empty")

    source = Path(csv_path)
    read_start = start - timedelta(hours=lookback_hours)
    points: list[NumericPoint] = []
    previous_timestamp: datetime | None = None

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in (timestamp_field, field) if name not in columns]
        if missing:
            raise MissingColumnError(f"Missing CSV column(s) {missing} in {source}")

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_timestamp(row[timestamp_field])
            except (TypeError, ValueError) as exc:
                raise TimeSeriesError(
                    f"Invalid timestamp at {source}:{line_number}: {row.get(timestamp_field)!r}"
                ) from exc

            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise NonMonotonicTimeError(
                    f"Timestamps must be strictly increasing at {source}:{line_number}"
                )
            previous_timestamp = timestamp

            if read_start <= timestamp < end:
                value = decimal_value(
                    row[field], context=f"{source}:{line_number} column {field!r}"
                )
                points.append(NumericPoint(timestamp=timestamp, value=value))

    return NumericWindow(
        field=field,
        start=start,
        end=end,
        lookback_hours=lookback_hours,
        points=tuple(points),
        source_path=str(source),
    )


def read_numeric_series(
    csv_path: str | Path,
    *,
    field: str,
    timestamp_field: str = "timestamp",
) -> tuple[NumericPoint, ...]:
    """Read and validate one complete numeric field for repeated windowing.

    Generators commonly evaluate several report-relative windows from the same
    device and metric. Parsing the CSV once keeps that operation deterministic
    without repeatedly paying for timestamp and Decimal conversion.
    """

    source = Path(csv_path)
    points: list[NumericPoint] = []
    previous_timestamp: datetime | None = None
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in (timestamp_field, field) if name not in columns]
        if missing:
            raise MissingColumnError(f"Missing CSV column(s) {missing} in {source}")
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_timestamp(row[timestamp_field])
            except (TypeError, ValueError) as exc:
                raise TimeSeriesError(
                    f"Invalid timestamp at {source}:{line_number}: {row.get(timestamp_field)!r}"
                ) from exc
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise NonMonotonicTimeError(
                    f"Timestamps must be strictly increasing at {source}:{line_number}"
                )
            previous_timestamp = timestamp
            points.append(
                NumericPoint(
                    timestamp=timestamp,
                    value=decimal_value(
                        row[field], context=f"{source}:{line_number} column {field!r}"
                    ),
                )
            )
    return tuple(points)


def numeric_window(
    points: Iterable[NumericPoint],
    *,
    field: str,
    start: datetime,
    end: datetime,
    lookback_hours: int = 0,
) -> NumericWindow:
    """Construct an in-memory window, primarily for deterministic composition."""

    window_hours(start, end)
    if isinstance(lookback_hours, bool) or not isinstance(lookback_hours, int):
        raise InvalidWindowError("lookback_hours must be an integer")
    if lookback_hours < 0:
        raise InvalidWindowError("lookback_hours must be non-negative")
    read_start = start - timedelta(hours=lookback_hours)
    ordered = tuple(
        sorted(
            (point for point in points if read_start <= point.timestamp < end),
            key=lambda point: point.timestamp,
        )
    )
    if len({point.timestamp for point in ordered}) != len(ordered):
        raise NonMonotonicTimeError("Duplicate timestamps in numeric window")
    return NumericWindow(
        field=field,
        start=start,
        end=end,
        lookback_hours=lookback_hours,
        points=ordered,
    )


def validate_complete_hourly_window(
    window: NumericWindow,
    *,
    required_lookback_hours: int = 0,
) -> tuple[NumericPoint, ...]:
    """Validate the hourly target grid and exact preceding lookback rows.

    The first target row must equal ``start`` and every following row must lie
    on the exact hourly grid. The returned tuple is the target window only;
    lookback rows remain accessible on ``window``.
    """

    if isinstance(required_lookback_hours, bool) or not isinstance(required_lookback_hours, int):
        raise InvalidWindowError("required_lookback_hours must be an integer")
    if required_lookback_hours < 0:
        raise InvalidWindowError("required_lookback_hours must be non-negative")
    if window.lookback_hours < required_lookback_hours:
        raise IncompleteHourlySeriesError(
            f"Loaded lookback ({window.lookback_hours}h) is shorter than required "
            f"lookback ({required_lookback_hours}h)"
        )

    targets = window.output_points
    timestamps = [point.timestamp for point in window.points]
    if len(set(timestamps)) != len(timestamps):
        raise NonMonotonicTimeError("Duplicate timestamps in numeric window")
    expected_count = window_hours(window.start, window.end)
    if len(targets) != expected_count:
        raise IncompleteHourlySeriesError(
            f"Expected {expected_count} hourly target rows in [start,end), found {len(targets)}"
        )
    if not targets:
        raise IncompleteHourlySeriesError("Evaluation window contains no target rows")

    if targets[0].timestamp != window.start:
        raise IncompleteHourlySeriesError(
            f"First target timestamp must equal window start {window.start}, "
            f"found {targets[0].timestamp}"
        )

    for previous, current in itertools.pairwise(targets):
        if current.timestamp - previous.timestamp != ONE_HOUR:
            raise IncompleteHourlySeriesError(
                f"Non-hourly target gap between {previous.timestamp} and {current.timestamp}"
            )

    first = targets[0].timestamp
    values = window.values_by_timestamp()
    for offset in range(1, required_lookback_hours + 1):
        required = first - timedelta(hours=offset)
        if required not in values:
            raise IncompleteHourlySeriesError(f"Missing required lookback row at {required}")

    return targets