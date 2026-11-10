from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Set
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_RE = re.compile(r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_manifest_sha256(
    root: str | Path,
    *,
    suffixes: Set[str] | None = None,
    include_mtime: bool = False,
) -> tuple[str, int]:
    """Hash a deterministic file manifest without loading large file bodies.

    Path and size make the digest portable across copies. An index may also
    request nanosecond mtimes to detect same-size local edits before querying.
    """

    source = Path(root).resolve()
    paths = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and (suffixes is None or path.suffix.lower() in suffixes)
        ),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        record = [path.relative_to(source).as_posix(), str(stat.st_size)]
        if include_mtime:
            record.append(str(stat.st_mtime_ns))
        digest.update("\x1f".join(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def directory_content_sha256(
    root: str | Path,
    *,
    suffixes: Set[str] | None = None,
) -> tuple[str, int]:
    """Hash relative paths and complete file contents in deterministic order."""

    source = Path(root).resolve()
    paths = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and (suffixes is None or path.suffix.lower() in suffixes)
        ),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), len(paths)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, text)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in rows)
    _atomic_write(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)


def format_timestamp(value: datetime) -> str:
    return value.strftime(TIMESTAMP_FORMAT)


def next_full_hour(value: datetime) -> datetime:
    floored = value.replace(minute=0, second=0, microsecond=0)
    return floored + timedelta(hours=1)


def timestamps_in_text(text: str) -> list[datetime]:
    return [parse_timestamp(value) for value in TIMESTAMP_RE.findall(text)]


def finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip().replace("%", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def round_number(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def safe_join(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes corpus root: {relative}") from exc
    return candidate