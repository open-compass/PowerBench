from __future__ import annotations

import copy
import csv
import io
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .inventory import build_inventory
from .models import CorpusInventory
from .util import (
    directory_manifest_sha256,
    format_timestamp,
    parse_timestamp,
    relative_posix,
    safe_join,
)

INDEXED_SUFFIXES = frozenset({".md", ".json", ".csv", ".xlsx"})
TEXT_SUFFIXES = {".md", ".json"}
BINARY_SUFFIXES = {".xlsx"}
DEVICE_SCOPED_DIRECTORIES = {"per_device", "device_archive", "check_report", "fault_report"}
FORBIDDEN_COMPONENTS = {"output", "private", "gen_qa"}
DEVICE_PREFIX_RE = re.compile(r"^(\d{4})(?:_|$)")
RULE_ID_RE = re.compile(r"(?<![A-Za-z0-9])K\s*0*(\d+)(?!\d)", re.IGNORECASE)
CORPUS_INDEX_VERSION = "0.1.0"


class CorpusAccessError(PermissionError):
    """Raised when a tool call attempts to leave its assigned corpus scope."""


class CorpusQueryError(ValueError):
    """Raised for malformed corpus tool arguments."""


class CorpusIndex:
    """A read-only-at-query-time FTS5 representation of the raw corpus.

    Markdown and JSON are indexed verbatim. CSV bodies are intentionally not
    indexed; only their original relative paths and headers are searchable.
    XLSX files are path-only binary artifacts.
    """

    def __init__(
        self,
        inventory: CorpusInventory,
        database_path: str | Path | None = None,
        *,
        rebuild: bool = True,
    ) -> None:
        self.inventory = inventory
        self.raw_root = inventory.raw_root.resolve()
        self.database_path = ":memory:" if database_path is None else str(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if rebuild:
            self._build()
        else:
            self._validate_existing()

    def _build(self) -> None:
        with self.connection:
            self.connection.execute("DROP TABLE IF EXISTS corpus_fts")
            self.connection.execute("DROP TABLE IF EXISTS corpus_files")
            self.connection.execute("DROP TABLE IF EXISTS corpus_meta")
            self.connection.execute(
                "CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute(
                """
                CREATE TABLE corpus_files (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    model_key TEXT NOT NULL,
                    device_uid TEXT,
                    kind TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    binary INTEGER NOT NULL
                )
                """
            )
            try:
                self.connection.execute(
                    """
                    CREATE VIRTUAL TABLE corpus_fts USING fts5(
                        path UNINDEXED,
                        model_key UNINDEXED,
                        device_uid UNINDEXED,
                        kind UNINDEXED,
                        path_text,
                        header,
                        body,
                        keywords,
                        tokenize='trigram'
                    )
                    """
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    "This Python SQLite build does not provide FTS5 trigram"
                ) from exc
            self.connection.execute(
                "INSERT INTO corpus_meta(key, value) VALUES ('raw_root', ?)",
                (str(self.raw_root),),
            )
            manifest_digest, manifest_count = directory_manifest_sha256(
                self.raw_root, suffixes=INDEXED_SUFFIXES
            )
            stat_digest, _ = directory_manifest_sha256(
                self.raw_root,
                suffixes=INDEXED_SUFFIXES,
                include_mtime=True,
            )
            self.connection.executemany(
                "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
                (
                    ("index_version", CORPUS_INDEX_VERSION),
                    ("raw_manifest_sha256", manifest_digest),
                    ("raw_stat_sha256", stat_digest),
                    ("raw_file_count", str(manifest_count)),
                ),
            )
            for record in self._source_records():
                cursor = self.connection.execute(
                    """
                    INSERT INTO corpus_files(
                        path, model_key, device_uid, kind, suffix, size_bytes, binary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["path"],
                        record["model_key"],
                        record["device_uid"],
                        record["kind"],
                        record["suffix"],
                        record["size_bytes"],
                        int(record["binary"]),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO corpus_fts(
                        rowid, path, model_key, device_uid, kind,
                        path_text, header, body, keywords
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        record["path"],
                        record["model_key"],
                        record["device_uid"],
                        record["kind"],
                        record["path"],
                        record["header"],
                        record["body"],
                        record["keywords"],
                    ),
                )

    def _validate_existing(self) -> None:
        try:
            integrity = self.connection.execute("PRAGMA quick_check").fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"Corpus index integrity check failed: {self.database_path}") from exc
        if integrity is None or str(integrity[0]).casefold() != "ok":
            raise ValueError(f"Corpus index integrity check failed: {integrity!r}")
        try:
            rows = self.connection.execute("SELECT key, value FROM corpus_meta").fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"Not a PowerBench corpus index: {self.database_path}") from exc
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        if Path(metadata.get("raw_root", "")).resolve() != self.raw_root:
            raise ValueError("Corpus index raw_root does not match the inventory")
        if metadata.get("index_version") != CORPUS_INDEX_VERSION:
            raise ValueError("Corpus index version is missing or stale; rebuild the index")
        manifest_digest, manifest_count = directory_manifest_sha256(
            self.raw_root, suffixes=INDEXED_SUFFIXES
        )
        stat_digest, _ = directory_manifest_sha256(
            self.raw_root,
            suffixes=INDEXED_SUFFIXES,
            include_mtime=True,
        )
        if metadata.get("raw_manifest_sha256") != manifest_digest:
            raise ValueError("Raw corpus path/size manifest changed; rebuild the index")
        if metadata.get("raw_stat_sha256") != stat_digest:
            raise ValueError("Raw corpus timestamps changed; rebuild the index before querying")
        if int(metadata.get("raw_file_count", -1)) != manifest_count:
            raise ValueError("Raw corpus file count changed; rebuild the index")
        try:
            file_rows = int(
                self.connection.execute("SELECT COUNT(*) FROM corpus_files").fetchone()[0]
            )
            search_rows = int(
                self.connection.execute("SELECT COUNT(*) FROM corpus_fts").fetchone()[0]
            )
        except sqlite3.Error as exc:
            raise ValueError("Corpus index tables are missing or unreadable") from exc
        if file_rows != manifest_count or search_rows != manifest_count:
            raise ValueError(
                "Corpus index row counts disagree with the frozen raw manifest; rebuild the index"
            )

    def _source_records(self) -> Iterable[dict[str, Any]]:
        known_models = set(self.inventory.metrics_by_model) | {
            device.model_key for device in self.inventory.devices.values()
        }
        for path in sorted(self.raw_root.rglob("*"), key=lambda item: item.as_posix()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() not in INDEXED_SUFFIXES
            ):
                continue
            relative = relative_posix(path, self.raw_root)
            parts = PurePosixPath(relative).parts
            if len(parts) < 2 or self._has_forbidden_component(parts):
                continue
            model_key = parts[0]
            if model_key not in known_models:
                continue

            artifact = self.inventory.artifacts.get(relative, {})
            device_uid = artifact.get("device_uid")
            kind = str(artifact.get("kind") or self._infer_kind(path))
            if device_uid is None and len(parts) >= 3 and parts[1] in DEVICE_SCOPED_DIRECTORIES:
                match = DEVICE_PREFIX_RE.match(parts[-1])
                if not match:
                    continue
                candidate_uid = f"{model_key}/{match.group(1)}"
                if candidate_uid not in self.inventory.devices:
                    continue
                device_uid = candidate_uid

            suffix = path.suffix.lower()
            body = ""
            header = ""
            if suffix in TEXT_SUFFIXES:
                body = path.read_text(encoding="utf-8-sig", errors="replace")
            elif suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    row = next(csv.reader(handle), [])
                header = ",".join(row)
            searchable = "\n".join((relative, header, body))
            rule_ids = sorted(
                {f"K{int(match.group(1))}" for match in RULE_ID_RE.finditer(searchable)},
                key=lambda value: int(value[1:]),
            )
            yield {
                "path": relative,
                "model_key": model_key,
                "device_uid": device_uid,
                "kind": kind,
                "suffix": suffix,
                "size_bytes": path.stat().st_size,
                "binary": suffix in BINARY_SUFFIXES,
                "header": header,
                "body": body,
                # Trigram cannot match two-character terms such as K1. Stable
                # sentinels make exact short rule-ID searches indexable.
                "keywords": " ".join(f"__{rule_id}__" for rule_id in rule_ids),
            }

    @staticmethod
    def _has_forbidden_component(parts: Sequence[str]) -> bool:
        return any(part.casefold() in FORBIDDEN_COMPONENTS for part in parts)

    @staticmethod
    def _infer_kind(path: Path) -> str:
        parent = path.parent.name
        if parent in DEVICE_SCOPED_DIRECTORIES:
            return {
                "per_device": "timeseries" if path.suffix.lower() == ".csv" else "annotation",
                "device_archive": "device_archive",
                "check_report": "check_report",
                "fault_report": "fault_report",
            }[parent]
        if path.suffix.lower() == ".xlsx":
            return "workbook"
        if path.name == "metric_catalog.json":
            return "metric_catalog"
        if path.name.startswith("rule_metric_stage"):
            return "rule_catalog"
        if path.name == "derived_metrics.json":
            return "derived_metrics"
        return "document"

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CorpusIndex:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


class CorpusTools:
    """Scope-enforcing read-only tools over a :class:`CorpusIndex`."""

    def __init__(
        self,
        inventory: CorpusInventory,
        scope: Mapping[str, Any] | Iterable[str],
        *,
        index: CorpusIndex | None = None,
        max_timeseries_rows: int = 500,
        max_file_lines: int = 240,
        max_file_bytes: int = 1_500_000,
        max_total_read_bytes: int = 1_500_000,
    ) -> None:
        self.inventory = inventory
        if isinstance(scope, Mapping):
            raw_uids = scope.get("device_uids")
            if not isinstance(raw_uids, (list, tuple, set)):
                raise CorpusQueryError("scope must contain device_uids")
        elif isinstance(scope, str):
            raw_uids = [scope]
        else:
            raw_uids = scope
        self.device_uids = frozenset(str(uid) for uid in raw_uids)
        if not self.device_uids:
            raise CorpusQueryError("Corpus scope cannot be empty")
        unknown = sorted(self.device_uids.difference(inventory.devices))
        if unknown:
            raise CorpusQueryError(f"Unknown device_uids in scope: {unknown}")
        self.model_keys = frozenset(inventory.device(uid).model_key for uid in self.device_uids)
        self.max_timeseries_rows = int(max_timeseries_rows)
        self.max_file_lines = int(max_file_lines)
        self.max_file_bytes = int(max_file_bytes)
        self.max_total_read_bytes = int(max_total_read_bytes)
        if (
            min(
                self.max_timeseries_rows,
                self.max_file_lines,
                self.max_file_bytes,
                self.max_total_read_bytes,
            )
            <= 0
        ):
            raise CorpusQueryError("Corpus read limits must be positive")

        self.index = index or CorpusIndex(inventory)
        if self.index.raw_root != inventory.raw_root.resolve():
            raise CorpusQueryError("CorpusIndex and CorpusInventory use different raw roots")
        self._owns_index = index is None
        self.trace: list[dict[str, Any]] = []
        self._call_number = 0
        self._materialized_number = 0
        self._materialized: dict[str, dict[str, Any]] = {}
        self._materialized_queries: dict[tuple[str, tuple[str, ...], str, str], str] = {}
        self._allowed_records_cache: list[dict[str, Any]] | None = None
        self._total_read_bytes = 0

    def _next_call_id(self) -> str:
        self._call_number += 1
        return f"call_{self._call_number:04d}"

    def _record_trace(self, **event: Any) -> None:
        self.trace.append({"sequence": len(self.trace) + 1, **event})

    def _charge_read_bytes(
        self,
        amount: int,
        *,
        tool: str,
        call_id: str,
        path: str,
    ) -> None:
        projected = self._total_read_bytes + amount
        if projected > self.max_total_read_bytes:
            self._record_trace(
                access_stage="Read",
                tool=tool,
                call_id=call_id,
                path=path,
                paths=[path],
                status="read_budget_exceeded",
                row_count=0,
                byte_count=0,
                attempted_byte_count=amount,
                total_read_bytes=self._total_read_bytes,
                max_total_read_bytes=self.max_total_read_bytes,
            )
            raise CorpusQueryError(
                "Session read budget exceeded: "
                f"{projected} > max_total_read_bytes={self.max_total_read_bytes}"
            )
        self._total_read_bytes = projected

    @staticmethod
    def _normalize_relative(path: str, *, allow_empty: bool = False) -> str:
        if not isinstance(path, str):
            raise CorpusAccessError("Corpus paths must be relative strings")
        candidate = path.strip().replace("\\", "/")
        if not candidate:
            if allow_empty:
                return ""
            raise CorpusAccessError("Corpus path cannot be empty")
        pure = PurePosixPath(candidate)
        parts = pure.parts
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in parts):
            raise CorpusAccessError(f"Unsafe corpus path: {path}")
        if any(part.casefold() in FORBIDDEN_COMPONENTS for part in parts):
            raise CorpusAccessError(f"Forbidden corpus path: {path}")
        return pure.as_posix().rstrip("/")

    def _is_allowed(self, row: Mapping[str, Any]) -> bool:
        # sqlite3.Row implements key lookup but not Mapping.get().
        device_uid = row["device_uid"]
        if device_uid:
            return str(device_uid) in self.device_uids
        return str(row["model_key"]) in self.model_keys

    def _allowed_records(self) -> list[dict[str, Any]]:
        if self._allowed_records_cache is None:
            rows = self.index.connection.execute(
                "SELECT * FROM corpus_files ORDER BY path"
            ).fetchall()
            self._allowed_records_cache = [dict(row) for row in rows if self._is_allowed(row)]
        return self._allowed_records_cache

    def _record_for_read(self, path: str) -> dict[str, Any]:
        relative = self._normalize_relative(path)
        row = self.index.connection.execute(
            "SELECT * FROM corpus_files WHERE path = ?", (relative,)
        ).fetchone()
        if row is None or not self._is_allowed(row):
            raise CorpusAccessError(f"Path is not available in this scope: {relative}")
        return dict(row)

    def list_corpus(
        self,
        path: str = "",
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        prefix = self._normalize_relative(path, allow_empty=True)
        if offset < 0 or limit <= 0:
            raise CorpusQueryError("offset must be non-negative and limit must be positive")
        call_id = self._next_call_id()
        children: dict[str, dict[str, Any]] = {}
        base = f"{prefix}/" if prefix else ""
        for record in self._allowed_records():
            relative = str(record["path"])
            if prefix and relative == prefix:
                remainder = PurePosixPath(relative).name
                child_path = relative
            elif relative.startswith(base):
                remainder = relative[len(base) :]
                first = remainder.partition("/")[0]
                child_path = f"{base}{first}" if base else first
            else:
                continue
            first, separator, _rest = remainder.partition("/")
            if separator:
                existing = children.setdefault(
                    child_path,
                    {"name": first, "path": child_path, "type": "directory", "artifact_count": 0},
                )
                existing["artifact_count"] += 1
            else:
                children[child_path] = {
                    "name": first,
                    "path": child_path,
                    "type": "file",
                    "kind": record["kind"],
                    "device_uid": record["device_uid"],
                    "size_bytes": record["size_bytes"],
                    "binary_unsupported": bool(record["binary"]),
                }
        ordered = sorted(
            children.values(), key=lambda item: (item["type"] != "directory", item["path"])
        )
        selected = ordered[offset : offset + limit]
        self._record_trace(
            access_stage="Retrieved",
            tool="list_corpus",
            call_id=call_id,
            path=prefix,
            offset=offset,
            limit=limit,
            results=copy.deepcopy(selected),
            paths=[entry["path"] for entry in selected if entry["type"] == "file"],
            row_count=len(selected),
            byte_count=0,
        )
        return {
            "path": prefix,
            "entries": selected,
            "offset": offset,
            "limit": limit,
            "total": len(ordered),
            "truncated": offset + len(selected) < len(ordered),
        }

    @staticmethod
    def _search_terms(query: str) -> list[str]:
        stripped = query.strip()
        if not stripped:
            raise CorpusQueryError("search query cannot be empty")
        terms = [term.strip('"') for term in re.findall(r'"[^"]+"|\S+', stripped)]
        result: list[str] = []
        for term in terms:
            rule_match = re.fullmatch(r"K\s*0*(\d+)", term, flags=re.IGNORECASE)
            if rule_match:
                term = f"__K{int(rule_match.group(1))}__"
            if term:
                result.append(term)
        if not result:
            raise CorpusQueryError("search query contains no searchable terms")
        return result

    @staticmethod
    def _fts_expression(terms: Iterable[str]) -> str:
        expressions: list[str] = []
        for term in terms:
            expressions.append('"' + term.replace('"', '""') + '"')
        return " AND ".join(expressions)

    def _scope_sql(self) -> tuple[str, list[str]]:
        uid_placeholders = ",".join("?" for _ in self.device_uids)
        model_placeholders = ",".join("?" for _ in self.model_keys)
        clause = (
            f"(f.device_uid IN ({uid_placeholders}) OR "
            f"(f.device_uid IS NULL AND f.model_key IN ({model_placeholders})))"
        )
        return clause, [*sorted(self.device_uids), *sorted(self.model_keys)]

    def search_corpus(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        if limit <= 0:
            raise CorpusQueryError("limit must be positive")
        terms = self._search_terms(query)
        scope_clause, scope_parameters = self._scope_sql()
        call_id = self._next_call_id()
        if all(len(term) >= 3 for term in terms):
            sql = f"""
                SELECT
                    f.path, f.model_key, f.device_uid, f.kind, f.suffix,
                    f.size_bytes, f.binary,
                    bm25(corpus_fts, 0, 0, 0, 0, 10, 5, 1, 8) AS raw_score,
                    snippet(corpus_fts, -1, '[', ']', ' … ', 24) AS snippet
                FROM corpus_fts
                JOIN corpus_files AS f ON f.id = corpus_fts.rowid
                WHERE corpus_fts MATCH ? AND {scope_clause}
                ORDER BY raw_score, f.path
                LIMIT ?
            """
            rows = self.index.connection.execute(
                sql, [self._fts_expression(terms), *scope_parameters, int(limit)]
            ).fetchall()
        else:
            # FTS5's trigram tokenizer cannot match one- or two-character
            # substrings (for example, the valid metric name "oil level"). Retain
            # strict scope filtering and use an indexed-table fallback only for
            # these short queries.
            conditions: list[str] = []
            like_parameters: list[str] = []
            for term in terms:
                conditions.append(
                    "(instr(lower(corpus_fts.path_text), ?) > 0 OR "
                    "instr(lower(corpus_fts.header), ?) > 0 OR "
                    "instr(lower(corpus_fts.body), ?) > 0 OR "
                    "instr(lower(corpus_fts.keywords), ?) > 0)"
                )
                like_parameters.extend([term.casefold()] * 4)
            sql = f"""
                SELECT
                    f.path, f.model_key, f.device_uid, f.kind, f.suffix,
                    f.size_bytes, f.binary, 0.0 AS raw_score,
                    corpus_fts.path_text, corpus_fts.header, corpus_fts.body,
                    corpus_fts.keywords
                FROM corpus_fts
                JOIN corpus_files AS f ON f.id = corpus_fts.rowid
                WHERE {" AND ".join(conditions)} AND {scope_clause}
                ORDER BY f.path
                LIMIT ?
            """
            rows = self.index.connection.execute(
                sql, [*like_parameters, *scope_parameters, int(limit)]
            ).fetchall()
        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            # sqlite3.Row.__contains__ checks values rather than column names.
            if "snippet" in set(row.keys()):
                snippet = str(row["snippet"] or "")
            else:
                needle = next((term for term in terms if len(term) < 3), terms[0])
                text = str(row["body"] or row["header"] or row["path_text"] or "")
                position = text.casefold().find(needle.casefold())
                start = max(0, position - 80) if position >= 0 else 0
                snippet = text[start : start + 240]
            result = {
                "rank": rank,
                "path": row["path"],
                "kind": row["kind"],
                "device_uid": row["device_uid"],
                "model_key": row["model_key"],
                "snippet": snippet,
                "score": -float(row["raw_score"]),
                "size_bytes": int(row["size_bytes"]),
                "binary_unsupported": bool(row["binary"]),
            }
            results.append(result)
        self._record_trace(
            access_stage="Retrieved",
            tool="search_corpus",
            call_id=call_id,
            query=query,
            limit=limit,
            results=copy.deepcopy(results),
            paths=[result["path"] for result in results],
            row_count=len(results),
            byte_count=sum(len(result["snippet"].encode("utf-8")) for result in results),
        )
        return {"query": query, "results": results, "count": len(results), "limit": limit}

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise CorpusQueryError("Invalid one-based inclusive line range")
        record = self._record_for_read(path)
        call_id = self._next_call_id()
        if record["suffix"] in BINARY_SUFFIXES:
            result = {
                "path": record["path"],
                "supported": False,
                "error": "binary unsupported; XLSX files can only be listed or searched by path",
                "line_count": 0,
                "byte_count": 0,
            }
            self._record_trace(
                access_stage="Read",
                tool="read_file",
                call_id=call_id,
                path=record["path"],
                paths=[record["path"]],
                status="binary_unsupported",
                row_count=0,
                byte_count=0,
            )
            return result
        if record["suffix"] == ".csv":
            raise CorpusQueryError("CSV bodies are available only through read_timeseries")
        if record["suffix"] not in TEXT_SUFFIXES:
            raise CorpusQueryError(f"Unsupported text artifact: {record['path']}")

        source = safe_join(self.inventory.raw_root, record["path"])
        lines: list[str] = []
        byte_count = 0
        actual_end = start_line - 1
        truncated = False
        requested_end = end_line if end_line is not None else start_line + self.max_file_lines - 1
        requested_end = min(requested_end, start_line + self.max_file_lines - 1)
        with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < start_line:
                    continue
                if line_number > requested_end:
                    truncated = True
                    break
                encoded = line.encode("utf-8")
                if byte_count + len(encoded) > self.max_file_bytes:
                    remaining = self.max_file_bytes - byte_count
                    if remaining > 0:
                        lines.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                        byte_count += len(lines[-1].encode("utf-8"))
                        actual_end = line_number
                    truncated = True
                    break
                lines.append(line)
                byte_count += len(encoded)
                actual_end = line_number
        content = "".join(lines)
        self._charge_read_bytes(
            byte_count,
            tool="read_file",
            call_id=call_id,
            path=record["path"],
        )
        result = {
            "path": record["path"],
            "supported": True,
            "start_line": start_line,
            "end_line": actual_end,
            "line_count": len(lines),
            "byte_count": byte_count,
            "total_read_bytes": self._total_read_bytes,
            "truncated": truncated,
            "content": content,
        }
        self._record_trace(
            access_stage="Read",
            tool="read_file",
            call_id=call_id,
            path=record["path"],
            paths=[record["path"]],
            start_line=start_line,
            end_line=actual_end,
            row_count=len(lines),
            byte_count=byte_count,
            total_read_bytes=self._total_read_bytes,
        )
        return result

    @staticmethod
    def _coerce_timestamp(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        try:
            return parse_timestamp(value)
        except (TypeError, ValueError) as exc:
            raise CorpusQueryError(f"Invalid timestamp: {value!r}") from exc

    def read_timeseries(
        self,
        device_uid: str,
        fields: str | Iterable[str],
        start: str | datetime,
        end: str | datetime,
    ) -> dict[str, Any]:
        if device_uid not in self.device_uids:
            raise CorpusAccessError(f"Device is not available in this scope: {device_uid}")
        call_id = self._next_call_id()
        device = self.inventory.device(device_uid)
        start_time = self._coerce_timestamp(start)
        end_time = self._coerce_timestamp(end)
        if start_time >= end_time:
            raise CorpusQueryError("read_timeseries requires start < end for [start, end)")
        requested = [fields] if isinstance(fields, str) else list(fields)
        requested = list(dict.fromkeys(str(field) for field in requested if str(field)))
        requested = [field for field in requested if field != "timestamp"]
        if not requested:
            raise CorpusQueryError("At least one non-timestamp field is required")
        missing = [field for field in requested if field not in device.csv_columns]
        if missing:
            raise CorpusQueryError(f"CSV fields not found for {device_uid}: {missing}")
        if "timestamp" not in device.csv_columns:
            raise CorpusQueryError(f"CSV has no timestamp column: {device.csv_path}")

        record = self._record_for_read(device.csv_path)
        formatted_start = format_timestamp(start_time)
        formatted_end = format_timestamp(end_time)
        query_key = (device_uid, tuple(requested), formatted_start, formatted_end)
        cached_handle = self._materialized_queries.get(query_key)
        if cached_handle is not None:
            cached = self._materialized[cached_handle]
            self._record_trace(
                access_stage="Read",
                tool="read_timeseries",
                call_id=call_id,
                path=record["path"],
                paths=[record["path"]],
                device_uid=device_uid,
                fields=requested,
                start=formatted_start,
                end=formatted_end,
                interval="[start,end)",
                materialized_handle=cached_handle,
                status="cache_hit",
                row_count=int(cached["row_count"]),
                byte_count=0,
                total_read_bytes=self._total_read_bytes,
            )
            result = {key: copy.deepcopy(value) for key, value in cached.items() if key != "csv"}
            result["cache_hit"] = True
            result["total_read_bytes"] = self._total_read_bytes
            return result
        source = safe_join(self.inventory.raw_root, device.csv_path)
        rows: list[dict[str, str]] = []
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_timestamp = row.get("timestamp", "")
                try:
                    timestamp = parse_timestamp(raw_timestamp)
                except ValueError as exc:
                    raise CorpusQueryError(
                        f"Invalid timestamp {raw_timestamp!r} in {device.csv_path}"
                    ) from exc
                if start_time <= timestamp < end_time:
                    rows.append(
                        {
                            "timestamp": raw_timestamp,
                            **{field: row.get(field, "") for field in requested},
                        }
                    )
                    if len(rows) > self.max_timeseries_rows:
                        raise CorpusQueryError(
                            f"read_timeseries exceeds max_timeseries_rows={self.max_timeseries_rows}"
                        )

        columns = ["timestamp", *requested]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        csv_text = buffer.getvalue()
        byte_count = len(csv_text.encode("utf-8"))
        self._charge_read_bytes(
            byte_count,
            tool="read_timeseries",
            call_id=call_id,
            path=record["path"],
        )
        self._materialized_number += 1
        materialized_handle = f"mat_{self._materialized_number:06d}"
        materialized = {
            "handle": materialized_handle,
            "kind": "timeseries_csv",
            "path": record["path"],
            "device_uid": device_uid,
            "start": format_timestamp(start_time),
            "end": format_timestamp(end_time),
            "interval": "[start,end)",
            "columns": columns,
            "rows": copy.deepcopy(rows),
            "csv": csv_text,
            "row_count": len(rows),
            "byte_count": byte_count,
            "total_read_bytes": self._total_read_bytes,
        }
        self._materialized[materialized_handle] = materialized
        self._materialized_queries[query_key] = materialized_handle
        self._record_trace(
            access_stage="Read",
            tool="read_timeseries",
            call_id=call_id,
            path=record["path"],
            paths=[record["path"]],
            device_uid=device_uid,
            fields=requested,
            start=format_timestamp(start_time),
            end=format_timestamp(end_time),
            interval="[start,end)",
            materialized_handle=materialized_handle,
            row_count=len(rows),
            byte_count=byte_count,
            total_read_bytes=self._total_read_bytes,
        )
        result = {key: copy.deepcopy(value) for key, value in materialized.items() if key != "csv"}
        result["cache_hit"] = False
        return result

    def get_materialized(self, handle: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._materialized[handle])
        except KeyError as exc:
            raise CorpusAccessError(f"Unknown materialized handle: {handle}") from exc

    def materialized_handles(self) -> dict[str, Any]:
        """Return a detached JSON-safe snapshot for the computation sandbox."""

        return copy.deepcopy(self._materialized)

    def export_trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.trace)

    def close(self) -> None:
        if self._owns_index:
            self.index.close()

    def __enter__(self) -> CorpusTools:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def build_corpus_index(*, raw_root: str | Path, database_path: str | Path) -> dict[str, Any]:
    """Build the persistent read-only corpus index used by benchmark runs."""

    inventory = build_inventory(raw_root)
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with CorpusIndex(inventory, path, rebuild=True) as index:
        count = int(index.connection.execute("SELECT COUNT(*) FROM corpus_files").fetchone()[0])
    return {"database_path": str(path.resolve()), "artifact_count": count}