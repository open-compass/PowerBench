from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ..util import canonical_json, read_jsonl

TERMINAL_RUN_STATUSES = {
    "completed",
    "format_error",
    "tool_budget_exhausted",
    "response_incomplete",
    "refusal",
    "time_budget_exhausted",
}
RETRYABLE_RUN_STATUSES = {"infrastructure_error", "api_error"}
MAX_CONSECUTIVE_INFRASTRUCTURE_ERRORS = 3
MAX_ATTEMPTS_PER_RUN = 3


class RunCircuitBreakerError(RuntimeError):
    """Stop a paid manifest after repeated infrastructure/API failures."""


class ResultsLockError(RuntimeError):
    """Reject concurrent writers for one append-only results destination."""


@contextmanager
def exclusive_results_lock(results_path: str | Path) -> Iterator[None]:
    """Hold an advisory lock for the complete manifest execution lifecycle.

    The sidecar is deliberately retained. ``flock`` ownership belongs to the
    open file descriptor, so a crashed process cannot leave a permanently stale
    lock, and retaining the inode avoids unlink/recreate races between runners.
    """

    destination = Path(results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f"{destination.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResultsLockError(
                f"Another runner already holds the results lock: {lock_path.resolve()}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def append_result(handle: TextIO, result: dict[str, Any]) -> None:
    """Durably append one attempt before execution advances or propagates an error."""

    handle.write(canonical_json(result) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


@dataclass(frozen=True)
class ResumeState:
    pending: list[dict[str, Any]]
    terminal_ids: set[str]
    retryable_ids: set[str]
    exhausted_ids: set[str]
    attempt_counts: dict[str, int]
    next_request_attempt_ordinals: dict[str, int]
    trailing_infrastructure_errors: int


def resume_state(jobs: list[dict[str, Any]], existing_rows: list[dict[str, Any]]) -> ResumeState:
    """Validate append-only history and recover retry/circuit state."""

    jobs_by_id = {job["run_id"]: job for job in jobs}
    attempts_by_run_id: dict[str, list[dict[str, Any]]] = {}
    for row in existing_rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Existing result row has no valid run_id")
        if run_id not in jobs_by_id:
            raise ValueError(
                f"Results destination contains run_id {run_id!r} outside the current manifest"
            )
        attempts_by_run_id.setdefault(run_id, []).append(row)

    latest_by_run_id: dict[str, dict[str, Any]] = {}
    valid_statuses = TERMINAL_RUN_STATUSES | RETRYABLE_RUN_STATUSES
    for run_id, attempts in attempts_by_run_id.items():
        job = jobs_by_id[run_id]
        for attempt_index, row in enumerate(attempts, start=1):
            disagreements = [key for key, value in job.items() if row.get(key) != value]
            if disagreements:
                raise ValueError(
                    f"Existing result {run_id} attempt {attempt_index} disagrees with its "
                    f"manifest job on: {sorted(disagreements)}"
                )
            status = row.get("status")
            if status not in valid_statuses:
                raise ValueError(
                    f"Existing result {run_id} attempt {attempt_index} has unsupported "
                    f"status {status!r}"
                )
        for attempt_index, row in enumerate(attempts[:-1], start=1):
            if row.get("status") not in RETRYABLE_RUN_STATUSES:
                raise ValueError(
                    f"Existing result {run_id} attempt {attempt_index} is terminal and "
                    "cannot be followed by another attempt"
                )
        cycle_start = max(
            (
                index
                for index, row in enumerate(attempts)
                if row.get("retry_exhausted_before_attempt") is True
            ),
            default=0,
        )
        current_cycle = attempts[cycle_start:]
        if len(current_cycle) > MAX_ATTEMPTS_PER_RUN:
            raise ValueError(
                f"Existing result {run_id} has {len(current_cycle)} attempts in its current "
                f"retry cycle, exceeding the maximum of {MAX_ATTEMPTS_PER_RUN}"
            )
        latest_by_run_id[run_id] = attempts[-1]

    terminal_ids = {
        run_id
        for run_id, row in latest_by_run_id.items()
        if row.get("status") in TERMINAL_RUN_STATUSES
    }
    retryable_ids = set(latest_by_run_id).difference(terminal_ids)
    attempt_counts: dict[str, int] = {}
    for run_id in jobs_by_id:
        attempts = attempts_by_run_id.get(run_id, [])
        cycle_start = max(
            (
                index
                for index, row in enumerate(attempts)
                if row.get("retry_exhausted_before_attempt") is True
            ),
            default=0,
        )
        attempt_counts[run_id] = len(attempts[cycle_start:])
    next_request_attempt_ordinals: dict[str, int] = {}
    for run_id in jobs_by_id:
        latest = latest_by_run_id.get(run_id)
        if latest is None:
            next_request_attempt_ordinals[run_id] = 1
            continue
        stored_ordinal = latest.get("request_attempt_ordinal")
        if (
            isinstance(stored_ordinal, bool)
            or not isinstance(stored_ordinal, int)
            or stored_ordinal < 1
        ):
            stored_ordinal = len(attempts_by_run_id.get(run_id, []))
        next_request_attempt_ordinals[run_id] = (
            stored_ordinal if latest.get("status") == "infrastructure_error" else stored_ordinal + 1
        )

    exhausted_ids = {
        run_id for run_id in retryable_ids if attempt_counts[run_id] >= MAX_ATTEMPTS_PER_RUN
    }
    pending = [
        job
        for job in jobs
        if job["run_id"] not in terminal_ids and job["run_id"] not in exhausted_ids
    ]
    # The circuit breaker counts exhausted logical jobs, not individual HTTP/job
    # attempts. A transient failure that later succeeds must never open it.
    trailing_infrastructure_errors = 0
    seen_attempts: dict[str, int] = {}
    for row in existing_rows:
        if row.get("circuit_breaker_reset_before_attempt") is True:
            trailing_infrastructure_errors = 0
        run_id = row["run_id"]
        if row.get("retry_exhausted_before_attempt") is True:
            seen_attempts[run_id] = 0
        seen_attempts[run_id] = seen_attempts.get(run_id, 0) + 1
        if row.get("status") in TERMINAL_RUN_STATUSES:
            trailing_infrastructure_errors = 0
        elif seen_attempts[run_id] >= MAX_ATTEMPTS_PER_RUN:
            trailing_infrastructure_errors += 1

    return ResumeState(
        pending=pending,
        terminal_ids=terminal_ids,
        retryable_ids=retryable_ids,
        exhausted_ids=exhausted_ids,
        attempt_counts=attempt_counts,
        next_request_attempt_ordinals=next_request_attempt_ordinals,
        trailing_infrastructure_errors=trailing_infrastructure_errors,
    )


def resume_plan(
    jobs: list[dict[str, Any]], existing_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Compatibility view of the complete append-only resume state."""

    state = resume_state(jobs, existing_rows)
    return state.pending, state.terminal_ids, state.retryable_ids


@dataclass
class RunStore:
    """Own append-only results, resume limits, and the paid-run circuit breaker."""

    destination: Path
    resume: ResumeState
    pending: list[dict[str, Any]]
    reset_applied: bool
    reopened_ids: set[str]
    reopened_count: int
    attempted: int = 0
    terminal_completed: int = 0
    retryable_failures: int = 0
    attempt_failures: list[dict[str, Any]] = field(default_factory=list)
    attempt_counts: dict[str, int] = field(default_factory=dict)
    exhausted_ids: set[str] = field(default_factory=set)
    consecutive_infrastructure_errors: int = 0

    @classmethod
    def prepare(
        cls,
        jobs: list[dict[str, Any]],
        results_path: str | Path,
        *,
        max_jobs: int | None,
        reset_circuit_breaker: bool,
        retry_exhausted: bool = False,
    ) -> RunStore:
        destination = Path(results_path)
        existing_rows = read_jsonl(destination) if destination.exists() else []
        resume = resume_state(jobs, existing_rows)
        breaker_was_open = (
            resume.trailing_infrastructure_errors >= MAX_CONSECUTIVE_INFRASTRUCTURE_ERRORS
        )
        reopened_ids = set(resume.exhausted_ids) if retry_exhausted else set()
        reset_applied = bool((reset_circuit_breaker or retry_exhausted) and breaker_was_open)
        pending = [
            job
            for job in jobs
            if job["run_id"] not in resume.terminal_ids
            and (job["run_id"] not in resume.exhausted_ids or job["run_id"] in reopened_ids)
        ]
        if max_jobs is not None:
            pending = pending[:max_jobs]
        attempt_counts = dict(resume.attempt_counts)
        for run_id in reopened_ids:
            attempt_counts[run_id] = 0
        return cls(
            destination=destination,
            resume=resume,
            pending=pending,
            reset_applied=reset_applied,
            reopened_ids=reopened_ids,
            reopened_count=len(reopened_ids),
            attempt_counts=attempt_counts,
            exhausted_ids=set(resume.exhausted_ids).difference(reopened_ids),
            consecutive_infrastructure_errors=(
                0 if reset_applied else resume.trailing_infrastructure_errors
            ),
        )

    @property
    def request_attempt_ordinals(self) -> dict[str, int]:
        return self.resume.next_request_attempt_ordinals

    def ensure_circuit_closed(self) -> None:
        if not self.pending:
            return
        if (
            self.resume.trailing_infrastructure_errors >= MAX_CONSECUTIVE_INFRASTRUCTURE_ERRORS
            and not self.reset_applied
        ):
            raise RunCircuitBreakerError(
                "Paid-run circuit breaker remains open because the append-only results end with "
                f"{self.resume.trailing_infrastructure_errors} consecutive jobs that exhausted "
                "their infrastructure/API retry budgets. Inspect the persisted failures, then "
                "explicitly pass "
                "reset_circuit_breaker=True to acknowledge and resume; per-run attempt limits "
                "remain in force."
            )

    @contextmanager
    def writer(self) -> Iterator[TextIO]:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8", newline="") as handle:
            yield handle

    def append_attempt(
        self,
        handle: TextIO,
        result: dict[str, Any],
        *,
        enforce_circuit_breaker: bool = True,
    ) -> None:
        """Persist one result and atomically advance in-memory run accounting."""

        if self.reset_applied and self.attempted == 0:
            result["circuit_breaker_reset_before_attempt"] = True
        run_id = result["run_id"]
        if run_id in self.reopened_ids and self.attempt_counts[run_id] == 0:
            result["retry_exhausted_before_attempt"] = True
            self.reopened_ids.remove(run_id)
        append_result(handle, result)
        self.attempted += 1

        self.attempt_counts[run_id] += 1
        if result.get("status") in RETRYABLE_RUN_STATUSES:
            self.retryable_failures += 1
            self.attempt_failures.append(
                {
                    "run_id": result.get("run_id"),
                    "qa_id": result.get("qa_id"),
                    "status": result.get("status"),
                    "error": result.get("error"),
                }
            )
            if self.attempt_counts[run_id] >= MAX_ATTEMPTS_PER_RUN:
                self.exhausted_ids.add(run_id)
                self.consecutive_infrastructure_errors += 1
        else:
            self.terminal_completed += 1
            self.consecutive_infrastructure_errors = 0

        if enforce_circuit_breaker:
            self.ensure_runtime_circuit_closed()

    def ensure_runtime_circuit_closed(self) -> None:
        """Stop after consecutive logical jobs exhaust their retry budgets.

        Concurrent execution calls this at worker-batch boundaries so every
        already-paid result in the batch is persisted before the runner stops.
        """

        if self.consecutive_infrastructure_errors >= MAX_CONSECUTIVE_INFRASTRUCTURE_ERRORS:
            raise RunCircuitBreakerError(
                "Paid-run circuit breaker opened after "
                f"{self.consecutive_infrastructure_errors} consecutive jobs exhausted their "
                "infrastructure/API retry budgets; inspect the persisted attempts and use the "
                "explicit circuit breaker reset acknowledgement before resuming"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "scheduled": len(self.pending),
            "attempted": self.attempted,
            "terminal_completed": self.terminal_completed,
            "retryable_failures": self.retryable_failures,
            "failures": self.attempt_failures,
            "exhausted_retryables": len(self.exhausted_ids),
            "exhausted_retryable_run_ids": sorted(self.exhausted_ids),
            "skipped_existing": len(self.resume.terminal_ids),
            "retryable_existing": len(self.resume.retryable_ids),
            "trailing_retryable_failures_before_run": (self.resume.trailing_infrastructure_errors),
            "circuit_breaker_reset_applied": self.reset_applied,
            "reopened_exhausted": self.reopened_count,
            "results_path": str(self.destination.resolve()),
        }