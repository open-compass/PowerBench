from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jsonschema

from ..config import PilotConfig
from ..corpus import CorpusAccessError, CorpusIndex, CorpusQueryError, CorpusTools
from ..inventory import build_inventory
from ..records import read_gold_records
from ..util import canonical_json, read_jsonl, stable_id
from .chat_transport import create_chat_response
from .oracle import build_oracle_bundle
from .preflight import preflight_oracle_evidence as _preflight_oracle_evidence
from .preflight import retrieval_settings as _retrieval_settings
from .preflight import validate_corpus_inputs as _preflight_corpus_inputs
from .preflight import validate_run_inputs as _preflight
from .preflight import verify_manifest_binding as _verify_manifest_binding
from .protocol import SYSTEM_INSTRUCTIONS, TOOL_DEFINITIONS
from .sandbox import SandboxError, run_python
from .store import (
    MAX_ATTEMPTS_PER_RUN,
    RETRYABLE_RUN_STATUSES,
    RunStore,
    exclusive_results_lock,
)

API_REQUEST_TIMEOUT_SECONDS = 600.0
JOB_RETRY_BACKOFF_SECONDS = (5.0, 15.0)


def _transport_parameters(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("api_format", "responses") == "responses":
        return {}
    return {
        "api_format": "chat-completions",
        "reasoning_effort": None if job["reasoning_effort"] == "unspecified" else job["reasoning_effort"],
        "max_tokens": 12_000,
        "store": None,
        "reasoning_summary": None,
        "verbosity": None,
    }


class RunInfrastructureError(RuntimeError):
    """Carry the observable partial trace through a failed model run."""

    def __init__(
        self,
        message: str,
        *,
        tool_trace: list[dict[str, Any]],
        usage: dict[str, int] | None = None,
        response_ids: list[str] | None = None,
        resolved_models: list[str] | None = None,
        prompt: str = "",
        response_text: str = "",
        format_retry_count: int = 0,
        elapsed_seconds: float = 0.0,
        retry_status: str = "infrastructure_error",
    ) -> None:
        super().__init__(message)
        if retry_status not in RETRYABLE_RUN_STATUSES:
            raise ValueError(f"Unsupported retry status: {retry_status!r}")
        self.tool_trace = list(tool_trace)
        self.usage = dict(usage or {})
        self.response_ids = list(response_ids or [])
        self.resolved_models = list(resolved_models or [])
        self.prompt = prompt
        self.response_text = response_text
        self.format_retry_count = int(format_retry_count)
        self.elapsed_seconds = float(elapsed_seconds)
        self.retry_status = retry_status


def _infrastructure_result(
    job: dict[str, Any],
    retrieval: dict[str, Any],
    exc: BaseException,
    *,
    prompt: str = "",
    response_text: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    elapsed_seconds: float = 0.0,
    response_ids: list[str] | None = None,
    resolved_models: list[str] | None = None,
    format_retry_count: int = 0,
    request_attempt_ordinal: int = 1,
) -> dict[str, Any]:
    """Build the same auditable result envelope for every runner-side failure."""

    return {
        **job,
        "status": getattr(exc, "retry_status", "infrastructure_error"),
        "prompt": prompt,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "response_text": response_text,
        "prediction": None,
        "tool_trace": list(tool_trace or []),
        "usage": {
            "input_tokens": int((usage or {}).get("input_tokens", 0)),
            "output_tokens": int((usage or {}).get("output_tokens", 0)),
            "total_tokens": int((usage or {}).get("total_tokens", 0)),
        },
        "latency_seconds": round(float(elapsed_seconds), 6),
        "response_ids": list(response_ids or []),
        "resolved_models": sorted(set(resolved_models or [])),
        "format_retry_count": int(format_retry_count),
        "request_attempt_ordinal": int(request_attempt_ordinal),
        "request_parameters": {
            "reasoning_effort": job["reasoning_effort"],
            "max_tool_rounds": int(retrieval["max_tool_rounds"]),
            "max_job_seconds": float(job.get("max_job_seconds", 300)),
            "max_output_tokens": 12_000,
            "answer_transport": "plain_json_prompt_schema_local_validation",
            "parallel_tool_calls": False,
            "store": False,
            "continuation": "stateless_plain_tool_transcript",
            "request_attempt_ordinal": int(request_attempt_ordinal),
            **_transport_parameters(job),
        },
        "time_budget_reached": False,
        "time_finalization_used": False,
        "time_finalization_succeeded": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _response_item_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    return {"type": getattr(item, "type", type(item).__name__)}


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _merge_usage(total: dict[str, int], current: dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] += int(current.get(key, 0))


def _scope_for_job(
    scopes: dict[tuple[str, int, int], dict[str, Any]], job: dict[str, Any]
) -> dict[str, Any]:
    key = (job["pair_id"], int(job["N"]), int(job["scope_seed"]))
    try:
        return scopes[key]
    except KeyError as exc:
        raise KeyError(f"No corpus scope for pair/N/seed={key}") from exc


def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: CorpusTools,
    retrieval_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_start = len(corpus.export_trace())
    sandbox_trace: list[dict[str, Any]] = []
    if name == "list_corpus":
        result = corpus.list_corpus(**arguments)
    elif name == "search_corpus":
        result = corpus.search_corpus(**arguments)
    elif name == "read_file":
        result = corpus.read_file(**arguments)
    elif name == "read_timeseries":
        result = corpus.read_timeseries(**arguments)
    elif name == "python":
        handles = corpus.materialized_handles()
        sandbox_result = run_python(
            arguments["code"],
            handles,
            timeout_seconds=float(retrieval_config["python_timeout_seconds"]),
            max_output_chars=int(retrieval_config["python_max_output_chars"]),
            trace=sandbox_trace,
        )
        result = sandbox_result.as_dict()
        result["available_handles"] = sorted(handles)
        if not sandbox_result.ok:
            result["usage_hint"] = (
                'Access a materialized table as handles["mat_000001"]["rows"] '
                "using the handle returned by read_timeseries. Use only direct calls "
                "to approved functions and print(json_dumps(result))."
            )
    else:
        raise CorpusQueryError(f"Unknown tool: {name}")
    access_events = corpus.export_trace()[trace_start:]
    return result, [*access_events, *sandbox_trace]


def _model_visible_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep large time-series rows local while exposing their sandbox handle.

    The complete result remains in ``tool_trace`` for audit.  Re-sending every
    row through each stateless Responses request is unnecessary because the
    Python tool receives the materialized table directly from ``CorpusTools``.
    """

    if name != "read_timeseries" or "error" in result:
        return result
    visible = {key: value for key, value in result.items() if key not in {"rows", "csv"}}
    handle = visible.get("handle")
    if isinstance(handle, str):
        visible["python_access"] = f'handles["{handle}"]["rows"]'
    return visible


def _make_prompt(
    question: dict[str, Any], job: dict[str, Any], oracle: dict[str, Any] | None
) -> str:
    lines = [
        f"condition: {job['condition']}",
        f"visible_device_count: {job['N']}",
        "question:",
        question["question"],
        "\nrequired_final_answer_json_schema:",
        canonical_json(question["answer_schema"]),
        (
            "Return only one JSON object matching this schema exactly. Do not wrap it "
            "in Markdown and do not add prose."
        ),
    ]
    if oracle is not None:
        public_oracle = {key: value for key, value in oracle.items() if key != "_audit"}
        lines.extend(
            [
                "\nOracle-Evidence condition: the evaluator has supplied one public evidence source per Gold evidence atom below. Corpus search/read tools are disabled; compute from this evidence and the materialized handles.",
                canonical_json(public_oracle),
            ]
        )
    return "\n".join(lines)


def _response_failure(response: Any, output_items: list[Any]) -> tuple[str | None, str | None]:
    response_status = str(getattr(response, "status", "completed") or "completed")
    if response_status in {"failed", "cancelled"}:
        details = getattr(response, "error", None)
        return "api_error", canonical_json(
            {
                "response_status": response_status,
                "details": _response_item_dict(details),
            }
        )
    if response_status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return "response_incomplete", canonical_json(_response_item_dict(details))
    if response_status in {"in_progress", "queued"}:
        return "api_error", canonical_json(
            {
                "response_status": response_status,
                "details": "Synchronous Responses call returned a non-terminal status",
            }
        )
    if response_status != "completed":
        return "api_error", canonical_json(
            {
                "response_status": response_status,
                "details": "Unrecognized Responses API status",
            }
        )
    for item in output_items:
        dumped = _response_item_dict(item)
        content = dumped.get("content", [])
        if any(
            isinstance(part, dict) and part.get("type") == "refusal"
            for part in (content if isinstance(content, list) else [])
        ):
            return "refusal", "Model returned a structured refusal"
    return None, None


def _api_exception_retry_status(exc: BaseException) -> str:
    """Separate definite HTTP failures from ambiguous transport failures.

    A response-bearing exception is safe to retry as a new logical request.
    Connection/time-out failures may have reached the service, so a resume
    reuses the same idempotency attempt ordinal.
    """

    if getattr(exc, "response", None) is not None or getattr(exc, "status_code", None) is not None:
        return "api_error"
    return "infrastructure_error"


def _is_timeout_exception(exc: BaseException) -> bool:
    """Recognize SDK, HTTP-client and built-in timeout errors without coupling."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.casefold():
            return True
        current = current.__cause__ or current.__context__
    return False


def run_one(
    *,
    client: Any,
    inventory: Any,
    index: CorpusIndex,
    config: PilotConfig,
    question: dict[str, Any],
    job: dict[str, Any],
    scope: dict[str, Any],
    gold: dict[str, Any] | None = None,
    request_attempt_ordinal: int = 1,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if request_attempt_ordinal < 1:
        raise ValueError("request_attempt_ordinal must be positive")
    retrieval = _retrieval_settings(config)
    started = monotonic()
    api_format = job.get("api_format", "responses")
    if api_format not in {"responses", "chat-completions"}:
        raise ValueError(f"Unsupported API format: {api_format}")
    max_job_seconds = float(job.get("max_job_seconds", 300))
    if not max_job_seconds >= 60:
        raise ValueError("job max_job_seconds must be at least 60")
    reasoning_deadline = started + max_job_seconds
    tool_trace: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response_ids: list[str] = []
    resolved_models: list[str] = []
    prompt = ""
    response_text = ""
    format_failures = 0
    tool_rounds = 0
    api_round = 0
    tool_budget_reached = False
    budget_finalization_used = False
    time_budget_reached = False
    time_finalization_used = False

    def infrastructure_error(
        message: str, *, retry_status: str = "infrastructure_error"
    ) -> RunInfrastructureError:
        return RunInfrastructureError(
            message,
            tool_trace=tool_trace,
            usage=usage,
            response_ids=response_ids,
            resolved_models=resolved_models,
            prompt=prompt,
            response_text=response_text,
            format_retry_count=format_failures,
            elapsed_seconds=monotonic() - started,
            retry_status=retry_status,
        )

    with CorpusTools(
        inventory,
        scope,
        index=index,
        max_timeseries_rows=int(retrieval["max_timeseries_rows"]),
        max_file_lines=int(retrieval["max_read_lines"]),
        max_total_read_bytes=int(retrieval["max_total_read_bytes"]),
    ) as corpus:
        oracle_bundle = None
        if job["condition"] == "oracle_evidence":
            if gold is None:
                raise ValueError("Oracle-Evidence run requires private Gold evidence annotations")
            try:
                oracle_bundle = build_oracle_bundle(inventory, corpus, gold)
            except Exception as exc:
                for event in corpus.export_trace():
                    tool_trace.append(
                        {
                            **event,
                            "event": "oracle_prefetch",
                            "step": 0,
                            "actor": "evaluator",
                            "source_access_stage": event.get("access_stage"),
                            "access_stage": "EvaluatorRead",
                        }
                    )
                tool_trace.append(
                    {
                        "event": "oracle_prefetch_error",
                        "step": 0,
                        "actor": "evaluator",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise infrastructure_error(
                    f"Oracle evidence materialization failed: {type(exc).__name__}: {exc}"
                ) from exc
            for event in corpus.export_trace():
                source_stage = event.get("access_stage")
                tool_trace.append(
                    {
                        **event,
                        "event": "oracle_prefetch",
                        "step": 0,
                        "actor": "evaluator",
                        "source_access_stage": source_stage,
                        "access_stage": "EvaluatorRead",
                    }
                )
            for provided in oracle_bundle.get("_audit", []):
                tool_trace.append(
                    {
                        "event": "oracle_evidence_provided",
                        "step": 0,
                        "actor": "evaluator",
                        "access_stage": "Provided",
                        "atom_id": provided["atom_id"],
                        "path": provided["path"],
                        "paths": [provided["path"]],
                        "selector": provided["selector"],
                    }
                )
        prompt = _make_prompt(question, job, oracle_bundle)
        input_items: list[Any] = [{"role": "user", "content": prompt}]
        available_tools = [TOOL_DEFINITIONS[-1]] if oracle_bundle is not None else TOOL_DEFINITIONS
        prediction: dict[str, Any] | None = None
        status = "completed"
        error = None

        def activate_time_finalization(reason: str) -> None:
            nonlocal available_tools, input_items, time_budget_reached, time_finalization_used
            if time_finalization_used:
                return
            time_budget_reached = True
            time_finalization_used = True
            available_tools = []
            tool_trace.append(
                {
                    "event": "time_budget_finalization",
                    "step": tool_rounds,
                    "elapsed_seconds": round(monotonic() - started, 6),
                    "max_job_seconds": max_job_seconds,
                    "reason": reason,
                }
            )
            input_items = [
                *input_items,
                {
                    "role": "user",
                    "content": canonical_json(
                        {
                            "type": "time_budget_finalization",
                            "max_job_seconds": max_job_seconds,
                            "instruction": (
                                "The retrieval/reasoning time budget is exhausted. No more tools "
                                "are available. Using only evidence already observed, immediately "
                                "return your best JSON answer. Fill every field you can support and "
                                "keep the required answer/evidence envelope."
                            ),
                        }
                    ),
                },
            ]

        while True:
            now = monotonic()
            if (
                now >= reasoning_deadline
                and not time_finalization_used
                and not budget_finalization_used
            ):
                activate_time_finalization("retrieval/reasoning time budget reached")
                now = monotonic()

            idempotency_key = stable_id(
                api_format,
                job["run_id"],
                request_attempt_ordinal,
                api_round,
                format_failures,
                length=32,
            )
            try:
                finalizing = time_finalization_used or budget_finalization_used
                request: dict[str, Any] = {
                    "model": job["model"],
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "input": input_items,
                    # A forced final answer still needs enough room for providers that
                    # count hidden reasoning against this limit. Tools, not output
                    # capacity, are what distinguish the finalization request.
                    "max_output_tokens": 12_000,
                    "text": {"verbosity": "low"},
                    "store": False,
                    "extra_headers": {"Idempotency-Key": idempotency_key},
                }
                if job["reasoning_effort"] != "unspecified":
                    request["reasoning"] = {"effort": job["reasoning_effort"], "summary": "auto"}
                if not finalizing:
                    request["timeout"] = min(
                        API_REQUEST_TIMEOUT_SECONDS,
                        max(0.25, reasoning_deadline - now),
                    )
                if available_tools:
                    request["tools"] = available_tools
                    request["parallel_tool_calls"] = False
                response = (
                    create_chat_response(client, request)
                    if api_format == "chat-completions"
                    else client.responses.create(**request)
                )
            except Exception as exc:
                deadline_reached = monotonic() >= reasoning_deadline - 0.5
                if not finalizing and _is_timeout_exception(exc) and deadline_reached:
                    tool_trace.append(
                        {
                            "event": "api_call_time_limit",
                            "step": tool_rounds,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    activate_time_finalization("model request consumed reasoning budget")
                    continue
                tool_trace.append(
                    {
                        "event": "api_call_error",
                        "step": tool_rounds,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise infrastructure_error(
                    f"{'Chat Completions' if api_format == 'chat-completions' else 'Responses'} API call failed: {type(exc).__name__}: {exc}",
                    retry_status=_api_exception_retry_status(exc),
                ) from exc
            api_round += 1
            response_ids.append(str(getattr(response, "id", "")))
            resolved_models.append(str(getattr(response, "model", job["model"])))
            _merge_usage(usage, _usage_dict(response))
            output_items = list(getattr(response, "output", []) or [])
            calls = [
                item for item in output_items if getattr(item, "type", None) == "function_call"
            ]
            tool_trace.append(
                {
                    "event": "model_response",
                    "step": tool_rounds,
                    "response_id": response_ids[-1],
                    "response_status": str(getattr(response, "status", "completed")),
                    "output_items": [_response_item_dict(item) for item in output_items],
                    **({"raw_chat_response": response.raw_chat_response}
                       if api_format == "chat-completions" else {}),
                }
            )
            response_failure, response_error = _response_failure(response, output_items)
            if response_failure is not None:
                status = (
                    "time_budget_exhausted" if time_finalization_used else response_failure
                )
                error = response_error
                response_text = str(getattr(response, "output_text", "") or "")
                break
            if (
                calls
                and monotonic() >= reasoning_deadline
                and not budget_finalization_used
                and not time_finalization_used
            ):
                for call in calls:
                    tool_trace.append(
                        {
                            "event": "tool_call_rejected",
                            "step": tool_rounds + 1,
                            "tool": str(call.name),
                            "arguments_json": str(call.arguments),
                            "error": "Per-question reasoning window reached",
                        }
                    )
                activate_time_finalization("reasoning budget reached before tool execution")
                continue
            if calls:
                if budget_finalization_used or time_finalization_used:
                    status = (
                        "time_budget_exhausted"
                        if time_finalization_used
                        else "tool_budget_exhausted"
                    )
                    error = "Model requested a tool during the no-tools finalization"
                    for call in calls:
                        tool_trace.append(
                            {
                                "event": "tool_call_rejected",
                                "step": tool_rounds + 1,
                                "tool": str(call.name),
                                "arguments_json": str(call.arguments),
                                "error": error,
                            }
                        )
                    break
                # The gateway cannot round-trip either response IDs or the
                # output items it created. Preserve tool evidence as ordinary
                # user-visible transcript messages instead.
                next_input_items = list(input_items)
                for call in calls:
                    if tool_rounds >= int(retrieval["max_tool_rounds"]):
                        tool_budget_reached = True
                        tool_trace.append(
                            {
                                "event": "tool_call_rejected",
                                "step": tool_rounds + 1,
                                "tool": str(call.name),
                                "arguments_json": str(call.arguments),
                                "error": "Tool-call budget reached; call was not executed",
                            }
                        )
                        break
                    tool_rounds += 1
                    name = str(call.name)
                    arguments: dict[str, Any] | None = None
                    corpus_trace_start = len(corpus.export_trace())
                    try:
                        arguments = json.loads(call.arguments)
                        if not isinstance(arguments, dict):
                            raise CorpusQueryError("tool arguments must be an object")
                        schema = next(
                            (
                                tool["parameters"]
                                for tool in available_tools
                                if tool["name"] == name
                            ),
                            None,
                        )
                        if schema is None:
                            raise CorpusQueryError(f"Unknown tool: {name}")
                        jsonschema.validate(
                            arguments,
                            schema,
                        )
                        result, access_events = _dispatch_tool(name, arguments, corpus, retrieval)
                        tool_error = None
                        if name == "python" and int(result.get("exit_code") or 0) != 0:
                            tool_error = str(result.get("stderr") or "Python computation failed")
                    except (
                        json.JSONDecodeError,
                        jsonschema.ValidationError,
                        CorpusAccessError,
                        CorpusQueryError,
                        SandboxError,
                    ) as exc:
                        # These are attributable to model-supplied arguments or
                        # code and can be corrected in a later tool turn.
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                        access_events = corpus.export_trace()[corpus_trace_start:]
                        tool_error = result["error"]
                    except Exception as exc:
                        access_events = corpus.export_trace()[corpus_trace_start:]
                        for event in access_events:
                            tool_trace.append(
                                {"event": "corpus_access", "step": tool_rounds, **event}
                            )
                        tool_trace.append(
                            {
                                "event": "tool_infrastructure_error",
                                "step": tool_rounds,
                                "tool": name,
                                "arguments": arguments,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        raise infrastructure_error(
                            f"Tool implementation failed: {type(exc).__name__}: {exc}"
                        ) from exc
                    tool_trace.append(
                        {
                            "event": "tool_call",
                            "step": tool_rounds,
                            "tool": name,
                            "arguments": arguments,
                            "result": result,
                            "error": tool_error,
                        }
                    )
                    for event in access_events:
                        tool_trace.append({"event": "corpus_access", "step": tool_rounds, **event})
                    next_input_items.append(
                        {
                            "role": "user",
                            "content": canonical_json(
                                {
                                    "type": "tool_result",
                                    "call_id": call.call_id,
                                    "tool": name,
                                    "arguments": arguments,
                                    "output": _model_visible_tool_result(name, result),
                                    "instruction": (
                                        "Continue solving the original question using this "
                                        "tool result."
                                    ),
                                }
                            ),
                        }
                    )
                    if tool_rounds >= int(retrieval["max_tool_rounds"]):
                        tool_budget_reached = True
                if tool_budget_reached:
                    budget_finalization_used = True
                    available_tools = []
                    next_input_items.append(
                        {
                            "role": "user",
                            "content": canonical_json(
                                {
                                    "type": "tool_budget_finalization",
                                    "tool_calls_used": tool_rounds,
                                    "tool_call_limit": int(retrieval["max_tool_rounds"]),
                                    "instruction": (
                                        "The tool-call budget is exhausted. No more tools are "
                                        "available. Using only evidence already observed, return "
                                        "your best final JSON answer satisfying the required schema."
                                    ),
                                }
                            ),
                        }
                    )
                input_items = next_input_items
                continue

            response_text = str(getattr(response, "output_text", "") or "")
            try:
                candidate = json.loads(response_text)
                if isinstance(candidate, dict):
                    # Retain structured partial work for Field Accuracy even
                    # when a finalization response misses required fields.
                    prediction = candidate
                jsonschema.validate(candidate, question["answer_schema"])
                break
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                if (
                    budget_finalization_used
                    or time_finalization_used
                    or format_failures >= int(retrieval["format_retries"])
                ):
                    status = (
                        "time_budget_exhausted" if time_finalization_used else "format_error"
                    )
                    error = str(exc)
                    break
                format_failures += 1
                input_items = [
                    *input_items,
                    {
                        "role": "user",
                        "content": canonical_json(
                            {
                                "type": "format_correction",
                                "invalid_answer": response_text,
                                "validation_error": str(exc),
                                "required_final_answer_json_schema": question["answer_schema"],
                                "instruction": (
                                    "Return only one corrected JSON object satisfying the "
                                    "required schema, without Markdown or explanatory text."
                                ),
                            }
                        ),
                    },
                ]

    return {
        **job,
        "status": status,
        "prompt": prompt,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "response_text": response_text,
        "prediction": prediction,
        "tool_trace": tool_trace,
        "usage": usage,
        "latency_seconds": round(monotonic() - started, 6),
        "response_ids": response_ids,
        "resolved_models": sorted(set(resolved_models)),
        "format_retry_count": format_failures,
        "tool_budget_reached": tool_budget_reached,
        "budget_finalization_used": budget_finalization_used,
        "budget_finalization_succeeded": bool(
            budget_finalization_used and status == "completed" and prediction is not None
        ),
        "time_budget_reached": time_budget_reached,
        "time_finalization_used": time_finalization_used,
        "time_finalization_succeeded": bool(
            time_finalization_used and status == "completed" and prediction is not None
        ),
        "request_attempt_ordinal": request_attempt_ordinal,
        "request_parameters": {
            "reasoning_effort": job["reasoning_effort"],
            "max_tool_rounds": int(retrieval["max_tool_rounds"]),
            "max_output_tokens": 12_000,
            "answer_transport": "plain_json_prompt_schema_local_validation",
            "parallel_tool_calls": False,
            "store": False,
            "continuation": "stateless_plain_tool_transcript",
            "tool_budget_finalization": "one_no_tools_response",
            "max_job_seconds": max_job_seconds,
            "time_finalization_reserve_seconds": 0,
            "time_finalization_timeout": "transport_default_only",
            "time_budget_finalization": "one_no_tools_response",
            "request_attempt_ordinal": request_attempt_ordinal,
            **_transport_parameters(job),
        },
        "error": error,
    }


def _run_manifest_locked(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    index_path: str | Path,
    questions_path: str | Path,
    scopes_path: str | Path,
    jobs_path: str | Path,
    results_path: str | Path,
    gold_path: str | Path | None = None,
    max_jobs: int | None = None,
    workers: int = 1,
    confirm_paid_run: bool = False,
    reset_circuit_breaker: bool = False,
    retry_exhausted: bool = False,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not confirm_paid_run:
        raise PermissionError(
            "Model execution is a paid external action. Re-run with explicit confirm_paid_run=True."
        )
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive when provided")
    if workers <= 0:
        raise ValueError("workers must be positive")
    retrieval = _retrieval_settings(config)
    question_rows = read_jsonl(Path(questions_path))
    scope_rows = read_jsonl(Path(scopes_path))
    jobs = read_jsonl(Path(jobs_path))
    gold_rows = read_gold_records(Path(gold_path)) if gold_path is not None else []
    _verify_manifest_binding(
        config,
        raw_root=raw_root,
        index_path=index_path,
        questions_path=questions_path,
        scopes_path=scopes_path,
        jobs_path=jobs_path,
        gold_path=gold_path,
        question_rows=question_rows,
        job_rows=jobs,
    )
    questions, scopes, gold = _preflight(
        question_rows=question_rows,
        scope_rows=scope_rows,
        job_rows=jobs,
        gold_rows=gold_rows,
    )
    store = RunStore.prepare(
        jobs,
        results_path,
        max_jobs=max_jobs,
        reset_circuit_breaker=reset_circuit_breaker,
        retry_exhausted=retry_exhausted,
    )
    if not store.pending:
        return {**store.summary(), "workers": workers}
    store.ensure_circuit_closed()

    try:
        api_key = os.environ[api_key_env]
    except KeyError as exc:
        raise OSError(f"{api_key_env} is not set") from exc
    if not api_key:
        raise OSError(f"{api_key_env} is empty")

    inventory = build_inventory(raw_root)
    _preflight_corpus_inputs(
        inventory=inventory,
        jobs=store.pending,
        scopes=scopes,
        gold=gold,
    )
    with CorpusIndex(inventory, index_path, rebuild=False) as index:
        _preflight_oracle_evidence(
            inventory=inventory,
            index=index,
            config=config,
            jobs=store.pending,
            scopes=scopes,
            gold=gold,
        )
    from openai import OpenAI

    client_options: dict[str, Any] = {
        "api_key": api_key,
        # Per-request SDK retries can multiply a nominal timeout and violate
        # the per-question retrieval/reasoning budget. Logical-job retries
        # below remain responsible for transient 429/5xx failures.
        "max_retries": 0,
        "timeout": API_REQUEST_TIMEOUT_SECONDS,
    }
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)

    next_request_ordinals = dict(store.request_attempt_ordinals)

    def execute(
        job: dict[str, Any], job_index: CorpusIndex, ordinal: int
    ) -> tuple[dict[str, Any], Exception | None]:
        question = questions[job["qa_id"]]
        try:
            return (
                run_one(
                    client=client,
                    inventory=inventory,
                    index=job_index,
                    config=config,
                    question=question,
                    job=job,
                    scope=_scope_for_job(scopes, job),
                    gold=gold.get(job["qa_id"]),
                    request_attempt_ordinal=ordinal,
                ),
                None,
            )
        except RunInfrastructureError as exc:
            return (
                _infrastructure_result(
                    job,
                    retrieval,
                    exc,
                    prompt=exc.prompt,
                    response_text=exc.response_text,
                    tool_trace=exc.tool_trace,
                    usage=exc.usage,
                    elapsed_seconds=exc.elapsed_seconds,
                    response_ids=exc.response_ids,
                    resolved_models=exc.resolved_models,
                    format_retry_count=exc.format_retry_count,
                    request_attempt_ordinal=ordinal,
                ),
                None,
            )
        except Exception as exc:
            return (
                _infrastructure_result(
                    job,
                    retrieval,
                    exc,
                    tool_trace=[
                        {
                            "event": "runner_unexpected_error",
                            "step": 0,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    request_attempt_ordinal=ordinal,
                ),
                exc,
            )

    total = len(store.pending)

    def report_result(position: int, result: dict[str, Any]) -> None:
        if progress is not None:
            progress(
                f"[{position}/{total} attempt "
                f"{store.attempt_counts[result['run_id']]}/{MAX_ATTEMPTS_PER_RUN}] "
                f"{result['qa_id']} "
                f"{result['status']} ({float(result.get('latency_seconds') or 0.0):.1f}s)"
            )

    def should_retry(job: dict[str, Any], result: dict[str, Any]) -> bool:
        if result.get("status") not in RETRYABLE_RUN_STATUSES:
            return False
        run_id = job["run_id"]
        if result["status"] == "api_error":
            next_request_ordinals[run_id] += 1
        return store.attempt_counts[run_id] < MAX_ATTEMPTS_PER_RUN

    def retry_delay(job: dict[str, Any]) -> float:
        completed_attempts = store.attempt_counts[job["run_id"]]
        index = min(max(completed_attempts - 1, 0), len(JOB_RETRY_BACKOFF_SECONDS) - 1)
        return JOB_RETRY_BACKOFF_SECONDS[index]

    def announce_retry(jobs: list[dict[str, Any]], delay: float) -> None:
        if progress is not None:
            label = "job" if len(jobs) == 1 else "jobs"
            progress(f"Retrying {len(jobs)} {label} after {delay:.0f}s")

    with store.writer() as handle:
        if workers == 1:
            with CorpusIndex(inventory, index_path, rebuild=False) as job_index:
                for position, job in enumerate(store.pending, start=1):
                    while True:
                        result, unexpected = execute(
                            job,
                            job_index,
                            next_request_ordinals[job["run_id"]],
                        )
                        # A retryable API outcome is not a circuit-breaker
                        # failure until this logical job exhausts its attempts.
                        store.append_attempt(handle, result, enforce_circuit_breaker=False)
                        report_result(position, result)
                        if unexpected is not None:
                            raise unexpected
                        if not should_retry(job, result):
                            store.ensure_runtime_circuit_closed()
                            break
                        delay = retry_delay(job)
                        announce_retry([job], delay)
                        time.sleep(delay)
        else:
            pool_size = min(workers, total)
            if progress is not None:
                progress(f"Starting {total} evaluation jobs with {pool_size} workers")
            with ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="powerbench-run",
            ) as executor:

                def execute_isolated(
                    item: tuple[int, dict[str, Any]],
                ) -> tuple[dict[str, Any], Exception | None]:
                    _position, job = item
                    with CorpusIndex(inventory, index_path, rebuild=False) as job_index:
                        return execute(
                            job,
                            job_index,
                            next_request_ordinals[job["run_id"]],
                        )

                for batch_start in range(0, total, pool_size):
                    batch = store.pending[batch_start : batch_start + pool_size]
                    retry_items = [
                        (batch_start + offset, job) for offset, job in enumerate(batch, start=1)
                    ]
                    while retry_items:
                        outcomes = list(executor.map(execute_isolated, retry_items))
                        unexpected_errors: list[Exception] = []
                        next_retry_items: list[tuple[int, dict[str, Any]]] = []
                        retry_delays: list[float] = []
                        for (position, job), (result, unexpected) in zip(
                            retry_items, outcomes, strict=True
                        ):
                            # All jobs in this wave have already contacted the
                            # paid service, so persist every outcome before retrying.
                            store.append_attempt(handle, result, enforce_circuit_breaker=False)
                            report_result(position, result)
                            if unexpected is not None:
                                unexpected_errors.append(unexpected)
                            elif should_retry(job, result):
                                next_retry_items.append((position, job))
                                retry_delays.append(retry_delay(job))
                        if unexpected_errors:
                            raise unexpected_errors[0]
                        if not next_retry_items:
                            break
                        delay = max(retry_delays)
                        announce_retry([job for _position, job in next_retry_items], delay)
                        time.sleep(delay)
                        retry_items = next_retry_items
                    store.ensure_runtime_circuit_closed()
    return {**store.summary(), "workers": workers}


def run_manifest(
    config: PilotConfig,
    *,
    raw_root: str | Path,
    index_path: str | Path,
    questions_path: str | Path,
    scopes_path: str | Path,
    jobs_path: str | Path,
    results_path: str | Path,
    gold_path: str | Path | None = None,
    max_jobs: int | None = None,
    workers: int = 1,
    confirm_paid_run: bool = False,
    reset_circuit_breaker: bool = False,
    retry_exhausted: bool = False,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not confirm_paid_run:
        raise PermissionError(
            "Model execution is a paid external action. Re-run with explicit confirm_paid_run=True."
        )
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive when provided")
    if workers <= 0:
        raise ValueError("workers must be positive")
    with exclusive_results_lock(results_path):
        return _run_manifest_locked(
            config,
            raw_root=raw_root,
            index_path=index_path,
            questions_path=questions_path,
            scopes_path=scopes_path,
            jobs_path=jobs_path,
            results_path=results_path,
            gold_path=gold_path,
            max_jobs=max_jobs,
            workers=workers,
            confirm_paid_run=True,
            reset_circuit_breaker=reset_circuit_breaker,
            retry_exhausted=retry_exhausted,
            base_url=base_url,
            api_key_env=api_key_env,
            progress=progress,
        )