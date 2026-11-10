from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, MutableSequence
from dataclasses import dataclass
from typing import Any


class SandboxError(ValueError):
    """Base class for safe-computation input and policy errors."""


class SandboxValidationError(SandboxError):
    """Raised when model-supplied Python is outside the allowed subset."""


class InvalidHandlesError(SandboxError):
    """Raised when materialized handles are not bounded JSON data."""


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    output_truncated: bool
    rejected: bool
    duration_ms: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.rejected

    def as_dict(self) -> dict[str, object]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "rejected": self.rejected,
            "duration_ms": self.duration_ms,
        }


_BANNED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "super",
    "type",
    "vars",
}

_SAFE_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "dict_get",
    "dict_items",
    "dict_keys",
    "dict_values",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "json_dumps",
    "len",
    "list",
    "map",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.Starred,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Lambda,
    ast.Call,
    ast.keyword,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.JoinedStr,
    ast.FormattedValue,
)


class _PolicyValidator(ast.NodeVisitor):
    def __init__(self, *, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.node_count = 0
        self.defined_functions: set[str] = set()

    def validate(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_identifier(node.name, "function name")
                self.defined_functions.add(node.name)
        self.visit(tree)

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > self.max_nodes:
            raise SandboxValidationError(f"Code exceeds the AST node limit ({self.max_nodes})")
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise SandboxValidationError(
                f"Syntax {type(node).__name__} is not allowed in the computation sandbox"
            )
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # pragma: no cover - direct dispatch
        raise SandboxValidationError("Attribute access is not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        self._check_identifier(node.id, "identifier")
        if node.id in _BANNED_NAMES:
            raise SandboxValidationError(f"Name {node.id!r} is forbidden")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_identifier(node.arg, "argument")
        if node.annotation is not None:
            raise SandboxValidationError("Function annotations are not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise SandboxValidationError(f"Constant type {type(value).__name__} is not allowed")
        if isinstance(value, str):
            if "__" in value:
                raise SandboxValidationError("Dunder strings are not allowed")
            if len(value) > 10_000:
                raise SandboxValidationError("String literal exceeds 10,000 characters")
        elif isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 256:
            raise SandboxValidationError("Integer literal exceeds 256 bits")
        elif isinstance(value, float) and not math.isfinite(value):
            raise SandboxValidationError("Non-finite float literals are not allowed")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise SandboxValidationError("Function decorators are not allowed")
        if node.returns is not None or node.type_comment is not None:
            raise SandboxValidationError("Function annotations are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise SandboxValidationError("Only direct calls to approved functions are allowed")
        name = node.func.id
        self._check_identifier(name, "called function")
        if name in _BANNED_NAMES:
            raise SandboxValidationError(f"Call to {name!r} is forbidden")
        if name not in _SAFE_CALLS and name not in self.defined_functions:
            raise SandboxValidationError(f"Call to unapproved function {name!r}")
        if any(keyword.arg is None for keyword in node.keywords):
            raise SandboxValidationError("Keyword unpacking is not allowed")
        if name == "print":
            unexpected = {
                keyword.arg for keyword in node.keywords if keyword.arg not in {"sep", "end"}
            }
            if unexpected:
                raise SandboxValidationError(f"Unsupported print keyword(s): {sorted(unexpected)}")
        self.generic_visit(node)

    @staticmethod
    def _check_identifier(value: str, label: str) -> None:
        if "__" in value:
            raise SandboxValidationError(f"Dunder {label}s are not allowed: {value!r}")


def validate_code(code: str, *, max_code_chars: int = 20_000, max_ast_nodes: int = 2_000) -> None:
    """Validate model Python against the small JSON-computation language."""

    if not isinstance(code, str):
        raise SandboxValidationError("Code must be text")
    if len(code) > max_code_chars:
        raise SandboxValidationError(f"Code exceeds {max_code_chars} characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxValidationError(f"Invalid Python syntax: {exc.msg}") from exc
    _PolicyValidator(max_nodes=max_ast_nodes).validate(tree)


_CHILD_RUNNER = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
limit = int(payload["max_output_chars"])
chunks = []
written = 0
truncated = False

def emit(text):
    global written, truncated
    text = str(text)
    remaining = limit - written
    if remaining <= 0:
        truncated = True
        return
    if len(text) > remaining:
        chunks.append(text[:remaining])
        written += remaining
        truncated = True
    else:
        chunks.append(text)
        written += len(text)

def safe_print(*values, sep=" ", end="\n"):
    if not isinstance(sep, str) or not isinstance(end, str):
        raise TypeError("print sep and end must be strings")
    emit(sep.join(str(value) for value in values) + end)

def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def dict_get(value, key, default=None):
    if not isinstance(value, dict):
        raise TypeError("dict_get expects a dictionary")
    return value.get(key, default)

def dict_items(value):
    if not isinstance(value, dict):
        raise TypeError("dict_items expects a dictionary")
    return list(value.items())

def dict_keys(value):
    if not isinstance(value, dict):
        raise TypeError("dict_keys expects a dictionary")
    return list(value.keys())

def dict_values(value):
    if not isinstance(value, dict):
        raise TypeError("dict_values expects a dictionary")
    return list(value.values())

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "dict_get": dict_get, "dict_items": dict_items, "dict_keys": dict_keys,
    "dict_values": dict_values, "enumerate": enumerate, "filter": filter,
    "float": float, "int": int, "isinstance": isinstance,
    "json_dumps": json_dumps, "len": len, "list": list, "map": map,
    "max": max, "min": min, "print": safe_print, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}
namespace = {"__builtins__": safe_builtins, "handles": payload["handles"]}
error = ""
model_exit_code = 0
try:
    compiled = compile(payload["code"], "<model-computation>", "exec")
    exec(compiled, namespace, namespace)
except BaseException as exc:
    model_exit_code = 1
    error = type(exc).__name__ + ": " + str(exc)
    if len(error) > limit:
        error = error[:limit]
        truncated = True

response = {
    "stdout": "".join(chunks),
    "stderr": error,
    "exit_code": model_exit_code,
    "output_truncated": truncated,
}
sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
"""


TraceSink = MutableSequence[dict[str, Any]] | Callable[[dict[str, Any]], None]


def _emit_trace(trace: TraceSink | None, event: dict[str, Any]) -> None:
    if trace is None:
        return
    if callable(trace):
        trace(dict(event))
    else:
        trace.append(dict(event))


def _handle_labels(handles: Any) -> list[str]:
    if isinstance(handles, Mapping):
        return sorted(str(key) for key in handles)
    if isinstance(handles, list):
        labels = []
        for index, item in enumerate(handles):
            if isinstance(item, Mapping) and "handle_id" in item:
                labels.append(str(item["handle_id"]))
            else:
                labels.append(str(index))
        return labels
    return []


def _validated_payload(
    code: str,
    handles: Any,
    *,
    max_output_chars: int,
    max_input_bytes: int,
) -> str:
    if not isinstance(handles, (Mapping, list)):
        raise InvalidHandlesError("Handles must be a JSON object or list")
    if isinstance(handles, Mapping) and any(not isinstance(key, str) for key in handles):
        raise InvalidHandlesError("Handle identifiers must be strings")
    try:
        payload = json.dumps(
            {"code": code, "handles": handles, "max_output_chars": max_output_chars},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidHandlesError("Handles must contain only finite JSON values") from exc
    if len(payload.encode("utf-8")) > max_input_bytes:
        raise InvalidHandlesError(f"Sandbox input exceeds {max_input_bytes} bytes")
    return payload


def _resource_limiter(*, timeout_seconds: float, memory_limit_mb: int) -> Callable[[], None]:
    def apply_limits() -> None:
        try:
            import resource

            cpu_seconds = max(1, math.ceil(timeout_seconds))
            limits = (
                (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
                (resource.RLIMIT_FSIZE, (0, 0)),
                (resource.RLIMIT_CORE, (0, 0)),
                (resource.RLIMIT_NOFILE, (16, 16)),
            )
            if hasattr(resource, "RLIMIT_AS"):
                memory_bytes = memory_limit_mb * 1024 * 1024
                limits += ((resource.RLIMIT_AS, (memory_bytes, memory_bytes)),)
            for resource_kind, value in limits:
                with contextlib.suppress(OSError, ValueError):
                    resource.setrlimit(resource_kind, value)
            os.umask(0o077)
        except (ImportError, OSError, ValueError):
            pass

    return apply_limits


def run_python(
    code: str,
    handles: Any,
    *,
    timeout_seconds: float = 8.0,
    max_output_chars: int = 12_000,
    max_input_bytes: int = 4_000_000,
    memory_limit_mb: int = 256,
    trace: TraceSink | None = None,
) -> SandboxResult:
    """Run approved Python over already-materialized JSON handles.

    The child receives no corpus paths, Gold data, ambient environment, site
    packages or import capability. Rejected code is returned as a structured
    result with exit code 2 and is still traceable.
    """

    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise SandboxError("timeout_seconds must be numeric")
    if not math.isfinite(float(timeout_seconds)) or not 0 < timeout_seconds <= 60:
        raise SandboxError("timeout_seconds must be in (0, 60]")
    for value, name in (
        (max_output_chars, "max_output_chars"),
        (max_input_bytes, "max_input_bytes"),
        (memory_limit_mb, "memory_limit_mb"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SandboxError(f"{name} must be a positive integer")
    if memory_limit_mb < 64:
        raise SandboxError("memory_limit_mb must be at least 64")

    started = time.monotonic()
    code_hash = hashlib.sha256(code.encode("utf-8") if isinstance(code, str) else b"").hexdigest()
    payload = _validated_payload(
        code,
        handles,
        max_output_chars=max_output_chars,
        max_input_bytes=max_input_bytes,
    )

    try:
        validate_code(code)
    except SandboxValidationError as exc:
        result = SandboxResult(
            stdout="",
            stderr=f"SandboxValidationError: {exc}",
            exit_code=2,
            timed_out=False,
            output_truncated=False,
            rejected=True,
            duration_ms=max(0, round((time.monotonic() - started) * 1000)),
        )
        _emit_trace(trace, _trace_event(result, code_hash, code, handles))
        return result
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    preexec_fn = (
        _resource_limiter(timeout_seconds=float(timeout_seconds), memory_limit_mb=memory_limit_mb)
        if os.name == "posix"
        else None
    )

    with tempfile.TemporaryDirectory(prefix="powerbench-sandbox-") as temporary:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _CHILD_RUNNER],
            cwd=temporary,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            close_fds=True,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
        try:
            child_stdout, child_stderr = process.communicate(
                payload, timeout=float(timeout_seconds)
            )
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            child_stdout, child_stderr = process.communicate()
            result = SandboxResult(
                stdout="",
                stderr=(child_stderr or "Sandbox execution timed out")[:max_output_chars],
                exit_code=None,
                timed_out=True,
                output_truncated=len(child_stderr or "") > max_output_chars,
                rejected=False,
                duration_ms=max(0, round((time.monotonic() - started) * 1000)),
            )
            _emit_trace(trace, _trace_event(result, code_hash, code, handles))
            return result

    result = _decode_child_result(
        child_stdout,
        child_stderr,
        process.returncode,
        max_output_chars=max_output_chars,
        duration_ms=max(0, round((time.monotonic() - started) * 1000)),
    )
    _emit_trace(trace, _trace_event(result, code_hash, code, handles))
    return result


def _decode_child_result(
    child_stdout: str,
    child_stderr: str,
    process_exit_code: int,
    *,
    max_output_chars: int,
    duration_ms: int,
) -> SandboxResult:
    try:
        payload = json.loads(child_stdout)
        if not isinstance(payload, dict):
            raise TypeError("child response is not an object")
        stdout = str(payload["stdout"])
        stderr = str(payload["stderr"])
        exit_code = int(payload["exit_code"])
        truncated = bool(payload["output_truncated"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        diagnostics = child_stderr or f"Invalid sandbox child response: {exc}"
        return SandboxResult(
            stdout="",
            stderr=diagnostics[:max_output_chars],
            exit_code=process_exit_code,
            timed_out=False,
            output_truncated=len(diagnostics) > max_output_chars,
            rejected=False,
            duration_ms=duration_ms,
        )

    if child_stderr:
        stderr = f"{stderr}\n{child_stderr}".strip()
    stdout_was_long = len(stdout) > max_output_chars
    stderr_was_long = len(stderr) > max_output_chars
    return SandboxResult(
        stdout=stdout[:max_output_chars],
        stderr=stderr[:max_output_chars],
        exit_code=exit_code if process_exit_code == 0 else process_exit_code,
        timed_out=False,
        output_truncated=truncated or stdout_was_long or stderr_was_long,
        rejected=False,
        duration_ms=duration_ms,
    )


def _trace_event(
    result: SandboxResult,
    code_hash: str,
    code: str,
    handles: Any,
) -> dict[str, Any]:
    return {
        "tool": "python",
        "code": code,
        "code_sha256": code_hash,
        "handle_ids": _handle_labels(handles),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "rejected": result.rejected,
        "output_truncated": result.output_truncated,
        "stdout_chars": len(result.stdout),
        "stderr_chars": len(result.stderr),
        "duration_ms": result.duration_ms,
    }