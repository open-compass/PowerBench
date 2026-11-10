from __future__ import annotations

from typing import Any

SYSTEM_INSTRUCTIONS = """You are taking the PowerBench autonomous heterogeneous retrieval and reasoning evaluation.
Use only the supplied read-only corpus tools and the data they return. Do not use outside knowledge or guess missing values.
Independently decide which corpus sources to locate, read, and combine and how to derive the requested result.
The user prompt contains the authoritative JSON Schema for the final answer. Return only one JSON object satisfying that schema, without Markdown fences or explanatory text.
In evidence, cite the raw-relative paths actually used. Do not reveal hidden reasoning or chain-of-thought.
"""


def _function_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        },
    }


TOOL_DEFINITIONS = [
    _function_tool(
        "list_corpus",
        "List files or directories visible in this question's device scope.",
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    ),
    _function_tool(
        "search_corpus",
        "Search visible raw paths, Markdown/JSON text, and CSV headers.",
        {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    ),
    _function_tool(
        "read_file",
        "Read visible Markdown or JSON lines. Use read_timeseries for CSV.",
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1},
        },
    ),
    _function_tool(
        "read_timeseries",
        'Materialize device CSV fields over [start,end) and return a Python handle. Use CSV field names, not natural-language metric names. Rows are available in python as handles[returned_handle]["rows"].',
        {
            "device_uid": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "start": {"type": "string"},
            "end": {"type": "string"},
        },
    ),
    _function_tool(
        "python",
        'Run restricted Python over materialized tables. Access a returned handle H as handles[H]["rows"]; use direct approved functions and print(json_dumps(...)).',
        {"code": {"type": "string", "minLength": 1}},
    ),
]

__all__ = ["SYSTEM_INSTRUCTIONS", "TOOL_DEFINITIONS"]