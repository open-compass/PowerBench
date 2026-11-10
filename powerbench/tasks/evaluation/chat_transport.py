from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def create_chat_response(client: Any, request: dict[str, Any]) -> Any:
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputRefusal,
        ResponseOutputText,
    )

    payload: dict[str, Any] = {
        "model": request["model"],
        "messages": [
            {"role": "system", "content": request["instructions"]},
            *request["input"],
        ],
        "max_tokens": request["max_output_tokens"],
    }
    for key in ("timeout", "extra_headers"):
        if key in request:
            payload[key] = request[key]
    if request.get("reasoning"):
        payload["reasoning_effort"] = request["reasoning"]["effort"]
    if request.get("tools"):
        payload["tools"] = [
            {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
            for tool in request["tools"]
        ]
        payload["parallel_tool_calls"] = False
    else:
        payload["tool_choice"] = "none"

    raw = client.chat.completions.create(**payload)
    if len(raw.choices) != 1:
        raise ValueError("Chat Completions must return exactly one choice")
    choice = raw.choices[0]
    message = choice.message
    output = []
    content = []
    if message.content:
        content.append(ResponseOutputText(type="output_text", text=message.content, annotations=[]))
    if message.refusal:
        content.append(ResponseOutputRefusal(type="refusal", refusal=message.refusal))
    if content:
        output.append(ResponseOutputMessage(
            id=raw.id + "-message", type="message", role="assistant",
            status="completed", content=content,
        ))
    for call in message.tool_calls or []:
        if call.type != "function":
            raise ValueError(f"Unsupported chat tool call type: {call.type}")
        output.append(ResponseFunctionToolCall(
            type="function_call", call_id=call.id,
            name=call.function.name, arguments=call.function.arguments,
        ))
    finish = choice.finish_reason
    if finish in {"stop", "tool_calls"}:
        status, details = "completed", None
    elif finish in {"length", "content_filter"}:
        status, details = "incomplete", {"reason": finish}
    else:
        status, details = "failed", {"reason": f"Unexpected finish_reason: {finish}"}
    usage = raw.usage
    return SimpleNamespace(
        id=raw.id, model=raw.model, output=output, output_text=message.content or "",
        status=status, incomplete_details=details, error=details,
        usage=None if usage is None else SimpleNamespace(
            input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        ),
        raw_chat_response=raw.model_dump(mode="json"),
    )