import time
import uuid
from typing import Any, AsyncIterator, Dict

_USAGE_FIELDS = (
    "promptTokenCount",
    "candidatesTokenCount",
    "totalTokenCount",
    "cachedContentTokenCount",
    "thoughtsTokenCount",
    "toolUsePromptTokenCount",
)


def add_usage_metadata(total: dict, current: dict) -> dict:
    merged = dict(total)
    for field in _USAGE_FIELDS:
        value = current.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[field] = merged.get(field, 0) + int(value)
    return merged


def openai_usage(metadata: dict) -> dict:
    prompt_tokens = metadata.get("promptTokenCount", 0) + metadata.get(
        "toolUsePromptTokenCount", 0
    )
    completion_tokens = metadata.get("candidatesTokenCount", 0) + metadata.get(
        "thoughtsTokenCount", 0
    )
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": metadata.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        ),
    }

    if "cachedContentTokenCount" in metadata:
        usage["prompt_tokens_details"] = {
            "cached_tokens": metadata["cachedContentTokenCount"]
        }
    if "thoughtsTokenCount" in metadata:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": metadata["thoughtsTokenCount"]
        }
    return usage


def _response_identity():
    return f"chatcmpl-{uuid.uuid4().hex}", int(time.time())


def _stream_chunk(response_id: str, created: int, model: str, delta: dict) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


async def stream_openai_response(
    events: AsyncIterator[Dict[str, Any]], model: str
) -> AsyncIterator[dict]:
    response_id, created = _response_identity()
    async for event in events:
        event_type = event.get("type")
        if event_type == "content":
            yield _stream_chunk(
                response_id, created, model, {"content": event.get("text", "")}
            )
        elif event_type == "reasoning":
            yield _stream_chunk(
                response_id,
                created,
                model,
                {"reasoning_content": event.get("text", "")},
            )
        elif event_type == "usage":
            yield {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": openai_usage(event.get("metadata", {})),
            }


async def collect_openai_response(
    events: AsyncIterator[Dict[str, Any]], model: str
) -> dict:
    content = []
    reasoning = []
    usage = None

    async for event in events:
        event_type = event.get("type")
        if event_type == "content":
            content.append(event.get("text", ""))
        elif event_type == "reasoning":
            reasoning.append(event.get("text", ""))
        elif event_type == "usage":
            usage = openai_usage(event.get("metadata", {}))

    response_id, created = _response_identity()
    message = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)

    response = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "logprobs": None, "finish_reason": "stop"}
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response
