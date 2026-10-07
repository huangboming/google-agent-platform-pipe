import asyncio
from typing import AsyncIterator, Callable, Dict, Any

from .grounding import emit_grounding
from .responses import add_usage_metadata
from .tools import execute_tool


async def run_agent(
    model: str,
    payload: Dict[str, Any],
    tools: dict,
    max_tool_iterations: int,
    event_emitter,
    stream_generate: Callable,
) -> AsyncIterator[Dict[str, Any]]:
    usage_metadata = {}

    for iteration in range(max_tool_iterations):
        model_parts = []
        function_calls = []
        prompt_feedback = None
        grounding_metadata = {}
        round_usage = {}

        async for response in stream_generate(model, payload):
            if response.get("promptFeedback"):
                prompt_feedback = response["promptFeedback"]
            if isinstance(response.get("usageMetadata"), dict):
                round_usage.update(response["usageMetadata"])

            candidates = response.get("candidates") or []
            if not candidates:
                continue
            candidate = candidates[0]
            if isinstance(candidate.get("groundingMetadata"), dict):
                grounding_metadata = candidate["groundingMetadata"]
            for part in (candidate.get("content") or {}).get("parts", []):
                model_parts.append(part)
                if "text" in part:
                    event_type = "reasoning" if part.get("thought") else "content"
                    yield {"type": event_type, "text": part["text"]}
                elif "inlineData" in part:
                    yield {
                        "type": "content",
                        "text": _render_inline_data(part["inlineData"]),
                    }
                if "functionCall" in part:
                    function_calls.append(part["functionCall"])

        usage_metadata = add_usage_metadata(usage_metadata, round_usage)
        if not function_calls:
            if not model_parts and prompt_feedback:
                reason = prompt_feedback.get("blockReason", "unknown reason")
                raise RuntimeError(f"Agent Platform blocked the prompt: {reason}")
            await emit_grounding(grounding_metadata, event_emitter)
            if usage_metadata:
                yield {"type": "usage", "metadata": usage_metadata}
            return

        if iteration + 1 >= max_tool_iterations:
            raise RuntimeError("Maximum tool-call iterations reached.")

        await _emit_status(
            event_emitter,
            "Running "
            + ", ".join(call.get("name", "unknown tool") for call in function_calls),
            False,
        )
        responses = await asyncio.gather(
            *(execute_tool(call, tools) for call in function_calls)
        )
        await _emit_status(event_emitter, "Tool execution complete", True)

        payload["contents"].append({"role": "model", "parts": model_parts})
        payload["contents"].append({"role": "user", "parts": responses})

    raise RuntimeError("Maximum tool-call iterations reached.")


def _render_inline_data(inline_data: dict) -> str:
    mime_type = inline_data.get("mimeType", "application/octet-stream")
    data = inline_data.get("data", "")
    if mime_type.startswith("image/"):
        return f"\n![Generated image](data:{mime_type};base64,{data})\n"
    return f"\n[Generated {mime_type}](data:{mime_type};base64,{data})\n"


async def _emit_status(emitter, description: str, done: bool) -> None:
    if not emitter:
        return
    try:
        await emitter(
            {
                "type": "status",
                "data": {"description": description, "done": done},
            }
        )
    except Exception:
        # UI status delivery must not fail the model request.
        pass
