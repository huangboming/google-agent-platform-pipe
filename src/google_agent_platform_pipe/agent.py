import asyncio
from typing import Any, AsyncIterator, Callable, Dict, Optional

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
    tool_approval_required: bool,
) -> AsyncIterator[Dict[str, Any]]:
    usage_metadata = {}

    for iteration in range(max_tool_iterations):
        model_parts = []
        function_calls = []
        prompt_feedback = None
        finish_reason = None
        finish_message = None
        has_answer = False
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
            if candidate.get("finishReason"):
                finish_reason = candidate["finishReason"]
                finish_message = candidate.get("finishMessage")
            if isinstance(candidate.get("groundingMetadata"), dict):
                grounding_metadata = candidate["groundingMetadata"]
            for part in (candidate.get("content") or {}).get("parts", []):
                model_parts.append(part)
                if "text" in part:
                    is_thought = bool(part.get("thought"))
                    event_type = "reasoning" if is_thought else "content"
                    if not is_thought and part["text"]:
                        has_answer = True
                    yield {"type": event_type, "text": part["text"]}
                elif "inlineData" in part:
                    has_answer = True
                    yield {
                        "type": "content",
                        "text": _render_inline_data(part["inlineData"]),
                    }
                if "functionCall" in part:
                    function_calls.append(part["functionCall"])

        usage_metadata = add_usage_metadata(usage_metadata, round_usage)
        if not function_calls:
            failure = _generation_failure(
                prompt_feedback, finish_reason, finish_message, has_answer
            )
            if failure:
                raise RuntimeError(failure)
            await emit_grounding(grounding_metadata, event_emitter)
            if usage_metadata:
                yield {"type": "usage", "metadata": usage_metadata}
            return

        if tool_approval_required:
            raise RuntimeError(
                "Tool execution requires approval, but this Pipe does not support "
                "Open WebUI's ask mode. No tools were executed."
            )

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


def _generation_failure(
    prompt_feedback: Optional[dict],
    finish_reason: Optional[str],
    finish_message: Optional[str],
    has_answer: bool,
) -> Optional[str]:
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        reason = prompt_feedback["blockReason"]
        message = prompt_feedback.get("blockReasonMessage")
        detail = f"{reason}: {message}" if message else reason
        return f"Agent Platform blocked the prompt: {detail}"

    successful_reasons = {None, "STOP", "FINISH_REASON_UNSPECIFIED"}
    if finish_reason in successful_reasons:
        return None
    if finish_reason == "MAX_TOKENS" and has_answer:
        return None

    detail = f"{finish_reason}: {finish_message}" if finish_message else finish_reason
    return f"Agent Platform stopped generation: {detail}"


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
