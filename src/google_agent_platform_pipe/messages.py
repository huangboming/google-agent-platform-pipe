import base64
import binascii
import copy
import re
from typing import Any, Dict


_DATA_URL_PATTERN = re.compile(
    r"data:([^;,]+);base64,(.*)", flags=re.IGNORECASE | re.DOTALL
)
_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}


def _decode_data_url(value: str):
    match = _DATA_URL_PATTERN.fullmatch(value)
    if not match:
        return None, 0
    try:
        decoded = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 media in a chat message.") from exc
    return {
        "inlineData": {
            "mimeType": match.group(1).lower(),
            "data": match.group(2),
        }
    }, len(decoded)


def message_parts(content: Any):
    if isinstance(content, str):
        return ([{"text": content}] if content else []), 0
    if not isinstance(content, list):
        return ([{"text": str(content)}] if content is not None else []), 0

    parts = []
    used_bytes = 0
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type in ("text", "input_text"):
            parts.append({"text": str(item.get("text", ""))})
            continue
        if item_type not in ("image_url", "input_image", "file"):
            continue

        source = item.get("image_url") or item.get("file") or item.get("url")
        if isinstance(source, dict):
            source = source.get("url") or source.get("data")
        if not isinstance(source, str):
            continue

        inline, size = _decode_data_url(source)
        if inline:
            parts.append(inline)
            used_bytes += size
        elif source.startswith("gs://"):
            mime_type = item.get("mime_type") or item.get("media_type")
            if mime_type:
                parts.append({"fileData": {"mimeType": mime_type, "fileUri": source}})
        else:
            # Agent Platform cannot consume arbitrary HTTP URLs as fileData.
            parts.append({"text": f"[Media URL: {source}]"})
    return parts, used_bytes


def restore_pre_source_messages(messages: list, metadata: Any) -> list:
    """Restore prompts captured before Open WebUI adds RAG source context."""
    if not isinstance(metadata, dict) or not metadata.get("sources"):
        return messages
    if metadata.get("user_prompt") is None:
        raise ValueError(
            "Open WebUI added source context without preserving the original prompt. "
            "Disable File Context for this model."
        )

    restored = copy.deepcopy(messages)
    original_user = metadata["user_prompt"]
    for message in reversed(restored):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in (
                    "text",
                    "input_text",
                ):
                    item["text"] = original_user
                    break
            else:
                content.insert(0, {"type": "text", "text": original_user})
        else:
            message["content"] = original_user
        break

    if "system_prompt" in metadata:
        original_system = metadata.get("system_prompt")
        without_system = [
            message for message in restored if message.get("role") != "system"
        ]
        if original_system:
            without_system.insert(
                0, {"role": "system", "content": original_system}
            )
        restored = without_system

    return restored


def generation_config(
    body: dict, thinking_level: str, include_thoughts: bool
) -> Dict[str, Any]:
    config = {}
    mappings = {
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "max_tokens": "maxOutputTokens",
        "max_completion_tokens": "maxOutputTokens",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "seed": "seed",
    }
    for source, target in mappings.items():
        if body.get(source) is not None:
            config[target] = body[source]

    stop = body.get("stop")
    if stop:
        config["stopSequences"] = [stop] if isinstance(stop, str) else stop

    thinking_level = thinking_level.strip().upper()
    if thinking_level and thinking_level not in _THINKING_LEVELS:
        raise ValueError(
            "THINKING_LEVEL must be MINIMAL, LOW, MEDIUM, HIGH, or empty."
        )

    thinking_config = {}
    if thinking_level:
        thinking_config["thinkingLevel"] = thinking_level
    if include_thoughts:
        thinking_config["includeThoughts"] = True
    if thinking_config:
        config["thinkingConfig"] = thinking_config
    return config
