import json
from typing import Any, AsyncIterator, Dict, List
from urllib.parse import quote

import httpx


class AgentPlatformClient:
    async def stream_generate(
        self,
        model: str,
        payload: Dict[str, Any],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        model_path = quote(model, safe="._-")
        url = (
            f"{base_url.rstrip('/')}/v1/publishers/google/models/"
            f"{model_path}:streamGenerateContent"
        )
        timeout = httpx.Timeout(timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                url,
                params={"alt": "sse", "key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    raw_error = await response.aread()
                    raise RuntimeError(_upstream_error(response.status_code, raw_error))

                data_lines = []
                async for line in response.aiter_lines():
                    if not line:
                        if data_lines:
                            yield _decode_sse(data_lines)
                            data_lines = []
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    yield _decode_sse(data_lines)


def _decode_sse(data_lines: List[str]) -> Dict[str, Any]:
    try:
        decoded = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Agent Platform returned malformed streaming data.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Agent Platform returned an unexpected streaming event.")
    return decoded


def _upstream_error(status_code: int, raw_error: bytes) -> str:
    message = raw_error.decode("utf-8", errors="replace")[:2000]
    try:
        error = json.loads(message).get("error", {})
        message = error.get("message") or message
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"Agent Platform request failed ({status_code}): {message}"
