import json
import re
from typing import Any, AsyncIterator, Dict, List
from urllib.parse import quote

import httpx

from .config import DEFAULT_API_BASE_URL


def resolve_model_base_url(base_url: str, project_id: str, location: str) -> str:
    project_id = project_id.strip()
    location = location.strip()
    if bool(project_id) != bool(location):
        raise ValueError(
            "PROJECT_ID and LOCATION must both be set for Full Agent Platform, "
            "or both be empty for express mode."
        )
    if project_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", project_id):
            raise ValueError("PROJECT_ID must be a Google Cloud project ID or number.")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", location):
            raise ValueError(
                "LOCATION must be global or a region ID using lowercase letters, "
                "digits, and hyphens."
            )

    base_url = base_url.strip().rstrip("/")
    if base_url in ("", DEFAULT_API_BASE_URL):
        base_url = DEFAULT_API_BASE_URL
        if location and location != "global":
            base_url = f"https://{location}-aiplatform.googleapis.com"

    scope = (
        f"projects/{quote(project_id, safe='')}/locations/{location}/"
        if project_id
        else ""
    )
    return f"{base_url}/v1/{scope}publishers/google/models"


class AgentPlatformClient:
    async def stream_generate(
        self,
        model: str,
        payload: Dict[str, Any],
        *,
        api_key: str,
        model_base_url: str,
        timeout_seconds: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        model_path = quote(model, safe="._-")
        url = f"{model_base_url}/{model_path}:streamGenerateContent"
        timeout = httpx.Timeout(timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                url,
                params={"alt": "sse"},
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    raw_error = await response.aread()
                    raise RuntimeError(
                        _upstream_error(response.status_code, raw_error, api_key)
                    )

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


def _upstream_error(status_code: int, raw_error: bytes, api_key: str) -> str:
    message = raw_error.decode("utf-8", errors="replace")
    try:
        error = json.loads(message).get("error", {})
        message = error.get("message") or message
    except (json.JSONDecodeError, AttributeError):
        pass
    message = str(message)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return f"Agent Platform request failed ({status_code}): {message[:2000]}"
