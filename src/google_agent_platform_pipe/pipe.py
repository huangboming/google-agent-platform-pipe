from typing import Any, Dict, List, Optional

from .agent import run_agent
from .client import AgentPlatformClient
from .config import Valves
from .files import ensure_media_limit, uploaded_file_parts
from .messages import generation_config, message_parts
from .models import parse_models, select_model
from .responses import collect_openai_response, stream_openai_response
from .tools import tool_declarations


_OPEN_WEBUI_SEARCH_TOOLS = {"search_web", "fetch_url"}


def native_google_search_enabled(
    metadata: Optional[dict], task: Optional[str]
) -> bool:
    features = metadata.get("features", {}) if isinstance(metadata, dict) else {}
    return not task and isinstance(features, dict) and features.get("web_search") is True


def tools_for_request(tools: dict, google_search: bool, internal_task: bool) -> dict:
    if not google_search and not internal_task:
        return tools

    without_open_webui_search = {
        name: tool
        for name, tool in tools.items()
        if name not in _OPEN_WEBUI_SEARCH_TOOLS
    }
    if internal_task:
        return without_open_webui_search

    remaining = {
        name: tool
        for name, tool in without_open_webui_search.items()
        if not isinstance(tool, dict) or tool.get("type") != "builtin"
    }
    if remaining:
        names = ", ".join(sorted(remaining))
        raise ValueError(
            "Gemini native Google Search cannot be combined with other tools: "
            f"{names}. Disable the selected tools or turn off Web Search."
        )
    return {}


class Pipe:
    """Open WebUI manifold pipe for the Google Agent Platform Model API."""

    name = "Google/"
    Valves = Valves

    def __init__(self, client=None):
        self.valves = self.Valves()
        self.client = client or AgentPlatformClient()

    def pipes(self) -> List[Dict[str, str]]:
        return [
            {"id": model, "name": model}
            for model in parse_models(self.valves.MODELS)
        ]

    async def pipe(
        self,
        body: dict,
        __tools__: Optional[dict] = None,
        __files__: Optional[list] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __task__: Optional[str] = None,
    ):
        if not self.valves.GOOGLE_API_KEY.strip():
            raise ValueError("Google Agent Platform API key is not configured.")

        google_search = native_google_search_enabled(__metadata__, __task__)
        tools = tools_for_request(__tools__ or {}, google_search, bool(__task__))
        model = select_model(body.get("model", ""), parse_models(self.valves.MODELS))
        payload = await self._build_request(
            body, tools, __files__ or [], google_search
        )

        async def stream_generate(selected_model, request):
            async for response in self.client.stream_generate(
                selected_model,
                request,
                api_key=self.valves.GOOGLE_API_KEY,
                base_url=self.valves.API_BASE_URL,
                timeout_seconds=self.valves.REQUEST_TIMEOUT_SECONDS,
            ):
                yield response

        output = run_agent(
            model,
            payload,
            tools,
            self.valves.MAX_TOOL_ITERATIONS,
            __event_emitter__,
            stream_generate,
        )
        if body.get("stream", False):
            return stream_openai_response(output, model)
        return await collect_openai_response(output, model)

    async def _build_request(
        self, body: dict, tools: dict, files: list, google_search: bool
    ) -> Dict[str, Any]:
        contents = []
        system_parts = []
        inline_bytes = 0
        max_inline_bytes = self.valves.MAX_INLINE_MEDIA_MB * 1024 * 1024

        for message in body.get("messages", []):
            role = message.get("role", "user")
            parts, used_bytes = message_parts(message.get("content", ""))
            inline_bytes += used_bytes
            ensure_media_limit(inline_bytes, max_inline_bytes)

            if role in ("system", "developer"):
                system_parts.extend(part for part in parts if "text" in part)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {"role": gemini_role, "parts": parts or [{"text": ""}]}
            )

        file_parts, used_bytes = await uploaded_file_parts(
            files, inline_bytes, max_inline_bytes
        )
        inline_bytes += used_bytes
        ensure_media_limit(inline_bytes, max_inline_bytes)
        if file_parts:
            for content in reversed(contents):
                if content["role"] == "user":
                    content["parts"].extend(file_parts)
                    break
            else:
                contents.append({"role": "user", "parts": file_parts})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        config = generation_config(
            body, self.valves.THINKING_LEVEL, self.valves.INCLUDE_THOUGHTS
        )
        if config:
            payload["generationConfig"] = config

        if google_search:
            payload["tools"] = [{"googleSearch": {}}]
        else:
            declarations = tool_declarations(tools)
            if declarations:
                payload["tools"] = [{"functionDeclarations": declarations}]
        return payload
