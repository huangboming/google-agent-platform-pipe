from typing import Any, Dict, List, Optional

from .agent import run_agent
from .client import AgentPlatformClient
from .config import Valves
from .files import ensure_media_limit, has_uploaded_files, uploaded_file_parts
from .messages import (
    generation_config,
    message_parts,
    restore_pre_source_messages,
)
from .models import parse_models, select_model
from .responses import collect_openai_response, stream_openai_response
from .tools import tool_declarations

_OPEN_WEBUI_SEARCH_TOOLS = {"search_web", "fetch_url"}
_OPEN_WEBUI_FILE_TOOLS = {
    "grep_chat_files",
    "list_chat_files",
    "query_chat_files",
    "view_file",
}


def native_google_search_enabled(metadata: Optional[dict], task: Optional[str]) -> bool:
    features = metadata.get("features", {}) if isinstance(metadata, dict) else {}
    return (
        not task and isinstance(features, dict) and features.get("web_search") is True
    )


def tool_approval_required(metadata: Optional[dict]) -> bool:
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    return isinstance(params, dict) and params.get("tool_approval_mode") == "ask"


def tools_for_request(
    tools: dict,
    google_search: bool,
    internal_task: bool,
    native_files: bool,
) -> dict:
    if internal_task:
        return {}

    selected = {
        name: tool
        for name, tool in tools.items()
        if not (
            native_files
            and name in _OPEN_WEBUI_FILE_TOOLS
            and isinstance(tool, dict)
            and tool.get("type") == "builtin"
        )
    }
    if not google_search:
        return selected

    without_open_webui_search = {
        name: tool
        for name, tool in selected.items()
        if name not in _OPEN_WEBUI_SEARCH_TOOLS
    }
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
            {"id": model, "name": model} for model in parse_models(self.valves.MODELS)
        ]

    async def pipe(
        self,
        body: dict,
        __tools__: Optional[dict] = None,
        __files__: Optional[list] = None,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __task__: Optional[str] = None,
        __user__: Optional[dict] = None,
    ):
        if not self.valves.GOOGLE_API_KEY.strip():
            raise ValueError("Google Agent Platform API key is not configured.")

        google_search = native_google_search_enabled(__metadata__, __task__)
        files = [] if __task__ else (__files__ or [])
        native_files = has_uploaded_files(files)
        tools = tools_for_request(
            __tools__ or {}, google_search, bool(__task__), native_files
        )
        model = select_model(body.get("model", ""), parse_models(self.valves.MODELS))
        payload = await self._build_request(
            body, tools, files, google_search, __metadata__, __user__
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
            tool_approval_required(__metadata__),
        )
        if body.get("stream", False):
            return stream_openai_response(output, model)
        return await collect_openai_response(output, model)

    async def _build_request(
        self,
        body: dict,
        tools: dict,
        files: list,
        google_search: bool,
        metadata: Optional[dict],
        user: Optional[dict],
    ) -> Dict[str, Any]:
        contents = []
        system_parts = []
        max_inline_bytes = self.valves.MAX_INLINE_MEDIA_MB * 1024 * 1024
        file_parts, inline_bytes = await uploaded_file_parts(
            files, max_inline_bytes, user
        )
        messages = body.get("messages", [])
        if file_parts:
            messages = restore_pre_source_messages(messages, metadata)

        for message in messages:
            role = message.get("role", "user")
            parts, used_bytes = message_parts(message.get("content", ""))
            inline_bytes += used_bytes
            ensure_media_limit(inline_bytes, max_inline_bytes)

            if role in ("system", "developer"):
                system_parts.extend(part for part in parts if "text" in part)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": parts or [{"text": ""}]})

        if file_parts:
            for content in reversed(contents):
                if content["role"] == "user":
                    content["parts"] = file_parts + content["parts"]
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
