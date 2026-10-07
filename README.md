# Google Agent Platform Pipe for Open WebUI

A lightweight [Open WebUI Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/)
for Google Agent Platform models.

It provides streaming, multimodal input, and native Open WebUI tool calling as
a single deployable Python file—with no additional runtime dependencies.

## Features

- Asynchronous SSE streaming
- Native text, image, audio, video, and original-PDF input
- Parallel and multi-turn Open WebUI tool calls
- Gemini-native Google Search with citations and Search Suggestions
- Collapsible Gemini thought summaries and token-usage reporting
- Gemini thought-signature preservation across tool turns
- Text and inline media output
- Per-request reasoning effort with a configurable thinking-level default
- Agent Platform API-key authentication

## Quick start

You need a [Google Agent Platform API key](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/api-keys).

1. In Open WebUI, go to **Admin Panel → Functions**.
2. Import [`dist/google_agent_platform.py`](./dist/google_agent_platform.py), or
   paste it into a new Function.
3. Save and enable the Function.
4. Set its `GOOGLE_API_KEY` Valve.
5. Start a chat with a model under `Google/`.

## Default models

- `gemini-3.8-flash`
- `gemini-3.5-flash-lite`
- `gemini-3.1-pro-preview` — preview

The Pipe uses explicit model IDs because Google's `*-latest` selectors can return
404 when an API key routes through a regional location such as `us-central1`.
You can replace the defaults through the `MODELS` Valve. Available models depend
on your Agent Platform project and region.

> After updating an existing Function, check `MODELS`: Open WebUI may retain its
> previously saved Valve value.

## Configuration

| Valve | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Agent Platform API key; required |
| `MODELS` | See above | Comma-separated model IDs exposed in Open WebUI |
| `INCLUDE_THOUGHTS` | `true` | Show available thought summaries in Open WebUI |
| `THINKING_LEVEL` | Empty | Default `MINIMAL`, `LOW`, `MEDIUM`, or `HIGH` when the request does not set `reasoning_effort` |
| `MAX_TOOL_ITERATIONS` | `8` | Maximum model/tool round trips |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Upstream request timeout |
| `MAX_INLINE_MEDIA_MB` | `18` | Total decoded inline-media limit per request |
| `API_BASE_URL` | `https://aiplatform.googleapis.com` | Agent Platform API endpoint |

Google exposes thought summaries rather than a model's raw chain of thought, and
only when available. Open WebUI stores returned summaries in the chat history.
Open WebUI's per-request `reasoning_effort` parameter takes precedence over the
`THINKING_LEVEL` Valve. Reported token usage includes all internal model calls
made during a multi-turn tool loop.

## Native Google Search

The Pipe uses Open WebUI's per-chat **Web Search** toggle to enable Agent
Platform's native `googleSearch` tool. Enable Web Search globally in Open WebUI,
ensure the model allows the Web Search capability, and use Native function
calling. The Pipe reads the toggle directly; Open WebUI's configured search
provider is not called.

Grounded responses include Open WebUI source cards and Google's Search
Suggestions widget. Internal tasks such as title generation never receive
files, Search, or tools.

Agent Platform does not allow `googleSearch` and function declarations in the
same request. When Web Search is on, Open WebUI's automatically injected
built-in tools are omitted and `googleSearch` takes precedence. Selecting an
additional Workspace or external tool returns an explicit error.

## Native file input

Attach a PDF in chat and ask Gemini about it. The Pipe sends the original PDF
bytes as `application/pdf` rather than using Open WebUI's extracted text. This
preserves pages, layout, tables, and images for Gemini's native document
understanding.

For the cleanest Open WebUI setup, create a Workspace Model preset for the Pipe:

1. Enable **File Upload**.
2. Disable **File Context** to prevent extracted content from being retrieved or
   injected during the chat, and to avoid Open WebUI source cards.
3. Optionally disable the built-in **Files** tool category.

Correctness does not depend on the second and third settings: when a native file
is attached, the Pipe restores the prompt captured before Open WebUI's source
wrapper and omits its chat-file retrieval tools.

> **Open WebUI still performs Content Extraction when uploading a document.**
> File Context controls chat-time retrieval and injection, not upload-time
> processing. The stock UI uploads documents with `process=true` before the Pipe
> runs; the Pipe ignores that extracted content and sends only the original file
> to Gemini. **Bypass Embedding and Retrieval** can avoid vector indexing but not
> extraction. Completely skipping extraction requires an Open WebUI client or UI
> change that uploads with `process=false`. If external extraction or embedding
> services are configured, consider their privacy and cost implications.

Input behavior:

- Local Open WebUI uploads are read server-side and sent as Gemini `inlineData`.
- `gs://` objects are sent as Gemini `fileData`.
- PDF, plain text, and supported image, audio, and video MIME types are accepted.
- Unsupported formats such as DOCX fail explicitly instead of silently falling
  back to extracted text.
- S3, Azure, and arbitrary HTTP objects are not downloaded. Use local Open WebUI
  storage or an Agent Platform-accessible `gs://` object.
- Message data URLs continue to be sent as Gemini `inlineData`.

The `MAX_INLINE_MEDIA_MB` Valve limits the combined decoded size of local files
and message media. Large documents should use `gs://`.

## Scope

This project targets only Google Agent Platform at `aiplatform.googleapis.com` using
an Agent Platform API key. The MVP does not support the Gemini Developer API,
ADC, service accounts, Google SDKs, or ADK.

## Development

```text
src/google_agent_platform_pipe/ source package
dist/google_agent_platform.py   generated Open WebUI Function
scripts/build.py                 deterministic single-file builder
tests/                           pytest behavior contracts by feature
pyproject.toml                   uv, pytest, and Ruff configuration
```

> `src/google_agent_platform_pipe/` is the source of truth. Do not edit `dist/`
> directly.

```bash
uv sync
uv run python scripts/build.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/build.py --check
```

The same pytest behavior contract runs against the source package and generated
single-file Function. Tests make no external API calls. The committed `uv.lock`
pins the development environment; it does not add dependencies to the deployed
single-file Function.
