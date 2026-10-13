# Google Agent Platform Pipe for Open WebUI

A lightweight [Open WebUI Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/) for Google Agent Platform models.

It provides streaming, multimodal input, and native Open WebUI tool calling as a single deployable Python file—with no additional runtime dependencies. Open WebUI integration targets version 0.11.4.

## Features

- Asynchronous SSE streaming
- Native text, image, audio, video, and original-PDF input
- Parallel and multi-turn Open WebUI tool calls
- Gemini-native Google Search with citations and Search Suggestions
- Collapsible Gemini thought summaries and token-usage reporting
- Per-request reasoning effort with a configurable thinking-level default
- Agent Platform API-key authentication

## Quick start

You need a [Google Agent Platform API key](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/api-keys).

1. In Open WebUI, go to **Admin Panel → Functions**.
2. Choose **Import From Link** and paste the latest Release URL:

   ```text
   https://github.com/huangboming/google-agent-platform-pipe/releases/latest/download/google_agent_platform.py
   ```

3. Save and enable the Function.
4. Set its `GOOGLE_API_KEY` Valve.
5. Start a chat with a model under `Google/`.

Alternatively, download `google_agent_platform.py` from the [latest Release](https://github.com/huangboming/google-agent-platform-pipe/releases/latest) and import it manually.

## Default models

- `gemini-3.8-flash`
- `gemini-3.5-flash-lite`
- `gemini-3.1-pro-preview` — preview

The Pipe uses explicit model IDs because Google's `*-latest` selectors can return 404 when an API key routes through a regional location such as `us-central1`. You can replace the defaults through the `MODELS` Valve. Available models depend on your Agent Platform project and region.

> After updating an existing Function, check `MODELS`: Open WebUI may retain its previously saved Valve value.

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

Google exposes thought summaries rather than a model's raw chain of thought, and only when available. Open WebUI stores returned summaries in the chat history. Open WebUI's per-request `reasoning_effort` parameter takes precedence over the `THINKING_LEVEL` Valve. Reported token usage includes all internal model calls made during a multi-turn tool loop.

## Tool execution

Open WebUI tools receive only arguments declared in their schemas; model-supplied `__`-prefixed context parameters cannot override the authenticated user or request. Tool exceptions return a generic failure to the model instead of forwarding private diagnostics.

The Pipe does not implement Open WebUI's native tool approval/resume flow. When Tool Permissions is enabled and the per-chat mode is **Ask for permission** (`ask`), any model turn requesting Open WebUI tools fails before executing the entire batch. Responses that do not call tools still work. Automatic execution is available only when the request is not in `ask` mode.

## Native Gemini capabilities

These capabilities are handled directly by Gemini through Agent Platform rather than by Open WebUI's equivalent processing.

### Native Google Search

Turn on Open WebUI's per-chat **Web Search** toggle to use Gemini-native Google Search. Web Search must be enabled globally, the model must allow the Web Search capability, and function calling must use Native mode.

Grounded responses include source cards and Google's Search Suggestions. The Pipe uses Agent Platform's `googleSearch`; it does not call the search provider configured in Open WebUI.

Google Search cannot be combined with selected Workspace or external tools in the same request. Turn off Web Search or deselect those tools when a conflict occurs.

### Native file input

Enable the **File Upload** capability, attach a supported file, and ask Gemini about it. PDFs are sent in their original form so Gemini can understand pages, layout, tables, and images instead of relying on extracted text.

For the cleanest Workspace Model setup:

1. Enable **File Upload**.
2. Disable **File Context** to avoid chat-time retrieval and duplicate source cards.
3. Optionally disable the built-in **Files** tool category.

Uploaded attachments are resolved by Open WebUI file ID and checked against the current user's read permissions, including shared-file access. Client-supplied server paths and file metadata are not trusted.

The Pipe always ignores Open WebUI's extracted document text and sends the original file. Open WebUI still performs Content Extraction during upload; disabling File Context prevents later retrieval but does not skip that upload-time processing.

Supported inputs include PDF, plain text, images, audio, and video. DOCX is not supported. Inline files and media share the `MAX_INLINE_MEDIA_MB` limit; use an Agent Platform-accessible `gs://` object for larger files.

## Development

`src/google_agent_platform_pipe/` is the source of truth; `dist/google_agent_platform.py` is generated for Open WebUI.

```bash
uv sync
uv run python scripts/build.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/build.py --check
```

## License

Licensed under the [Apache License 2.0](LICENSE).
