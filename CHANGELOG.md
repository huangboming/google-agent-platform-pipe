# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Send API keys in the `x-goog-api-key` header instead of URL query parameters to prevent exposure in HTTP request URL logs.

## [0.4.1] - 2026-10-09

Initial public release.

### Added

- Google Agent Platform API-key authentication with explicit regional model IDs.
- Asynchronous streaming for responses, thought summaries, and token usage.
- Native multimodal input for text, images, audio, video, and original PDF files.
- Parallel and multi-turn Open WebUI tool calling with Gemini thought-signature preservation.
- Gemini-native Google Search with source cards and Search Suggestions.
- Per-request reasoning effort and configurable thinking defaults.
- A single-file Open WebUI distribution with no Google SDK or ADK dependency.

### Fixed

- Surface blocked and abnormal model responses instead of returning empty messages.
- Preserve references and definitions in complex tool schemas.
- Prevent files, Search, and tools from leaking into Open WebUI internal tasks.

[Unreleased]: https://github.com/huangboming/google-agent-platform-pipe/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/huangboming/google-agent-platform-pipe/releases/tag/v0.4.1
