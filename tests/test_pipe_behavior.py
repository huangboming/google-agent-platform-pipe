import base64
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google_agent_platform_pipe import Pipe as SourcePipe  # noqa: E402


def load_distribution_pipe():
    path = ROOT / "dist" / "google_agent_platform.py"
    spec = importlib.util.spec_from_file_location(
        "google_agent_platform_pipe_distribution", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Pipe


DistributionPipe = load_distribution_pipe()


class RecordingAgentPlatformClient:
    """Records the public upstream contract without making network requests."""

    def __init__(self, response_streams=None):
        self.response_streams = iter(response_streams or [])
        self.requests = []

    async def stream_generate(
        self, model, payload, *, api_key, base_url, timeout_seconds
    ):
        self.requests.append(
            {
                "model": model,
                "payload": copy.deepcopy(payload),
                "api_key": api_key,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
            }
        )
        for response in next(self.response_streams):
            yield response


def text_response(*parts, usage=None):
    response = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": part} for part in parts]}}
        ]
    }
    if usage is not None:
        response["usageMetadata"] = usage
    return response


class PipeBehaviorContract:
    PipeClass = None

    def make_pipe(self, response_streams=None):
        client = RecordingAgentPlatformClient(response_streams)
        pipe = self.PipeClass(client=client)
        pipe.valves.GOOGLE_API_KEY = "test-key"
        return pipe, client

    def test_exposes_expected_default_models(self):
        pipe, _ = self.make_pipe()

        self.assertEqual(
            [model["id"] for model in pipe.pipes()],
            [
                "gemini-3.8-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-pro-preview",
            ],
        )

    def test_exposes_configured_models_without_duplicates(self):
        pipe, _ = self.make_pipe()
        pipe.valves.MODELS = (
            "gemini-3.8-flash, gemini-3.5-flash-lite,gemini-3.8-flash"
        )

        self.assertEqual(
            pipe.pipes(),
            [
                {"id": "gemini-3.8-flash", "name": "gemini-3.8-flash"},
                {
                    "id": "gemini-3.5-flash-lite",
                    "name": "gemini-3.5-flash-lite",
                },
            ],
        )

    async def test_sends_multimodal_messages_parameters_and_tools(self):
        pipe, client = self.make_pipe([[text_response("A small image.")]])
        image = base64.b64encode(b"image bytes").decode()

        async def weather(city):
            return {"city": city, "temperature": 20}

        result = await pipe.pipe(
            {
                "model": "google_agent_platform.gemini-3.8-flash",
                "stream": False,
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            __tools__={
                "weather": {
                    "callable": weather,
                    "spec": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "title": "City"}
                            },
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            },
        )

        self.assertEqual(
            result["choices"][0]["message"]["content"], "A small image."
        )
        request = client.requests[0]
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertEqual(request["base_url"], "https://aiplatform.googleapis.com")
        payload = request["payload"]
        self.assertEqual(
            payload["systemInstruction"]["parts"], [{"text": "Be concise."}]
        )
        self.assertEqual(
            payload["contents"][0]["parts"][1]["inlineData"]["data"], image
        )
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 512)
        self.assertTrue(
            payload["generationConfig"]["thinkingConfig"]["includeThoughts"]
        )
        declaration = payload["tools"][0]["functionDeclarations"][0]
        self.assertEqual(declaration["name"], "weather")
        self.assertNotIn("additionalProperties", declaration["parameters"])
        self.assertNotIn(
            "title", declaration["parameters"]["properties"]["city"]
        )

    async def test_sends_original_pdf_without_open_webui_rag_context(self):
        pipe, client = self.make_pipe([[text_response("The original PDF.")]])
        pdf = b"%PDF-1.7\noriginal document bytes"
        image = base64.b64encode(b"image bytes").decode()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.pdf"
            path.write_bytes(pdf)
            completion = await pipe.pipe(
                {
                    "model": "pipe.gemini-3.8-flash",
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Base system prompt. EXTRACTED SYSTEM CONTEXT",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "EXTRACTED PDF TEXT FROM OPEN WEBUI",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image}"
                                    },
                                },
                            ],
                        },
                    ],
                },
                __files__=[
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "path": str(path),
                            "meta": {"content_type": "application/pdf"},
                            "data": {
                                "content": "EXTRACTED FILE DATA MUST NOT BE USED"
                            },
                        },
                    }
                ],
                __metadata__={
                    "user_prompt": "Read the original PDF.",
                    "system_prompt": "Base system prompt.",
                    "sources": [
                        {"document": ["EXTRACTED RETRIEVAL CHUNK"]}
                    ],
                },
                __tools__={
                    name: {"type": "builtin"}
                    for name in (
                        "list_chat_files",
                        "query_chat_files",
                        "grep_chat_files",
                        "view_file",
                    )
                }
                | {
                    "weather": {
                        "spec": {
                            "name": "weather",
                            "parameters": {"type": "object"},
                        }
                    }
                },
            )

        self.assertEqual(
            completion["choices"][0]["message"]["content"],
            "The original PDF.",
        )
        payload = client.requests[0]["payload"]
        self.assertEqual(
            payload["systemInstruction"]["parts"],
            [{"text": "Base system prompt."}],
        )
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0]["inlineData"]["mimeType"], "application/pdf")
        self.assertEqual(base64.b64decode(parts[0]["inlineData"]["data"]), pdf)
        self.assertEqual(parts[1], {"text": "Read the original PDF."})
        self.assertEqual(parts[2]["inlineData"]["data"], image)
        self.assertNotIn("EXTRACTED", json.dumps(payload))
        self.assertEqual(
            [
                declaration["name"]
                for declaration in payload["tools"][0]["functionDeclarations"]
            ],
            ["weather"],
        )

    async def test_rejects_source_context_without_the_original_prompt(self):
        pipe, client = self.make_pipe()
        pdf = b"%PDF-1.7\noriginal document bytes"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.pdf"
            path.write_bytes(pdf)
            with self.assertRaisesRegex(
                ValueError, "without preserving the original prompt"
            ):
                await pipe.pipe(
                    {
                        "model": "pipe.gemini-3.8-flash",
                        "stream": False,
                        "messages": [
                            {
                                "role": "user",
                                "content": "EXTRACTED PDF TEXT",
                            }
                        ],
                    },
                    __files__=[
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.pdf",
                                "path": str(path),
                                "meta": {"content_type": "application/pdf"},
                            },
                        }
                    ],
                    __metadata__={
                        "sources": [{"document": ["EXTRACTED PDF TEXT"]}]
                    },
                )

        self.assertEqual(client.requests, [])

    async def test_rejects_an_unavailable_uploaded_file(self):
        pipe, client = self.make_pipe()

        with self.assertRaisesRegex(
            ValueError, "is not available on the Open WebUI server"
        ):
            await pipe.pipe(
                {
                    "model": "pipe.gemini-3.8-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Read it"}],
                },
                __files__=[
                    {
                        "type": "file",
                        "file": {
                            "filename": "missing.pdf",
                            "path": "/missing/document.pdf",
                            "meta": {"content_type": "application/pdf"},
                        },
                    }
                ],
            )

        self.assertEqual(client.requests, [])

    async def test_rejects_an_unsupported_uploaded_file_type(self):
        pipe, client = self.make_pipe()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.docx"
            path.write_bytes(b"not a native Gemini document")
            with self.assertRaisesRegex(
                ValueError, "unsupported media type.*application/vnd"
            ):
                await pipe.pipe(
                    {
                        "model": "pipe.gemini-3.8-flash",
                        "stream": False,
                        "messages": [
                            {"role": "user", "content": "Read it"}
                        ],
                    },
                    __files__=[
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.docx",
                                "path": str(path),
                                "meta": {
                                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                },
                            },
                        }
                    ],
                )

        self.assertEqual(client.requests, [])

    async def test_executes_tool_and_continues_the_model_turn(self):
        tool_call_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": "weather",
                                    "args": {"city": "Paris"},
                                },
                                "thoughtSignature": "signature",
                            }
                        ],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 13,
            },
        }
        pipe, client = self.make_pipe(
            [
                [tool_call_response],
                [
                    text_response(
                        "Paris is 20 C.",
                        usage={
                            "promptTokenCount": 20,
                            "candidatesTokenCount": 5,
                            "thoughtsTokenCount": 3,
                            "totalTokenCount": 28,
                        },
                    )
                ],
            ]
        )
        calls = []
        statuses = []

        async def weather(city):
            calls.append(city)
            return {"temperature": 20, "unit": "C"}

        async def emit(event):
            statuses.append(event)

        result = await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "Weather?"}],
            },
            __tools__={
                "weather": {
                    "callable": weather,
                    "spec": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            },
            __event_emitter__=emit,
        )

        self.assertEqual(
            result["choices"][0]["message"]["content"], "Paris is 20 C."
        )
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": 30,
                "completion_tokens": 11,
                "total_tokens": 41,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        )
        self.assertEqual(calls, ["Paris"])
        self.assertEqual([event["data"]["done"] for event in statuses], [False, True])

        follow_up = client.requests[1]["payload"]
        model_part = follow_up["contents"][-2]["parts"][0]
        self.assertEqual(model_part["thoughtSignature"], "signature")
        function_response = follow_up["contents"][-1]["parts"][0]["functionResponse"]
        self.assertEqual(function_response["id"], "call-1")
        self.assertEqual(function_response["response"]["temperature"], 20)

    async def test_streams_text_chunks_to_open_webui(self):
        pipe, _ = self.make_pipe([[text_response("hello", " world")]])

        stream = await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )

        chunks = [chunk async for chunk in stream]
        self.assertEqual(
            [chunk["choices"][0]["delta"]["content"] for chunk in chunks],
            ["hello", " world"],
        )

    async def test_exposes_reasoning_and_token_usage(self):
        response_stream = [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "Considering the options...",
                                    "thought": True,
                                }
                            ],
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "thoughtsTokenCount": 1,
                    "totalTokenCount": 11,
                },
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Use the second option."}],
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "toolUsePromptTokenCount": 2,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 3,
                    "cachedContentTokenCount": 5,
                    "totalTokenCount": 19,
                },
            },
        ]
        pipe, _ = self.make_pipe([response_stream])
        body = {
            "model": "pipe.gemini-3.8-flash",
            "stream": True,
            "messages": [{"role": "user", "content": "Which option?"}],
        }

        stream = await pipe.pipe(body)
        chunks = [chunk async for chunk in stream]

        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"reasoning_content": "Considering the options..."},
        )
        self.assertEqual(
            chunks[1]["choices"][0]["delta"],
            {"content": "Use the second option."},
        )
        self.assertEqual(chunks[2]["choices"], [])
        expected_usage = {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
        self.assertEqual(chunks[2]["usage"], expected_usage)

        pipe, _ = self.make_pipe([response_stream])
        completion = await pipe.pipe({**body, "stream": False})
        self.assertEqual(
            completion["choices"][0]["message"]["reasoning_content"],
            "Considering the options...",
        )
        self.assertEqual(
            completion["choices"][0]["message"]["content"],
            "Use the second option.",
        )
        self.assertEqual(completion["usage"], expected_usage)

    async def test_uses_native_google_search_and_emits_grounding(self):
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "The grounded answer."}],
                    },
                    "groundingMetadata": {
                        "webSearchQueries": ["current answer"],
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/source",
                                    "title": "Example Source",
                                    "domain": "example.com",
                                }
                            }
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"text": "The grounded answer."},
                                "groundingChunkIndices": [0],
                            }
                        ],
                        "searchEntryPoint": {
                            "renderedContent": "<div>Google Search</div>"
                        },
                    },
                }
            ]
        }
        pipe, client = self.make_pipe([[response]])
        events = []

        async def emit(event):
            events.append(event)

        completion = await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "What is current?"}],
            },
            __tools__={
                "search_web": {"type": "builtin"},
                "fetch_url": {"type": "builtin"},
                "ask_user": {"type": "builtin"},
                "query_knowledge_files": {"type": "builtin"},
            },
            __metadata__={"features": {"web_search": True}},
            __event_emitter__=emit,
        )

        self.assertEqual(
            completion["choices"][0]["message"]["content"],
            "The grounded answer.",
        )
        self.assertEqual(
            client.requests[0]["payload"]["tools"], [{"googleSearch": {}}]
        )
        self.assertEqual(events[0]["type"], "source")
        self.assertEqual(events[0]["data"]["document"], ["The grounded answer."])
        self.assertEqual(
            events[0]["data"]["metadata"][0]["url"],
            "https://example.com/source",
        )
        self.assertEqual(
            events[1],
            {
                "type": "embeds",
                "data": {
                    "embeds": ["<div>Google Search</div>"],
                    "replace": False,
                },
            },
        )

    async def test_rejects_other_tools_with_native_google_search(self):
        pipe, client = self.make_pipe()

        with self.assertRaisesRegex(
            ValueError, "cannot be combined with other tools: weather"
        ):
            await pipe.pipe(
                {
                    "model": "pipe.gemini-3.8-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Weather?"}],
                },
                __tools__={"search_web": {}, "weather": {}},
                __metadata__={"features": {"web_search": True}},
            )

        self.assertEqual(client.requests, [])

    async def test_internal_tasks_do_not_use_native_google_search(self):
        pipe, client = self.make_pipe([[text_response("A short title")]])

        await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "Create a title"}],
            },
            __tools__={"search_web": {}, "fetch_url": {}},
            __metadata__={"features": {"web_search": True}},
            __task__="title_generation",
        )

        self.assertNotIn("tools", client.requests[0]["payload"])

    async def test_rejects_a_model_that_is_not_exposed(self):
        pipe, client = self.make_pipe()

        with self.assertRaisesRegex(ValueError, "Unsupported Agent Platform model"):
            await pipe.pipe(
                {
                    "model": "pipe.unconfigured-model",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            )

        self.assertEqual(client.requests, [])

    async def test_rejects_inline_media_over_the_configured_limit(self):
        pipe, client = self.make_pipe()
        pipe.valves.MAX_INLINE_MEDIA_MB = 1
        image = base64.b64encode(b"x" * (1024 * 1024 + 1)).decode()

        with self.assertRaisesRegex(ValueError, "exceeds the configured 1 MB limit"):
            await pipe.pipe(
                {
                    "model": "pipe.gemini-3.8-flash",
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image}"
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

        self.assertEqual(client.requests, [])


class SourcePipeBehaviorTests(PipeBehaviorContract, unittest.IsolatedAsyncioTestCase):
    PipeClass = SourcePipe


class DistributionPipeBehaviorTests(
    PipeBehaviorContract, unittest.IsolatedAsyncioTestCase
):
    PipeClass = DistributionPipe


if __name__ == "__main__":
    unittest.main()
