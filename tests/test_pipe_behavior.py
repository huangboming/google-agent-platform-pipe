import base64
import copy
import importlib.util
import sys
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


def text_response(*parts):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": part} for part in parts]}}
        ]
    }


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

        self.assertEqual(result, "A small image.")
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
        declaration = payload["tools"][0]["functionDeclarations"][0]
        self.assertEqual(declaration["name"], "weather")
        self.assertNotIn("additionalProperties", declaration["parameters"])
        self.assertNotIn(
            "title", declaration["parameters"]["properties"]["city"]
        )

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
            ]
        }
        pipe, client = self.make_pipe(
            [[tool_call_response], [text_response("Paris is 20 C.")]]
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

        self.assertEqual(result, "Paris is 20 C.")
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

        self.assertEqual([chunk async for chunk in stream], ["hello", " world"])

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
