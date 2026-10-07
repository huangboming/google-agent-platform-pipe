import base64

import pytest

from tests.helpers import text_response


async def test_sends_multimodal_messages_parameters_and_tools(make_pipe):
    pipe, client = make_pipe([[text_response("A small image.")]])
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
                            "image_url": {"url": f"data:image/png;base64,{image}"},
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
                        "properties": {"city": {"type": "string", "title": "City"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        },
    )

    assert result["choices"][0]["message"]["content"] == "A small image."
    request = client.requests[0]
    assert request["model"] == "gemini-3.8-flash"
    assert request["base_url"] == "https://aiplatform.googleapis.com"
    payload = request["payload"]
    assert payload["systemInstruction"]["parts"] == [{"text": "Be concise."}]
    assert payload["contents"][0]["parts"][1]["inlineData"]["data"] == image
    assert payload["generationConfig"]["maxOutputTokens"] == 512
    assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "weather"
    assert "additionalProperties" not in declaration["parameters"]
    assert "title" not in declaration["parameters"]["properties"]["city"]


async def test_request_reasoning_effort_overrides_valve_default(make_pipe):
    pipe, client = make_pipe([[text_response("A considered answer.")]])
    pipe.valves.THINKING_LEVEL = "LOW"

    await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": False,
            "messages": [{"role": "user", "content": "Think carefully."}],
            "reasoning_effort": "high",
        }
    )

    thinking_config = client.requests[0]["payload"]["generationConfig"][
        "thinkingConfig"
    ]
    assert thinking_config["thinkingLevel"] == "HIGH"


async def test_rejects_inline_media_over_the_configured_limit(make_pipe):
    pipe, client = make_pipe()
    pipe.valves.MAX_INLINE_MEDIA_MB = 1
    image = base64.b64encode(b"x" * (1024 * 1024 + 1)).decode()

    with pytest.raises(ValueError, match="exceeds the configured 1 MB limit"):
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
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            }
                        ],
                    }
                ],
            }
        )

    assert client.requests == []
