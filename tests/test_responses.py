import pytest

from tests.helpers import text_response


async def test_streams_text_chunks_to_open_webui(make_pipe):
    pipe, _ = make_pipe([[text_response("hello", " world")]])

    stream = await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )

    chunks = [chunk async for chunk in stream]
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == [
        "hello",
        " world",
    ]


async def test_surfaces_an_abnormal_generation_stop(make_pipe):
    pipe, _ = make_pipe(
        [
            [
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": []},
                            "finishReason": "SAFETY",
                            "finishMessage": "Blocked by safety policy",
                        }
                    ]
                }
            ]
        ]
    )

    with pytest.raises(RuntimeError, match="SAFETY: Blocked by safety policy"):
        await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )


async def test_returns_partial_content_stopped_at_the_token_limit(make_pipe):
    pipe, _ = make_pipe(
        [
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Partial answer"}],
                            },
                            "finishReason": "MAX_TOKENS",
                        }
                    ]
                }
            ]
        ]
    )

    completion = await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": False,
            "messages": [{"role": "user", "content": "Write a lot"}],
        }
    )

    assert completion["choices"][0]["message"]["content"] == "Partial answer"


async def test_exposes_reasoning_and_token_usage(make_pipe):
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
    pipe, _ = make_pipe([response_stream])
    body = {
        "model": "pipe.gemini-3.8-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "Which option?"}],
    }

    stream = await pipe.pipe(body)
    chunks = [chunk async for chunk in stream]

    assert chunks[0]["choices"][0]["delta"] == {
        "reasoning_content": "Considering the options..."
    }
    assert chunks[1]["choices"][0]["delta"] == {"content": "Use the second option."}
    assert chunks[2]["choices"] == []
    expected_usage = {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    assert chunks[2]["usage"] == expected_usage

    pipe, _ = make_pipe([response_stream])
    completion = await pipe.pipe({**body, "stream": False})
    assert (
        completion["choices"][0]["message"]["reasoning_content"]
        == "Considering the options..."
    )
    assert completion["choices"][0]["message"]["content"] == "Use the second option."
    assert completion["usage"] == expected_usage
