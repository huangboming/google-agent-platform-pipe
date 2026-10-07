import pytest


async def test_uses_native_google_search_and_emits_grounding(make_pipe):
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
                    "searchEntryPoint": {"renderedContent": "<div>Google Search</div>"},
                },
            }
        ]
    }
    pipe, client = make_pipe([[response]])
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

    assert completion["choices"][0]["message"]["content"] == "The grounded answer."
    assert client.requests[0]["payload"]["tools"] == [{"googleSearch": {}}]
    assert events[0]["type"] == "source"
    assert events[0]["data"]["document"] == ["The grounded answer."]
    assert events[0]["data"]["metadata"][0]["url"] == "https://example.com/source"
    assert events[1] == {
        "type": "embeds",
        "data": {
            "embeds": ["<div>Google Search</div>"],
            "replace": False,
        },
    }


async def test_rejects_other_tools_with_native_google_search(make_pipe):
    pipe, client = make_pipe()

    with pytest.raises(
        ValueError, match="cannot be combined with other tools: weather"
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

    assert client.requests == []
