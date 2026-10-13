from tests.helpers import text_response


async def test_preserves_reusable_tool_schema_definitions(make_pipe):
    pipe, client = make_pipe([[text_response("Found it.")]])

    await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": False,
            "messages": [{"role": "user", "content": "Find Paris."}],
        },
        __tools__={
            "lookup": {
                "spec": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"$ref": "#/$defs/Location"}},
                        "$defs": {
                            "Location": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            }
                        },
                    },
                }
            }
        },
    )

    parameters = client.requests[0]["payload"]["tools"][0]["functionDeclarations"][0][
        "parameters"
    ]
    assert parameters["properties"]["location"] == {"ref": "#/defs/Location"}
    assert parameters["defs"]["Location"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


async def test_executes_tool_and_continues_the_model_turn(make_pipe):
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
    pipe, client = make_pipe(
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
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        },
        __event_emitter__=emit,
    )

    assert result["choices"][0]["message"]["content"] == "Paris is 20 C."
    assert result["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 11,
        "total_tokens": 41,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    assert calls == ["Paris"]
    assert [event["data"]["done"] for event in statuses] == [False, True]

    follow_up = client.requests[1]["payload"]
    model_part = follow_up["contents"][-2]["parts"][0]
    assert model_part["thoughtSignature"] == "signature"
    function_response = follow_up["contents"][-1]["parts"][0]["functionResponse"]
    assert function_response["id"] == "call-1"
    assert function_response["response"]["temperature"] == 20


async def test_internal_tasks_do_not_receive_tools(make_pipe):
    pipe, client = make_pipe([[text_response("A short title")]])

    await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": False,
            "messages": [{"role": "user", "content": "Create a title"}],
        },
        __tools__={
            "search_web": {"type": "builtin"},
            "fetch_url": {"type": "builtin"},
            "ask_user": {
                "type": "builtin",
                "spec": {"name": "ask_user"},
            },
            "weather": {
                "spec": {
                    "name": "weather",
                    "parameters": {"type": "object"},
                }
            },
        },
        __metadata__={"features": {"web_search": True}},
        __task__="title_generation",
    )

    assert "tools" not in client.requests[0]["payload"]
