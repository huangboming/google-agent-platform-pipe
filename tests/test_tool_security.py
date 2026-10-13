from functools import partial

import pytest

from tests.helpers import text_response

BODY = {
    "model": "pipe.gemini-3.8-flash",
    "stream": False,
    "messages": [{"role": "user", "content": "Use the selected tool"}],
}


def tool_calls_response(*calls):
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": call} for call in calls],
                }
            }
        ]
    }


@pytest.mark.parametrize("nested_spec", [False, True])
async def test_filters_tool_arguments_without_overriding_injected_context(
    make_pipe, nested_spec
):
    query = {"__user__": "ordinary nested document data"}
    trusted_user = {"id": "trusted-user", "role": "user"}
    trusted_request = object()
    observed = []

    async def lookup(query, __user__, __request__, **extra):
        observed.append((query, __user__, __request__, extra))
        return {"user_id": __user__["id"]}

    spec = {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "object"},
                "__user__": {"type": "object"},
                "__request__": {"type": "string"},
            },
        },
    }
    if nested_spec:
        spec = {"type": "function", "function": spec}
    pipe, client = make_pipe(
        [
            [
                tool_calls_response(
                    {
                        "name": "lookup",
                        "args": {
                            "query": query,
                            "__user__": {"id": "forged-user", "role": "admin"},
                            "__request__": "forged-request",
                            "undeclared": "extra argument",
                        },
                    }
                )
            ],
            [text_response("Done.")],
        ]
    )

    await pipe.pipe(
        BODY,
        __tools__={
            "lookup": {
                "spec": spec,
                "callable": partial(
                    lookup, __user__=trusted_user, __request__=trusted_request
                ),
            }
        },
    )

    assert observed == [(query, trusted_user, trusted_request, {})]
    response = client.requests[1]["payload"]["contents"][-1]["parts"][0]
    assert response["functionResponse"]["response"] == {"user_id": "trusted-user"}


@pytest.mark.parametrize("parameters", [None, {}])
async def test_tools_without_declared_parameters_do_not_receive_extra_arguments(
    make_pipe, parameters
):
    observed = []

    async def tool(**kwargs):
        observed.append(kwargs)
        return {"ok": True}

    spec = {"name": "tool"}
    if parameters is not None:
        spec["parameters"] = parameters
    pipe, _ = make_pipe(
        [
            [tool_calls_response({"name": "tool", "args": {"undeclared": "value"}})],
            [text_response("Done.")],
        ]
    )

    await pipe.pipe(BODY, __tools__={"tool": {"spec": spec, "callable": tool}})

    assert observed == [{}]


@pytest.mark.parametrize("arguments", [None, [], 0, False, "invalid"])
async def test_malformed_tool_arguments_do_not_execute_the_tool(make_pipe, arguments):
    observed = []

    async def tool():
        observed.append("executed")
        return {"ok": True}

    pipe, client = make_pipe(
        [
            [tool_calls_response({"name": "tool", "args": arguments})],
            [text_response("Done.")],
        ]
    )

    await pipe.pipe(
        BODY, __tools__={"tool": {"spec": {"name": "tool"}, "callable": tool}}
    )

    assert observed == []
    response = client.requests[1]["payload"]["contents"][-1]["parts"][0]
    assert response["functionResponse"]["response"] == {
        "error": "Tool arguments must be an object."
    }


@pytest.mark.parametrize("stream", [False, True])
async def test_ask_mode_refuses_the_whole_tool_batch_without_execution(
    make_pipe, stream
):
    observed = []
    events = []

    async def tool():
        observed.append("executed")
        return {"ok": True}

    async def emit(event):
        events.append(event)

    pipe, client = make_pipe(
        [
            [
                tool_calls_response(
                    {"name": "write", "args": {}}, {"name": "read", "args": {}}
                )
            ]
        ]
    )

    with pytest.raises(RuntimeError, match="requires approval.*No tools were executed"):
        output = await pipe.pipe(
            BODY | {"stream": stream},
            __tools__={
                name: {"spec": {"name": name}, "callable": tool}
                for name in ("write", "read")
            },
            __metadata__={"params": {"tool_approval_mode": "ask"}},
            __event_emitter__=emit,
        )
        if stream:
            _ = [chunk async for chunk in output]

    assert observed == []
    assert events == []
    assert len(client.requests) == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_ask_mode_still_allows_answers_without_tool_execution(make_pipe, stream):
    pipe, _ = make_pipe([[text_response("A normal answer.")]])

    output = await pipe.pipe(
        BODY | {"stream": stream},
        __metadata__={"params": {"tool_approval_mode": "ask"}},
    )

    if stream:
        chunks = [chunk async for chunk in output]
        assert (
            "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks)
            == "A normal answer."
        )
    else:
        assert output["choices"][0]["message"]["content"] == "A normal answer."


async def test_full_mode_preserves_authorized_tool_execution(make_pipe):
    observed = []

    async def tool():
        observed.append("executed")
        return {"ok": True}

    pipe, _ = make_pipe(
        [
            [tool_calls_response({"name": "tool", "args": {}})],
            [text_response("Done.")],
        ]
    )

    await pipe.pipe(
        BODY,
        __tools__={"tool": {"spec": {"name": "tool"}, "callable": tool}},
        __metadata__={"params": {"tool_approval_mode": "full"}},
    )

    assert observed == ["executed"]


@pytest.mark.parametrize("failure_stage", ["callable", "serialization"])
async def test_tool_errors_do_not_forward_private_diagnostics(make_pipe, failure_stage):
    secret = "private-tool-token"

    class UnserializableResult:
        def __str__(self):
            raise RuntimeError(f"Private serialization diagnostic: {secret}")

    async def tool():
        if failure_stage == "callable":
            raise RuntimeError(f"Private tool diagnostic: {secret}")
        return UnserializableResult()

    pipe, client = make_pipe(
        [
            [tool_calls_response({"name": "tool", "args": {}})],
            [text_response("The tool failed.")],
        ]
    )

    completion = await pipe.pipe(
        BODY, __tools__={"tool": {"spec": {"name": "tool"}, "callable": tool}}
    )

    assert completion["choices"][0]["message"]["content"] == "The tool failed."
    response = client.requests[1]["payload"]["contents"][-1]["parts"][0]
    assert response["functionResponse"]["response"] == {
        "error": "Tool execution failed."
    }
    assert secret not in str(client.requests)
