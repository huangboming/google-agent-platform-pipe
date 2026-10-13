import json
import logging

import httpx
import pytest

from tests.helpers import text_response


@pytest.mark.parametrize("stream", [False, True])
async def test_authenticates_without_exposing_api_key_in_urls_or_logs(
    pipe_class, monkeypatch, caplog, stream
):
    api_key = "test-key-not-for-urls"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"data: {json.dumps(text_response('hello'))}\n\n",
        )

    original_client = httpx.AsyncClient

    def mock_client(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    caplog.set_level(logging.INFO, logger="httpx")
    pipe = pipe_class()
    pipe.valves.GOOGLE_API_KEY = api_key
    pipe.valves.API_BASE_URL = "https://aiplatform.googleapis.com"
    pipe.valves.MODELS = "gemini-3.8-flash"

    output = await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": stream,
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    if stream:
        chunks = [chunk async for chunk in output]
        assert (
            "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks)
            == "hello"
        )
    else:
        assert output["choices"][0]["message"]["content"] == "hello"

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        "gemini-3.8-flash:streamGenerateContent?alt=sse"
    )
    assert request.headers["x-goog-api-key"] == api_key

    log_messages = [
        record.getMessage() for record in caplog.records if record.name == "httpx"
    ]
    assert any("HTTP Request: POST" in message for message in log_messages)
    assert all(api_key not in message for message in log_messages)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "error_format", ["json", "text", "long-json", "truncation-boundary", "escaped-json"]
)
async def test_upstream_errors_redact_api_key_before_truncating(
    pipe_class, monkeypatch, caplog, stream, error_format
):
    api_key = (
        'test-"key-not-for-errors'
        if error_format == "escaped-json"
        else "test-key-not-for-errors"
    )
    message = f"Invalid API key: {api_key}"
    if error_format == "long-json":
        message += " More diagnostics." * 200
    elif error_format == "truncation-boundary":
        message = "x" * 1985 + api_key
    raw_error = (
        json.dumps({"error": {"message": message}}).encode()
        if error_format in ("json", "long-json", "escaped-json")
        else message.encode()
    )

    def respond(request):
        return httpx.Response(403, content=raw_error)

    original_client = httpx.AsyncClient

    def mock_client(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    caplog.set_level(logging.INFO, logger="httpx")
    pipe = pipe_class()
    pipe.valves.GOOGLE_API_KEY = api_key
    pipe.valves.MODELS = "gemini-3.8-flash"

    with pytest.raises(RuntimeError, match=r"request failed \(403\)") as raised:
        output = await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": stream,
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        if stream:
            _ = [chunk async for chunk in output]

    error = str(raised.value)
    assert "[REDACTED]" in error
    assert api_key not in error
    assert api_key[:12] not in error
    assert api_key not in caplog.text
    if error_format != "truncation-boundary":
        assert "Invalid API key" in error
