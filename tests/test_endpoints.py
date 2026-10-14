import json

import httpx
import pytest

from tests.helpers import text_response


@pytest.fixture
def http_requests(monkeypatch):
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
    return requests


@pytest.mark.parametrize(
    ("project_id", "location", "base_url", "expected_url"),
    [
        (
            "",
            "",
            "",
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            " \t ",
            "\n",
            " https://aiplatform.googleapis.com/// ",
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            " test-project ",
            " us-central1 ",
            " https://aiplatform.googleapis.com/// ",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/us-central1/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "test-project",
            "europe-west4",
            "",
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/europe-west4/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "123456789",
            "global",
            "",
            "https://aiplatform.googleapis.com/v1/projects/123456789/"
            "locations/global/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "example.com:test-project",
            "global",
            "",
            "https://aiplatform.googleapis.com/v1/projects/example.com%3Atest-project/"
            "locations/global/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "",
            "",
            " https://proxy.example/agent-platform/// ",
            "https://proxy.example/agent-platform/v1/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "test-project",
            "global",
            " https://proxy.example/agent-platform/// ",
            "https://proxy.example/agent-platform/v1/projects/test-project/"
            "locations/global/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
        (
            "test-project",
            "us-central1",
            " https://proxy.example/agent-platform/// ",
            "https://proxy.example/agent-platform/v1/projects/test-project/"
            "locations/us-central1/publishers/google/models/"
            "gemini-3.8-flash:streamGenerateContent?alt=sse",
        ),
    ],
    ids=[
        "express-auto",
        "express-whitespace",
        "full-whitespace",
        "full-auto-regional",
        "full-project-number",
        "full-domain-project",
        "express-proxy",
        "full-global-proxy",
        "full-regional-proxy",
    ],
)
async def test_routes_project_location_and_custom_endpoints(
    pipe_class, http_requests, project_id, location, base_url, expected_url
):
    pipe = pipe_class()
    pipe.valves.GOOGLE_API_KEY = "test-key"
    pipe.valves.PROJECT_ID = project_id
    pipe.valves.LOCATION = location
    pipe.valves.API_BASE_URL = base_url
    pipe.valves.MODELS = "gemini-3.8-flash"

    output = await pipe.pipe({"messages": [{"role": "user", "content": "Hi"}]})

    assert output["choices"][0]["message"]["content"] == "hello"
    assert len(http_requests) == 1
    assert str(http_requests[0].url) == expected_url
    assert http_requests[0].headers["x-goog-api-key"] == "test-key"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("project_id", "location", "error"),
    [
        ("test-project", "", "PROJECT_ID and LOCATION must both be set"),
        ("", "global", "PROJECT_ID and LOCATION must both be set"),
        ("test-project", " \t ", "PROJECT_ID and LOCATION must both be set"),
        ("\n", "us-central1", "PROJECT_ID and LOCATION must both be set"),
        ("..", "global", "PROJECT_ID"),
        ("../test-project", "global", "PROJECT_ID"),
        ("test-project?key=secret", "global", "PROJECT_ID"),
        ("test-project", "..", "LOCATION"),
        ("test-project", "US-CENTRAL1", "LOCATION"),
        ("test-project", "evil.example/path", "LOCATION"),
        ("test-project", "global?key=secret", "LOCATION"),
    ],
)
async def test_rejects_incomplete_or_unsafe_configuration_before_a_request(
    pipe_class, http_requests, stream, project_id, location, error
):
    pipe = pipe_class()
    pipe.valves.GOOGLE_API_KEY = "test-key"
    pipe.valves.PROJECT_ID = project_id
    pipe.valves.LOCATION = location

    with pytest.raises(ValueError, match=error):
        await pipe.pipe(
            {"stream": stream, "messages": [{"role": "user", "content": "Hi"}]}
        )

    assert http_requests == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("location", ["global", "us-central1"])
@pytest.mark.parametrize("status_code", [403, 404, 429, 500])
async def test_full_request_errors_do_not_fall_back_to_express(
    pipe_class, monkeypatch, stream, location, status_code
):
    api_key = "test-key-not-for-errors"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status_code, json={"error": {"message": f"Request rejected: {api_key}"}}
        )

    original_client = httpx.AsyncClient

    def mock_client(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    pipe = pipe_class()
    pipe.valves.GOOGLE_API_KEY = api_key
    pipe.valves.PROJECT_ID = "test-project"
    pipe.valves.LOCATION = location
    pipe.valves.MODELS = "gemini-3.8-flash"

    with pytest.raises(
        RuntimeError, match=rf"request failed \({status_code}\)"
    ) as raised:
        output = await pipe.pipe(
            {"stream": stream, "messages": [{"role": "user", "content": "Hi"}]}
        )
        if stream:
            _ = [chunk async for chunk in output]

    assert len(requests) == 1
    assert requests[0].url.path == (
        f"/v1/projects/test-project/locations/{location}/publishers/google/models/"
        "gemini-3.8-flash:streamGenerateContent"
    )
    assert api_key not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)
