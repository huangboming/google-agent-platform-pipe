import pytest


def test_exposes_expected_default_models(make_pipe):
    pipe, _ = make_pipe()

    assert [model["id"] for model in pipe.pipes()] == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
    ]


def test_exposes_configured_models_without_duplicates(make_pipe):
    pipe, _ = make_pipe()
    pipe.valves.MODELS = "gemini-3.8-flash, gemini-3.5-flash-lite,gemini-3.8-flash"

    assert pipe.pipes() == [
        {"id": "gemini-3.8-flash", "name": "gemini-3.8-flash"},
        {
            "id": "gemini-3.5-flash-lite",
            "name": "gemini-3.5-flash-lite",
        },
    ]


async def test_rejects_a_model_that_is_not_exposed(make_pipe):
    pipe, client = make_pipe()

    with pytest.raises(ValueError, match="Unsupported Agent Platform model"):
        await pipe.pipe(
            {
                "model": "pipe.unconfigured-model",
                "stream": False,
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )

    assert client.requests == []
