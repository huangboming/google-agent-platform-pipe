import copy
import importlib.util
import sys
from pathlib import Path

from google_agent_platform_pipe import Pipe as SourcePipe

__all__ = (
    "DistributionPipe",
    "RecordingAgentPlatformClient",
    "SourcePipe",
    "text_response",
)

ROOT = Path(__file__).resolve().parents[1]


def load_distribution_pipe():
    path = ROOT / "dist" / "google_agent_platform.py"
    spec = importlib.util.spec_from_file_location(
        "google_agent_platform_pipe_distribution", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generated Pipe from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Pipe


DistributionPipe = load_distribution_pipe()


class RecordingAgentPlatformClient:
    """Record the public upstream contract without making network requests."""

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
