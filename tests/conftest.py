import pytest

from tests.helpers import (
    DistributionPipe,
    RecordingAgentPlatformClient,
    SourcePipe,
)


@pytest.fixture(
    params=(
        pytest.param(SourcePipe, id="source"),
        pytest.param(DistributionPipe, id="distribution"),
    )
)
def pipe_class(request):
    return request.param


@pytest.fixture
def make_pipe(pipe_class):
    def factory(response_streams=None):
        client = RecordingAgentPlatformClient(response_streams)
        pipe = pipe_class(client=client)
        pipe.valves.GOOGLE_API_KEY = "test-key"
        return pipe, client

    return factory
