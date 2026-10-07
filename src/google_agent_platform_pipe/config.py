import os

from pydantic import BaseModel, Field


class Valves(BaseModel):
    GOOGLE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Agent Platform API key.",
        json_schema_extra={"input_type": "password"},
    )
    API_BASE_URL: str = Field(
        default="https://aiplatform.googleapis.com",
        description="Google Agent Platform API base URL.",
    )
    MODELS: str = Field(
        default=(
            "gemini-3.8-flash,gemini-3.5-flash-lite,"
            "gemini-3.1-pro-preview"
        ),
        description="Comma-separated Agent Platform model IDs.",
    )
    THINKING_LEVEL: str = Field(
        default="",
        description="Optional Gemini thinking level: MINIMAL, LOW, MEDIUM, or HIGH.",
    )
    MAX_TOOL_ITERATIONS: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum model/tool round trips per response.",
    )
    REQUEST_TIMEOUT_SECONDS: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Upstream connection and response timeout in seconds.",
    )
    MAX_INLINE_MEDIA_MB: int = Field(
        default=18,
        ge=1,
        le=100,
        description="Maximum total size of inline message and uploaded media.",
    )
