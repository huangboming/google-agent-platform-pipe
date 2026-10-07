import re
from typing import List


_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def parse_models(value: str) -> List[str]:
    models = []
    for item in value.split(","):
        model = item.strip()
        if not model or model in models:
            continue
        if not _MODEL_ID_PATTERN.fullmatch(model):
            raise ValueError(f"Invalid Agent Platform model ID: {model!r}")
        models.append(model)
    if not models:
        raise ValueError("At least one Agent Platform model must be configured.")
    return models


def select_model(requested: str, configured: List[str]) -> str:
    if not requested:
        return configured[0]
    if requested in configured:
        return requested
    for model in configured:
        if requested.endswith(f".{model}"):
            return model
    raise ValueError(f"Unsupported Agent Platform model: {requested!r}")
