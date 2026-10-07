import asyncio
import base64
import mimetypes
from pathlib import Path


async def uploaded_file_parts(files: list, initial_bytes: int, max_bytes: int):
    parts = []
    used_bytes = 0
    for wrapper in files:
        file_info = wrapper.get("file", wrapper) if isinstance(wrapper, dict) else {}
        if not isinstance(file_info, dict):
            continue

        path_value = file_info.get("path")
        meta = file_info.get("meta") or {}
        mime_type = (
            meta.get("content_type")
            or file_info.get("content_type")
            or mimetypes.guess_type(file_info.get("filename", ""))[0]
        )
        if not path_value or not mime_type:
            continue
        if str(path_value).startswith("gs://"):
            parts.append(
                {"fileData": {"mimeType": mime_type, "fileUri": path_value}}
            )
            continue

        path = Path(path_value)
        if not path.is_file():
            continue
        file_size = path.stat().st_size
        ensure_media_limit(initial_bytes + used_bytes + file_size, max_bytes)
        data = await asyncio.to_thread(path.read_bytes)
        used_bytes += len(data)
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            }
        )
    return parts, used_bytes


def ensure_media_limit(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise ValueError(
            f"Inline media exceeds the configured {max_bytes // (1024 * 1024)} MB limit."
        )
