import asyncio
import base64
import mimetypes
from pathlib import Path

try:
    from open_webui.models.files import Files
    from open_webui.models.users import UserModel
    from open_webui.utils.access_control.files import has_access_to_file
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("open_webui"):
        raise
    Files = UserModel = has_access_to_file = None

_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "audio/mp3",
    "audio/mpeg",
    "audio/wav",
    "image/jpeg",
    "image/png",
    "image/webp",
    "text/plain",
    "video/avi",
    "video/flv",
    "video/mov",
    "video/mp4",
    "video/mpeg",
    "video/mpegps",
    "video/mpg",
    "video/wmv",
}
_MIME_ALIASES = {
    "audio/x-wav": "audio/wav",
    "image/jpg": "image/jpeg",
    "video/quicktime": "video/mov",
    "video/x-flv": "video/flv",
    "video/x-msvideo": "video/avi",
    "video/x-ms-wmv": "video/wmv",
}


def _file_info(wrapper):
    if not isinstance(wrapper, dict):
        return None
    if wrapper.get("type") not in (None, "file"):
        return None
    file_info = wrapper.get("file", wrapper)
    return file_info if isinstance(file_info, dict) else None


def has_uploaded_files(files: list) -> bool:
    return any(_file_info(wrapper) is not None for wrapper in files)


def _display_name(wrapper: dict, file_info: dict) -> str:
    return str(
        file_info.get("filename")
        or wrapper.get("name")
        or file_info.get("path")
        or "uploaded file"
    )


def _mime_type(file_info: dict) -> str:
    meta = file_info.get("meta") or {}
    value = (
        meta.get("content_type")
        or file_info.get("content_type")
        or mimetypes.guess_type(str(file_info.get("filename", "")))[0]
        or ""
    )
    normalized = str(value).split(";", 1)[0].strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)


async def uploaded_file_parts(files: list, max_bytes: int, user: dict | None):
    if not has_uploaded_files(files):
        return [], 0
    if not isinstance(user, dict) or not user.get("id"):
        raise ValueError(
            "Open WebUI user context is required to access uploaded files."
        )

    if Files is None or UserModel is None or has_access_to_file is None:
        raise ValueError("Uploaded files require Open WebUI file access APIs.")

    current_user = UserModel(**user)
    parts = []
    used_bytes = 0
    for wrapper in files:
        file_info = _file_info(wrapper)
        if file_info is None:
            continue

        file_id = file_info.get("id") or wrapper.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Uploaded files must have an Open WebUI file ID.")

        file = await Files.get_file_by_id(file_id)
        if file is None or not (
            file.user_id == current_user.id
            or current_user.role == "admin"
            or await has_access_to_file(file_id, "read", current_user)
        ):
            raise ValueError("Uploaded file is unavailable or access is denied.")

        file_info = file.model_dump()
        name = _display_name(wrapper, file_info)
        path_value = file_info.get("path")
        if not path_value:
            raise ValueError(f'Uploaded file "{name}" has no accessible path.')

        mime_type = _mime_type(file_info)
        if mime_type not in _SUPPORTED_MIME_TYPES:
            label = mime_type or "unknown"
            raise ValueError(
                f'Uploaded file "{name}" has unsupported media type "{label}".'
            )

        path_text = str(path_value)
        if path_text.startswith("gs://"):
            parts.append({"fileData": {"mimeType": mime_type, "fileUri": path_text}})
            continue

        path = Path(path_value)
        if not path.is_file():
            raise ValueError(
                f'Uploaded file "{name}" is not available on the Open WebUI server. '
                "Use local file storage or a gs:// object accessible to Agent Platform."
            )

        try:
            file_size = path.stat().st_size
            ensure_media_limit(used_bytes + file_size, max_bytes)
            data = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise ValueError(f'Uploaded file "{name}" could not be read.') from exc

        used_bytes += len(data)
        ensure_media_limit(used_bytes, max_bytes)
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
