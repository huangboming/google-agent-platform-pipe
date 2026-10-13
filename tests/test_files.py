import base64
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from tests.helpers import text_response

BODY = {
    "model": "pipe.gemini-3.8-flash",
    "stream": False,
    "messages": [{"role": "user", "content": "Read it"}],
}


@pytest.fixture
def webui_files(monkeypatch, pipe_class):
    class UserModel(BaseModel):
        id: str
        role: str

    class FileModel(BaseModel):
        id: str
        user_id: str
        filename: str
        path: str | None
        meta: dict
        data: dict

    records = {}
    lookup = AsyncMock(side_effect=lambda file_id: records.get(file_id))
    access = AsyncMock(return_value=False)
    module_name = (
        "google_agent_platform_pipe.files"
        if pipe_class.__module__ == "google_agent_platform_pipe.pipe"
        else pipe_class.__module__
    )
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "Files", SimpleNamespace(get_file_by_id=lookup))
    monkeypatch.setattr(module, "UserModel", UserModel)
    monkeypatch.setattr(module, "has_access_to_file", access)

    def add(path, *, file_id="file-id", owner="user", content_type="application/pdf"):
        record = FileModel(
            id=file_id,
            user_id=owner,
            filename="document.pdf",
            path=str(path) if path is not None else None,
            meta={"content_type": content_type},
            data={"content": "EXTRACTED FILE DATA MUST NOT BE USED"},
        )
        records[file_id] = record
        return {"type": "file", "id": file_id, "file": record.model_dump()}

    return SimpleNamespace(
        add=add,
        records=records,
        lookup=lookup,
        access=access,
        user={"id": "user", "role": "user"},
        module=module,
    )


async def test_sends_original_pdf_without_open_webui_rag_context(
    make_pipe, tmp_path, webui_files
):
    pipe, client = make_pipe([[text_response("The original PDF.")]])
    pdf = b"%PDF-1.7\noriginal document bytes"
    image = base64.b64encode(b"image bytes").decode()
    path = tmp_path / "document.pdf"
    path.write_bytes(pdf)

    completion = await pipe.pipe(
        {
            "model": "pipe.gemini-3.8-flash",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Base system prompt. EXTRACTED SYSTEM CONTEXT",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "EXTRACTED PDF TEXT FROM OPEN WEBUI",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        },
                    ],
                },
            ],
        },
        __files__=[webui_files.add(path)],
        __user__=webui_files.user,
        __metadata__={
            "user_prompt": "Read the original PDF.",
            "system_prompt": "Base system prompt.",
            "sources": [{"document": ["EXTRACTED RETRIEVAL CHUNK"]}],
        },
        __tools__={
            name: {"type": "builtin"}
            for name in (
                "list_chat_files",
                "query_chat_files",
                "grep_chat_files",
                "view_file",
            )
        }
        | {
            "weather": {
                "spec": {
                    "name": "weather",
                    "parameters": {"type": "object"},
                }
            }
        },
    )

    assert completion["choices"][0]["message"]["content"] == "The original PDF."
    payload = client.requests[0]["payload"]
    assert payload["systemInstruction"]["parts"] == [{"text": "Base system prompt."}]
    parts = payload["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "application/pdf"
    assert base64.b64decode(parts[0]["inlineData"]["data"]) == pdf
    assert parts[1] == {"text": "Read the original PDF."}
    assert parts[2]["inlineData"]["data"] == image
    assert "EXTRACTED" not in json.dumps(payload)
    assert [
        declaration["name"]
        for declaration in payload["tools"][0]["functionDeclarations"]
    ] == ["weather"]


@pytest.mark.parametrize("shape", ["nested", "flat", "id-only"])
async def test_uses_authorized_server_file_instead_of_client_metadata(
    make_pipe, tmp_path, webui_files, shape
):
    pipe, client = make_pipe([[text_response("Read the authorized file.")]])
    path = tmp_path / "authorized.pdf"
    path.write_bytes(b"authorized file bytes")
    other_path = tmp_path / "private.txt"
    other_path.write_bytes(b"PRIVATE SERVER DATA")
    attachment = webui_files.add(path)
    attachment["file"].update(
        {
            "path": str(other_path),
            "filename": "private.txt",
            "meta": {"content_type": "text/plain"},
        }
    )
    if shape == "flat":
        attachment = attachment["file"]
    elif shape == "id-only":
        attachment = {"type": "file", "id": "file-id", "path": str(other_path)}

    await pipe.pipe(BODY, __files__=[attachment], __user__=webui_files.user)

    part = client.requests[0]["payload"]["contents"][0]["parts"][0]["inlineData"]
    assert part["mimeType"] == "application/pdf"
    assert base64.b64decode(part["data"]) == b"authorized file bytes"


async def test_rejects_another_users_file_despite_forged_ownership(
    make_pipe, tmp_path, webui_files
):
    pipe, client = make_pipe()
    path = tmp_path / "private.pdf"
    path.write_bytes(b"private bytes")
    attachment = webui_files.add(path, owner="other-user")
    attachment["file"]["user_id"] = "user"

    with pytest.raises(ValueError, match="unavailable or access is denied"):
        await pipe.pipe(BODY, __files__=[attachment], __user__=webui_files.user)

    assert client.requests == []
    webui_files.access.assert_awaited_once()
    file_id, permission, user = webui_files.access.await_args.args
    assert (file_id, permission, user.id) == ("file-id", "read", "user")


@pytest.mark.parametrize("access_type", ["shared", "admin"])
async def test_preserves_open_webui_shared_and_admin_file_access(
    make_pipe, tmp_path, webui_files, access_type
):
    pipe, client = make_pipe([[text_response("Read it.")]])
    path = tmp_path / "shared.pdf"
    path.write_bytes(b"shared file bytes")
    attachment = webui_files.add(path, owner="other-user")
    user = webui_files.user.copy()
    if access_type == "shared":
        webui_files.access.return_value = True
    else:
        user["role"] = "admin"

    await pipe.pipe(BODY, __files__=[attachment], __user__=user)

    data = client.requests[0]["payload"]["contents"][0]["parts"][0]["inlineData"][
        "data"
    ]
    assert base64.b64decode(data) == b"shared file bytes"


@pytest.mark.parametrize("failure", ["missing-id", "unknown-id", "missing-user"])
async def test_rejects_unverifiable_uploaded_files(
    make_pipe, tmp_path, webui_files, failure
):
    pipe, client = make_pipe()
    path = tmp_path / "private.pdf"
    path.write_bytes(b"private bytes")
    attachment = webui_files.add(path)
    user = webui_files.user
    if failure == "missing-id":
        del attachment["id"]
        del attachment["file"]["id"]
        message = "must have an Open WebUI file ID"
    elif failure == "unknown-id":
        webui_files.records.clear()
        message = "unavailable or access is denied"
    else:
        user = None
        message = "user context is required"

    with pytest.raises(ValueError, match=message):
        await pipe.pipe(BODY, __files__=[attachment], __user__=user)

    assert client.requests == []


async def test_rejects_files_when_platform_access_apis_are_unavailable(
    make_pipe, webui_files, monkeypatch
):
    pipe, client = make_pipe()
    monkeypatch.setattr(webui_files.module, "Files", None)

    with pytest.raises(ValueError, match="require Open WebUI file access APIs"):
        await pipe.pipe(
            BODY,
            __files__=[{"type": "file", "id": "file-id", "path": "/private"}],
            __user__=webui_files.user,
        )

    webui_files.lookup.assert_not_awaited()
    assert client.requests == []


async def test_rejects_source_context_without_the_original_prompt(
    make_pipe, tmp_path, webui_files
):
    pipe, client = make_pipe()
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.7\noriginal document bytes")

    with pytest.raises(ValueError, match="without preserving the original prompt"):
        await pipe.pipe(
            BODY,
            __files__=[webui_files.add(path)],
            __user__=webui_files.user,
            __metadata__={"sources": [{"document": ["EXTRACTED PDF TEXT"]}]},
        )

    assert client.requests == []


@pytest.mark.parametrize("path", [None, "/missing/document.pdf"])
async def test_rejects_an_unavailable_uploaded_file(make_pipe, webui_files, path):
    pipe, client = make_pipe()
    message = "has no accessible path" if path is None else "is not available"

    with pytest.raises(ValueError, match=message):
        await pipe.pipe(
            BODY, __files__=[webui_files.add(path)], __user__=webui_files.user
        )

    assert client.requests == []


async def test_rejects_an_unsupported_server_file_type_despite_client_mime(
    make_pipe, tmp_path, webui_files
):
    pipe, client = make_pipe()
    path = tmp_path / "document.docx"
    path.write_bytes(b"not a native Gemini document")
    attachment = webui_files.add(
        path,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    attachment["file"]["meta"]["content_type"] = "application/pdf"

    with pytest.raises(ValueError, match="unsupported media type.*application/vnd"):
        await pipe.pipe(BODY, __files__=[attachment], __user__=webui_files.user)

    assert client.requests == []


async def test_uses_only_the_authorized_server_gcs_uri(make_pipe, webui_files):
    pipe, client = make_pipe([[text_response("Read the object.")]])
    attachment = webui_files.add("gs://authorized-bucket/document.pdf")
    attachment["file"]["path"] = "gs://untrusted-bucket/private.pdf"

    await pipe.pipe(BODY, __files__=[attachment], __user__=webui_files.user)

    assert client.requests[0]["payload"]["contents"][0]["parts"][0] == {
        "fileData": {
            "mimeType": "application/pdf",
            "fileUri": "gs://authorized-bucket/document.pdf",
        }
    }


async def test_internal_tasks_do_not_resolve_uploaded_files(make_pipe, webui_files):
    pipe, client = make_pipe([[text_response("A short title")]])

    await pipe.pipe(
        BODY,
        __files__=[{"type": "file", "id": "private-file", "path": "/private"}],
        __task__="title_generation",
    )

    webui_files.lookup.assert_not_awaited()
    assert client.requests[0]["payload"]["contents"][0]["parts"] == [
        {"text": "Read it"}
    ]
