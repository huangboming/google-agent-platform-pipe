import base64
import json

import pytest

from tests.helpers import text_response


async def test_sends_original_pdf_without_open_webui_rag_context(make_pipe, tmp_path):
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
        __files__=[
            {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "path": str(path),
                    "meta": {"content_type": "application/pdf"},
                    "data": {"content": "EXTRACTED FILE DATA MUST NOT BE USED"},
                },
            }
        ],
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


async def test_rejects_source_context_without_the_original_prompt(make_pipe, tmp_path):
    pipe, client = make_pipe()
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.7\noriginal document bytes")

    with pytest.raises(ValueError, match="without preserving the original prompt"):
        await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "EXTRACTED PDF TEXT"}],
            },
            __files__=[
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "path": str(path),
                        "meta": {"content_type": "application/pdf"},
                    },
                }
            ],
            __metadata__={"sources": [{"document": ["EXTRACTED PDF TEXT"]}]},
        )

    assert client.requests == []


async def test_rejects_an_unavailable_uploaded_file(make_pipe):
    pipe, client = make_pipe()

    with pytest.raises(ValueError, match="is not available on the Open WebUI server"):
        await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "Read it"}],
            },
            __files__=[
                {
                    "type": "file",
                    "file": {
                        "filename": "missing.pdf",
                        "path": "/missing/document.pdf",
                        "meta": {"content_type": "application/pdf"},
                    },
                }
            ],
        )

    assert client.requests == []


async def test_rejects_an_unsupported_uploaded_file_type(make_pipe, tmp_path):
    pipe, client = make_pipe()
    path = tmp_path / "document.docx"
    path.write_bytes(b"not a native Gemini document")

    with pytest.raises(ValueError, match="unsupported media type.*application/vnd"):
        await pipe.pipe(
            {
                "model": "pipe.gemini-3.8-flash",
                "stream": False,
                "messages": [{"role": "user", "content": "Read it"}],
            },
            __files__=[
                {
                    "type": "file",
                    "file": {
                        "filename": "document.docx",
                        "path": str(path),
                        "meta": {
                            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        },
                    },
                }
            ],
        )

    assert client.requests == []
