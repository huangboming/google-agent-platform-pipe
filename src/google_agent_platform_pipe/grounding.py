from typing import Any, Dict, List


def grounding_events(metadata: dict) -> List[Dict[str, Any]]:
    support_texts = _support_texts_by_chunk(metadata.get("groundingSupports") or [])
    events = []

    for index, chunk in enumerate(metadata.get("groundingChunks") or []):
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not isinstance(web, dict) or not web.get("uri"):
            continue

        uri = web["uri"]
        title = web.get("title") or web.get("domain") or uri
        document = "\n\n".join(support_texts.get(index, [])) or title
        events.append(
            {
                "type": "source",
                "data": {
                    "source": {"name": title, "id": uri, "url": uri},
                    "document": [document],
                    "metadata": [{"source": uri, "name": title, "url": uri}],
                },
            }
        )

    search_entry_point = metadata.get("searchEntryPoint") or {}
    rendered_content = search_entry_point.get("renderedContent")
    if rendered_content:
        events.append(
            {
                "type": "embeds",
                "data": {"embeds": [rendered_content], "replace": False},
            }
        )
    return events


def _support_texts_by_chunk(supports: list) -> Dict[int, List[str]]:
    texts = {}
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") or {}
        text = segment.get("text")
        if not text:
            continue
        for index in support.get("groundingChunkIndices") or []:
            if not isinstance(index, int):
                continue
            chunk_texts = texts.setdefault(index, [])
            if text not in chunk_texts:
                chunk_texts.append(text)
    return texts


async def emit_grounding(metadata: dict, emitter) -> None:
    if not emitter:
        return
    for event in grounding_events(metadata):
        try:
            await emitter(event)
        except Exception:
            # Citation delivery must not fail the model response.
            pass
