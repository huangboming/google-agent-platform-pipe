import json
from typing import Any, Dict, List

from pydantic import BaseModel

_SCHEMA_FIELDS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "propertyOrdering",
    "default",
    "ref",
    "defs",
}
_SCHEMA_FIELD_ALIASES = {
    "$ref": "ref",
    "$defs": "defs",
    "definitions": "defs",
}


def _clean_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_clean_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned = {}
    for key, item in value.items():
        field = _SCHEMA_FIELD_ALIASES.get(key, key)
        if field not in _SCHEMA_FIELDS:
            continue
        if field in ("properties", "defs") and isinstance(item, dict):
            cleaned[field] = {
                name: _clean_schema(schema) for name, schema in item.items()
            }
        elif field == "ref" and isinstance(item, str):
            cleaned[field] = item.replace("#/$defs/", "#/defs/").replace(
                "#/definitions/", "#/defs/"
            )
        else:
            cleaned[field] = _clean_schema(item)

    schema_type = cleaned.get("type")
    if isinstance(schema_type, list) and "null" in schema_type:
        non_null = [item for item in schema_type if item != "null"]
        cleaned["type"] = non_null[0] if len(non_null) == 1 else non_null
        cleaned["nullable"] = True
    return cleaned


def _tool_spec(tool: dict) -> dict:
    spec = tool.get("spec") or {}
    if spec.get("type") == "function":
        spec = spec.get("function") or {}
    return spec


def tool_arguments(arguments: dict, tool: dict) -> dict:
    parameters = _tool_spec(tool).get("parameters") or {}
    properties = parameters.get("properties") or {}
    return {
        name: value
        for name, value in arguments.items()
        if name in properties and not name.startswith("__")
    }


def tool_declarations(tools: dict) -> List[Dict[str, Any]]:
    declarations = []
    for fallback_name, tool in tools.items():
        if not isinstance(tool, dict):
            continue
        spec = _tool_spec(tool)

        declaration = {"name": spec.get("name") or fallback_name}
        if spec.get("description"):
            declaration["description"] = spec["description"]
        if isinstance(spec.get("parameters"), dict):
            declaration["parameters"] = _clean_schema(spec["parameters"])
        declarations.append(declaration)
    return declarations


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    return json.loads(json.dumps(value, default=str))


async def execute_tool(call: dict, tools: dict) -> Dict[str, Any]:
    name = call.get("name", "")
    tool = tools.get(name)
    response: Dict[str, Any]

    if not isinstance(tool, dict) or not callable(tool.get("callable")):
        response = {"error": f"Tool is unavailable: {name}"}
    else:
        arguments = call.get("args", {})
        if not isinstance(arguments, dict):
            response = {"error": "Tool arguments must be an object."}
        else:
            try:
                result = await tool["callable"](**tool_arguments(arguments, tool))
                normalized = _json_value(result)
                response = (
                    normalized
                    if isinstance(normalized, dict)
                    else {"result": normalized}
                )
            except Exception as exc:
                response = {"error": str(exc)}

    function_response = {"name": name, "response": response}
    if call.get("id"):
        function_response["id"] = call["id"]
    return {"functionResponse": function_response}
