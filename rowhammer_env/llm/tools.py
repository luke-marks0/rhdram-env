from __future__ import annotations

from typing import Any


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dram.info",
            "description": "Return disclosed DRAM topology, command, mitigation, task, and budget metadata.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dram.read",
            "description": "Read bytes from a disclosed DRAM address.",
            "parameters": {
                "type": "object",
                "required": ["addr"],
                "properties": {
                    "addr": {"type": "object"},
                    "length": {"type": "integer", "minimum": 1, "maximum": 4096},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dram.write",
            "description": "Write base64 data to a disclosed DRAM address.",
            "parameters": {
                "type": "object",
                "required": ["addr", "data_b64"],
                "properties": {"addr": {"type": "object"}, "data_b64": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dram.issue",
            "description": "Issue RD, WR, or WAIT commands through the simulator worker.",
            "parameters": {
                "type": "object",
                "required": ["commands"],
                "properties": {"commands": {"type": "array", "items": {"type": "object"}, "maxItems": 4096}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "script.run",
            "description": "Run isolated python-rh-sdk code whose rh calls are brokered to the same tools.",
            "parameters": {
                "type": "object",
                "required": ["language", "code"],
                "properties": {
                    "language": {"const": "python-rh-sdk"},
                    "code": {"type": "string", "maxLength": 200000},
                    "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 60000},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "episode.finish",
            "description": "Finish the episode and receive reward from trusted environment state.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def tool_schema_by_name() -> dict[str, dict[str, Any]]:
    return {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}
