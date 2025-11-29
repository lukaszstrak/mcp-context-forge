#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/format_converters.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

# Future
from __future__ import annotations

# Standard
import json
from typing import Any


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        # Try to locate the first/last braces as a recovery attempt
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return None


def _clip(s: str, n: int) -> str:
    if n is None or n <= 0:
        return s
    return s[:n]


def normalize_response(
    raw_text: str,
    analysis_type: str,
    language: str,
    include_confidence: bool,
    max_desc: int,
) -> dict[str, Any]:
    """Normalize provider text into a consistent JSON result schema."""
    parsed = _safe_json_loads(raw_text)

    summary = ""
    objects: list[str] = []
    scene = ""
    text_in_image = ""
    confidence = None

    if isinstance(parsed, dict):
        summary = str(parsed.get("summary", ""))
        scene = str(parsed.get("scene", ""))
        text_in_image = str(parsed.get("text_in_image", ""))
        # objects can be array or comma-delimited string
        obj_val = parsed.get("objects", [])
        if isinstance(obj_val, list):
            objects = [str(x) for x in obj_val]
        elif isinstance(obj_val, str):
            objects = [x.strip() for x in obj_val.split(",") if x.strip()]
        if include_confidence:
            c = parsed.get("confidence")
            try:
                confidence = float(c) if c is not None else None
            except Exception:
                confidence = None
    else:
        # Fallback: no JSON — treat raw text as summary
        summary = raw_text

    summary = _clip(summary, max_desc)

    result: dict[str, Any] = {
        "summary": summary,
        "objects": objects,
        "scene": scene,
        "text_in_image": text_in_image,
        "language": language,
    }
    if include_confidence:
        result["confidence"] = confidence

    # Always include raw for debugging/traceability
    result["raw"] = raw_text

    return result
