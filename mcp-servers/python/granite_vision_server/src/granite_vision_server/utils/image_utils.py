#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/image_utils.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

# Future
from __future__ import annotations

# Standard
import base64
import os
import re
from typing import Tuple

try:
    # Third-Party
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,(?P<b64>[A-Za-z0-9+/=]+)$")


def _strip_data_uri(value: str) -> Tuple[bool, str]:
    m = DATA_URI_RE.match(value)
    if m:
        return True, m.group("b64")
    return False, value


def _is_probable_base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
        return True
    except Exception:
        return False


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _download_bytes(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("'requests' is required to fetch images from URLs")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def ensure_base64(image_data: str) -> str:
    """Accepts base64/data URI, filesystem path, or URL. Returns raw base64 string (no data URI header)."""
    image_data = image_data.strip()

    # 1) data URI
    is_data_uri, payload = _strip_data_uri(image_data)
    if is_data_uri:
        return payload

    # 2) looks like pure base64 already
    if _is_probable_base64(image_data):
        return image_data

    # 3) file path
    if os.path.exists(image_data):
        data = _read_file_bytes(image_data)
        return base64.b64encode(data).decode("ascii")

    # 4) URL
    if image_data.lower().startswith(("http://", "https://")):
        data = _download_bytes(image_data)
        return base64.b64encode(data).decode("ascii")

    raise ValueError("Unsupported image_data format. Provide base64, data URI, file path, or URL.")
