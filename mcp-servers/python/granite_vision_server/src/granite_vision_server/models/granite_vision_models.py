#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/models/granite_vision_models.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

SUPPORTED_MODELS = [
    "granite-vision-document-v1",
    "granite-docvqa-8b",
    "granite-vision-general-v1",
    "granite-multimodal-8b",
    "granite-vision-ocr-v1",
    "granite-text-extraction",
    "granite-vision-chart-v1",
    "granite-vision-table-v1"
]

def validate_model(model):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")