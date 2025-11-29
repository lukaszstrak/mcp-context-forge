#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/models/model_capabilities.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

CAPABILITIES = {
    "granite-vision-document-v1": ["document_understanding", "extraction"],
    "granite-docvqa-8b": ["docvqa"],
    "granite-vision-general-v1": ["image_analysis"],
    "granite-multimodal-8b": ["vqa", "multimodal"],
    "granite-vision-ocr-v1": ["ocr"],
    "granite-text-extraction": ["text_extraction"],
    "granite-vision-chart-v1": ["chart_analysis"],
    "granite-vision-table-v1": ["table_processing"]
}

def get_capabilities(model):
    return CAPABILITIES.get(model, [])
