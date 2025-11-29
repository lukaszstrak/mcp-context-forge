#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/image_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass


@dataclass
class ImageAnalysisRequest:
    image_data: str  # base64, file path, or URL
    model: str = "granite-vision-general-v1"
    provider: str = "ollama"
    analysis_type: str = "general"  # general, detailed, objects, scene, text
    include_confidence: bool = True
    max_description_length: int = 200
    language: str = "en"
