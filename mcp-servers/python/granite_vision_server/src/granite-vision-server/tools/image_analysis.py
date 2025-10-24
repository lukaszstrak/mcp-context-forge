#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/image_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from image_processor import process_image_analysis


@dataclass
class ImageAnalysisRequest:
    image_data: str  # base64, file path, or URL
    model: str = "granite-vision-general-v1"
    provider: str = "ollama"
    analysis_type: str = "general"  # general, detailed, objects, scene, text
    include_confidence: bool = True
    max_description_length: int = 200
    language: str = "en"


async def analyze_image(req: ImageAnalysisRequest) -> Dict:
    """Entry point used by the MCP tool handler."""
    return await process_image_analysis(req)