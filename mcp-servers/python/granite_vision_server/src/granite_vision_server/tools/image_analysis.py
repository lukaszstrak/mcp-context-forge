#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/image_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from dataclasses import dataclass
from ..providers import get_provider
from ..processing.image_processor import ImageProcessor
from ..models import validate_model

@dataclass
class ImageAnalysisRequest:
    image_data: str
    model: str = "granite-vision-general-v1"
    provider: str = "ollama"
    analysis_type: str = "general"
    include_confidence: bool = True
    max_description_length: int = 200
    language: str = "en"

class ImageAnalysisTool:
    name = "analyze_image"

    async def execute(self, req: ImageAnalysisRequest):
        validate_model(req.model)
        provider = get_provider(req.provider)
        processor = ImageProcessor()
        image = processor.preprocess(req.image_data)
        prompt = f"Analyze the image in {req.analysis_type} mode, language {req.language}"
        result = provider.infer(req.model, image, prompt)
        if req.include_confidence:
            result += " (confidence: high)"  # Placeholder
        return result[:req.max_description_length]