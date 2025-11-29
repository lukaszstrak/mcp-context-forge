#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/image_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
from pydantic import BaseModel, Field

from ..models import validate_model
from ..processing.image_processor import ImageProcessor
from ..providers import get_provider


class ImageAnalysisRequest(BaseModel):
    image_data: str = Field(..., description="base64, file path, or URL")
    model: str = Field(default="granite-vision-general-v1")
    provider: str = Field(default="ollama")
    analysis_type: str = Field(default="general")  # general, detailed, objects, scene, text
    include_confidence: bool = Field(default=True)
    max_description_length: int = Field(default=200)
    language: str = Field(default="en")

async def analyze_image(req: ImageAnalysisRequest) -> str:
    validate_model(req.model)
    provider = get_provider(req.provider)
    processor = ImageProcessor()
    image = processor.preprocess(req.image_data)
    prompt = f"Analyze the image in {req.analysis_type} mode, language {req.language}"
    result = provider.infer(req.model, image, prompt)
    if req.include_confidence:
        result += " (confidence: high)"  # Placeholder
    return result[:req.max_description_length]
