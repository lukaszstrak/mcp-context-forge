#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/ocr.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""


from pydantic import BaseModel, Field

from ..models import validate_model
from ..processing.ocr_processor import OCRProcessor
from ..providers import get_provider


class OCRRequest(BaseModel):
    image_data: str = Field(...)
    model: str = Field(default="granite-vision-ocr-v1")
    provider: str = Field(default="huggingface")
    languages: list[str] = Field(default_factory=lambda: ["en"])  # Supported languages
    preserve_formatting: bool = Field(default=True)
    extract_handwriting: bool = Field(default=False)
    confidence_filtering: bool = Field(default=True)
    bounding_boxes: bool = Field(default=False)
    text_regions: list[dict] | None = Field(default=None)  # Specific regions to process

async def ocr_text_extraction(req: OCRRequest) -> str | dict:
    validate_model(req.model)
    provider = get_provider(req.provider)
    ocr = OCRProcessor()
    text = ocr.extract_text(req.image_data, req.languages)
    if req.bounding_boxes:
        # Placeholder
        return {"text": text, "boxes": []}
    return text
