#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/tools/batch_processing.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

import concurrent.futures

from pydantic import BaseModel, Field

from ..models import validate_model
from ..providers import get_provider


class BatchImageRequest(BaseModel):
    images: list[str] = Field(...)  # Multiple image data sources
    model: str = Field(default="granite-vision-general-v1")
    provider: str = Field(default="ollama")
    processing_type: str = Field(default="analyze")  # analyze, ocr, extract, classify
    parallel_processing: bool = Field(default=True)
    max_concurrent: int = Field(default=4)
    aggregate_results: bool = Field(default=True)

async def batch_process_images(req: BatchImageRequest) -> list[str] | Dict:
    validate_model(req.model)
    provider = get_provider(req.provider)
    def process_image(img):
        return provider.infer(req.model, img, f"Process with {req.processing_type}")

    if req.parallel_processing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=req.max_concurrent) as executor:
            results = list(executor.map(process_image, req.images))
    else:
        results = [process_image(img) for img in req.images]

    if req.aggregate_results:
        return {"results": results}
    return results
