#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/vqa.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from pydantic import BaseModel, Field
from typing import Optional
from ..providers import get_provider
from ..models import validate_model

class VQARequest(BaseModel):
    image_data: str = Field(...)
    question: str = Field(...)
    model: str = Field(default="granite-multimodal-8b")
    provider: str = Field(default="ollama")
    context: Optional[str] = Field(default=None)  # Additional context
    answer_type: str = Field(default="natural")  # natural, short, boolean, multiple_choice
    confidence_threshold: float = Field(default=0.7)

async def visual_question_answering(req: VQARequest) -> str:
    validate_model(req.model)
    provider = get_provider(req.provider)
    prompt = req.question
    if req.context:
        prompt += f" Context: {req.context}"
    result = provider.infer(req.model, req.image_data, prompt)
    # Check confidence placeholder
    if "confidence" in result and result["confidence"] < req.confidence_threshold:
        return "Low confidence answer"
    return result