#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/processing/image_processor.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""
# Future
from __future__ import annotations

# Local
from ..models.granite_vision_models import build_prompts, resolve_model_for_provider
from ..providers.ollama_vision import OllamaVisionClient
from ..tools.image_analysis import ImageAnalysisRequest
from ..utils.format_converters import normalize_response
from ..utils.image_utils import ensure_base64
from ..utils.validations import validate_request


async def process_image_analysis(req: ImageAnalysisRequest):
    """Orchestrates: validation -> prompt building -> provider call -> normalization.
    Returns a provider-agnostic normalized JSON dict.
    """
    validate_request(req)

    # Preprocess image to base64 string (no data URI header)
    img_b64 = ensure_base64(req.image_data)

    # Resolve model & prompts
    model_name = resolve_model_for_provider(req.model, req.provider)
    system_prompt, user_prompt = build_prompts(
        analysis_type=req.analysis_type,
        language=req.language,
        max_desc=req.max_description_length,
        include_confidence=req.include_confidence,
    )

    # Provider call (only Ollama for now)
    if req.provider == "ollama":
        client = OllamaVisionClient()
        raw_text = client.analyze_image(
            model=model_name,
            image_b64=img_b64,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    else:
        raise ValueError(f"Unsupported provider: {req.provider}")

    # Normalize output
    normalized = normalize_response(
        raw_text=raw_text,
        analysis_type=req.analysis_type,
        language=req.language,
        include_confidence=req.include_confidence,
        max_desc=req.max_description_length,
    )

    return {
        "ok": True,
        "provider": req.provider,
        "model": model_name,
        "requested_model": req.model,
        "analysis_type": req.analysis_type,
        "language": req.language,
        "result": normalized,
    }
