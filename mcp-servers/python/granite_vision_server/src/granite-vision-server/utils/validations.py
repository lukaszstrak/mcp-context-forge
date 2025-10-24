#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/validations.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from __future__ import annotations
from .processing.image_analysis import ImageAnalysisRequest
from granite_vision_models import SUPPORTED_ANALYSIS_TYPES

SUPPORTED_PROVIDERS = {"ollama"}


def validate_request(req: ImageAnalysisRequest) -> None:
    if req.provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {req.provider}")
    if req.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported analysis_type: {req.analysis_type}. "
            f"Choose from: {sorted(SUPPORTED_ANALYSIS_TYPES)}"
        )
    if not isinstance(req.max_description_length, int) or req.max_description_length <= 0:
        raise ValueError("max_description_length must be a positive integer")
    if not isinstance(req.include_confidence, bool):
        raise ValueError("include_confidence must be a boolean")
    if not isinstance(req.language, str) or not req.language:
        raise ValueError("language must be a non-empty string (BCP-47 code)")