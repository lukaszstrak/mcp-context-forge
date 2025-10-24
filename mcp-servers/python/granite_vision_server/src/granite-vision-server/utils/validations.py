#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/validations.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from __future__ import annotations
from ..tools.image_analysis import ImageAnalysisRequest
from ..models.granite_vision_models import SUPPORTED_ANALYSIS_TYPES

SUPPORTED_PROVIDERS = {"ollama"}


def validate_request(req: ImageAnalysisRequest) -> None:
    if req.provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {req.provider}")
    if req.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported analysis_type: {req.analysis_type}. "
            f"Choose from: {sorted(SUPPORTED_ANALYSIS_TYPES)}"
        )