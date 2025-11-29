#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/models/granite_vision_models.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

# Future
from __future__ import annotations

# Standard
from typing import Tuple

# Supported analysis types
SUPPORTED_ANALYSIS_TYPES = {"general", "detailed", "objects", "scene", "text"}

# Map logical model names to Ollama model identifiers. You can adjust these
# to match the local models you have pulled into Ollama.
MODEL_ALIASES_OLLAMA = {
    # Logical Granite default -> pick a strong vision model available in Ollama
    "granite-vision-general-v1": "granite3.3-vision:2b",
    "granite-vision-document-v1": "granite3.3-vision:2b",
    "granite-vision-ocr-v1": "granite3.3-vision:2b",
    "granite-vision-chart-v1": "granite3.3-vision:2b",
    "granite-vision-table-v1": "granite3.3-vision:2b",
    "granite-multimodal-8b": "granite3.3-vision:2b",
}


def resolve_model_for_provider(model: str, provider: str) -> str:
    if provider != "ollama":
        raise ValueError(f"Unsupported provider: {provider}")
    # Pass-through if caller already gave a valid Ollama tag with colon
    if ":" in model:
        return model
    # Check if it's a known Granite model alias
    if model in MODEL_ALIASES_OLLAMA:
        return MODEL_ALIASES_OLLAMA[model]
    # Pass through unknown models as-is
    return model


SYSTEM_PROMPT_TEMPLATE = "You are a meticulous, multilingual computer vision analyst. " "Always return **only** strict JSON. No prose, no markdown."

USER_PROMPT_TEMPLATE = (
    "You are given an image. Perform a {analysis_label} analysis and respond in JSON with keys: "
    "'summary' (string, <= {max_len} chars), "
    "'objects' (array of strings), 'scene' (string), 'text_in_image' (string), "
    "{confidence_key} "
    "'language' (BCP-47 code).\n\n"
    "Guidelines:\n"
    "- Language of the response: {language}.\n"
    "- Keep 'summary' concise (<= {max_len} chars).\n"
    "- If unsure about any field, use an empty string or empty array.\n"
    "- Output strict JSON only (no markdown)."
)

ANALYSIS_LABELS = {
    "general": "general high-level description",
    "detailed": "detailed, fine-grained description including attributes and relations",
    "objects": "list of salient objects/classes and their roles",
    "scene": "scene and setting description",
    "text": "OCR-like extraction of visible text and its context",
}


def build_prompts(analysis_type: str, language: str, max_desc: int, include_confidence: bool) -> Tuple[str, str]:
    label = ANALYSIS_LABELS.get(analysis_type, ANALYSIS_LABELS["general"])
    confidence_key = "'confidence' (float in [0,1]), " if include_confidence else ""
    user = USER_PROMPT_TEMPLATE.format(
        analysis_label=label,
        max_len=max_desc,
        language=language,
        confidence_key=confidence_key,
    )
    return SYSTEM_PROMPT_TEMPLATE, user
