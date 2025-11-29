#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/huggingface_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi, Uzma Siddiqui
Granite Vision MCP Server - FastMCP Implementation

"""

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from typing import Dict, Any

# https://huggingface.co/ibm-granite/granite-vision-3.2-2b
# pip install transformers>=4.49


# ------------------------
# Image Analysis
# ------------------------
async def analyze_image(
    model: str,
    image_data: str, #path to image file
    analysis_type: str,
    include_conf: bool,
    lang: str) -> Dict[str, Any]:
    """
    Calls Ollama vision model (granite3.2-vision) to analyze an image.
    """
    if not model:
        model = "ibm-granite/granite-vision-3.2-2b"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_str, use_fast=True)
    vision_model = AutoModelForImageTextToText.from_pretrained(model).to(device)

    text_prompt = "Describe the image in detail. If the image has animate or inanimate objects, provide a detailed description of the scene and all visible objects, and List all objects and their relationships, and describe the setting, environment, and lighting of the scene. Describe any text present in the image. If there is text in the image, extrapolate the meaning and purpose of the document, given its text content"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_data},
                {"type": "text", "text": text_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # autoregressively complete prompt
    output = vision_model.generate(**inputs, max_new_tokens=100)
    result = processor.decode(output[0], skip_special_tokens=True)

    return {
        "description": result.strip(),
        "tags": [],
        "objects": [],
        "confidences": []
    }