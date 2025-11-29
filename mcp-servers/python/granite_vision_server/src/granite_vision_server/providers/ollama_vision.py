#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/ollama_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
from ollama import Client

class OllamaVisionProvider:
    def __init__(self, config):
        self.client = Client(host=config["base_url"], timeout=config["timeout"])

    def infer(self, model, image_data, prompt, **kwargs):
        # Implement Ollama vision inference
        response = self.client.generate(model=model, prompt=prompt, images=[image_data], **kwargs)
        return response["response"]

    # Add methods for other tool types