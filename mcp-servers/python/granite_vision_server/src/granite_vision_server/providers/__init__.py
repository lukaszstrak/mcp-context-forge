#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/providers/__init__.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import yaml

from .custom_endpoints import CustomEndpointsProvider
from .huggingface_vision import HuggingFaceVisionProvider
from .ollama_vision import OllamaVisionProvider
from .watsonx_vision import WatsonxVisionProvider


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

config = load_config()

providers = {}

# Register providers
if config["providers"]["ollama"]["vision_models_enabled"]:
    providers["ollama"] = OllamaVisionProvider(config["providers"]["ollama"])
providers["watsonx"] = WatsonxVisionProvider(config["providers"]["watsonx"])
providers["huggingface"] = HuggingFaceVisionProvider(config["providers"]["huggingface"])
providers["custom"] = CustomEndpointsProvider()

def get_provider(name):
    return providers.get(name)
