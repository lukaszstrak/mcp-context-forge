#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/validations.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
from ..config import processing_config

def validate_input(image_data):
    # Check size, format
    if not image_data.split(".")[-1] in processing_config["supported_formats"]:
        raise ValueError("Unsupported format")
    # Size check placeholder
    return True