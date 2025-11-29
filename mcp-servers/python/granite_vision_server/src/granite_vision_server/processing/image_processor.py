#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/processing/image_processor.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import cv2

from ..utils.image_utils import load_image


class ImageProcessor:
    def __init__(self):
        pass

    def preprocess(self, image_data):
        image = load_image(image_data)
        # Resize, normalize, etc.
        image = cv2.resize(image, (224, 224))
        return image

    def postprocess(self, output):
        return output.strip()
