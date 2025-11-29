#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/image_utils.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
import base64
import io

import requests
from PIL import Image


def load_image(image_data):
    if image_data.startswith("http"):
        response = requests.get(image_data)
        return Image.open(io.BytesIO(response.content))
    elif image_data.startswith("data:"):
        # base64
        _, data = image_data.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data)))
    else:
        # file path
        return Image.open(image_data)

def save_image(image, path):
    image.save(path)
