#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/processing/ocr_processor.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import pytesseract

from ..config import ocr_config


class OCRProcessor:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = ocr_config["tesseract_path"]

    def extract_text(self, image, languages):
        lang = "+".join(languages)
        return pytesseract.image_to_string(image, lang=lang)
