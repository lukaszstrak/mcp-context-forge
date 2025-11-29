#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/processing/document_processor.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

from pdf2image import convert_from_path
from pypdf import PdfReader

class DocumentProcessor:
    def __init__(self):
        pass

    def extract_pages(self, document_data):
        if document_data.endswith(".pdf"):
            return convert_from_path(document_data)
        else:
            raise ValueError("Unsupported document format")

    def process_multi_page(self, pages, func):
        results = []
        for page in pages:
            results.append(func(page))
        return results