#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/init.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from .image_analysis import ImageAnalysisTool
from .document_extraction import DocumentExtractionTool
from .vqa import VQATool
from .batch_processing import BatchProcessingTool
# Add missing ones
from .ocr import OCRTool
from .chart_analysis import ChartAnalysisTool
from .table_processing import TableProcessingTool