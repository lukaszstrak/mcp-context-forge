#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/tools/table_processing.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from pydantic import BaseModel, Field
from ..providers import get_provider
from ..processing.table_processor import TableProcessor
from ..models import validate_model

class TableProcessingRequest(BaseModel):
    image_data: str = Field(...)
    model: str = Field(default="granite-vision-table-v1")
    provider: str = Field(default="ollama")
    extract_structure: bool = Field(default=True)
    extract_content: bool = Field(default=True)
    preserve_formatting: bool = Field(default=True)
    output_format: str = Field(default="csv")  # csv, json, html, markdown
    header_detection: bool = Field(default=True)

async def process_tables(req: TableProcessingRequest) -> str:
    validate_model(req.model)
    provider = get_provider(req.provider)
    processor = TableProcessor()
    result = processor.extract_table(req.image_data, req.output_format)
    return result