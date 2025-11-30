#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/document_extraction.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from pydantic import BaseModel, Field

from ..models import validate_model
from ..processing.document_processor import DocumentProcessor
from ..providers import get_provider


class DocumentExtractionRequest(BaseModel):
    document_data: str = Field(..., description="PDF, image, or multi-page document")
    model: str = Field(default="granite-vision-document-v1")
    provider: str = Field(default="watsonx")
    extraction_type: str = Field(default="full")  # full, text_only, structured, tables, forms
    preserve_layout: bool = Field(default=True)
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    output_format: str = Field(default="markdown")  # markdown, json, xml, plain_text

async def extract_document_content(req: DocumentExtractionRequest) -> str:
    validate_model(req.model)
    provider = get_provider(req.provider)
    processor = DocumentProcessor()
    pages = processor.extract_pages(req.document_data)
    results = processor.process_multi_page(pages, lambda p: provider.infer(req.model, p, f"Extract {req.extraction_type}"))
    # Combine results, apply format
    combined = "\n".join(results)
    if req.output_format == "markdown":
        return combined
    return combined
