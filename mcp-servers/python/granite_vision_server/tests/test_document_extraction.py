#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_document_extraction.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

from unittest.mock import MagicMock, patch

import pytest

from granite_vision_server.tools.document_extraction import (
    DocumentExtractionRequest,
    extract_document_content,
)


@pytest.mark.asyncio
async def test_document_extraction():
    req = DocumentExtractionRequest(document_data="fake.pdf", model="granite-vision-document-v1", provider="watsonx")

    mock_provider = MagicMock()
    mock_provider.infer.return_value = "extracted_page"

    mock_processor = MagicMock()
    mock_processor.extract_pages.return_value = ["page1", "page2"]
    mock_processor.process_multi_page.return_value = ["extracted_page1", "extracted_page2"]

    with patch('granite_vision_server.tools.document_extraction.get_provider', return_value=mock_provider), \
         patch('granite_vision_server.tools.document_extraction.DocumentProcessor', return_value=mock_processor):

        result = await extract_document_content(req)

        assert result == "extracted_page1\nextracted_page2"
        mock_processor.process_multi_page.assert_called_once()

@pytest.mark.asyncio
async def test_document_extraction_unsupported_format():
    req = DocumentExtractionRequest(document_data="fake.txt")
    with pytest.raises(ValueError):
        await extract_document_content(req)
