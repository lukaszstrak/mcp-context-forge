#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_ocr.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import pytest
from unittest.mock import patch, MagicMock
from granite_vision_server.tools.ocr import ocr_text_extraction, OCRRequest

@pytest.mark.asyncio
async def test_ocr():
    req = OCRRequest(image_data="fake.png")
    
    mock_ocr = MagicMock()
    mock_ocr.extract_text.return_value = "Extracted text"
    
    with patch('granite_vision_server.tools.ocr.get_provider'), \
         patch('granite_vision_server.tools.ocr.OCRProcessor', return_value=mock_ocr):
        
        result = await ocr_text_extraction(req)
        
        assert result == "Extracted text"
        mock_ocr.extract_text.assert_called_once_with("fake.png", ["en"])

@pytest.mark.asyncio
async def test_ocr_with_boxes():
    req = OCRRequest(image_data="fake.png", bounding_boxes=True)
    
    mock_ocr = MagicMock()
    mock_ocr.extract_text.return_value = "Text"
    
    with patch('granite_vision_server.tools.ocr.get_provider'), \
         patch('granite_vision_server.tools.ocr.OCRProcessor', return_value=mock_ocr):
        
        result = await ocr_text_extraction(req)
        
        assert result == {"text": "Text", "boxes": []}