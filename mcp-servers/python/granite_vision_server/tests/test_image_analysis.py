#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_image_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import pytest
from unittest.mock import patch, MagicMock
from granite_vision_server.tools.image_analysis import analyze_image, ImageAnalysisRequest
from granite_vision_server.models.granite_vision_models import validate_model

@pytest.mark.asyncio
async def test_image_analysis():
    req = ImageAnalysisRequest(image_data="fake.jpg", model="granite-vision-general-v1", provider="ollama")
    
    mock_provider = MagicMock()
    mock_provider.infer.return_value = "Mock analysis result"
    
    mock_processor = MagicMock()
    mock_processor.preprocess.return_value = "processed_image"
    
    with patch('granite_vision_server.tools.image_analysis.get_provider', return_value=mock_provider), \
         patch('granite_vision_server.tools.image_analysis.ImageProcessor', return_value=mock_processor):
        
        result = await analyze_image(req)
        
        assert result == "Mock analysis result"[:req.max_description_length]
        mock_provider.infer.assert_called_once_with(req.model, "processed_image", "Analyze the image in general mode, language en")

@pytest.mark.asyncio
async def test_image_analysis_invalid_model():
    req = ImageAnalysisRequest(image_data="fake.jpg", model="invalid")
    with pytest.raises(ValueError):
        await analyze_image(req)