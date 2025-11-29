#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_vqa.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

from unittest.mock import MagicMock, patch

import pytest

from granite_vision_server.tools.vqa import VQARequest, visual_question_answering


@pytest.mark.asyncio
async def test_vqa():
    req = VQARequest(image_data="fake.jpg", question="What is this?")

    mock_provider = MagicMock()
    mock_provider.infer.return_value = "It's a mock"

    with patch('granite_vision_server.tools.vqa.get_provider', return_value=mock_provider):

        result = await visual_question_answering(req)

        assert result == "It's a mock"
        mock_provider.infer.assert_called_once_with(req.model, req.image_data, "What is this?")

@pytest.mark.asyncio
async def test_vqa_with_context():
    req = VQARequest(image_data="fake.jpg", question="What?", context="Test context")

    mock_provider = MagicMock()
    mock_provider.infer.return_value = "Answer"

    with patch('granite_vision_server.tools.vqa.get_provider', return_value=mock_provider):

        result = await visual_question_answering(req)

        mock_provider.infer.assert_called_once_with(req.model, req.image_data, "What? Context: Test context")
