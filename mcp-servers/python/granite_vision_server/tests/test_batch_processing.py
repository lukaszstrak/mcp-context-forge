#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_batch_processing.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

from unittest.mock import MagicMock, patch
import os.path

import pytest

from granite_vision_server.tools.batch_processing import BatchImageRequest, batch_process_images


@pytest.mark.asyncio
async def test_batch_processing():
    req = BatchImageRequest(images=[os.path.join("documents", "img1.jpg"), os.path.join("documents", "img2.jpg")])

    mock_provider = MagicMock()
    mock_provider.infer.side_effect = ["result1", "result2"]

    with patch('granite_vision_server.tools.batch_processing.get_provider', return_value=mock_provider), \
         patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:

        mock_map = MagicMock()
        mock_map.return_value = ["result1", "result2"]
        mock_executor.return_value.__enter__.return_value.map = mock_map

        result = await batch_process_images(req)

        assert result == {"results": ["result1", "result2"]}
        mock_map.assert_called_once()

@pytest.mark.asyncio
async def test_batch_processing_sequential():
    req = BatchImageRequest(images=["img1.jpg", "img2.jpg"], parallel_processing=False)

    mock_provider = MagicMock()
    mock_provider.infer.side_effect = ["result1", "result2"]

    with patch('granite_vision_server.tools.batch_processing.get_provider', return_value=mock_provider):

        result = await batch_process_images(req)

        assert result == {"results": ["result1", "result2"]}
