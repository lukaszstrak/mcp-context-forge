#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_chart_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

from unittest.mock import MagicMock, patch

import pytest

from granite_vision_server.tools.chart_analysis import ChartAnalysisRequest, analyze_charts_graphs


@pytest.mark.asyncio
async def test_chart_analysis():
    req = ChartAnalysisRequest(image_data="fake.jpg")

    mock_provider = MagicMock()
    mock_provider.infer.return_value = "Chart data"

    with patch('granite_vision_server.tools.chart_analysis.get_provider', return_value=mock_provider):

        result = await analyze_charts_graphs(req)

        assert result == {"data": "Chart data"}
        mock_provider.infer.assert_called_once_with(req.model, "fake.jpg", "Analyze chart of type auto")

@pytest.mark.asyncio
async def test_chart_analysis_different_format():
    req = ChartAnalysisRequest(image_data="fake.jpg", output_format="description")

    mock_provider = MagicMock()
    mock_provider.infer.return_value = "Description"

    with patch('granite_vision_server.tools.chart_analysis.get_provider', return_value=mock_provider):

        result = await analyze_charts_graphs(req)

        assert result == "Description"
