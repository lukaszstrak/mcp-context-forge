#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_table_processing.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

import pytest
from unittest.mock import patch, MagicMock
from granite_vision_server.tools.table_processing import process_tables, TableProcessingRequest

@pytest.mark.asyncio
async def test_table_processing():
    req = TableProcessingRequest(image_data="fake.png")
    
    mock_processor = MagicMock()
    mock_processor.extract_table.return_value = "table_csv"
    
    with patch('granite_vision_server.tools.table_processing.get_provider'), \
         patch('granite_vision_server.tools.table_processing.TableProcessor', return_value=mock_processor):
        
        result = await process_tables(req)
        
        assert result == "table_csv"
        mock_processor.extract_table.assert_called_once_with("fake.png", "csv")