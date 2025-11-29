#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/utils/format_converters.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
import json


def to_markdown(data):
    return str(data)  # Placeholder

def to_json(data):
    return json.dumps(data)

def to_csv(df):
    return df.to_csv(index=False)
