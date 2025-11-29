#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/tests/test_huggingface_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - FastMCP Implementation
"""

# Standard
import asyncio

# Third-Party
from granite_vision_server.providers import huggingface_vision


async def main():

    image_data = "documents/INVOICE-3.png"

    print("Testing analyze_image for", image_data)
    res1 = await huggingface_vision.analyze_image(model="granite3.2-vision:latest", image_data=image_data, analysis_type="general", include_conf=True, lang="en")


if __name__ == "__main__":
    asyncio.run(main())
