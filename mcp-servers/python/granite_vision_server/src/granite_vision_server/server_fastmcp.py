#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/server.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi, Uzma Siddiqui
Granite Vision MCP Server - FastMCP Implementation

"""
import logging
import sys
from typing import Any, Dict

from fastmcp import FastMCP
from pydantic import Field

from .tools.image_analysis import ImageAnalysisRequest
from .processing.image_processor import process_image_analysis

# Configure logging to stderr to avoid MCP protocol interference
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Create FastMCP serpipver instance
mcp = FastMCP("granite-vision-server")


@mcp.tool(description="General image analysis and understanding")
async def analyze_image(
    image_data: str = Field(..., description="base64, file path, or URL"),
    model: str = Field("granite-vision-general-v1", description="Model to use for analysis"),
    provider: str = Field("ollama", description="Provider to use for analysis"),
    analysis_type: str = Field("general", description="general, detailed, objects, scene, text"),
    include_confidence: bool = Field(True, description="Whether to include confidence scores in the output"),
    max_description_length: int = Field(200, description="Maximum length of the description"),
    language: str = Field("en", description="Language for the output")
) -> Dict[str, Any]:
    
    # Placeholder implementation
    logger.info(f"Analyzing image with model {model} from provider {provider}")

    req = ImageAnalysisRequest(
        image_data=image_data,
        model=model,
        provider=provider,
        analysis_type=analysis_type,
        include_confidence=include_confidence,
        max_description_length=max_description_length,
        language=language,
    )

    return await process_image_analysis(req)    

def main():
    """Main entry point for the FastMCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Granite Vision MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (stdio or http)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=9004, help="HTTP port")

    args = parser.parse_args()

    if args.transport == "http":
        logger.info(f"Starting Granite Vision MCP Server on HTTP at {args.host}:{args.port}")
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        logger.info("Starting Granite Vision MCP Server on stdio")
        mcp.run()


if __name__ == "__main__":
    main()
