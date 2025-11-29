#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/server.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

import yaml
import logging
from fastmcp import FastMCP

from .tools.image_analysis import analyze_image
from .tools.document_extraction import extract_document_content
from .tools.vqa import visual_question_answering
from .tools.ocr import ocr_text_extraction
from .tools.chart_analysis import analyze_charts_graphs
from .tools.table_processing import process_tables
from .tools.batch_processing import batch_process_images
from .providers.ollama_vision import OllamaVisionProvider
from .providers.watsonx_vision import WatsonxVisionProvider
from .providers.huggingface_vision import HuggingFaceVisionProvider
from .providers.custom_endpoints import CustomEndpointsProvider


# Configure logging to stderr to avoid MCP protocol interference
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Create FastMCP serpipver instance
mcp = FastMCP("granite-vision-server")

providers = {}

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Register providers
if config["providers"]["ollama"]["vision_models_enabled"]:
    providers["ollama"] = OllamaVisionProvider(config["providers"]["ollama"])
providers["watsonx"] = WatsonxVisionProvider(config["providers"]["watsonx"])
providers["huggingface"] = HuggingFaceVisionProvider(config["providers"]["huggingface"])
providers["custom"] = CustomEndpointsProvider()

def get_provider(name):
    return providers.get(name)

mcp = FastMCP("granite-vision-server")

mcp.tool(analyze_image)
mcp.tool(extract_document_content)
mcp.tool(visual_question_answering)
mcp.tool(ocr_text_extraction)
mcp.tool(analyze_charts_graphs)
mcp.tool(process_tables)
mcp.tool(batch_process_images)

def main():
    """Main entry point for the FastMCP server."""
    # Standard
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