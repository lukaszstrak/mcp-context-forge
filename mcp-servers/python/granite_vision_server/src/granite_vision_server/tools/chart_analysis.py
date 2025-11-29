#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite-vision-server/src/granite-vision-server/tools/chart_analysis.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""


from pydantic import BaseModel, Field

from ..models import validate_model
from ..providers import get_provider


class ChartAnalysisRequest(BaseModel):
    image_data: str = Field(...)
    model: str = Field(default="granite-vision-chart-v1")
    provider: str = Field(default="watsonx")
    chart_type: str = Field(default="auto")  # auto, bar, line, pie, scatter, histogram
    extract_data: bool = Field(default=True)
    extract_labels: bool = Field(default=True)
    extract_legend: bool = Field(default=True)
    output_format: str = Field(default="json")  # json, csv, description

async def analyze_charts_graphs(req: ChartAnalysisRequest) -> str | dict:
    validate_model(req.model)
    provider = get_provider(req.provider)
    prompt = f"Analyze chart of type {req.chart_type}"
    result = provider.infer(req.model, req.image_data, prompt)
    # Format output
    if req.output_format == "json":
        return {"data": result}
    return result
