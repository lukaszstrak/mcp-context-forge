#!/usr/bin/env python3
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/watsonx_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
from ibm_watson_machine_learning import APIClient


class WatsonxVisionProvider:
    def __init__(self, config):
        self.client = APIClient({
            "url": config["url"],
            "apikey": config["apikey"]
        })
        self.project_id = config["project_id"]
        self.vision_endpoint = config["vision_endpoint"]

    def infer(self, model, image_data, prompt, **kwargs):
        # Implement Watsonx vision inference
        params = {
            "model_id": model,
            "input": prompt,
            "image": image_data,
            "project_id": self.project_id,
            **kwargs
        }
        response = self.client.inferences.create(deployment_id="vision", params=params)
        return response["results"]
