#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/custom_endpoints.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""
import requests

class CustomEndpointsProvider:
    def __init__(self):
        self.endpoints = {}  # Dict of custom endpoints

    def register_endpoint(self, name, url):
        self.endpoints[name] = url

    def infer(self, endpoint_name, image_data, prompt, **kwargs):
        url = self.endpoints.get(endpoint_name)
        if not url:
            raise ValueError("Endpoint not registered")
        payload = {"prompt": prompt, "image": image_data, **kwargs}
        response = requests.post(url, json=payload)
        return response.json()["result"]