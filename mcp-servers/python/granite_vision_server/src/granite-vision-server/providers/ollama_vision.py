#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/ollama_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi
Granite Vision MCP Server - FastMCP Implementation

"""

from __future__ import annotations
import os
import requests
from typing import Optional, Dict, Any


class OllamaVisionClient:
    """Simple client for Ollama's /api/chat endpoint with image support."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

    def _chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Ollama error {resp.status_code}: {detail}")
        return resp.json()

    def analyze_image(
        self,
        model: str,
        image_b64: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_prompt,
            "images": [image_b64],  # raw base64 string
        })

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data = self._chat(payload)
        # Expected structure: { message: { role: 'assistant', content: '...' }, ... }
        msg = data.get("message") or {}
        content = msg.get("content", "")
        if not content:
            # Some models return an array of messages; fallback attempt
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                content = choices[0].get("message", {}).get("content", "")
        return content