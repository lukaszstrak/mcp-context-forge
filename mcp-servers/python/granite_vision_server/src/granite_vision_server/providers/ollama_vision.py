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
import logging
from typing import Optional, Dict, Any, List

try:
    import ollama
    from ollama import Client
except ImportError:
    raise ImportError(
        "The 'ollama' package is required. Install it with: pip install ollama"
    )

logger = logging.getLogger(__name__)


class OllamaVisionClient:
    """Client for Ollama vision models using the official Ollama Python SDK."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        host = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = Client(host=host)
        self.timeout = timeout
        logger.info(f"Initialized OllamaVisionClient with host: {host}")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model dictionaries with at least 'model' or 'name' key
        """
        try:
            response = self.client.list()

            # The Ollama SDK (v0.4+) returns a ListResponse object with models attribute
            if hasattr(response, "models"):
                models = response.models
                # Convert Model objects to dicts for consistency
                result = []
                for m in models:
                    if hasattr(m, "model"):
                        # Extract key attributes from Model object
                        model_dict = {
                            "model": m.model,
                            "name": m.model,  # Add name alias for compatibility
                        }
                        if hasattr(m, "size"):
                            model_dict["size"] = m.size
                        if hasattr(m, "modified_at"):
                            model_dict["modified_at"] = str(m.modified_at)
                        if hasattr(m, "details"):
                            model_dict["details"] = str(m.details)
                        result.append(model_dict)
                    else:
                        # Fallback: add as-is
                        result.append(m if isinstance(m, dict) else {"model": str(m), "name": str(m)})
                return result
            # Legacy format: dict with 'models' key
            elif isinstance(response, dict) and "models" in response:
                return response["models"]
            else:
                logger.warning(f"Unexpected response format from ollama.list(): {type(response)}")
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            import traceback
            traceback.print_exc()
            return []

    def check_model_exists(self, model: str) -> bool:
        """
        Check if a specific model is available in Ollama.

        Args:
            model: Model name to check (e.g., "granite3.3-vision:2b")

        Returns:
            True if model exists, False otherwise
        """
        try:
            models = self.list_models()

            # Extract model names from dict entries
            model_names = []
            for m in models:
                if isinstance(m, dict):
                    # Try both 'model' and 'name' keys
                    name = m.get("model") or m.get("name", "")
                else:
                    # Fallback for non-dict items
                    name = str(m)

                if name:
                    model_names.append(name)

            exists = model in model_names

            if not exists:
                logger.warning(f"Model '{model}' not found in Ollama")
                logger.warning(f"Available models: {', '.join(model_names) if model_names else 'none'}")

            return exists
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_image(
        self,
        model: str,
        image_b64: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Analyze an image using Ollama vision model.

        Args:
            model: Model name (e.g., "granite3.3-vision:2b")
            image_b64: Base64-encoded image string (without data URI prefix)
            system_prompt: Optional system prompt
            user_prompt: User prompt for the analysis
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Model response as string

        Raises:
            RuntimeError: If the API call fails
        """
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": user_prompt,
                "images": [image_b64],  # Ollama SDK expects raw base64 strings
            })

            # Build options
            options: Dict[str, Any] = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            logger.info(f"Sending request to Ollama with model: {model}")

            # Call Ollama API using official SDK
            response = self.client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False,
            )

            # Extract content from response
            # The Ollama SDK returns a ChatResponse object
            if hasattr(response, "message"):
                # ChatResponse object with message attribute
                message = response.message
                if hasattr(message, "content"):
                    content = message.content
                    if content:
                        logger.info(f"Received response from Ollama ({len(content)} chars)")
                        return content
                    else:
                        logger.warning("Empty content in Ollama response")
                        return ""
                else:
                    logger.warning(f"Message has no content attribute: {type(message)}")
                    return str(message)
            elif isinstance(response, dict):
                # Legacy dict format
                message = response.get("message", {})
                content = message.get("content", "")
                if content:
                    logger.info(f"Received response from Ollama ({len(content)} chars)")
                    return content
                else:
                    logger.warning("Empty content in Ollama response")
                    return ""
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return str(response)

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to analyze image with Ollama: {e}") from e
