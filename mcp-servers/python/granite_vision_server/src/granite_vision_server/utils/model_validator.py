#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/granite_vision_server/src/granite_vision_server/utils/model_validator.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Granite Vision MCP Server - Model Validation Utilities
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any

from ..providers.ollama_vision import OllamaVisionClient

logger = logging.getLogger(__name__)


def validate_model_available(
    model: str,
    provider: str = "ollama",
    base_url: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate that a model is available in the specified provider.

    Args:
        model: Model name to validate (e.g., "granite3.3-vision:2b")
        provider: Provider name (currently only "ollama" is supported)
        base_url: Optional custom base URL for the provider

    Returns:
        Tuple of (is_valid, message):
            - is_valid: True if model exists and is ready
            - message: Success message or error description
    """
    if provider != "ollama":
        return False, f"Unsupported provider: {provider}. Only 'ollama' is supported."

    try:
        client = OllamaVisionClient(base_url=base_url)

        # Check if model exists
        exists = client.check_model_exists(model)

        if exists:
            logger.info(f"Model '{model}' is available in Ollama")
            return True, f"Model '{model}' is available and ready to use"
        else:
            available_models = client.list_models()
            model_list = [m.get("name", "") for m in available_models]
            msg = (
                f"Model '{model}' is not available in Ollama. "
                f"Available models: {', '.join(model_list) if model_list else 'none'}. "
                f"Pull the model with: ollama pull {model}"
            )
            logger.warning(msg)
            return False, msg

    except Exception as e:
        error_msg = f"Failed to validate model '{model}': {e}"
        logger.error(error_msg)
        return False, error_msg


def get_model_info(
    model: str,
    provider: str = "ollama",
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a model.

    Args:
        model: Model name to query
        provider: Provider name (currently only "ollama" is supported)
        base_url: Optional custom base URL for the provider

    Returns:
        Dictionary with model information, or None if not found
    """
    if provider != "ollama":
        logger.error(f"Unsupported provider: {provider}")
        return None

    try:
        client = OllamaVisionClient(base_url=base_url)
        models = client.list_models()

        for model_info in models:
            if model_info.get("name") == model:
                return model_info

        logger.warning(f"Model '{model}' not found")
        return None

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def suggest_pull_command(model: str, provider: str = "ollama") -> str:
    """
    Generate a command to pull/download the model.

    Args:
        model: Model name
        provider: Provider name

    Returns:
        Command string to pull the model
    """
    if provider == "ollama":
        return f"ollama pull {model}"
    else:
        return f"# Model pulling not supported for provider: {provider}"


def validate_granite_vision_setup() -> tuple[bool, str]:
    """
    Validate that the default Granite 3.3 Vision model is set up correctly.

    Returns:
        Tuple of (is_valid, message)
    """
    model = "granite3.3-vision:2b"
    return validate_model_available(model, provider="ollama")
