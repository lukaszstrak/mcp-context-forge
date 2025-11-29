"""Location: ./mcp-servers/python/granite_vision_server/src/granite-vision-server/providers/huggingface_vision.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi, Uzma Siddiqui
Granite Vision MCP Server - FastMCP Implementation

"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


class HuggingFaceVisionProvider:
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.device = config["device"] if config["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = config["cache_dir"]

    def infer(self, model, image_data, prompt, **kwargs):
        # Load model and processor
        processor = AutoProcessor.from_pretrained(model, cache_dir=self.cache_dir)
        model_obj = AutoModelForVision2Seq.from_pretrained(model, cache_dir=self.cache_dir).to(self.device)
        inputs = processor(text=prompt, images=image_data, return_tensors="pt").to(self.device)
        outputs = model_obj.generate(**inputs, **kwargs)
        return processor.decode(outputs[0], skip_special_tokens=True)
