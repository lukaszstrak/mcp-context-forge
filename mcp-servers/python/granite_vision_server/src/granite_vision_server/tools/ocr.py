from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from ..providers import get_provider
from ..models import validate_model
from ..utils.image_utils import load_image

class OCRRequest(BaseModel):
    image_data: str = Field(...)
    model: str = Field(default="granite-vision-ocr-v1")
    provider: str = Field(default="huggingface")
    languages: List[str] = Field(default_factory=lambda: ["en"])  # Supported languages
    preserve_formatting: bool = Field(default=True)
    extract_handwriting: bool = Field(default=False)
    confidence_filtering: bool = Field(default=True)
    bounding_boxes: bool = Field(default=False)
    text_regions: Optional[List[Dict]] = Field(default=None)  # Specific regions to process

async def ocr_text_extraction(req: OCRRequest) -> str | Dict:
    validate_model(req.model)
    provider = get_provider(req.provider)
    image = load_image(req.image_data)
    prompt = "Extract all text from the image accurately."
    if req.preserve_formatting:
        prompt += " Preserve original formatting and layout."
    if req.extract_handwriting:
        prompt += " Include recognition of handwritten text."
    prompt += f" Languages: {', '.join(req.languages)}"
    if req.text_regions:
        # Placeholder for processing specific regions (e.g., crop image)
        prompt += " Focus on specified regions."
    text = provider.infer(req.model, image, prompt)
    if req.confidence_filtering:
        # Placeholder: filter low-confidence text (requires model support)
        pass
    if req.bounding_boxes:
        # Placeholder: assume model returns boxes if supported
        return {"text": text, "boxes": []}
    return text