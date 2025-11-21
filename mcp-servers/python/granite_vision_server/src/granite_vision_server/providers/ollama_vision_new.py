import httpx
from typing import Dict, Any, List, Optional

OLLAMA_URL = "http://localhost:11434/api"

# Shared async HTTP client
async def _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=180.0) as client:

        resp = await client.post(f"{OLLAMA_URL}/{endpoint}", json=payload)

        if resp.status_code != 200:
            print("❌ Ollama request failed:")
            print("URL:", resp.url)
            print("Status:", resp.status_code)
            print("Response text:", resp.text[:300])
            print("Payload keys:", list(payload.keys()))
            raise httpx.HTTPStatusError(
                f"Ollama error {resp.status_code}: {resp.text[:300]}",
                request=resp.request,
                response=resp,
        )       
        
        
        #resp.raise_for_status()
        return resp.json()


# ------------------------
# 1. Image Analysis
# ------------------------
async def analyze_image(
    model: str,
    image_data: str,
    analysis_type: str,
    include_conf: bool,
    lang: str
) -> Dict[str, Any]:
    """
    Calls Ollama vision model (granite3.2-vision) to analyze an image.
    """
    prompt_map = {
        "general": "Describe the image in detail.",
        "detailed": "Provide a detailed description of the scene and all visible objects.",
        "objects": "List all objects and their relationships.",
        "scene": "Describe the setting, environment, and lighting of the scene.",
        "text": "Describe any text present in the image."
    }
    prompt = prompt_map.get(analysis_type, "Describe this image.")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    result = await _post("generate", payload)
    return {
        "description": result.get("response", "").strip(),
        "tags": [],
        "objects": [],
        "confidences": []
    }


# ------------------------
# 2. Document Extraction
# ------------------------
async def extract_document(
    model: str,
    document_data: str,
    extraction_type: str,
    preserve_layout: bool,
    extract_tables: bool,
    extract_images: bool,
    output_format: str
) -> Dict[str, Any]:
    """
    Uses Ollama vision model for simple doc-understanding extraction.
    For complex structured extraction, Granite Vision Watsonx provider is preferred.
    """
    prompt = f"Extract and summarize the document ({extraction_type}). Provide key content and structure."

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [document_data],
        "stream": False
    }

    result = await _post("generate", payload)
    text = result.get("response", "").strip()
    return {"content": text, "resources": []}


# ------------------------
# 3. Visual Question Answering (VQA)
# ------------------------
async def vqa(
    model: str,
    image_data: str,
    question: str,
    context: Optional[str],
    answer_type: str,
    conf: float
) -> Dict[str, Any]:
    """
    Ask a question about the image and get the answer.
    """
    prompt = f"Question: {question}\nAnswer briefly and clearly."
    if context:
        prompt = f"Context: {context}\n\n{prompt}"

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    result = await _post("generate", payload)
    answer = result.get("response", "").strip()
    return {"answer": answer, "confidence": None}


# ------------------------
# 4. OCR (text extraction)
# ------------------------
async def ocr(
    model: str,
    image_data: str,
    languages: List[str],
    preserve_formatting: bool,
    handwriting: bool,
    conf_filter: bool,
    bboxes: bool,
    regions
) -> Dict[str, Any]:
    """
    Uses Granite3.2-Vision to transcribe visible text (OCR-like behavior).
    """
    prompt = "Extract all visible text from this image and return it as plain text."
    if handwriting:
        prompt += " Include any handwritten content if present."

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    result = await _post("generate", payload)
    text = result.get("response", "").strip()
    return {"text": text, "blocks": []}


# ------------------------
# 5. Charts / Graphs Analysis
# ------------------------
async def charts(
    model: str,
    image_data: str,
    chart_type: str = "auto",
    extract_data: bool = True,
    extract_labels: bool = True,
    extract_legend: bool = True,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Analyzes chart or graph images and extracts data series.
    """
    prompt = (
        "Analyze the chart and describe its data trends, labels, and legend. "
        "If possible, summarize data values in a JSON-like structure."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    result = await _post("generate", payload)
    return {"series": [result.get("response", "").strip()], "csv_resource": None}


# ------------------------
# 6. Table Extraction
# ------------------------
async def tables(
    model: str,
    image_data: str,
    extract_structure: bool = True,
    extract_content: bool = True,
    preserve_formatting: bool = True,
    output_format: str = "csv",
    header_detection: bool = True
) -> Dict[str, Any]:
    """
    Extracts tables from a document or image using Granite Vision.
    """
    prompt = (
        "Extract any tables visible in the image and output them in CSV format. "
        "Include headers if detected."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    result = await _post("generate", payload)
    text = result.get("response", "").strip()
    return {"tables": [text], "resources": []}
