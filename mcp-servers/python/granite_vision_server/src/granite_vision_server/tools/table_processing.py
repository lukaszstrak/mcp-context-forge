from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
from ..providers import get_provider
from ..models import validate_model
from ..utils.image_utils import load_image
from ..utils.format_converters import to_csv, to_json, to_markdown  # Assuming html can use markdown or add to_html if needed

class TableProcessingRequest(BaseModel):
    image_data: str = Field(..., description="Image containing the table")
    model: str = Field(default="granite-vision-table-v1")
    provider: str = Field(default="ollama")
    extract_structure: bool = Field(default=True)
    extract_content: bool = Field(default=True)
    preserve_formatting: bool = Field(default=True)
    output_format: str = Field(default="csv")  # csv, json, html, markdown
    header_detection: bool = Field(default=True)

async def process_tables(req: TableProcessingRequest) -> str | Dict[str, Any]:
    validate_model(req.model)
    provider = get_provider(req.provider)
    image = load_image(req.image_data)
    prompt = "Extract the table from the image."
    if req.extract_structure:
        prompt += " Include table structure."
    if req.extract_content:
        prompt += " Extract all content accurately."
    if req.preserve_formatting:
        prompt += " Preserve original formatting where possible."
    if req.header_detection:
        prompt += " Detect and include headers."
    
    # Assume model returns a markdown or json representation of the table
    raw_result = provider.infer(req.model, image, prompt)
    
    # Post-process: Parse raw_result into a DataFrame (placeholder logic)
    # For example, if raw_result is markdown table, parse it
    # Here, using placeholder data; in real impl, use table parsing libs if needed
    data = [["col1", "col2"], ["val1", "val2"]]  # Replace with parsed raw_result
    df = pd.DataFrame(data[1:], columns=data[0] if req.header_detection else None)
    
    if req.output_format == "csv":
        return to_csv(df)
    elif req.output_format == "json":
        return to_json(df.to_dict(orient="records"))
    elif req.output_format == "html":
        return df.to_html()
    elif req.output_format == "markdown":
        return to_markdown(df)
    else:
        raise ValueError("Unsupported output format")