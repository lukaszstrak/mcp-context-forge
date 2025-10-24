# Granite Vision Model

> Author: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Create a comprehensive MCP Server for IBM Granite Vision Models supporting document understanding, image analysis, OCR, visual question answering, and multi-modal processing with enterprise deployment capabilities.

## Configuration

## Tools

1. `analyze_image`

General image analysis and understanding

```python
@dataclass
class ImageAnalysisRequest:
  image_data: str  # base64, file path, or URL
  model: str = "granite-vision-general-v1"
  provider: str = "ollama"
  analysis_type: str = "general"  # general, detailed, objects, scene, text
  include_confidence: bool = True
  max_description_length: int = 200
  language: str = "en"
```

2. `extract_document_content`

Document understanding and content extraction

```python
@dataclass
class DocumentExtractionRequest:
    document_data: str  # PDF, image, or multi-page document
    model: str = "granite-vision-document-v1"
    provider: str = "watsonx"
    extraction_type: str = "full"  # full, text_only, structured, tables, forms
    preserve_layout: bool = True
    extract_tables: bool = True
    extract_images: bool = False
    output_format: str = "markdown"  # markdown, json, xml, plain_text
```

3. `visual_question_answering`

Answer questions about images and documents

```python
@dataclass
class VQARequest:
    image_data: str
    question: str
    model: str = "granite-multimodal-8b"
    provider: str = "ollama"
    context: Optional[str] = None  # Additional context
    answer_type: str = "natural"  # natural, short, boolean, multiple_choice
    confidence_threshold: float = 0.7
```

4. `ocr_text_extraction`

Optical Character Recognition with advanced features

```python
@dataclass
class OCRRequest:
    image_data: str
    model: str = "granite-vision-ocr-v1"
    provider: str = "huggingface"
    languages: List[str] = ["en"]  # Supported languages
    preserve_formatting: bool = True
    extract_handwriting: bool = False
    confidence_filtering: bool = True
    bounding_boxes: bool = False
    text_regions: Optional[List[Dict]] = None  # Specific regions to process
```

5. `analyze_charts_graphs`

Chart and graph analysis and data extraction

```python
@dataclass
class ChartAnalysisRequest:
    image_data: str
    model: str = "granite-vision-chart-v1"
    provider: str = "watsonx"
    chart_type: str = "auto"  # auto, bar, line, pie, scatter, histogram
    extract_data: bool = True
    extract_labels: bool = True
    extract_legend: bool = True
    output_format: str = "json"  # json, csv, description
```

6. `process_tables`

Table detection, extraction, and structure analysis

```python
@dataclass
class TableProcessingRequest:
    image_data: str
    model: str = "granite-vision-table-v1"
    provider: str = "ollama"
    extract_structure: bool = True
    extract_content: bool = True
    preserve_formatting: bool = True
    output_format: str = "csv"  # csv, json, html, markdown
    header_detection: bool = True
```

7. `multi_modal_chat`

Interactive multi-modal conversation

```python
@dataclass
class MultiModalChatRequest:
    messages: List[Dict[str, Any]]  # Text and image messages
    model: str = "granite-multimodal-8b"
    provider: str = "watsonx"
    max_tokens: int = 512
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    maintain_context: bool = True
```

8. `batch_process_images`

Efficient batch processing of multiple images

```python
@dataclass
class BatchImageRequest:
    images: List[str]  # Multiple image data sources
    model: str = "granite-vision-general-v1"
    provider: str = "ollama"
    processing_type: str = "analyze"  # analyze, ocr, extract, classify
    parallel_processing: bool = True
    max_concurrent: int = 4
    aggregate_results: bool = True
```

## Installation

```bash
# Install in development mode
make dev-install

# Or install normally
make install
```

## Usage

### Running the FastMCP Server

```bash
# Start the server
make dev

# Or directly
python -m granite_vision_server.server
```

### HTTP Bridge

Expose the server over HTTP for REST API access:

```bash
make serve-http
```

### MCP Client Configuration

```json
{
  "mcpServers": {
    "docx-server": {
      "command": "python",
      "args": ["-m", "granite_vision_server.server"],
      "cwd": "/path/to/granite_vision_server"
    }
  }
}
```

### Test Tools

```bash
# Test tool listing
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m granite_vision_server.server

# Create a document
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_document","arguments":{"file_path":"test.docx","title":"Test Document"}}}' | python -m granite_vision_server.server
```

## FastMCP Advantages

The FastMCP implementation provides:

1. **Type-Safe Parameters**: Automatic validation using Pydantic Field constraints
2. **Range Validation**: Ensures heading level is between 1-9 with `ge=1, le=9`
3. **Cleaner Code**: Decorator-based tool definitions (`@mcp.tool`)
4. **Better Error Handling**: Built-in exception management
5. **Automatic Schema Generation**: No manual JSON schema definitions

## Development

```bash
# Format code
make format

# Run tests
make test

# Lint code
make lint
```

## Requirements

- Python 3.11+
- python-docx library for document manipulation
- MCP framework for protocol implementation
