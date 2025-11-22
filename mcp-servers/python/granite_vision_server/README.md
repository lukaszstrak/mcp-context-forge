# Granite Vision Model

> Author: Anna Topol, Łukasz Strąk, Hong Wei Jia, Lisette Contreras, Mohammed Kazmi

Create a comprehensive MCP Server for IBM Granite Vision Models supporting document understanding, image analysis, OCR, visual question answering, and multi-modal processing with enterprise deployment capabilities.

## Prerequisites

### Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com):

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download installer from ollama.com
```

### Pull IBM Granite 3.3 Vision Model

```bash
ollama pull granite3.3-vision:2b
```

Verify the model is available:

```bash
ollama list | grep granite
# Should show: granite3.3-vision:2b
```

### Verify Ollama is Running

```bash
# Test Ollama server
curl http://localhost:11434/api/tags

# Should return JSON with model list
```

## Quick Start

```bash
# Clone or navigate to the project
cd granite_vision_server

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using uv (recommended)

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv pip install -e .
```

## Quick Test

```bash
# Test the server
python -m granite_vision_server.server_fastmcp --help

# List available tools
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m granite_vision_server.server_fastmcp
```

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
# Start with HTTP transport
python -m granite_vision_server.server_fastmcp --transport http --port 9004

# Or using make
make serve-http
```

### MCP Client Configuration

```json
{
  "mcpServers": {
    "granite-vision": {
      "command": "python",
      "args": ["-m", "granite_vision_server.server_fastmcp"],
      "cwd": "/path/to/granite_vision_server"
    }
  }
}
```

### Test Tools

```bash
# Test tool listing
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m granite_vision_server.server_fastmcp

# Analyze an image
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"analyze_image","arguments":{"image_data":"<BASE64_IMAGE_DATA>","analysis_type":"general"}}}' | python -m granite_vision_server.server_fastmcp
```

## FastMCP Advantages

The FastMCP implementation provides:

1. **Type-Safe Parameters**: Automatic validation using Pydantic Field constraints
2. **Range Validation**: Ensures heading level is between 1-9 with `ge=1, le=9`
3. **Cleaner Code**: Decorator-based tool definitions (`@mcp.tool`)
4. **Better Error Handling**: Built-in exception management
5. **Automatic Schema Generation**: No manual JSON schema definitions

## Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking
mypy src/granite_vision_server
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=granite_vision_server --cov-report=html

# Run specific test file
pytest tests/test_server.py -v
```

### Import Errors

```bash
# Reinstall dependencies
pip install -e .

# Or use uv
uv pip install -e .

# Verify installations
pip list | grep -E "ollama|fastmcp|Pillow"
```

## Requirements

- **Python**: 3.11+
- **Ollama**: Latest version with granite3.3-vision:2b model
- **Dependencies**:
  - `fastmcp>=2.11.3` - MCP protocol implementation
  - `ollama>=0.4.0` - Official Ollama Python SDK
  - `Pillow>=10.0.0` - Image processing
  - `requests>=2.31.0` - HTTP client for image URLs
  - `pydantic>=2.5.0` - Data validation
