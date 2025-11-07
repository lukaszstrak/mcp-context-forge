# Makefile for DOCX MCP Server

.PHONY: help install dev-install format lint test dev mcp-info serve-http serve-sse test-http example-create clean

PYTHON ?= python #python3 (I have python pointing to python3)
HTTP_PORT ?= 9004
HTTP_HOST ?= localhost

help: ## Show help
	@echo "Granite Vision MCP Server - FastMCP Implementation"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install          Install FastMCP server"
	@echo "  make dev              Run FastMCP server"
	@echo ""
	@echo "Available Commands:"
	@awk 'BEGIN {FS=":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install in editable mode
	$(PYTHON) -m pip install -e .

dev-install: ## Install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

format: ## Format (black + ruff --fix)
	black . && ruff --fix .

lint: ## Lint (ruff, mypy)
	ruff check . && mypy src/granite_vision_server

test: ## Run tests
	pytest -v --cov=granite_vision_server --cov-report=term-missing

dev: ## Run FastMCP server (stdio)
	@echo "Starting Granite Vision MCP Server..."
	$(PYTHON) -m granite_vision_server.server_fastmcp

mcp-info: ## Show MCP client config
	@echo "==================== MCP CLIENT CONFIGURATION ===================="
	@echo ""
	@echo "FastMCP Server:"
	@echo '{"command": "python", "args": ["-m", "granite_vision_server.server_fastmcp"], "cwd": "'$(PWD)'"}'
	@echo ""
	@echo "=================================================================="

serve-http: ## Run with native FastMCP HTTP
	@echo "Starting FastMCP server with native HTTP support..."
	@echo "HTTP endpoint: http://$(HTTP_HOST):$(HTTP_PORT)/mcp/"
	@echo "API docs: http://$(HTTP_HOST):$(HTTP_PORT)/docs"
	$(PYTHON) -m granite_vision_server.server_fastmcp --transport http --host $(HTTP_HOST) --port $(HTTP_PORT)

serve-sse: ## Run with mcpgateway.translate (SSE bridge)
	@echo "Starting with translate SSE bridge..."
	@echo "SSE endpoint: http://$(HTTP_HOST):$(HTTP_PORT)/sse"
	@echo "HTTP endpoint: http://$(HTTP_HOST):$(HTTP_PORT)/"
	$(PYTHON) -m mcpgateway.translate --stdio "$(PYTHON) -m granite_vision_server.server_fastmcp" --host $(HTTP_HOST) --port $(HTTP_PORT) --expose-sse

test-http: ## Basic HTTP checks
	curl -s http://$(HTTP_HOST):$(HTTP_PORT)/ | head -20 || true
	curl -s -X POST -H 'Content-Type: application/json' \
	  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
	  http://$(HTTP_HOST):$(HTTP_PORT)/ | head -40 || true

validate-model: ## Validate Granite 3.3 Vision model availability
	@echo "Validating Granite 3.3 Vision model..."
	@$(PYTHON) -c "from granite_vision_server.utils.model_validator import validate_granite_vision_setup; \
	is_valid, msg = validate_granite_vision_setup(); \
	print('OK' if is_valid else 'NOT OK', msg); \
	exit(0 if is_valid else 1)"

list-models: ## List available Ollama models
	@echo "Available Ollama models:"
	@$(PYTHON) -c "from granite_vision_server.providers.ollama_vision import OllamaVisionClient; client = OllamaVisionClient(); models = client.list_models(); [print(f\"  - {m.get('model', m.get('name', 'unknown'))} ({m.get('size', 0) / (1024**3):.1f} GB)\") for m in models]; print(f\"\\nTotal models: {len(models)}\")"

debug-ollama: ## Debug Ollama API response
	@echo "Debugging Ollama API response..."
	@$(PYTHON) -c "from granite_vision_server.providers.ollama_vision import OllamaVisionClient; \
	import json; \
	client = OllamaVisionClient(); \
	response = client.client.list(); \
	print('Response type:', type(response)); \
	print('Response keys:', list(response.keys()) if isinstance(response, dict) else 'Not a dict'); \
	print('Full response:'); \
	print(json.dumps(response, indent=2, default=str))"

example-analyze: ## Example: Analyze an image (Usage: make example-analyze IMAGE=path/to/image.jpg)
	@$(PYTHON) examples/image_analysis.py $(IMAGE)

pull-model: ## Pull Granite 3.3 Vision model
	@echo "Pulling Granite 3.3 Vision model from Ollama..."
	ollama pull granite3.3-vision:2b

check-ollama: ## Check if Ollama is running
	@echo "Checking Ollama status..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama is running" || echo "Ollama is not running"

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ */__pycache__ *.egg-info build/ dist/
