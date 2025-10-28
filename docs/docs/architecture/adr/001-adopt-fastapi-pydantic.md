# ADR-0001: Adopt FastAPI + Pydantic

- *Status:* Accepted
- *Date:* 2025-02-01
- *Deciders:* Mihai Criveti

## Context

The MCP Gateway must serve both human and machine clients with low-latency HTTP and WebSocket endpoints. Payloads require runtime validation and schema documentation, while internal data types must align with environment-driven settings and JSON models.

We explored Python-native frameworks that support async-first operation, data validation, OpenAPI generation, and modular service layout.

## Decision

We will adopt:

- **FastAPI** as the core web framework for routing HTTP, WebSocket, and streaming endpoints.
- **Pydantic v2.11+** for all settings, schemas, and typed data models (e.g., `Tool`, `Resource`, `GatewayMetadata`, etc.).

These will form the foundation for the application layer and public API.

Pydantic v2 uses a **Rust-based core** (`pydantic-core`) that provides 5-50x performance improvements over Pydantic v1 for schema validation, type coercion, and model serialization. The Rust components release the GIL during execution, enabling better concurrency even within single worker processes.

## Consequences

- ✨ Strong typing, runtime validation, and auto-generated OpenAPI specs.
- 🧩 Unified model structure across internal logic, external APIs, and config parsing.
- 🚀 Excellent async performance with Uvicorn and Starlette compatibility.
- ⚡ Rust-powered validation (5-50x faster than Pydantic v1) with GIL release during validation.
- 🔒 Tight coupling to Pydantic means future transitions (e.g., to dataclasses or attrs) would be non-trivial.

## Alternatives Considered

| Option | Why Not |
|--------|---------|
| **Flask + Marshmallow** | Sync-first architecture, weak async support, manual OpenAPI generation. |
| **Django REST Framework** | Heavyweight, monolithic, tightly bound to Django ORM, not async-native. |
| **Tornado or Starlette alone** | More boilerplate to assemble middlewares, validators, and routing. |
| **Node.js + Fastify** | Excellent performance but requires a split language/runtime and loss of shared model code. |
| **Pure `httpx` + `uvicorn` + `pydantic-core`** | Too low-level; duplicating FastAPI features manually. |

## Status

This decision has been implemented in the current architecture.
