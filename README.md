# Tiny RAG

A lightweight Retrieval-Augmented Generation (RAG) system built with Python, featuring document processing, vector search, and caching capabilities.

## Features

- **Document Processing**: Automatic monitoring and processing of documents
- **Vector Search**: Efficient semantic search using embeddings
- **Caching**: Redis-based caching for improved performance
- **Object Storage**: MinIO integration for document storage
- **RESTful API**: FastAPI-based search service
- **Dockerized**: Easy deployment with Docker Compose

## Architecture

- **Search Service**: FastAPI server providing search endpoints (port 4500)
- **Document Monitor**: Background service for processing documents
- **Redis**: Caching layer (port 23456)
- **MinIO**: Object storage for documents (ports 6100, 6101)

## Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)

## Quick Start

### Using Docker Compose

```bash
# Build the image
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f tiny_rag

# Stop services
docker compose down

# Local Development
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# Set environment variables
export TINY_RAG@@search_service_port=4500
export TINY_RAG@@cache_config@@conn_url=redis://localhost:23456/0
export TINY_RAG@@object_store_config@@conn_url=localhost:6100

# Run the search service
python -m rag.service

# Run the document monitor (in another terminal)
python -m rag.document
```