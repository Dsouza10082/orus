# Orus API

A Go-based API server that provides text embedding and LLM capabilities using Ollama and BGE-M3 models.

## Overview

Orus API is a REST API that offers:
- Text embedding using BGE-M3 and Ollama models (nomic-embed-text)
- LLM chat completions via Ollama
- Model management (list and pull models)
- System information endpoint

## Prerequisites

- Docker and Docker Compose installed
- Go 1.21+ (for local development)
- Sufficient disk space for AI models (10GB+ recommended)
- 8GB+ RAM recommended

## Quick Start

### Using Docker Compose (Recommended)

1. **Start the services**:
   ```bash
   docker-compose up -d
   ```

2. **Wait for services to be healthy** (30-60 seconds):
   ```bash
   docker-compose ps
   ```

3. **Verify the API is running**:
   ```bash
   curl http://localhost:8081/orus-api/v1/system-info
   ```

### Using Setup Scripts

We provide automated setup scripts for convenience:

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```powershell
.\setup.ps1
```

## ⚠️ Important Setup Requirements

Before using the API, you **MUST** complete these steps:

### 1. Download the Embedding Model (MANDATORY)

The `nomic-embed-text` model is **required** for the `/embed-text` endpoint. Download it using:

```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "nomic-embed-text"}'
```

This will stream the download progress. Wait for the "success" message.

### 2. Download an LLM Model (MANDATORY for Chat)

Before using `/call-llm`, you must download at least one Ollama model. Popular choices:

```bash
# Llama 3.1 (8B parameters, ~4.7GB)
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1:8b"}'

# Or Mistral (7B parameters, ~4.1GB)
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "mistral:7b"}'

# Or Phi-3 (3.8B parameters, ~2.2GB)
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "phi3:mini"}'
```

### 3. Verify Downloaded Models

Check which models are available:

```bash
curl http://localhost:8081/orus-api/v1/ollama-model-list
```

## Architecture

```
┌─────────────┐         ┌─────────────┐
│  Orus API   │ ◄─────► │   Ollama    │
│  (Port 8081)│         │ (Port 11434)│
└─────────────┘         └─────────────┘
       │
       ├─ BGE-M3 Embedder (ONNX)
       └─ Ollama Client
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORUS_API_PORT` | `8081` | API server port |
| `ORUS_API_OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama service URL |
| `ORUS_API_AGENT_MEMORY_PATH` | `./agent_memory/` | BGE-M3 memory path |
| `ORUS_API_ONNX_PATH` | `onnx/model.onnx` | ONNX model path |
| `ORUS_API_TOK_PATH` | `onnx/tokenizer.json` | Tokenizer path |
| `ORUS_API_ONNX_RUNTIME_PATH` | `onnx/aarch64/libonnxruntime.so` | ONNX runtime library |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Default embedding model |

## API Documentation

See [API.md](./API.md) for detailed endpoint documentation.

### Quick API Examples

**Get System Info:**
```bash
curl http://localhost:8081/orus-api/v1/system-info
```

**Embed Text:**
```bash
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "nomic-embed-text",
      "text": "Hello, world!"
    }
  }'
```

**Chat with LLM:**
```bash
curl -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "llama3.1:8b",
      "stream": false,
      "messages": [
        {
          "role": "user",
          "content": "Tell me a story about a brave knight."
        }
      ]
    }
  }'
```

## Troubleshooting

### Service Not Starting

Check logs:
```bash
docker-compose logs -f
```

### Out of Memory

Reduce the number of parallel models in `docker-compose.yml`:
```yaml
environment:
  - OLLAMA_NUM_PARALLEL=1
  - OLLAMA_MAX_LOADED_MODELS=1
```

### Model Download Fails

Ensure you have:
- Active internet connection
- Sufficient disk space
- Ollama service is healthy

Check Ollama health:
```bash
curl http://localhost:11434/api/tags
```

### API Returns 500 Error

Common causes:
1. Model not downloaded
2. Ollama service not ready
3. Insufficient memory

Check if models are loaded:
```bash
curl http://localhost:8081/orus-api/v1/ollama-model-list
```

## Development

### Building Locally

```bash
go mod download
go build -o orus-api
./orus-api
```

### Running Tests

```bash
go test ./...
```

## Ports

- **8081**: Orus API
- **11434**: Ollama service

## Storage

- `ollama_dev_data`: Stores downloaded Ollama models
- `./models`: Read-only model directory mount

## Stopping Services

```bash
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
```

## Support

- GitHub: [Dsouza10082](https://github.com/Dsouza10082)
- Version: 1.0.0

## License

See LICENSE file for details.
