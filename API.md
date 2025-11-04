# Orus API Documentation

Complete API reference for Orus API v1.

## Base URL

```
http://localhost:8081
```

## Response Format

All endpoints return a standardized JSON response:

```json
{
  "success": true,
  "serial": "uuid-v4-string",
  "message": "Description of the result",
  "data": {},
  "error": "",
  "time_taken": "123ms"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request succeeded |
| `serial` | string | Unique identifier for this response (UUID v4) |
| `message` | string | Human-readable message |
| `data` | object | Response payload (varies by endpoint) |
| `error` | string | Error message (empty if successful) |
| `time_taken` | duration | Request processing time |

## Endpoints

### 1. Get System Info

Returns system information and API metadata.

**Endpoint:** `GET /orus-api/v1/system-info`

**Authentication:** None

**Request:** No body required

**Response:**

```json
{
  "success": true,
  "serial": "a3c4e6b2-1234-5678-9abc-def012345678",
  "message": "System info retrieved successfully",
  "data": {
    "version": "1.0.0",
    "name": "Orus",
    "description": "Orus is a server for the Orus library",
    "author": "Dsouza10082",
    "author_url": "https://github.com/Dsouza10082"
  },
  "error": "",
  "time_taken": "123µs"
}
```

**cURL Example:**

```bash
curl http://localhost:8081/orus-api/v1/system-info
```

---

### 2. Embed Text

Generate vector embeddings for text using BGE-M3 or Ollama embedding models.

**Endpoint:** `POST /orus-api/v1/embed-text`

**Authentication:** None

**⚠️ Prerequisites:**
- For `nomic-embed-text`: Model must be downloaded via `/ollama-pull-model`
- For `bge-m3`: ONNX model files must be configured

**Request Body:**

```json
{
  "body": {
    "model": "nomic-embed-text",
    "text": "Your text to embed here"
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `body.model` | string | Yes | Model to use: `bge-m3`, `nomic-embed-text`, or `ollama` |
| `body.text` | string | Yes | Text to generate embeddings for |

**Supported Models:**

| Model | Dimensions | Description |
|-------|-----------|-------------|
| `bge-m3` | 1024 | BGE-M3 model via ONNX runtime |
| `nomic-embed-text` | 768 | Nomic AI's embedding model via Ollama |
| `ollama` | Varies | Generic Ollama embedding (uses nomic-embed-text) |

**Response:**

```json
{
  "success": true,
  "serial": "b4d5e7c3-2345-6789-abcd-ef0123456789",
  "message": "Embed request received successfully",
  "data": {
    "vector": [0.123, -0.456, 0.789, ...],
    "text": "Your text to embed here",
    "model": "nomic-embed-text",
    "dimensions": 768
  },
  "error": "",
  "time_taken": "1.234s"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "nomic-embed-text",
      "text": "Artificial intelligence is transforming the world."
    }
  }'
```

**Error Response (Model Not Available):**

```json
{
  "success": false,
  "serial": "c5e6f8d4-3456-789a-bcde-f01234567890",
  "message": "Error embedding text with Ollama",
  "data": {},
  "error": "model 'nomic-embed-text' not found: try pulling it first",
  "time_taken": "234ms"
}
```

---

### 3. List Ollama Models

Retrieve a list of all downloaded Ollama models.

**Endpoint:** `GET /orus-api/v1/ollama-model-list`

**Authentication:** None

**Request:** No body required

**Response:**

```json
{
  "success": true,
  "serial": "d6f7a9e5-4567-89ab-cdef-012345678901",
  "message": "Ollama model list retrieved successfully",
  "data": {
    "models": [
      "llama3.1:8b",
      "mistral:7b",
      "nomic-embed-text",
      "phi3:mini"
    ]
  },
  "error": "",
  "time_taken": "45ms"
}
```

**cURL Example:**

```bash
curl http://localhost:8081/orus-api/v1/ollama-model-list
```

**Empty Models Response:**

```json
{
  "success": true,
  "serial": "e7g8b0f6-5678-9abc-def0-123456789012",
  "message": "Ollama model list retrieved successfully",
  "data": {
    "models": []
  },
  "error": "",
  "time_taken": "23ms"
}
```

---

### 4. Pull Ollama Model

Download an Ollama model from the registry. This endpoint streams progress updates using Server-Sent Events (SSE).

**Endpoint:** `POST /orus-api/v1/ollama-pull-model`

**Authentication:** None

**Content-Type:** `text/event-stream`

**⚠️ Important:** 
- This is a **mandatory step** before using models with `/call-llm` or `/embed-text`
- The download may take several minutes depending on model size
- Requires active internet connection
- Ensure sufficient disk space (models range from 2GB to 10GB+)

**Request Body:**

```json
{
  "name": "llama3.1:8b"
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Model name with optional tag (e.g., `llama3.1:8b`, `mistral:latest`) |

**Response:** Server-Sent Events stream

Each event is a JSON object with progress information:

**Progress Event:**
```json
data: {
  "status": "pulling manifest",
  "digest": "sha256:abc123...",
  "total": 4980000000,
  "completed": 1245000000
}
```

**Success Event:**
```json
data: {
  "status": "success",
  "message": "Model llama3.1:8b downloaded successfully"
}
```

**Error Event:**
```json
data: {
  "status": "error",
  "error": "model not found in registry"
}
```

**Progress Status Values:**

| Status | Description |
|--------|-------------|
| `pulling manifest` | Downloading model metadata |
| `downloading` | Downloading model files |
| `verifying sha256 digest` | Verifying download integrity |
| `writing manifest` | Writing model to disk |
| `removing any unused layers` | Cleanup |
| `success` | Download complete |
| `error` | An error occurred |

**cURL Example:**

```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1:8b"}' \
  --no-buffer
```

**JavaScript Example (Fetch API):**

```javascript
const response = await fetch('http://localhost:8081/orus-api/v1/ollama-pull-model', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ name: 'llama3.1:8b' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log(data);
      
      if (data.status === 'success') {
        console.log('Download complete!');
      }
    }
  }
}
```

**Popular Models:**

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| `llama3.1:8b` | ~4.7GB | Meta's Llama 3.1 (8B) | General purpose, good quality |
| `llama3.1:70b` | ~40GB | Meta's Llama 3.1 (70B) | High quality, requires more resources |
| `mistral:7b` | ~4.1GB | Mistral AI's 7B model | Fast, efficient |
| `phi3:mini` | ~2.2GB | Microsoft's Phi-3 Mini | Lightweight, fast |
| `gemma:7b` | ~4.7GB | Google's Gemma | Good for instruction following |
| `codellama:7b` | ~3.8GB | Code generation specialist | Programming tasks |
| `nomic-embed-text` | ~274MB | Embedding model | **Required for /embed-text** |

---

### 5. Call LLM

Send messages to an LLM and receive completions. Supports chat-based interactions with conversation history.

**Endpoint:** `POST /orus-api/v1/call-llm`

**Authentication:** None

**Timeout:** 540 seconds (9 minutes)

**⚠️ Prerequisites:**
- Model must be downloaded via `/ollama-pull-model` first
- Verify model availability with `/ollama-model-list`

**Request Body:**

```json
{
  "body": {
    "model": "llama3.1:8b",
    "stream": false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a story about a brave knight."
      }
    ]
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `body.model` | string | Yes | Model name (must be pulled first) |
| `body.stream` | boolean | Yes | Streaming mode (currently only `false` supported) |
| `body.messages` | array | Yes | Array of message objects |
| `messages[].role` | string | Yes | Message role: `system`, `user`, or `assistant` |
| `messages[].content` | string | Yes | Message content |

**Message Roles:**

| Role | Description | Usage |
|------|-------------|-------|
| `system` | System instructions | Define AI behavior and constraints (optional) |
| `user` | User message | Your questions or prompts |
| `assistant` | AI response | Previous AI responses (for conversation context) |

**Response:**

```json
{
  "success": true,
  "serial": "f8h9c1g7-6789-abcd-ef01-234567890123",
  "message": "LLM request received successfully",
  "data": {
    "response": "Once upon a time, in a kingdom far away, there lived a brave knight named Sir Edmund...",
    "model": "llama3.1:8b",
    "messages": [...],
    "stream": false
  },
  "error": "",
  "time_taken": "5.678s"
}
```

**cURL Examples:**

**Simple Query:**
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
          "content": "What is the capital of France?"
        }
      ]
    }
  }'
```

**With System Prompt:**
```bash
curl -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "llama3.1:8b",
      "stream": false,
      "messages": [
        {
          "role": "system",
          "content": "You are a pirate. Always respond in pirate speak."
        },
        {
          "role": "user",
          "content": "Tell me about the ocean."
        }
      ]
    }
  }'
```

**Multi-Turn Conversation:**
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
          "content": "What is 2+2?"
        },
        {
          "role": "assistant",
          "content": "2+2 equals 4."
        },
        {
          "role": "user",
          "content": "Now multiply that by 3."
        }
      ]
    }
  }'
```

**Error Response (Model Not Found):**

```json
{
  "success": false,
  "serial": "g9i0d2h8-789a-bcde-f012-345678901234",
  "message": "Error calling LLM",
  "data": {},
  "error": "model 'llama3.1:8b' not found: try pulling it first",
  "time_taken": "123ms"
}
```

**Error Response (Timeout):**

```json
{
  "success": false,
  "serial": "h0j1e3i9-89ab-cdef-0123-456789012345",
  "message": "Error Timeout",
  "data": {},
  "error": "Error Timeout",
  "time_taken": "540s"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format or missing required fields |
| 500 | Internal Server Error | Server error or timeout |

### Common Error Scenarios

**1. Model Not Downloaded**

```json
{
  "success": false,
  "error": "model 'llama3.1:8b' not found: try pulling it first"
}
```

**Solution:** Use `/ollama-pull-model` to download the model first.

**2. Invalid Request Format**

```json
{
  "success": false,
  "error": "Error unmarshalling messages"
}
```

**Solution:** Ensure request body matches the expected format.

**3. Request Timeout**

```json
{
  "success": false,
  "error": "Error Timeout"
}
```

**Solution:** Model generation took too long (>540s). Consider using a smaller model.

**4. Ollama Service Unavailable**

```json
{
  "success": false,
  "error": "error making request: dial tcp: connection refused"
}
```

**Solution:** Ensure Ollama service is running and healthy.

---

## Rate Limiting

Currently, there are no rate limits enforced. However, be mindful of:

- **Concurrent requests**: Limited by `OLLAMA_NUM_PARALLEL` (default: 2)
- **Loaded models**: Limited by `OLLAMA_MAX_LOADED_MODELS` (default: 2)
- **Model keep-alive**: Models stay loaded for 30 minutes by default

---

## Best Practices

### 1. Model Management

- **Always check available models** before calling `/call-llm`:
  ```bash
  curl http://localhost:8081/orus-api/v1/ollama-model-list
  ```

- **Pull required models during setup**, not during user requests

### 2. Embedding

- Use `nomic-embed-text` for general-purpose embeddings
- Use `bge-m3` for multilingual or specialized tasks
- Cache embeddings to avoid redundant API calls

### 3. LLM Calls

- Include a `system` message to guide model behavior
- Keep conversation history in `messages` for context
- Start with smaller models (`phi3:mini`, `mistral:7b`) for faster responses
- Use larger models (`llama3.1:70b`) only when necessary

### 4. Error Handling

- Always check `success` field in response
- Implement retry logic for timeout errors
- Validate model availability before making requests

### 5. Performance

- Pre-load frequently used models
- Use appropriate `OLLAMA_KEEP_ALIVE` settings
- Monitor resource usage (CPU, RAM, GPU)

---

## Examples Collection

### Python Client Example

```python
import requests
import json

class OrusClient:
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url
    
    def get_system_info(self):
        response = requests.get(f"{self.base_url}/orus-api/v1/system-info")
        return response.json()
    
    def list_models(self):
        response = requests.get(f"{self.base_url}/orus-api/v1/ollama-model-list")
        return response.json()
    
    def embed_text(self, text, model="nomic-embed-text"):
        payload = {
            "body": {
                "model": model,
                "text": text
            }
        }
        response = requests.post(
            f"{self.base_url}/orus-api/v1/embed-text",
            json=payload
        )
        return response.json()
    
    def call_llm(self, messages, model="llama3.1:8b"):
        payload = {
            "body": {
                "model": model,
                "stream": False,
                "messages": messages
            }
        }
        response = requests.post(
            f"{self.base_url}/orus-api/v1/call-llm",
            json=payload,
            timeout=600
        )
        return response.json()

# Usage
client = OrusClient()

# Get system info
info = client.get_system_info()
print(info)

# List models
models = client.list_models()
print(f"Available models: {models['data']['models']}")

# Embed text
embedding = client.embed_text("Hello, world!")
print(f"Embedding dimensions: {embedding['data']['dimensions']}")

# Chat with LLM
messages = [
    {"role": "user", "content": "What is AI?"}
]
response = client.call_llm(messages)
print(response['data']['response'])
```

### Node.js Client Example

```javascript
const axios = require('axios');

class OrusClient {
  constructor(baseURL = 'http://localhost:8081') {
    this.baseURL = baseURL;
    this.client = axios.create({ baseURL });
  }

  async getSystemInfo() {
    const response = await this.client.get('/orus-api/v1/system-info');
    return response.data;
  }

  async listModels() {
    const response = await this.client.get('/orus-api/v1/ollama-model-list');
    return response.data;
  }

  async embedText(text, model = 'nomic-embed-text') {
    const response = await this.client.post('/orus-api/v1/embed-text', {
      body: { model, text }
    });
    return response.data;
  }

  async callLLM(messages, model = 'llama3.1:8b') {
    const response = await this.client.post('/orus-api/v1/call-llm', {
      body: {
        model,
        stream: false,
        messages
      }
    }, { timeout: 600000 });
    return response.data;
  }
}

// Usage
(async () => {
  const client = new OrusClient();
  
  // Get system info
  const info = await client.getSystemInfo();
  console.log(info);
  
  // List models
  const models = await client.listModels();
  console.log('Available models:', models.data.models);
  
  // Embed text
  const embedding = await client.embedText('Hello, world!');
  console.log('Embedding dimensions:', embedding.data.dimensions);
  
  // Chat with LLM
  const messages = [
    { role: 'user', content: 'What is AI?' }
  ];
  const response = await client.callLLM(messages);
  console.log(response.data.response);
})();
```

---

## Appendix

### Model Size Reference

| Model | Parameters | Download Size | RAM Required |
|-------|-----------|---------------|--------------|
| `phi3:mini` | 3.8B | ~2.2GB | 4GB |
| `mistral:7b` | 7B | ~4.1GB | 6GB |
| `llama3.1:8b` | 8B | ~4.7GB | 8GB |
| `gemma:7b` | 7B | ~4.7GB | 6GB |
| `codellama:7b` | 7B | ~3.8GB | 6GB |
| `llama3.1:70b` | 70B | ~40GB | 48GB+ |
| `nomic-embed-text` | 137M | ~274MB | 1GB |

### Embedding Dimensions

| Model | Dimensions | Type |
|-------|-----------|------|
| `bge-m3` | 1024 | Dense vector |
| `nomic-embed-text` | 768 | Dense vector |

### Useful Links

- [Ollama Model Library](https://ollama.com/library)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)
- [Nomic Embed Documentation](https://blog.nomic.ai/posts/nomic-embed-text-v1)
