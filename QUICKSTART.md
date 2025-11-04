# Orus API Quick Start Guide

This guide will help you get started with Orus API quickly with practical examples.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Model Management](#model-management)
3. [Text Embedding](#text-embedding)
4. [LLM Chat](#llm-chat)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### Step 1: Start the Services

**Linux/Mac:**
```bash
./setup.sh
```

**Windows:**
```powershell
.\setup.ps1
```

**Manual (any OS):**
```bash
docker-compose up -d
```

### Step 2: Verify Services are Running

```bash
# Check Orus API
curl http://localhost:8081/orus-api/v1/system-info

# Check Ollama
curl http://localhost:11434/api/tags
```

---

## Model Management

### Download Embedding Model (Required First)

This is **mandatory** before using the embedding functionality.

```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "nomic-embed-text"}'
```

**Output:**
```
data: {"status":"pulling manifest"}
data: {"status":"downloading","digest":"sha256:...","total":274000000,"completed":68500000}
data: {"status":"verifying sha256 digest"}
data: {"status":"success","message":"Model nomic-embed-text downloaded successfully"}
```

### Download LLM Models

Choose based on your needs:

**Recommended for most users (Llama 3.1 8B):**
```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1:8b"}'
```

**Faster, smaller model (Phi-3 Mini):**
```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "phi3:mini"}'
```

**For coding tasks (CodeLlama):**
```bash
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "codellama:7b"}'
```

### List Available Models

```bash
curl http://localhost:8081/orus-api/v1/ollama-model-list
```

**Example Response:**
```json
{
  "success": true,
  "serial": "uuid...",
  "message": "Ollama model list retrieved successfully",
  "data": {
    "models": [
      "llama3.1:8b",
      "nomic-embed-text",
      "phi3:mini"
    ]
  },
  "error": "",
  "time_taken": "45ms"
}
```

---

## Text Embedding

### Basic Embedding Example

```bash
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "nomic-embed-text",
      "text": "Machine learning is transforming technology."
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "vector": [0.123, -0.456, 0.789, ...],
    "text": "Machine learning is transforming technology.",
    "model": "nomic-embed-text",
    "dimensions": 768
  },
  "time_taken": "234ms"
}
```

### Use Case: Document Similarity

**Embed multiple documents:**

```bash
# Document 1
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "nomic-embed-text",
      "text": "Python is a programming language."
    }
  }' > doc1.json

# Document 2
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "nomic-embed-text",
      "text": "JavaScript is used for web development."
    }
  }' > doc2.json
```

**Calculate similarity in Python:**

```python
import json
import numpy as np

# Load embeddings
with open('doc1.json') as f:
    doc1 = json.load(f)
with open('doc2.json') as f:
    doc2 = json.load(f)

# Extract vectors
v1 = np.array(doc1['data']['vector'])
v2 = np.array(doc2['data']['vector'])

# Calculate cosine similarity
similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"Similarity: {similarity:.4f}")
```

---

## LLM Chat

### Simple Question

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

### With System Prompt

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
          "content": "You are a helpful assistant that provides concise answers."
        },
        {
          "role": "user",
          "content": "Explain quantum computing in one sentence."
        }
      ]
    }
  }'
```

### Multi-Turn Conversation

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
          "content": "I need a creative name for a coffee shop."
        },
        {
          "role": "assistant",
          "content": "How about \"Brew Haven\" or \"The Daily Grind\"?"
        },
        {
          "role": "user",
          "content": "I like Brew Haven. Can you suggest a tagline?"
        }
      ]
    }
  }'
```

### Code Generation Example

```bash
curl -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "codellama:7b",
      "stream": false,
      "messages": [
        {
          "role": "system",
          "content": "You are an expert programmer. Provide clean, well-documented code."
        },
        {
          "role": "user",
          "content": "Write a Python function to calculate the Fibonacci sequence up to n terms."
        }
      ]
    }
  }'
```

---

## Common Workflows

### Workflow 1: Semantic Search

**Goal:** Find the most relevant document for a query.

```bash
# 1. Embed your documents (do this once)
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{"body":{"model":"nomic-embed-text","text":"Document 1 content"}}' \
  | jq -r '.data.vector' > doc1_vector.json

curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{"body":{"model":"nomic-embed-text","text":"Document 2 content"}}' \
  | jq -r '.data.vector' > doc2_vector.json

# 2. Embed user query
curl -X POST http://localhost:8081/orus-api/v1/embed-text \
  -H "Content-Type: application/json" \
  -d '{"body":{"model":"nomic-embed-text","text":"User search query"}}' \
  | jq -r '.data.vector' > query_vector.json

# 3. Calculate similarities (use Python/Node.js/etc.)
# 4. Return most similar document
```

### Workflow 2: RAG (Retrieval Augmented Generation)

**Goal:** Answer questions using your own documents.

```bash
# 1. Find relevant document (using semantic search above)

# 2. Create context-aware prompt
curl -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "llama3.1:8b",
      "stream": false,
      "messages": [
        {
          "role": "system",
          "content": "Answer questions based only on the provided context. Context: [Retrieved document content here]"
        },
        {
          "role": "user",
          "content": "What does the document say about topic X?"
        }
      ]
    }
  }'
```

### Workflow 3: Chatbot with Memory

```python
import requests
import json

class Chatbot:
    def __init__(self, model="llama3.1:8b"):
        self.model = model
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.base_url = "http://localhost:8081"
    
    def chat(self, user_message):
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Call API
        response = requests.post(
            f"{self.base_url}/orus-api/v1/call-llm",
            json={
                "body": {
                    "model": self.model,
                    "stream": False,
                    "messages": self.messages
                }
            }
        )
        
        result = response.json()
        assistant_message = result['data']['response']
        
        # Add assistant response to history
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message

# Usage
bot = Chatbot()
print(bot.chat("Hi! I'm learning Python."))
print(bot.chat("Can you recommend a good project?"))
print(bot.chat("What resources should I use?"))
```

### Workflow 4: Content Generation Pipeline

```bash
#!/bin/bash

# 1. Generate blog outline
OUTLINE=$(curl -s -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "model": "llama3.1:8b",
      "stream": false,
      "messages": [{
        "role": "user",
        "content": "Create an outline for a blog post about AI in healthcare."
      }]
    }
  }' | jq -r '.data.response')

echo "Outline:"
echo "$OUTLINE"

# 2. Generate full content based on outline
curl -X POST http://localhost:8081/orus-api/v1/call-llm \
  -H "Content-Type: application/json" \
  -d "{
    \"body\": {
      \"model\": \"llama3.1:8b\",
      \"stream\": false,
      \"messages\": [{
        \"role\": \"user\",
        \"content\": \"Write a detailed blog post based on this outline: $OUTLINE\"
      }]
    }
  }" | jq -r '.data.response' > blog_post.md

echo "Blog post generated: blog_post.md"
```

---

## Troubleshooting

### Issue: "Model not found"

**Error:**
```json
{
  "success": false,
  "error": "model 'llama3.1:8b' not found: try pulling it first"
}
```

**Solution:**
```bash
# Pull the model first
curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1:8b"}'
```

### Issue: Request Timeout

**Error:**
```json
{
  "success": false,
  "error": "Error Timeout"
}
```

**Solutions:**
1. Use a smaller model (phi3:mini instead of llama3.1:70b)
2. Reduce the complexity of your prompt
3. Increase available RAM/CPU resources

### Issue: Connection Refused

**Error:**
```
curl: (7) Failed to connect to localhost port 8081
```

**Solution:**
```bash
# Check if services are running
docker-compose ps

# Start services if not running
docker-compose up -d

# Wait for services to be ready
sleep 30

# Check logs
docker-compose logs -f
```

### Issue: Out of Memory

**Symptoms:** Container crashes or Docker becomes unresponsive

**Solution:**

Edit `docker-compose.yml` and reduce resources:
```yaml
environment:
  - OLLAMA_NUM_PARALLEL=1
  - OLLAMA_MAX_LOADED_MODELS=1

deploy:
  resources:
    limits:
      memory: 4G  # Reduce from 8G
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

---

## Performance Tips

1. **Keep Models Loaded**: Set appropriate `OLLAMA_KEEP_ALIVE` (default: 30m)
2. **Parallel Requests**: Increase `OLLAMA_NUM_PARALLEL` for concurrent requests
3. **Cache Embeddings**: Store embeddings in a database instead of regenerating
4. **Use Appropriate Models**: Small models (phi3:mini) for simple tasks
5. **Monitor Resources**: Use `docker stats` to monitor resource usage

---

## Next Steps

- Read the [API Documentation](./API.md) for complete endpoint reference
- Check the [README](./README.md) for architecture details
- Explore [Ollama Model Library](https://ollama.com/library) for more models
- Join the community for support and examples

Happy coding! 🚀
