# FreeGPT

Free access to GPT-4, using only a Github Copilot subscription.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package and project manager
- Python 3.12+
- GitHub Copilot subscription

### For Embeddings Feature

- [Ollama](https://ollama.com/) - Local AI model runner
- Pull the embedding model: `ollama pull mxbai-embed-large`

## Usage

### Interactive Chat CLI

```bash
uv run chat.py
```

This command will automatically download dependencies and execute the interactive chat program.

### API Server (OpenAI-compatible)

Start the API server:

```bash
uv run api.py
```

The server will start on `http://localhost:8000` and provide an OpenAI-compatible API endpoint.

#### Authentication

On first run, you'll need to authenticate with GitHub. The server provides endpoints to handle the authentication flow:

**Step 1: Start Authentication**
```bash
curl -X POST http://localhost:8000/auth/device | jq
```

Response:
```json
{
  "device_code": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
  "user_code": "XXXX-XXXX",
  "verification_uri": "https://github.com/login/device",
  "expires_in": 900,
  "message": "Please visit https://github.com/login/device and enter code XXXX-XXXX"
}
```

**Step 2: Complete Authentication**
1. Visit the `verification_uri` in your browser
2. Enter the `user_code` when prompted
3. Authorize the application

**Step 3: Verify Authentication**
```bash
# Replace DEVICE_CODE with the actual device_code from step 1
curl -X POST http://localhost:8000/auth/device/DEVICE_CODE/complete | jq
```

Responses:
- While waiting: `{"status": "pending", "error": "authorization_pending"}`
- On success: `{"status": "completed", "message": "Authentication successful"}`
- On error: `{"status": "error", "message": "Error description"}`

The authentication token will be saved and reused for future requests.

#### API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /v1/embeddings` - OpenAI-compatible embeddings generation
- `GET /v1/models` - List available models
- `GET /health` - Health check endpoint
- `POST /auth/device` - Start GitHub device authentication
- `POST /auth/device/{device_code}/complete` - Check/complete authentication

#### Example Usage with curl

```bash
# Non-streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }' | jq

# Streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

#### Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key is not used but required by the client
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Embeddings Examples

**Note**: Requires Ollama to be running (`ollama serve`) with the `mxbai-embed-large` model.

##### Using curl

```bash
# Single string input
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The sky is blue because of Rayleigh scattering",
    "model": "mxbai-embed-large"
  }' | jq

# Multiple strings input
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "The sky is blue because of Rayleigh scattering",
      "Embeddings are useful for semantic search"
    ],
    "model": "mxbai-embed-large"
  }' | jq
```

##### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Single string embedding
response = client.embeddings.create(
    input="The sky is blue because of Rayleigh scattering",
    model="mxbai-embed-large"
)

# Multiple strings embedding
response = client.embeddings.create(
    input=[
        "The sky is blue because of Rayleigh scattering",
        "Embeddings are useful for semantic search"
    ],
    model="mxbai-embed-large"
)

# Access the embedding vectors
for data in response.data:
    print(f"Embedding {data.index}: {len(data.embedding)} dimensions")
```

### Docker

Run the API server using Docker:

```bash
# Copy and configure environment variables
cp .env.sample .env
# Edit .env to customize settings if needed

# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f copilot-api

# Stop the service
docker-compose down
```

The token will be stored persistently in the `./data` directory.

#### Using Embeddings with Docker

When running the API server in Docker, it needs to connect to Ollama running on your host machine. The `docker-compose.yaml` is pre-configured for this:

**macOS/Windows:**
- Default configuration uses `host.docker.internal:11434`
- No changes needed if Ollama is running on default port

**Linux:**
- Option 1: Replace `host.docker.internal` with your machine's IP address
- Option 2: Use host network mode by adding `network_mode: host` to docker-compose.yaml
- Option 3: Set environment variable: `OLLAMA_HOST=http://172.17.0.1:11434` (Docker's default gateway)

#### Docker Authentication

When running via Docker, use the authentication endpoints:

```bash
# Start authentication
curl -X POST http://localhost:8000/auth/device | jq

# Complete authentication (replace DEVICE_CODE)
curl -X POST http://localhost:8000/auth/device/DEVICE_CODE/complete | jq
```

Alternatively, check the container logs for authentication instructions:
```bash
docker-compose logs copilot-api
```

## Troubleshooting

### Authentication Issues

**Token not persisting in Docker:**
- Ensure the `./data` directory exists: `mkdir -p data`
- Check permissions: `ls -la data/`
- Verify volume mount in docker-compose.yaml

**Authentication timeout:**
- The authentication process has a 5-minute timeout
- If it expires, restart the authentication flow

**Invalid token errors:**
- Delete the existing token: `rm data/.copilot_token`
- Re-authenticate using the `/auth/device` endpoint

### Common Errors

- **401 Unauthorized**: Token is invalid or expired. Re-authenticate.
- **Connection refused**: Ensure the server is running on port 8000
- **No logs visible**: Add `-f` flag to `docker-compose logs -f copilot-api`

### Embeddings Errors

**503 Service Unavailable - Ollama not running:**
- Start Ollama: `ollama serve`
- Verify it's running: `curl http://localhost:11434/api/tags`

**Model not found:**
- Pull the model: `ollama pull mxbai-embed-large`
- List available models: `ollama list`

**Connection errors:**
- Check if Ollama is running on the default port (11434)
- Ensure no firewall is blocking the connection
- Try restarting Ollama service

**Docker-specific: Cannot connect to Ollama:**
- Ensure Ollama is running on the host machine (not inside Docker)
- Check OLLAMA_HOST environment variable in docker-compose.yaml
- For Linux users:
  ```bash
  # Find your host IP
  ip addr show docker0
  # Update docker-compose.yaml with your IP
  OLLAMA_HOST=http://172.17.0.1:11434
  ```
- Test connectivity from container:
  ```bash
  docker-compose exec copilot-api curl http://host.docker.internal:11434/api/tags
  ```