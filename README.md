# FreeGPT

Free access to GPT-4, using only a Github Copilot subscription.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package and project manager
- Python 3.12+
- GitHub Copilot subscription

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

#### API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check endpoint

#### Example Usage with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
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