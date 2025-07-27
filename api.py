import requests
import json
import time
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FreeGPT API Proxy", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global token storage
github_token = None
copilot_token = None
token_expiry = None

# OpenAI-compatible request/response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4")
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str]

class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageStats = Field(default_factory=lambda: UsageStats())

# Authentication functions from chat.py
def setup():
    """Setup GitHub OAuth device flow authentication"""
    resp = requests.post('https://github.com/login/device/code', headers={
        'accept': 'application/json',
        'editor-version': 'Neovim/0.6.1',
        'editor-plugin-version': 'copilot.vim/1.16.0',
        'content-type': 'application/json',
        'user-agent': 'GithubCopilot/1.155.0',
        'accept-encoding': 'gzip,deflate,br'
    }, data='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}')

    resp_json = resp.json()
    device_code = resp_json.get('device_code')
    user_code = resp_json.get('user_code')
    verification_uri = resp_json.get('verification_uri')

    print(f'Please visit {verification_uri} and enter code {user_code} to authenticate.')

    while True:
        time.sleep(5)
        resp = requests.post('https://github.com/login/oauth/access_token', headers={
            'accept': 'application/json',
            'editor-version': 'Neovim/0.6.1',
            'editor-plugin-version': 'copilot.vim/1.16.0',
            'content-type': 'application/json',
            'user-agent': 'GithubCopilot/1.155.0',
            'accept-encoding': 'gzip,deflate,br'
        }, data=f'{{"client_id":"Iv1.b507a08c87ecfe98","device_code":"{device_code}","grant_type":"urn:ietf:params:oauth:grant-type:device_code"}}')

        resp_json = resp.json()
        access_token = resp_json.get('access_token')

        if access_token:
            break

    # Save the access token to a file
    with open('.copilot_token', 'w') as f:
        f.write(access_token)

    print('Authentication success!')
    return access_token

def get_github_token():
    """Get GitHub access token from file or setup"""
    global github_token
    try:
        with open('.copilot_token', 'r') as f:
            github_token = f.read().strip()
    except FileNotFoundError:
        github_token = setup()
    return github_token

def refresh_copilot_token():
    """Get or refresh Copilot API token"""
    global copilot_token, token_expiry
    
    if not github_token:
        get_github_token()
    
    # Get a new Copilot token
    resp = requests.get('https://api.github.com/copilot_internal/v2/token', headers={
        'authorization': f'token {github_token}',
        'editor-version': 'Neovim/0.6.1',
        'editor-plugin-version': 'copilot.vim/1.16.0',
        'user-agent': 'GithubCopilot/1.155.0'
    })
    
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Failed to get Copilot token")
    
    resp_json = resp.json()
    copilot_token = resp_json.get('token')
    # Token typically expires in 30 minutes, refresh at 25 minutes
    token_expiry = time.time() + (25 * 60)
    
    return copilot_token

def ensure_valid_token():
    """Ensure we have a valid Copilot token"""
    global copilot_token, token_expiry
    
    if not copilot_token or not token_expiry or time.time() >= token_expiry:
        refresh_copilot_token()
    
    return copilot_token

# Map OpenAI models to Copilot models
def map_model(openai_model: str) -> str:
    """Map OpenAI model names to Copilot model names"""
    model_mapping = {
        "gpt-4": "gpt-4.1",
        "gpt-4-turbo": "gpt-4.1",
        "gpt-4-turbo-preview": "gpt-4.1",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }
    return model_mapping.get(openai_model, "gpt-4.1")

async def stream_copilot_response(resp) -> AsyncGenerator[str, None]:
    """Convert Copilot streaming response to OpenAI format"""
    async def read_stream():
        for line in resp.iter_lines():
            if line:
                yield line.decode('utf-8')
    
    created = int(time.time())
    model = "gpt-4"
    
    async for line in read_stream():
        if line.startswith('data: {'):
            try:
                json_data = json.loads(line[6:])
                choices = json_data.get('choices', [])
                
                if choices and 'delta' in choices[0]:
                    delta = choices[0]['delta']
                    content = delta.get('content', '')
                    
                    # Format as OpenAI streaming response
                    chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except json.JSONDecodeError:
                continue
    
    # Send final chunk with usage info
    final_chunk = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,  # Will be calculated properly in production
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Ensure we have a valid token
        token = ensure_valid_token()
        
        # Convert messages to format expected by Copilot
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Map model name
        copilot_model = map_model(request.model)
        
        # Prepare request to Copilot
        copilot_request = {
            'intent': False,
            'model': copilot_model,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'n': request.n,
            'stream': request.stream,
            'messages': messages
        }
        
        # Make request to Copilot
        resp = requests.post('https://api.githubcopilot.com/chat/completions', 
            headers={
                'authorization': f'Bearer {token}',
                'Editor-Version': 'vscode/1.80.1',
                'content-type': 'application/json'
            }, 
            json=copilot_request,
            stream=request.stream
        )
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Copilot API error: {resp.text}")
        
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_copilot_response(resp),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response - handle both streaming and non-streaming formats
            result = ""
            
            # First try to parse as direct JSON response
            try:
                resp_json = resp.json()
                if 'choices' in resp_json and resp_json['choices']:
                    choice = resp_json['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        result = choice['message']['content']
                    elif 'delta' in choice and 'content' in choice['delta']:
                        result = choice['delta']['content']
            except (json.JSONDecodeError, KeyError):
                # Fallback: parse as streaming format
                for line in resp.text.split('\n'):
                    if line.startswith('data: {'):
                        try:
                            json_data = json.loads(line[6:])
                            choices = json_data.get('choices', [])
                            if choices and 'delta' in choices[0]:
                                content = choices[0]['delta'].get('content', '')
                                if content:
                                    result += content
                        except json.JSONDecodeError:
                            continue
            
            # Calculate token usage (rough estimation)
            prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
            completion_tokens = len(result.split())
            total_tokens = prompt_tokens + completion_tokens
            
            # Format as OpenAI response (return dict to avoid Pydantic issues)
            return {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4.1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "github-copilot"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "github-copilot"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "freegpt-api-proxy"}

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting FreeGPT API Proxy...")
    try:
        get_github_token()
        logger.info("GitHub token loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load GitHub token on startup: {e}")
        logger.info("Token will be obtained on first request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)