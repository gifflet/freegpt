import requests
import json
import time
import asyncio
import os
import sys
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging
import ollama

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

# Token file path from environment or default
TOKEN_PATH = os.getenv('COPILOT_TOKEN_PATH', '.copilot_token')

# Ollama host configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# OpenAI-compatible request/response models
class Message(BaseModel):
    role: str
    content: str

class ResponseFormat(BaseModel):
    type: str  # "json_object" or "json_schema"
    json_schema: Optional[Dict[str, Any]] = None

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
    response_format: Optional[ResponseFormat] = None

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

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = Field(default="text-embedding-3-small")
    encoding_format: Optional[str] = Field(default="float")
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

# Authentication functions from chat.py
def setup():
    """Setup GitHub OAuth device flow authentication"""
    logger.info("Starting GitHub OAuth device flow authentication...")
    
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

    logger.critical(f'AUTHENTICATION REQUIRED: Please visit {verification_uri} and enter code {user_code}')
    logger.critical(f'Waiting for authentication... (timeout: 5 minutes)')
    sys.stdout.flush()  # Force flush to ensure Docker sees the logs

    start_time = time.time()
    timeout = 300  # 5 minutes timeout
    
    while True:
        if time.time() - start_time > timeout:
            logger.error("Authentication timeout - no response received within 5 minutes")
            raise HTTPException(status_code=408, detail="Authentication timeout. Please restart the authentication process.")
            
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
        error = resp_json.get('error')
        
        if error == 'authorization_pending':
            logger.debug("Authorization pending, checking again in 5 seconds...")
        elif error:
            logger.error(f"Authentication error: {error}")
            raise HTTPException(status_code=400, detail=f"Authentication failed: {error}")

        if access_token:
            break

    # Save the access token to a file
    try:
        token_dir = os.path.dirname(TOKEN_PATH)
        if token_dir:  # Only create directory if path has a directory component
            os.makedirs(token_dir, exist_ok=True)
            logger.info(f"Created token directory: {token_dir}")
        
        with open(TOKEN_PATH, 'w') as f:
            f.write(access_token)
        logger.critical(f"Token saved successfully to {TOKEN_PATH}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save authentication token: {str(e)}")

    logger.critical('Authentication successful!')
    sys.stdout.flush()
    return access_token

def get_github_token():
    """Get GitHub access token from file or setup"""
    global github_token
    try:
        with open(TOKEN_PATH, 'r') as f:
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
    
    if resp.status_code == 401:
        logger.error("GitHub token is invalid or expired. Re-authentication required.")
        raise HTTPException(status_code=401, detail="GitHub token is invalid. Please re-authenticate using POST /auth/device")
    elif resp.status_code != 200:
        logger.error(f"Failed to get Copilot token: {resp.status_code} - {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Failed to get Copilot token: {resp.text}")
    
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
        
        # Prepare request to Copilot
        copilot_request = {
            'intent': False,
            'model': request.model,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'n': request.n,
            'stream': request.stream,
            'messages': messages
        }
        
        # Add response_format if provided
        if request.response_format:
            copilot_request['response_format'] = {
                'type': request.response_format.type
            }
            if request.response_format.json_schema:
                copilot_request['response_format']['json_schema'] = request.response_format.json_schema

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

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint using Ollama"""
    try:
        # Create Ollama client with configured host
        client = ollama.Client(host=OLLAMA_HOST)
        
        # Normalize input to list
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input
        
        # Use the model from the request
        ollama_model = request.model
        
        embeddings_data = []
        total_tokens = 0
        
        # Generate embeddings for each input
        for idx, text in enumerate(inputs):
            try:
                # Call Ollama embeddings API with client
                response = client.embeddings(
                    model=ollama_model,
                    prompt=text
                )
                
                # Extract embedding from response
                embedding = response.get('embedding', [])
                
                # Create embedding data object
                embeddings_data.append(EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=idx
                ))
                
                # Rough token estimation (4 chars per token)
                total_tokens += len(text) // 4
                
            except Exception as e:
                logger.error(f"Error generating embedding for input {idx}: {str(e)}")
                # Check if Ollama is running
                if "connection" in str(e).lower() or "refused" in str(e).lower():
                    error_msg = f"Cannot connect to Ollama at {OLLAMA_HOST}. "
                    if "docker" in OLLAMA_HOST.lower() or OLLAMA_HOST != "http://localhost:11434":
                        error_msg += "When running in Docker, ensure OLLAMA_HOST is set correctly (e.g., 'host.docker.internal:11434' for Mac/Windows)"
                    else:
                        error_msg += "Please ensure Ollama is installed and running with 'ollama serve'"
                    raise HTTPException(
                        status_code=503,
                        detail=error_msg
                    )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate embedding: {str(e)}"
                )
        
        # Create response
        response = EmbeddingResponse(
            object="list",
            data=embeddings_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
        
        return response.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/device")
async def start_device_auth():
    """Start GitHub device authentication flow"""
    try:
        logger.info("Starting device authentication flow...")
        
        resp = requests.post('https://github.com/login/device/code', headers={
            'accept': 'application/json',
            'editor-version': 'Neovim/0.6.1',
            'editor-plugin-version': 'copilot.vim/1.16.0',
            'content-type': 'application/json',
            'user-agent': 'GithubCopilot/1.155.0',
            'accept-encoding': 'gzip,deflate,br'
        }, data='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}')
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")
        
        resp_json = resp.json()
        device_code = resp_json.get('device_code')
        user_code = resp_json.get('user_code')
        verification_uri = resp_json.get('verification_uri')
        expires_in = resp_json.get('expires_in', 900)
        
        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri,
            "expires_in": expires_in,
            "message": f"Please visit {verification_uri} and enter code {user_code}"
        }
    except Exception as e:
        logger.error(f"Failed to start device auth: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/device/{device_code}/complete")
async def complete_device_auth(device_code: str):
    """Complete GitHub device authentication flow"""
    try:
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
        error = resp_json.get('error')
        
        if error:
            return {"status": "pending", "error": error}
        
        if access_token:
            # Save the token
            global github_token
            github_token = access_token
            
            try:
                token_dir = os.path.dirname(TOKEN_PATH)
                if token_dir:
                    os.makedirs(token_dir, exist_ok=True)
                
                with open(TOKEN_PATH, 'w') as f:
                    f.write(access_token)
                    
                logger.info("Authentication completed successfully")
                return {"status": "completed", "message": "Authentication successful"}
            except Exception as e:
                logger.error(f"Failed to save token: {e}")
                return {"status": "error", "message": f"Authentication successful but failed to save token: {str(e)}"}
                
        return {"status": "pending", "message": "Authentication still pending"}
        
    except Exception as e:
        logger.error(f"Failed to complete device auth: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting FreeGPT API Proxy...")
    logger.info(f"Token path: {TOKEN_PATH}")
    logger.info(f"Ollama host: {OLLAMA_HOST}")
    sys.stdout.flush()
    
    # Check if token exists but don't try to authenticate at startup
    if os.path.exists(TOKEN_PATH):
        logger.info("Token file found. Will validate on first request.")
    else:
        logger.critical("No token file found. Authentication required.")
        logger.critical("To authenticate:")
        logger.critical("1. Use POST /auth/device to start authentication")
        logger.critical("2. Or make an API request to trigger automatic authentication")
        sys.stdout.flush()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)