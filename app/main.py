from fastapi import FastAPI, HTTPException
from app.structured_output import (
    StructuredOutputError,
    generate_structured_response,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    StructuredChatAPIResponse,
)
from app.ollama_client import (
    MODEL_NAME,
    generate_response,
)

from app.schemas import (
    ChatRequest,
    ChatResponse,
)


app = FastAPI(
    title="Local AI Assistant",
    description=(
        "Offline AI assistant powered by "
        "a local Ollama model."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Local AI Assistant",
        "status": "running",
        "model": MODEL_NAME,
        "mode": "offline",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "ollama": "localhost:11434",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        result = generate_response(
            prompt=request.prompt,
            temperature=request.temperature,
        )

        return ChatResponse(
            response=result.response,
            model=MODEL_NAME,
            temperature=request.temperature,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,
            output_tokens=result.output_tokens,
            tokens_per_second=result.tokens_per_second,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Local model inference failed: "
                f"{str(exc)}"
            ),
        )
@app.post(
    "/structured-chat",
    response_model=StructuredChatAPIResponse,
)
def structured_chat(request: ChatRequest):

    try:

        result = generate_structured_response(
            prompt=request.prompt,
            max_retries=1,
        )

        return StructuredChatAPIResponse(
            result=result.data,
            model=MODEL_NAME,
            attempts=result.attempts,
        )

    except StructuredOutputError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )