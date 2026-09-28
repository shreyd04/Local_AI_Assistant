from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )


class ChatResponse(BaseModel):
    response: str
    model: str
    temperature: float

    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    tokens_per_second: float