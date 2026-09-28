import time
from dataclasses import dataclass

from ollama import Client


OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

client = Client(host=OLLAMA_HOST)


@dataclass
class GenerationResult:
    response: str
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    tokens_per_second: float


def generate_response(
    prompt: str,
    temperature: float = 0.0,
) -> GenerationResult:

    start_time = time.perf_counter()

    first_token_time = None
    response_parts = []
    output_tokens = 0

    stream = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": temperature,
        },
        stream=True,
    )

    for chunk in stream:

        message = chunk.get("message", {})
        content = message.get("content", "")

        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            response_parts.append(content)

        # Ollama provides token counts in the final chunk.
        if chunk.get("done"):
            output_tokens = chunk.get(
                "eval_count",
                0
            )

    end_time = time.perf_counter()

    total_latency_ms = (
        end_time - start_time
    ) * 1000

    if first_token_time is not None:
        ttft_ms = (
            first_token_time - start_time
        ) * 1000
    else:
        ttft_ms = total_latency_ms

    generation_time_seconds = (
        end_time -
        first_token_time
        if first_token_time is not None
        else end_time - start_time
    )

    if generation_time_seconds > 0:
        tokens_per_second = (
            output_tokens /
            generation_time_seconds
        )
    else:
        tokens_per_second = 0.0

    return GenerationResult(
        response="".join(response_parts),
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        output_tokens=output_tokens,
        tokens_per_second=tokens_per_second,
    )