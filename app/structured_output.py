import json
from dataclasses import dataclass

from pydantic import ValidationError

from ollama import Client

from app.ollama_client import (
    MODEL_NAME,
    OLLAMA_HOST,
)

from app.schemas import StructuredAssistantResponse


client = Client(host=OLLAMA_HOST)


class StructuredOutputError(RuntimeError):
    """Raised when structured output remains invalid after retry."""


@dataclass
class StructuredGenerationResult:
    data: StructuredAssistantResponse
    attempts: int
    raw_response: str


def _call_model(
    prompt: str,
) -> str:
    """
    Request JSON from the local LLM using
    the Pydantic-generated JSON schema.
    """

    schema = (
        StructuredAssistantResponse
        .model_json_schema()
    )

    schema_text = json.dumps(
        schema,
        indent=2,
    )

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY valid JSON that "
                    "matches the provided schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Required JSON schema:\n"
                    f"{schema_text}"
                ),
            },
        ],
        format=schema,
        options={
            "temperature": 0.0,
        },
        stream=False,
    )

    return response.message.content


def generate_structured_response(
    prompt: str,
    max_retries: int = 1,
) -> StructuredGenerationResult:

    attempts = 0
    last_error = None
    raw_response = ""

    for attempt in range(
        max_retries + 1
    ):

        attempts += 1

        retry_instruction = ""

        if attempt > 0:
            retry_instruction = (
                "\n\nYour previous response failed "
                "validation. Return ONLY JSON that "
                "matches the required schema."
            )

        try:

            raw_response = _call_model(
                prompt + retry_instruction
            )

            data = (
                StructuredAssistantResponse
                .model_validate_json(
                    raw_response
                )
            )

            return StructuredGenerationResult(
                data=data,
                attempts=attempts,
                raw_response=raw_response,
            )

        except (
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            last_error = exc

            if attempt >= max_retries:
                break

    raise StructuredOutputError(
        "Structured output remained invalid "
        f"after {attempts} attempt(s). "
        f"Validation error: {last_error}"
    )