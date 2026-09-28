import csv
from datetime import datetime

from app.ollama_client import MODEL_NAME
from app.structured_output import (
    StructuredOutputError,
    generate_structured_response,
)


PROMPTS = [
    {
        "id": "structure_001",
        "prompt": (
            "Explain what an API is. "
            "Give the most important concepts."
        ),
    },
    {
        "id": "structure_002",
        "prompt": (
            "Explain the difference between "
            "a process and a thread."
        ),
    },
    {
        "id": "structure_003",
        "prompt": (
            "Explain what a database index is "
            "and why it is useful."
        ),
    },
    {
        "id": "structure_004",
        "prompt": (
            "Explain overfitting in machine learning "
            "and give methods to reduce it."
        ),
    },
    {
        "id": "structure_005",
        "prompt": (
            "Explain REST API architecture "
            "and its main principles."
        ),
    },
    {
        "id": "structure_006",
        "prompt": (
            "Explain what normalization means "
            "in relational databases."
        ),
    },
    {
        "id": "structure_007",
        "prompt": (
            "Explain the difference between "
            "SQL DELETE, DROP, and TRUNCATE."
        ),
    },
    {
        "id": "structure_008",
        "prompt": (
            "Explain binary search and when "
            "it is useful."
        ),
    },
    {
        "id": "structure_009",
        "prompt": (
            "Explain supervised versus "
            "unsupervised learning."
        ),
    },
    {
        "id": "structure_010",
        "prompt": (
            "Explain what an HTTP status code is "
            "and give examples."
        ),
    },
]


def run_benchmark():

    results = []

    print("=" * 60)
    print("STRUCTURED OUTPUT BENCHMARK")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(PROMPTS)}")

    print("=" * 60)

    for item in PROMPTS:

        prompt_id = item["id"]
        prompt = item["prompt"]

        print()
        print(f"Prompt: {prompt_id}")
        print(f"Input: {prompt}")

        try:

            result = generate_structured_response(
                prompt=prompt,
                max_retries=1,
            )

            results.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "success": True,
                    "attempts": result.attempts,
                    "ttft_ms": "",
                    "latency_ms": "",
                    "output": result.raw_response,
                    "error": "",
                }
            )

            print(
                f"Success: True"
            )

            print(
                f"Attempts: "
                f"{result.attempts}"
            )

            print("Validated output:")
            print(result.data.model_dump_json(
                indent=2
            ))

        except StructuredOutputError as exc:

            results.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "model": MODEL_NAME,
                    "prompt_id": prompt_id,
                    "success": False,
                    "attempts": 2,
                    "ttft_ms": "",
                    "latency_ms": "",
                    "output": "",
                    "error": str(exc),
                }
            )

            print("Success: False")
            print(f"Error: {exc}")

    filename = "structured_results.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "model",
                "prompt_id",
                "success",
                "attempts",
                "ttft_ms",
                "latency_ms",
                "output",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    successful = [
        row
        for row in results
        if row["success"] is True
    ]

    retry_count = [
        row
        for row in successful
        if row["attempts"] == 2
    ]

    failures = [
        row
        for row in results
        if row["success"] is False
    ]

    print()
    print("=" * 60)
    print("STRUCTURED OUTPUT SUMMARY")
    print("=" * 60)

    print(
        f"Total prompts: {len(results)}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Retry required: {len(retry_count)}"
    )

    print(
        f"Final failures: {len(failures)}"
    )

    if results:
        print(
            f"First-attempt success rate: "
            f"{(len(successful) - len(retry_count)) / len(results) * 100:.2f}%"
        )

        print(
            f"Final success rate: "
            f"{len(successful) / len(results) * 100:.2f}%"
        )

    print("=" * 60)

    print(
        f"Results saved to: {filename}"
    )


if __name__ == "__main__":
    run_benchmark()
    