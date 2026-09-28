import csv
import statistics
from datetime import datetime

from app.ollama_client import (
    MODEL_NAME,
    generate_response,
)


PROMPTS = [
    "Explain what a database index is and why it improves query performance.",
    "Explain the difference between a process and a thread.",
    "What is REST API architecture? Give three important principles.",
    "Explain overfitting in machine learning and give two ways to reduce it.",
    "What is the purpose of a primary key in a relational database?",
]


TEMPERATURES = [
    0.0,
    0.7,
]


REPETITIONS = 3


def run_experiment():

    results = []

    print("=" * 60)
    print("TEMPERATURE VARIANCE BENCHMARK")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Repetitions: {REPETITIONS}")
    print(
        f"Temperatures: {TEMPERATURES}"
    )

    print("=" * 60)

    for temperature in TEMPERATURES:

        print(
            f"\nTEMPERATURE = {temperature}"
        )

        print("-" * 60)

        for prompt_index, prompt in enumerate(
            PROMPTS,
            start=1,
        ):

            for repetition in range(
                1,
                REPETITIONS + 1,
            ):

                print(
                    f"\nPrompt {prompt_index}/"
                    f"{len(PROMPTS)} | "
                    f"Run {repetition}/"
                    f"{REPETITIONS}"
                )

                result = generate_response(
                    prompt=prompt,
                    temperature=temperature,
                )

                print(
                    f"TTFT: "
                    f"{result.ttft_ms:.2f} ms"
                )

                print(
                    f"Latency: "
                    f"{result.total_latency_ms:.2f} ms"
                )

                print(
                    f"Tokens: "
                    f"{result.output_tokens}"
                )

                print(
                    f"Tokens/sec: "
                    f"{result.tokens_per_second:.2f}"
                )

                print(
                    "\nOUTPUT:"
                )

                print(
                   result.response
                )

                results.append(
                    {
                        "timestamp": (
                            datetime.now()
                            .isoformat()
                        ),
                        "model": MODEL_NAME,
                        "prompt_id": prompt_index,
                        "prompt": prompt,
                        "repetition": repetition,
                        "temperature": temperature,
                        "ttft_ms": result.ttft_ms,
                        "latency_ms": (
                            result.total_latency_ms
                        ),
                        "output_tokens": (
                            result.output_tokens
                        ),
                        "tokens_per_second": (
                            result.tokens_per_second
                        ),
                        "output": result.response,
                    }
                )
    return results


def save_results(results):

    filename = (
        "temperature_results.csv"
    )

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
                "prompt",
                "repetition",
                "temperature",
                "ttft_ms",
                "latency_ms",
                "output_tokens",
                "tokens_per_second",
                "output",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    return filename


def print_summary(results):

    print("\n")
    print("=" * 60)
    print("TEMPERATURE BENCHMARK SUMMARY")
    print("=" * 60)

    for temperature in TEMPERATURES:

        rows = [
            row
            for row in results
            if row["temperature"]
            == temperature
        ]

        ttft = [
            row["ttft_ms"]
            for row in rows
        ]

        latency = [
            row["latency_ms"]
            for row in rows
        ]

        tokens_per_second = [
            row["tokens_per_second"]
            for row in rows
        ]

        output_tokens = [
            row["output_tokens"]
            for row in rows
        ]

        print(
            f"\nTemperature: "
            f"{temperature}"
        )

        print(
            f"Runs: {len(rows)}"
        )

        print(
            f"TTFT mean: "
            f"{statistics.mean(ttft):.2f} ms"
        )

        print(
            f"Latency mean: "
            f"{statistics.mean(latency):.2f} ms"
        )

        print(
            f"Tokens/sec mean: "
            f"{statistics.mean(tokens_per_second):.2f}"
        )

        print(
            f"Output tokens mean: "
            f"{statistics.mean(output_tokens):.2f}"
        )

    print("=" * 60)


def main():

    results = run_experiment()

    print_summary(results)

    filename = save_results(
        results
    )

    print(
        f"\nResults saved to: "
        f"{filename}"
    )


if __name__ == "__main__":
    main()
