import argparse
import csv
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from app.ollama_client import generate_response
from app.system_metrics import (
    get_ollama_model_memory_mb,
)


PROMPT_FILE = (
    Path("benchmarks") /
    "prompts.json"
)

DEFAULT_REPETITIONS = 1
DEFAULT_NUM_PREDICT = 256


def load_prompts(limit=None):

    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        prompts = json.load(file)

    if limit is not None:
        prompts = prompts[:limit]

    return prompts


def unload_model(model):

    subprocess.run(
        [
            "ollama",
            "stop",
            model,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def wait_for_model_unloaded(
    timeout_seconds=30,
):

    start = time.perf_counter()

    while (
        time.perf_counter() - start
        < timeout_seconds
    ):

        memory = (
            get_ollama_model_memory_mb()
        )

        if memory == 0:
            return True

        time.sleep(0.5)

    return False


def warm_up(model, num_predict):

    print("\nRunning warm-up...")

    result = generate_response(
        prompt=(
            "Give a one-sentence explanation "
            "of what an API is."
        ),
        temperature=0.0,
        model=model,
        num_predict=num_predict,
    )

    print(
        f"Warm-up TTFT: "
        f"{result.ttft_ms:.2f} ms"
    )

    print(
        f"Warm-up latency: "
        f"{result.total_latency_ms:.2f} ms"
    )

    print(
        f"Warm-up tokens/sec: "
        f"{result.tokens_per_second:.2f}"
    )


def safe_stdev(values):

    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def calculate_summary(results):

    ttft = [
        r["ttft_ms"]
        for r in results
    ]

    latency = [
        r["latency_ms"]
        for r in results
    ]

    tokens_per_second = [
        r["tokens_per_second"]
        for r in results
    ]

    output_tokens = [
        r["output_tokens"]
        for r in results
    ]

    memory = [
        r["model_memory_mb"]
        for r in results
        if r["model_memory_mb"] > 0
    ]

    summary = {
        "model": results[0]["model"],
        "prompt_count": len(
            set(
                r["prompt_id"]
                for r in results
            )
        ),
        "total_runs": len(results),

        "ttft_mean_ms": statistics.mean(
            ttft
        ),
        "ttft_median_ms": statistics.median(
            ttft
        ),
        "ttft_std_dev_ms": safe_stdev(
            ttft
        ),

        "latency_mean_ms": statistics.mean(
            latency
        ),
        "latency_median_ms": statistics.median(
            latency
        ),
        "latency_std_dev_ms": safe_stdev(
            latency
        ),

        "tokens_per_second_mean": (
            statistics.mean(
                tokens_per_second
            )
        ),
        "tokens_per_second_median": (
            statistics.median(
                tokens_per_second
            )
        ),
        "tokens_per_second_std_dev": (
            safe_stdev(
                tokens_per_second
            )
        ),

        "output_tokens_mean": (
            statistics.mean(
                output_tokens
            )
        ),

        "model_memory_mean_mb": (
            statistics.mean(memory)
            if memory
            else 0.0
        ),

        "model_memory_max_mb": (
            max(memory)
            if memory
            else 0.0
        ),
    }

    return summary


def calculate_category_summary(
    results,
):

    categories = sorted(
        set(
            r["category"]
            for r in results
        )
    )

    category_results = []

    for category in categories:

        rows = [
            r
            for r in results
            if r["category"] == category
        ]

        category_results.append(
            {
                "model": rows[0]["model"],
                "category": category,
                "runs": len(rows),
                "mean_ttft_ms": statistics.mean(
                    r["ttft_ms"]
                    for r in rows
                ),
                "mean_latency_ms": statistics.mean(
                    r["latency_ms"]
                    for r in rows
                ),
                "mean_tokens_per_second": (
                    statistics.mean(
                        r["tokens_per_second"]
                        for r in rows
                    )
                ),
                "mean_output_tokens": (
                    statistics.mean(
                        r["output_tokens"]
                        for r in rows
                    )
                ),
            }
        )

    return category_results


def save_results(
    results,
    summary,
    category_summary,
):

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    model_slug = (
        summary["model"]
        .replace(":", "_")
        .replace("/", "_")
    )

    raw_file = (
        RESULTS_DIR /
        f"{model_slug}_raw.csv"
    )

    summary_file = (
        RESULTS_DIR /
        f"{model_slug}_summary.csv"
    )

    category_file = (
        RESULTS_DIR /
        f"{model_slug}_category_summary.csv"
    )

    with open(
        raw_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys(),
        )

        writer.writeheader()
        writer.writerow(summary)

    with open(
        category_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=category_summary[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            category_summary
        )

    return (
        raw_file,
        summary_file,
        category_file,
    )


def run_benchmark(
    model,
    repetitions,
    limit,
    num_predict,
):

    prompts = load_prompts(limit)

    if not prompts:
        raise RuntimeError(
            "No benchmark prompts loaded."
        )

    print()
    print("=" * 65)
    print("STANDARDIZED LOCAL LLM BENCHMARK")
    print("=" * 65)

    print(f"Model: {model}")
    print(
        f"Prompts: {len(prompts)}"
    )
    print(
        f"Repetitions: {repetitions}"
    )
    print(
        f"Max output tokens: "
        f"{num_predict}"
    )

    print(
        f"Total measured runs: "
        f"{len(prompts) * repetitions}"
    )

    print("=" * 65)

    # Ensure only this model is active.
    print("\nUnloading model if currently loaded...")

    unload_model(model)

    unloaded = wait_for_model_unloaded()

    if unloaded:
        print("Model unloaded.")

    else:
        print(
            "Warning: model may still "
            "be resident."
        )

    # Warm-up is intentionally excluded.
    warm_up(
        model,
        num_predict,
    )

    results = []

    for prompt_index, item in enumerate(
        prompts,
        start=1,
    ):

        prompt_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        for repetition in range(
            1,
            repetitions + 1,
        ):

            print(
                f"\n[{prompt_index}/"
                f"{len(prompts)}] "
                f"{prompt_id} | "
                f"run {repetition}/"
                f"{repetitions}"
            )

            result = generate_response(
                prompt=prompt,
                temperature=0.0,
                model=model,
                num_predict=num_predict,
            )

            memory = (
                get_ollama_model_memory_mb()
            )

            row = {
                "timestamp": (
                    datetime.now()
                    .isoformat()
                ),
                "model": model,
                "prompt_id": prompt_id,
                "category": category,
                "repetition": repetition,
                "temperature": 0.0,
                "num_predict": num_predict,
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
                "model_memory_mb": memory,
                "output": result.response,
            }

            results.append(row)

            print(
                f"TTFT: "
                f"{result.ttft_ms:.2f} ms"
            )

            print(
                f"Latency: "
                f"{result.total_latency_ms:.2f} ms"
            )

            print(
                f"Output tokens: "
                f"{result.output_tokens}"
            )

            print(
                f"Tokens/sec: "
                f"{result.tokens_per_second:.2f}"
            )

            print(
                f"Model memory: "
                f"{memory:.2f} MB"
            )

    summary = calculate_summary(
        results
    )

    category_summary = (
        calculate_category_summary(
            results
        )
    )

    files = save_results(
        results,
        summary,
        category_summary,
    )

    print()
    print("=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    print(
        f"Model: "
        f"{summary['model']}"
    )

    print(
        f"Prompts: "
        f"{summary['prompt_count']}"
    )

    print(
        f"Runs: "
        f"{summary['total_runs']}"
    )

    print(
        f"TTFT mean: "
        f"{summary['ttft_mean_ms']:.2f} ms"
    )

    print(
        f"TTFT median: "
        f"{summary['ttft_median_ms']:.2f} ms"
    )

    print(
        f"Latency mean: "
        f"{summary['latency_mean_ms']:.2f} ms"
    )

    print(
        f"Latency median: "
        f"{summary['latency_median_ms']:.2f} ms"
    )

    print(
        f"Tokens/sec mean: "
        f"{summary['tokens_per_second_mean']:.2f}"
    )

    print(
        f"Output tokens mean: "
        f"{summary['output_tokens_mean']:.2f}"
    )

    print(
        f"Model memory mean: "
        f"{summary['model_memory_mean_mb']:.2f} MB"
    )

    print(
        f"Model memory max: "
        f"{summary['model_memory_max_mb']:.2f} MB"
    )

    print("=" * 65)

    for file in files:
        print(
            f"Saved: {file}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized "
            "local LLM benchmark."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Ollama model name, e.g. "
            "llama3.2:3b"
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N prompts "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=(
            "Maximum generated tokens."
        ),
    )

    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError(
            "Repetitions must be >= 1."
        )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "Limit must be >= 1."
            )

    if args.num_predict < 1:
        raise ValueError(
            "num-predict must be >= 1."
        )

    run_benchmark(
        model=args.model,
        repetitions=args.repetitions,
        limit=args.limit,
        num_predict=args.num_predict,
    )


if __name__ == "__main__":
    main()