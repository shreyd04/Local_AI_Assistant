import csv
import os
import statistics
import time
from datetime import datetime

import psutil

from app.ollama_client import (
    MODEL_NAME,
    generate_response,
)

from app.system_metrics import (
    get_ollama_model_memory_mb,
    get_python_memory_mb,
)

PROMPTS = [
    "Explain what an API is in simple terms.",
    "Explain the difference between TCP and UDP.",
    "What is a binary search tree?",
    "Explain overfitting in machine learning.",
    "What is the purpose of normalization in databases.",
]


REPETITIONS = 3

def safe_stdev(values):
    """
    Calculate standard deviation when at least
    two measurements are available.
    """
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)
def get_memory_mb() -> float:
    """
    Return the current Python process RSS memory in MB.
    """
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss

    return memory_bytes / (1024 * 1024)


def run_single_test(
    prompt: str,
    temperature: float,
):
    python_memory_before = (
        get_python_memory_mb()
    )

    ollama_memory_before = (
        get_ollama_model_memory_mb()
    )

    result = generate_response(
        prompt=prompt,
        temperature=temperature,
    )

    python_memory_after = (
        get_python_memory_mb()
    )

    ollama_memory_after = (
        get_ollama_model_memory_mb()
    )

    return {
        "ttft_ms": result.ttft_ms,
        "total_latency_ms": result.total_latency_ms,
        "output_tokens": result.output_tokens,
        "tokens_per_second": result.tokens_per_second,

        "python_memory_before_mb": (
            python_memory_before
        ),

        "python_memory_after_mb": (
            python_memory_after
        ),

        "ollama_memory_before_mb": (
            ollama_memory_before
        ),

        "ollama_memory_after_mb": (
            ollama_memory_after
        ),
    }


def calculate_statistics(values):
    """
    Calculate basic descriptive statistics.
    """

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def run_benchmark():

    raw_results = []

    print()
    print("=" * 60)
    print("LOCAL AI ASSISTANT BENCHMARK")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Repetitions: {REPETITIONS}")
    print(
        f"Total inference runs: "
        f"{len(PROMPTS) * REPETITIONS}"
    )

    print("=" * 60)

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

            result = run_single_test(
                prompt=prompt,
                temperature=0.0,
            )

            row = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "prompt_id": prompt_index,
            "repetition": repetition,
            "temperature": 0.0,
        
            "ttft_ms": result["ttft_ms"],
            "total_latency_ms": result["total_latency_ms"],
            "output_tokens": result["output_tokens"],
            "tokens_per_second": result["tokens_per_second"],
        
            "python_memory_before_mb": (
                result["python_memory_before_mb"]
            ),
        
            "python_memory_after_mb": (
                result["python_memory_after_mb"]
            ),
        
            "ollama_memory_before_mb": (
                result["ollama_memory_before_mb"]
            ),
        
            "ollama_memory_after_mb": (
                result["ollama_memory_after_mb"]
            ),
            }

            raw_results.append(row)
            print(
                f"TTFT: "
                f"{result['ttft_ms']:.2f} ms"
            )
            print(
                f"Latency: "
                f"{result['total_latency_ms']:.2f} ms"
            )
            print(
                f"Tokens: "
                f"{result['output_tokens']}"
            )
            print(
                f"Tokens/sec: "
                f"{result['tokens_per_second']:.2f}"
            )
            print(
                f"Ollama/model memory: "
                f"{result['ollama_memory_after_mb']:.2f} MB"
            )
            print(
                f"Python memory: "
                f"{result['python_memory_after_mb']:.2f} MB"
            )

    # --------------------------------------------------
    # Save raw results
    # --------------------------------------------------

    raw_filename = (
        "benchmark_results_raw.csv"
    )

    with open(
        raw_filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=raw_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(raw_results)

    # --------------------------------------------------
    # Calculate overall statistics
    # --------------------------------------------------

    ttft_values = [
    row["ttft_ms"]
    for row in raw_results
    ]

    latency_values = [
        row["total_latency_ms"]
        for row in raw_results
    ]
    
    token_speed_values = [
        row["tokens_per_second"]
        for row in raw_results
    ]
    
    ollama_memory_values = [
        row["ollama_memory_after_mb"]
        for row in raw_results
    ]
    
    python_memory_values = [
        row["python_memory_after_mb"]
        for row in raw_results
    ]
    print(
    f"DEBUG: collected "
    f"{len(raw_results)} results"
    )
    statistics_data = {
    "model": MODEL_NAME,
    "number_of_runs": len(raw_results),

    "ttft_mean_ms": statistics.mean(
        ttft_values
    ),

    "ttft_median_ms": statistics.median(
        ttft_values
    ),

    "ttft_min_ms": min(ttft_values),

    "ttft_max_ms": max(ttft_values),

    "ttft_std_dev_ms": safe_stdev(
        ttft_values
    ),

    "latency_mean_ms": statistics.mean(
        latency_values
    ),

    "latency_median_ms": statistics.median(
        latency_values
    ),

    "latency_min_ms": min(
        latency_values
    ),

    "latency_max_ms": max(
        latency_values
    ),

    "latency_std_dev_ms": safe_stdev(
        latency_values
    ),

    "tokens_per_second_mean": (
        statistics.mean(
            token_speed_values
        )
    ),

    "tokens_per_second_median": (
        statistics.median(
            token_speed_values
        )
    ),

    "tokens_per_second_min": min(
        token_speed_values
    ),

    "tokens_per_second_max": max(
        token_speed_values
    ),

    "tokens_per_second_std_dev": (
        safe_stdev(
            token_speed_values
        )
    ),

    "ollama_memory_mean_mb": (
        statistics.mean(
            ollama_memory_values
        )
    ),

    "ollama_memory_max_mb": (
        max(
            ollama_memory_values
        )
    ),

    "python_memory_mean_mb": (
        statistics.mean(
            python_memory_values
        )
    ),

    "python_memory_max_mb": (
        max(
            python_memory_values
        )
    ),
   }

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    summary_filename = (
        "benchmark_summary.csv"
    )

    with open(
        summary_filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=statistics_data.keys(),
        )

        writer.writeheader()
        writer.writerow(statistics_data)

    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(
        f"TTFT mean: "
        f"{statistics_data['ttft_mean_ms']:.2f} ms"
    )

    print(
        f"TTFT median: "
        f"{statistics_data['ttft_median_ms']:.2f} ms"
    )

    print(
        f"Latency mean: "
        f"{statistics_data['latency_mean_ms']:.2f} ms"
    )

    print(
        f"Latency median: "
        f"{statistics_data['latency_median_ms']:.2f} ms"
    )

    print(
        f"Tokens/sec mean: "
        f"{statistics_data['tokens_per_second_mean']:.2f}"
    )

    print(
        f"Tokens/sec median: "
        f"{statistics_data['tokens_per_second_median']:.2f}"
    )

    print(
        f"Ollama/model maximum memory: "
        f"{statistics_data['ollama_memory_max_mb']:.2f} MB"
    )
    
    print(
        f"Python mean memory: "
        f"{statistics_data['python_memory_mean_mb']:.2f} MB"
    )
    
    print(
        f"Python maximum memory: "
        f"{statistics_data['python_memory_max_mb']:.2f} MB"
    )
    
    print()
    print(
        f"Raw results: {raw_filename}"
    )

    print(
        f"Summary: {summary_filename}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()