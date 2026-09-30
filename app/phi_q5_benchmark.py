import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.ollama_client import generate_response
from app.system_metrics import (
    get_ollama_model_memory_mb,
    get_python_memory_mb,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPTS_FILE = PROJECT_ROOT / "benchmarks" / "prompts.json"
RESULTS_FILE = PROJECT_ROOT / "results" / "phi3_5_q5_raw.csv"
SUMMARY_FILE = PROJECT_ROOT / "results" / "llama3_2_3b_q5_summary.csv"

MODEL_NAME = "phi3.5:3.8b-mini-instruct-q5_K_M"

TEMPERATURE = 0.0
NUM_PREDICT = 256


def load_prompts():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("prompts", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Could not find prompt list in prompts.json")


def normalize_prompt(item):
    if isinstance(item, str):
        return {
            "prompt_id": "",
            "category": "",
            "prompt": item,
        }

    return {
        "prompt_id": item.get("id", item.get("prompt_id", "")),
        "category": item.get("category", ""),
        "prompt": item.get("prompt", item.get("text", "")),
    }


def main():
    raw_prompts = load_prompts()
    prompts = [normalize_prompt(p) for p in raw_prompts]

    print("=" * 70)
    print("Q5 QUANTIZATION BENCHMARK")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(prompts)}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max output tokens: {NUM_PREDICT}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------
    print("\nRunning warm-up...")
    warmup = generate_response(
        "Reply with exactly: warmup",
        temperature=TEMPERATURE,
        model=MODEL_NAME,
        num_predict=16,
    )

    print(
        f"Warm-up complete: "
        f"{warmup.output_tokens} tokens, "
        f"{warmup.tokens_per_second:.2f} tok/s"
    )

    # Memory after model has been loaded.
    warmup_model_memory = get_ollama_model_memory_mb()
    warmup_python_memory = get_python_memory_mb()

    print(
        f"Q5 model RSS after warm-up: "
        f"{warmup_model_memory:.2f} MB"
    )
    print(
        f"Python RSS after warm-up: "
        f"{warmup_python_memory:.2f} MB"
    )

    rows = []

    # ---------------------------------------------------------
    # 40-prompt benchmark
    # ---------------------------------------------------------
    for index, item in enumerate(prompts, start=1):

        prompt_id = item["prompt_id"]
        category = item["category"]
        prompt = item["prompt"]

        print("\n" + "-" * 70)
        print(f"Prompt {index}/{len(prompts)}")
        print(f"Category: {category}")
        print(f"Prompt ID: {prompt_id}")

        model_memory_before = get_ollama_model_memory_mb()
        python_memory_before = get_python_memory_mb()

        result = generate_response(
            prompt,
            temperature=TEMPERATURE,
            model=MODEL_NAME,
            num_predict=NUM_PREDICT,
        )

        model_memory_after = get_ollama_model_memory_mb()
        python_memory_after = get_python_memory_mb()

        # Keep the larger observed RSS value for this run.
        model_memory_mb = max(
            model_memory_before,
            model_memory_after,
        )

        python_memory_mb = max(
            python_memory_before,
            python_memory_after,
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_NAME,
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "ttft_ms": result.ttft_ms,
            "latency_ms": result.total_latency_ms,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.tokens_per_second,
            "python_memory_mb": python_memory_mb,
            "ollama_model_memory_mb": model_memory_mb,
            "output": result.response,
        }

        rows.append(row)

        print(f"TTFT: {result.ttft_ms:.2f} ms")
        print(f"Latency: {result.total_latency_ms:.2f} ms")
        print(f"Tokens: {result.output_tokens}")
        print(f"Tokens/sec: {result.tokens_per_second:.2f}")
        print(f"Model RSS: {model_memory_mb:.2f} MB")
        print(f"Python RSS: {python_memory_mb:.2f} MB")

    # ---------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Calculate summary
    # ---------------------------------------------------------
    ttft_values = [r["ttft_ms"] for r in rows]
    latency_values = [r["latency_ms"] for r in rows]
    token_speed_values = [r["tokens_per_second"] for r in rows]
    output_token_values = [r["output_tokens"] for r in rows]
    memory_values = [r["ollama_model_memory_mb"] for r in rows]
    python_memory_values = [r["python_memory_mb"] for r in rows]

    summary = {
        "model": MODEL_NAME,
        "runs": len(rows),

        "ttft_mean_ms": statistics.mean(ttft_values),
        "ttft_median_ms": statistics.median(ttft_values),

        "latency_mean_ms": statistics.mean(latency_values),
        "latency_median_ms": statistics.median(latency_values),

        "tokens_per_second_mean": statistics.mean(
            token_speed_values
        ),
        "tokens_per_second_median": statistics.median(
            token_speed_values
        ),

        "output_tokens_mean": statistics.mean(
            output_token_values
        ),

        "model_memory_mean_mb": statistics.mean(
            memory_values
        ),
        "model_memory_max_mb": max(
            memory_values
        ),

        "python_memory_mean_mb": statistics.mean(
            python_memory_values
        ),
        "python_memory_max_mb": max(
            python_memory_values
        ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys(),
        )
        writer.writeheader()
        writer.writerow(summary)

    # ---------------------------------------------------------
    # Print final summary
    # ---------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("Q5 QUANTIZATION BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"Runs: {len(rows)}")

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
        f"Tokens/sec median: "
        f"{summary['tokens_per_second_median']:.2f}"
    )

    print(
        f"Output tokens mean: "
        f"{summary['output_tokens_mean']:.2f}"
    )

    print(
        f"Model RSS mean: "
        f"{summary['model_memory_mean_mb']:.2f} MB"
    )
    print(
        f"Model RSS max: "
        f"{summary['model_memory_max_mb']:.2f} MB"
    )

    print(
        f"Python RSS mean: "
        f"{summary['python_memory_mean_mb']:.2f} MB"
    )
    print(
        f"Python RSS max: "
        f"{summary['python_memory_max_mb']:.2f} MB"
    )

    print("=" * 70)

    print(f"\nRaw results:")
    print(RESULTS_FILE)

    print(f"Summary:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()