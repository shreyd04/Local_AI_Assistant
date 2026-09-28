import csv
import statistics
import subprocess
import time
from datetime import datetime

from app.ollama_client import MODEL_NAME, generate_response
from app.system_metrics import get_ollama_model_memory_mb


TEST_PROMPT = """
Explain what a database index is and why it improves query performance.
Give a concise technical explanation with one example.
""".strip()


WARM_RUNS = 10


def unload_model():
    """
    Ask Ollama to unload the model from memory.
    """

    subprocess.run(
        [
            "ollama",
            "stop",
            MODEL_NAME,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def wait_for_model_unloaded(timeout_seconds=30):
    """
    Wait until the llama-server process is no longer present.
    """

    start = time.perf_counter()

    while (
        time.perf_counter() - start
        < timeout_seconds
    ):

        memory = get_ollama_model_memory_mb()

        if memory == 0:
            return True

        time.sleep(0.5)

    return False


def run_single_inference():
    """
    Execute one inference and return measured metrics.
    """

    result = generate_response(
        prompt=TEST_PROMPT,
        temperature=0.0,
    )

    return {
        "ttft_ms": result.ttft_ms,
        "latency_ms": result.total_latency_ms,
        "tokens": result.output_tokens,
        "tokens_per_second": result.tokens_per_second,
        "memory_mb": get_ollama_model_memory_mb(),
    }


def main():

    print("=" * 60)
    print("COLD vs WARM LOCAL LLM BENCHMARK")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Warm measured runs: {WARM_RUNS}")

    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: Unload model
    # --------------------------------------------------

    print("\n[1/4] Unloading model...")

    unload_model()

    unloaded = wait_for_model_unloaded()

    if not unloaded:

        print(
            "WARNING: Model may still be loaded."
        )

    else:

        print(
            "Model successfully unloaded."
        )

    # --------------------------------------------------
    # STEP 2: Cold start
    # --------------------------------------------------

    print("\n[2/4] Running cold-start inference...")

    cold_start_time = time.perf_counter()

    cold_result = run_single_inference()

    cold_end_time = time.perf_counter()

    cold_wall_time_ms = (
        cold_end_time - cold_start_time
    ) * 1000

    print("\nCOLD START")
    print("-" * 40)

    print(
        f"TTFT: "
        f"{cold_result['ttft_ms']:.2f} ms"
    )

    print(
        f"Latency: "
        f"{cold_result['latency_ms']:.2f} ms"
    )

    print(
        f"Tokens: "
        f"{cold_result['tokens']}"
    )

    print(
        f"Tokens/sec: "
        f"{cold_result['tokens_per_second']:.2f}"
    )

    print(
        f"Ollama/model memory: "
        f"{cold_result['memory_mb']:.2f} MB"
    )

    print(
        f"Wall-clock time: "
        f"{cold_wall_time_ms:.2f} ms"
    )

    # --------------------------------------------------
    # STEP 3: Warm-up
    # --------------------------------------------------

    print("\n[3/4] Running warm-up inference...")

    warmup_result = run_single_inference()

    print(
        f"Warm-up TTFT: "
        f"{warmup_result['ttft_ms']:.2f} ms"
    )

    print(
        "Warm-up result will NOT be included "
        "in warm statistics."
    )

    # --------------------------------------------------
    # STEP 4: Warm measurements
    # --------------------------------------------------

    print(
        f"\n[4/4] Running {WARM_RUNS} "
        "warm measurements..."
    )

    warm_results = []

    for i in range(WARM_RUNS):

        print(
            f"\nWarm run "
            f"{i + 1}/{WARM_RUNS}"
        )

        result = run_single_inference()

        warm_results.append(result)

        print(
            f"TTFT: "
            f"{result['ttft_ms']:.2f} ms"
        )

        print(
            f"Latency: "
            f"{result['latency_ms']:.2f} ms"
        )

        print(
            f"Tokens: "
            f"{result['tokens']}"
        )

        print(
            f"Tokens/sec: "
            f"{result['tokens_per_second']:.2f}"
        )

        print(
            f"Ollama/model memory: "
            f"{result['memory_mb']:.2f} MB"
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    warm_ttft = [
        x["ttft_ms"]
        for x in warm_results
    ]

    warm_latency = [
        x["latency_ms"]
        for x in warm_results
    ]

    warm_tokens_per_second = [
        x["tokens_per_second"]
        for x in warm_results
    ]

    warm_memory = [
        x["memory_mb"]
        for x in warm_results
    ]

    print("\n")
    print("=" * 60)
    print("COLD vs WARM SUMMARY")
    print("=" * 60)

    print("\nCOLD START")

    print(
        f"TTFT: "
        f"{cold_result['ttft_ms']:.2f} ms"
    )

    print(
        f"Latency: "
        f"{cold_result['latency_ms']:.2f} ms"
    )

    print(
        f"Tokens/sec: "
        f"{cold_result['tokens_per_second']:.2f}"
    )

    print(
        f"Model memory: "
        f"{cold_result['memory_mb']:.2f} MB"
    )

    print("\nWARM START")

    print(
        f"TTFT mean: "
        f"{statistics.mean(warm_ttft):.2f} ms"
    )

    print(
        f"TTFT median: "
        f"{statistics.median(warm_ttft):.2f} ms"
    )

    print(
        f"Latency mean: "
        f"{statistics.mean(warm_latency):.2f} ms"
    )

    print(
        f"Latency median: "
        f"{statistics.median(warm_latency):.2f} ms"
    )

    print(
        f"Tokens/sec mean: "
        f"{statistics.mean(warm_tokens_per_second):.2f}"
    )

    print(
        f"Tokens/sec median: "
        f"{statistics.median(warm_tokens_per_second):.2f}"
    )

    print(
        f"Model memory mean: "
        f"{statistics.mean(warm_memory):.2f} MB"
    )

    print(
        f"Model memory max: "
        f"{max(warm_memory):.2f} MB"
    )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    print("\nCOMPARISON")

    print(
        f"Cold TTFT: "
        f"{cold_result['ttft_ms']:.2f} ms"
    )

    print(
        f"Warm TTFT mean: "
        f"{statistics.mean(warm_ttft):.2f} ms"
    )

    print(
        f"Cold latency: "
        f"{cold_result['latency_ms']:.2f} ms"
    )

    print(
        f"Warm latency mean: "
        f"{statistics.mean(warm_latency):.2f} ms"
    )

    print("=" * 60)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    filename = (
        "cold_warm_benchmark.csv"
    )

    with open(
        filename,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "model",
                "phase",
                "run",
                "ttft_ms",
                "latency_ms",
                "tokens",
                "tokens_per_second",
                "model_memory_mb",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "phase": "cold",
                "run": 1,
                "ttft_ms": cold_result["ttft_ms"],
                "latency_ms": cold_result["latency_ms"],
                "tokens": cold_result["tokens"],
                "tokens_per_second": cold_result[
                    "tokens_per_second"
                ],
                "model_memory_mb": cold_result[
                    "memory_mb"
                ],
            }
        )

        for i, result in enumerate(
            warm_results,
            start=1,
        ):

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": MODEL_NAME,
                    "phase": "warm",
                    "run": i,
                    "ttft_ms": result["ttft_ms"],
                    "latency_ms": result["latency_ms"],
                    "tokens": result["tokens"],
                    "tokens_per_second": result[
                        "tokens_per_second"
                    ],
                    "model_memory_mb": result[
                        "memory_mb"
                    ],
                }
            )

    print(
        f"\nResults saved to: {filename}"
    )


if __name__ == "__main__":
    main()