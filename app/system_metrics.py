import os
import subprocess

import psutil


def get_python_memory_mb() -> float:
    """Return RSS memory of the current Python process."""
    process = psutil.Process(os.getpid())

    return process.memory_info().rss / (1024 * 1024)


def get_ollama_model_memory_mb() -> float:
    """
    Return RSS memory of Ollama's llama-server process.
    """

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,rss=,command="],
            capture_output=True,
            text=True,
            check=True,
        )

        total_rss_kb = 0

        for line in result.stdout.splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split(None, 2)

            if len(parts) < 3:
                continue

            pid_text, rss_text, command = parts

            if "llama-server" not in command:
                continue

            try:
                pid = int(pid_text)
                rss_kb = int(rss_text)

                # Make sure the process still exists.
                psutil.Process(pid)

                total_rss_kb += rss_kb

            except (
                ValueError,
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                continue

        return total_rss_kb / 1024

    except (
        subprocess.SubprocessError,
        OSError,
    ):
        return 0.0