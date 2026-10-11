# Local AI Assistant

An offline local-LLM assistant for low-latency inference, structured JSON generation, and reproducible benchmarking of local model behavior using Ollama.

### 1. Overview

This project is a local-first LLM experimentation and benchmarking workspace built around Ollama. It allows a developer to run a local model without a cloud dependency, measure generation behavior, validate structured output, and compare model/runtime characteristics under controlled conditions.

The repository focuses on a practical engineering question: how do you reliably run small language models locally while measuring latency, throughput, memory usage, and output validity? The project addresses that by combining:

a FastAPI backend for local inference
a lightweight frontend for interactive prompting
structured JSON generation with Pydantic validation
retry-on-validation-failure logic
benchmark scripts for runtime and memory measurement
automated quality-scoring workflows using a local LLM as an evaluator
The emphasis is on local/offline inference, privacy, reproducibility, and measurement rather than production deployment or cloud orchestration.

### 2. Key Features

The following features are implemented in the repository and verified in source:

Local Ollama inference
FastAPI backend
Lightweight web frontend
Streaming generation for TTFT and latency measurement
TTFT (time to first token) capture
Total latency measurement
Tokens/sec calculation
Output token counts
Model memory and Python process memory measurement
Structured JSON generation
Pydantic schema validation
Retry-on-validation-failure
Graceful error handling for invalid structured output
Temperature benchmarking
Cold/warm model benchmarking
Multi-model benchmark script support
Automated quality scoring workflow
Test coverage for structured-output validation

### 3. System Architecture

Mermaid
graph TD
    A[Web Frontend<br/>frontend/index.html + app.js] --> B[FastAPI API<br/>app/main.py]
    B --> C[Ollama Client<br/>app/ollama_client.py]
    C --> D[Local Ollama Server<br/>localhost:11434]
    D --> E[Local LLM Model]
    B --> F[Structured Output Validation<br/>app/structured_output.py]
    B --> G[System Metrics<br/>app/system_metrics.py]
    F --> H[Pydantic Schema Validation]
    G --> I[Model RSS + Python RSS Memory]
    E --> J[Generated text + metrics]
    J --> B
    B --> A
Components

Frontend: a simple HTML/CSS/JS interface that sends prompts to the backend and displays metrics.
FastAPI backend: exposes the chat and structured-chat endpoints and validates incoming payloads.
Ollama client: handles model inference, streaming, and timing measurements.
Structured output module: builds a JSON schema from a Pydantic model, submits it to the model, validates the result, and retries once on failure.
System metrics: checks CPU/RSS memory usage for the current Python process and the running llama-server process.
Local model: the Ollama model served locally on the machine.

### 4. Repository Structure

<img width="676" height="540" alt="Screenshot 2026-10-11 at 11 24 55 AM" src="https://github.com/user-attachments/assets/0b4b3a5c-e9a3-4fe6-a7a6-497e5816c620" />
<img width="676" height="253" alt="Screenshot 2026-10-11 at 11 25 07 AM" src="https://github.com/user-attachments/assets/3ea9ccf1-9606-4ac7-bcd7-aa947422f736" />


### Important Files :

app/main.py: FastAPI app and actual HTTP endpoints
app/ollama_client.py: model invocation, streaming response handling, and timing calculations
app/structured_output.py: schema-driven structured generation and validation retry flow
app/system_metrics.py: memory measurement logic for Python and Ollama model memory
app/model_benchmark.py: standardized benchmark runner that loads prompts and writes raw + summary CSVs
app/temperature_benchmark.py: compares behavior across temperature values
app/cold_warm_benchmark.py: compares cold-start and warm-start performance
benchmarks/prompts.json: benchmark prompt set
benchmarks/quality_rubric.json: scoring rubric for automated quality evaluation
frontend/app.js: browser-side API client and metrics display
tests/test_structured_output.py: tests for structured output validation and retry logic


### 5. Models Evaluated

The repository includes benchmark scripts and result files for the following model names that are directly confirmed in code and CSV outputs:

<img width="1148" height="202" alt="Screenshot 2026-10-11 at 10 58 19 AM" src="https://github.com/user-attachments/assets/54f28923-7532-4fbf-b51f-28f7335fb0e7" />


The repository is explicit about model-level benchmarking and quantization experiments, but it does not claim that any model is universally “best.” It documents performance observations rather than ranking models.

### 6. Benchmark Methodology

The benchmark system is implemented primarily in:

app/benchmark.py
app/model_benchmark.py
app/cold_warm_benchmark.py
app/temperature_benchmark.py
app/structured_benchmark.py
app/quantization_benchmark.py
Prompt categories in benchmarks/prompts.json

The benchmark set includes these categories:

factual
reasoning
summarization
instruction_following
classification
information_extraction
coding
structured_output
Benchmark behavior

Within each benchmark script, the project measures:

TTFT (time to first token)
total latency
output tokens
tokens/sec
Python memory
Ollama model memory
cold vs warm behavior
temperature effect
prompt-by-prompt and aggregate statistics
The repository indicates the same prompt set and settings were reused within a benchmark run for comparison. It does not make universal claims across all hardware; benchmark output is hardware- and runtime-dependent.

### Q4 memory note
The repository includes Q5 quantization scripts and verified Q5 outputs, but no validated Q4 Mistral memory summary was found in the checked files. For that reason, this README does not use unverified Q4 Mistral memory values for quantitative comparison.

### 7. Quantitative Results

The repository contains actual CSV files that provide verified benchmark data.

Sample benchmark rows from benchmark_results.csv

<img width="715" height="219" alt="Screenshot 2026-10-11 at 11 00 19 AM" src="https://github.com/user-attachments/assets/6209a06d-f078-40bb-add8-4c43f785650e" />


These values are directly from the repository CSV and are not inferred.

Cold vs. warm measurement from cold_warm_benchmark.csv

<img width="1218" height="429" alt="Screenshot 2026-10-11 at 11 20 38 AM" src="https://github.com/user-attachments/assets/a355b354-d6f4-412d-881a-567ec767b2fe" />


This is a good example of the cold-start penalty: the cold TTFT is about 42× larger than the warm minimum and more than 40× larger than the warm median range shown in the file.

### Summary file notes

The repository includes benchmark_summary.csv, but the line in the tool output was truncated. Because the project instructions emphasize verifying numbers before publishing them, this README does not include exact summary values beyond the explicit values above. The benchmark scripts themselves are the authoritative source for how those aggregate metrics are generated.

### 8. Quality Evaluation

The project includes an automated quality-evaluation workflow:

app/quality_evaluator.py
app/quality_rubric.json
app/auto_quality_score.py
app/update_quality_scores.py
Scoring dimensions

The rubric in benchmarks/quality_rubric.json defines three score dimensions:

correctness
instruction_adherence
completeness
Each dimension is scored on a 0–2 scale:

0 = fails criterion or contains a material error
1 = partially satisfies criterion
2 = fully satisfies the criterion
Automated evaluation mechanism

The evaluator is designed to:

load benchmark results from results/*_raw.csv
match each result to a prompt in benchmarks/prompts.json
use a rubric from benchmarks/quality_rubric.json
calculate a quality score as the mean of the three dimensions
save evaluation and summary CSVs
This workflow is not presented as human ground truth. It is an automated evaluation process with its own biases and limits.

### Limitations

The repository explicitly frames this as an automated evaluation workflow rather than a substitute for human judgment. Important limitations include:

model-generated evaluation introduces evaluator/model bias
the score reflects local LLM behavior and prompt phrasing
the benchmark set is limited
one model acting as judge can be biased toward similar outputs
results are useful for relative comparison, not absolute truth

### 9. API

The FastAPI app in app/main.py implements the endpoints below.

### GET /

<img width="676" height="139" alt="Screenshot 2026-10-11 at 11 27 33 AM" src="https://github.com/user-attachments/assets/1bfe8405-21b1-4521-85f0-45068c5ae065" />


### GET /health

<img width="676" height="139" alt="Screenshot 2026-10-11 at 11 27 42 AM" src="https://github.com/user-attachments/assets/96ca6086-b0c5-4dd6-9f6e-b7e0f860a759" />


### POST /chat

Request body:

<img width="676" height="107" alt="Screenshot 2026-10-11 at 11 27 51 AM" src="https://github.com/user-attachments/assets/7eb85b2f-4e47-43ad-923a-320573e20919" />


Response model:

<img width="676" height="191" alt="Screenshot 2026-10-11 at 11 28 04 AM" src="https://github.com/user-attachments/assets/22c5ce84-82dd-4552-9aa9-f7f3499cfa87" />


### POST /structured-chat

Request body:

<img width="676" height="106" alt="Screenshot 2026-10-11 at 11 28 13 AM" src="https://github.com/user-attachments/assets/87ccd587-2941-421a-9f68-29fb028e2890" />


Response model:

<img width="676" height="219" alt="Screenshot 2026-10-11 at 11 28 19 AM" src="https://github.com/user-attachments/assets/3a3098e4-9b50-41f4-826d-10edafe3c78d" />


### Example curl calls

<img width="676" height="188" alt="Screenshot 2026-10-11 at 11 28 27 AM" src="https://github.com/user-attachments/assets/8e19c9e7-3ef7-442b-a082-b984ccc1f513" />


### 10. Structured Output

The project’s structured output flow is implemented in app/structured_output.py and validated by tests/test_structured_output.py.

#### How it works

A Pydantic model is defined in app/schemas.py
JSON schema is derived with model_json_schema()
The schema is embedded in the prompt sent to Ollama
The response is parsed with model_validate_json()
Validation errors trigger a retry
If validation still fails, the code raises a StructuredOutputError
Schema

Python

<img width="676" height="226" alt="Screenshot 2026-10-11 at 11 28 36 AM" src="https://github.com/user-attachments/assets/1af3f091-042a-4e0f-878f-bc5430139034" />


#### Representative response

### JSON
{
  "answer": "A database index improves lookup performance by reducing the amount of data scanned.",
  "key_points": [
    "Faster lookups",
    "Reduced scanning",
    "Query optimization"
  ],
  "category": "explanation"
}

#### Failure behavior

The code retries once by default and then raises:

#### Python

StructuredOutputError(
    "Structured output remained invalid after 2 attempt(s). Validation error: ..."
)


### 11. Frontend

The frontend is implemented under frontend/ and is a plain web app with no npm dependencies.

#### Features in the repo

chat interface
backend status indicator
temperature control
structured JSON toggle
metrics panel
model display
responsive layout
markdown rendering for assistant responses
Run locally

#### bash
python -m http.server 5500 --directory frontend

#### Then open:
#### Text
http://127.0.0.1:5500

#### The frontend calls the backend at:
#### Text
http://127.0.0.1:8000


### 12. Installation

#### Prerequisites

Python 3.9 or later
Ollama installed locally
Git
Clone

#### bash
git clone https://github.com/shreyd04/Local_AI_Assistant.git
cd Local_AI_Assistant

#### Create a virtual environment

#### bash
python3 -m venv .venv
source .venv/bin/activate


#### Install dependencies

#### bash
pip install -r requirements.txt

#### The project dependencies are exactly those in requirements.txt:

#### Text
fastapi
uvicorn[standard]
ollama
pydantic
psutil
httpx
pytest

#### Ollama prerequisite

Install Ollama using the official installer instructions at:

#### Text
https://ollama.ai

#### Then pull the default model used in the project:

bash
ollama pull mistral:7b-instruct-v0.3-q5_K_M


## 13. Running the Application

Start Ollama

#### bash
ollama serve
Start the backend

#### bash
source .venv/bin/activate

cd Local_AI_Assistant

#### python 

-m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

#### Test the backend

#### bash

curl http://127.0.0.1:8000/health

#### Start the frontend

#### bash

#### python

-m http.server 5500 --directory frontend

#### Text

http://127.0.0.1:5500

#### Try a prompt

Example prompts:

“Explain what an API is.”
“What is a database index and why is it useful?”
“Compare a process and a thread in operating systems.”

#### Toggle Structured JSON mode


## 14. Benchmark Reproduction

These commands are the ones implemented in the project by script names under app/:

Baseline benchmark

#### bash
python -m app.benchmark

What it measures:

TTFT
total latency
output token count
tokens/sec
Python and Ollama memory usage
Output:

benchmark_results_raw.csv
benchmark_summary.csv
Cold/warm benchmark

#### bash

python 

-m app.cold_warm_benchmark

What it measures:

cold start after unloading the model
warm measurements after warm-up
TTFT, latency, throughput, memory

Output:

cold_warm_benchmark.csv
Temperature benchmark

bash

python

-m app.temperature_benchmark


What it measures:

temperature = 0.0 and 0.7
prompt-by-prompt repeated runs
TTFT, latency, throughput, output token count


Output:

temperature_results.csv
Structured output benchmark

bash

python 
-m app.structured_benchmark

What it measures:

success/failure on structured-output prompts
number of attempts
validation retry behavior

Output:

structured_results.csv
Model benchmark

bash
python -m app.model_benchmark --model mistral:7b-instruct-v0.3-q5_K_M --repetitions 1 --num-predict 256
What it measures:

multi-model benchmark runs
standardized prompt workload
TTFT, latency, tokens/sec, output tokens, model memory
Output:

results/<model>_raw.csv
results/<model>_summary.csv
results/<model>_category_summary.csv
15. Testing

The repository includes a test for structured-output reliability:

bash
pytest tests/
Test file

tests/test_structured_output.py verifies:

valid output is accepted on the first attempt
invalid output triggers a retry
retry failure raises StructuredOutputError
the final failure path is handled gracefully
16. Engineering Challenges and Solutions

Challenge: structured output validation failures

Problem → Investigation → Solution → Engineering lesson

Problem: local models sometimes return malformed or out-of-schema JSON.
Investigation: the project checks the model output as Pydantic JSON and identifies validation failures.
Solution: app/structured_output.py adds a retry path with explicit instructions to return valid JSON only.
Engineering lesson: validation-aware prompting plus bounded retry is a practical pattern for local LLMs.
Challenge: cold start overhead

Problem → Investigation → Solution → Engineering lesson

Problem: cold model startup is much slower than warm inference.
Investigation: cold_warm_benchmark.py unloads the model and measures the difference between cold and warm runs.
Solution: the script polls process memory and waits for the local llama-server process to unload.
Engineering lesson: model loading costs add real latency and should be explicitly measured.
Challenge: model memory measurement

Problem → Investigation → Solution → Engineering lesson

Problem: RSS memory must be measured outside the model runtime itself.
Investigation: app/system_metrics.py inspects ps output to find llama-server and sums RSS.
Solution: the code uses process-level memory measurement and guards against missing or inaccessible processes.
Engineering lesson: system-level metrics are important, but they require careful platform-specific handling.
Challenge: frontend/backend connectivity

Problem → Investigation → Solution → Engineering lesson

Problem: the frontend and backend are on different local ports and may fail due to CORS or backend health issues.
Investigation: app/main.py includes CORS middleware and the frontend checks backend health.
Solution: the app exposes a local health endpoint and the browser displays backend status.
Engineering lesson: local tools need clear connection diagnostics.
17. Performance Trade-offs

The repository shows several real trade-offs without claiming a universal winner:

Smaller models can require less memory but may produce shorter or more variable outputs depending on prompt and system conditions.
Quantized models reduce memory usage but depend on the local model runtime and hardware environment.
Temperature changes output variance; higher temperature tends to increase variety, while lower values make output more deterministic.
TTFT and total latency are not equivalent. TTFT reflects the time to first token; total latency reflects the full generation time.
More complex prompts can produce longer outputs, which lengthen total latency.
Benchmark numbers strongly depend on the hardware, system load, and local model runtime.
18. Limitations

The repository is honest about the constraints of the project:

local hardware limits strongly affect latency, memory, and throughput
automated LLM-based evaluation is not human ground truth
benchmark datasets are limited
benchmark numbers are runtime- and hardware-dependent
structure validation can fail on some prompts
some scripts use the same local LLM as both subject and judge, which introduces evaluation bias
memory measurement is process-based and is sensitive to OS behavior
19. Future Improvements

These are clearly labeled as future work and are not presented as existing features:

more robust model discovery
configurable generation limits
larger benchmark suites
human evaluation
additional hardware tests
richer observability
persistent chat history
authentication if deployment is required
containerization
streaming token display in the frontend
20. Screenshots

Chat Interface

<img width="1440" height="825" alt="Screenshot 2026-09-30 at 5 19 02 PM" src="https://github.com/user-attachments/assets/02180f22-e690-4ead-8fe7-41eff4ee2b2e" />

Structured Output

<img width="1440" height="825" alt="Screenshot 2026-09-30 at 5 22 16 PM" src="https://github.com/user-attachments/assets/f2c668da-c127-4ee9-9987-91f6bfd3d894" />


Benchmark Results

<img width="252" height="818" alt="Screenshot 2026-09-30 at 8 33 22 PM" src="https://github.com/user-attachments/assets/de15a5a0-5069-4467-8e62-6f85226ad349" />

21. Demo

<!-- TODO: Add screen-recorded demo link -->
22. Technical Highlights for Recruiters

Backend

Python
FastAPI
Pydantic
Local process inspection
HTTP API design
LLM / inference

Ollama
local LLM inference
streaming generation
generation controls
structured output validation
TTFT and latency measurement
Evaluation

benchmarking
latency analysis
memory profiling
quantization experiments
automated quality evaluation
Engineering

validation
retry handling
error handling
testing
reproducibility
working with real system constraints
23. Technologies Used

Technology	Role
Python	Backend implementation
FastAPI	API framework
Uvicorn	ASGI server
Ollama	Local model runtime
Pydantic	Request/response validation and schema generation
psutil	Process-level memory measurement
httpx	HTTP testing/client support
pytest	Testing
HTML	Frontend structure
CSS	Frontend styling
JavaScript	Frontend logic
CSV	Benchmark data storage
JSON	Configuration and evaluation data
24. License

License: Not specified.

25. Author / Links

GitHub: GitHub profile
LinkedIn: LinkedIn profile
Portfolio: Portfolio
26. Summary

Local AI Assistant is a compact, practical engineering project focused on local inference, benchmarking, and structured-output reliability. It demonstrates hands-on work with Python, FastAPI, Ollama, Pydantic, benchmarking scripts, and local system metrics while staying grounded in truly local execution and reproducible data collection.

This is not a generic “AI app.” It is a small but credible research-and-build project that interrogates the real constraints of local inference: latency, memory pressure, model reliability, temperature effects, structured output quality, and benchmark reproducibility.

If you want, I can also provide:

a shorter recruiter-friendly version
a more technical version for ML/AI engineering audiences
a version optimized for GitHub with tighter markdown formatting and less narrative text
