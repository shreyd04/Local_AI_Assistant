Local AI Assistant

An offline local-LLM assistant for low-latency inference, structured JSON generation, and reproducible benchmarking of local model behavior using Ollama.

1. Overview

This project is a local-first LLM experimentation and benchmarking workspace built around Ollama. It allows a developer to run a local model without a cloud dependency, measure generation behavior, validate structured output, and compare model/runtime characteristics under controlled conditions.

The repository focuses on a practical engineering question: how do you reliably run small language models locally while measuring latency, throughput, memory usage, and output validity? The project addresses that by combining:

a FastAPI backend for local inference
a lightweight frontend for interactive prompting
structured JSON generation with Pydantic validation
retry-on-validation-failure logic
benchmark scripts for runtime and memory measurement
automated quality-scoring workflows using a local LLM as an evaluator
The emphasis is on local/offline inference, privacy, reproducibility, and measurement rather than production deployment or cloud orchestration.

2. Key Features

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
3. System Architecture

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
4. Repository Structure

Text
Local_AI_Assistant/
├── app/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cold_warm_benchmark.py
│   ├── main.py
│   ├── model_benchmark.py
│   ├── mistral_q5_benchmark.py
│   ├── ollama_client.py
│   ├── phi_q5_benchmark.py
│   ├── quality_evaluator.py
│   ├── quantization_benchmark.py
│   ├── schemas.py
│   ├── structured_benchmark.py
│   ├── structured_output.py
│   ├── system_metrics.py
│   ├── temperature_benchmark.py
│   └── update_quality_scores.py
├── benchmarks/
│   ├── prompts.json
│   └── quality_rubric.json
├── frontend/
│   ├── README.md
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── tests/
│   └── test_structured_output.py
├── benchmark_results.csv
├── benchmark_results_raw.csv
├── benchmark_summary.csv
├── cold_warm_benchmark.csv
├── structured_results.csv
├── temperature_results.csv
├── requirements.txt
├── .gitignore
├── README.md
└── results/
Important files

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
5. Models Evaluated

The repository includes benchmark scripts and result files for the following model names that are directly confirmed in code and CSV outputs:

Model	Variant / Note	Confirmed in
mistral:7b-instruct-v0.3-q5_K_M	Q5 quantized Mistral 7B	app/ollama_client.py, frontend/index.html, benchmark results
llama3.2:3b	Llama 3.2 3B	cold_warm_benchmark.csv, benchmark_summary.csv, benchmark_results.csv
phi3.5 / Phi benchmark script	Q5-related benchmark script exists	app/phi_q5_benchmark.py
The repository is explicit about model-level benchmarking and quantization experiments, but it does not claim that any model is universally “best.” It documents performance observations rather than ranking models.

6. Benchmark Methodology

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

Q4 memory note

The repository includes Q5 quantization scripts and verified Q5 outputs, but no validated Q4 Mistral memory summary was found in the checked files. For that reason, this README does not use unverified Q4 Mistral memory values for quantitative comparison.

7. Quantitative Results

The repository contains actual CSV files that provide verified benchmark data.

Sample benchmark rows from benchmark_results.csv

prompt_id	ttft_ms	total_latency_ms	output_tokens	tokens_per_second
1	413.4662499418482	22589.642124949023	251	11.318452803347403
2	499.6465001022443	63405.58783302549	592	9.41087578464013
3	383.1620829878375	49594.605625025	601	12.212606595996656
4	387.7634999807924	42134.84874996357	531	12.719450874722304
5	317.02695903368294	30416.23641701881	391	12.990374399892096
These values are directly from the repository CSV and are not inferred.

Cold vs. warm measurement from cold_warm_benchmark.csv

Phase	run	ttft_ms	latency_ms	tokens	tokens_per_second	model_memory_mb
Cold	1	4182.542749913409	33228.12741599046	341	11.740166497604061	2358.640625
Warm	1	97.81224990729243	24646.449416992255	313	12.750198630972202	2358.390625
Warm	2	97.9731660336256	24191.763541079126	313	12.990899112502506	2359.265625
Warm	3	88.43891602009535	27758.42662504874	313	11.311895158445225	2355.71875
Warm	4	147.56958303041756	40324.58508305717	313	7.790523912852401	2354.0625
Warm	5	142.9639169946313	32579.142041970044	313	9.64971886620003	2354.4375
Warm	6	120.57579203974456	26940.324292052537	313	11.670504665614247	2355.09375
Warm	7	98.05558307562023	25470.435666036792	313	12.336249062034003	2355.15625
Warm	8	97.45500003919005	25957.180291996337	313	12.10376353446217	2355.03125
Warm	9	81.6492090234533	26353.887042030692	313	11.913716752623223	2355.28125
Warm	10	80.37933299783617	25480.349082965404	313	12.322849321519355	2355.59375
This is a good example of the cold-start penalty: the cold TTFT is about 42× larger than the warm minimum and more than 40× larger than the warm median range shown in the file.

Summary file notes

The repository includes benchmark_summary.csv, but the line in the tool output was truncated. Because the project instructions emphasize verifying numbers before publishing them, this README does not include exact summary values beyond the explicit values above. The benchmark scripts themselves are the authoritative source for how those aggregate metrics are generated.

8. Quality Evaluation

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

Limitations

The repository explicitly frames this as an automated evaluation workflow rather than a substitute for human judgment. Important limitations include:

model-generated evaluation introduces evaluator/model bias
the score reflects local LLM behavior and prompt phrasing
the benchmark set is limited
one model acting as judge can be biased toward similar outputs
results are useful for relative comparison, not absolute truth
9. API

The FastAPI app in app/main.py implements the endpoints below.

GET /

JSON
{
  "name": "Local AI Assistant",
  "status": "running",
  "model": "mistral:7b-instruct-v0.3-q5_K_M",
  "mode": "offline"
}
GET /health

JSON
{
  "status": "healthy",
  "model": "mistral:7b-instruct-v0.3-q5_K_M",
  "ollama": "localhost:11434"
}
POST /chat

Request body:

JSON
{
  "prompt": "Explain what an API is.",
  "temperature": 0.0
}
Response model:

JSON
{
  "response": "string",
  "model": "string",
  "temperature": 0.0,
  "ttft_ms": 0.0,
  "total_latency_ms": 0.0,
  "output_tokens": 0,
  "tokens_per_second": 0.0
}
POST /structured-chat

Request body:

JSON
{
  "prompt": "Explain what an API is.",
  "temperature": 0.0
}
Response model:

JSON
{
  "result": {
    "answer": "string",
    "key_points": ["string"],
    "category": "definition"
  },
  "model": "string",
  "attempts": 1
}
Example curl calls

bash
curl -X GET http://127.0.0.1:8000/
curl -X GET http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain what an API is.","temperature":0.0}'
curl -X POST http://127.0.0.1:8000/structured-chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain what an API is.","temperature":0.0}'
10. Structured Output

The project’s structured output flow is implemented in app/structured_output.py and validated by tests/test_structured_output.py.

How it works

A Pydantic model is defined in app/schemas.py
JSON schema is derived with model_json_schema()
The schema is embedded in the prompt sent to Ollama
The response is parsed with model_validate_json()
Validation errors trigger a retry
If validation still fails, the code raises a StructuredOutputError
Schema

Python
class StructuredAssistantResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    key_points: list[str] = Field(..., min_length=1, max_length=5)
    category: Literal[
        "definition",
        "explanation",
        "comparison",
        "problem_solving",
        "other",
    ]
Representative response

JSON
{
  "answer": "A database index improves lookup performance by reducing the amount of data scanned.",
  "key_points": [
    "Faster lookups",
    "Reduced scanning",
    "Query optimization"
  ],
  "category": "explanation"
}
Failure behavior

The code retries once by default and then raises:

Python
StructuredOutputError(
    "Structured output remained invalid after 2 attempt(s). Validation error: ..."
)
11. Frontend

The frontend is implemented under frontend/ and is a plain web app with no npm dependencies.

Features in the repo

chat interface
backend status indicator
temperature control
structured JSON toggle
metrics panel
model display
responsive layout
markdown rendering for assistant responses
Run locally

bash
python -m http.server 5500 --directory frontend
Then open:

Text
http://127.0.0.1:5500
The frontend calls the backend at:

Text
http://127.0.0.1:8000
12. Installation

Prerequisites

Python 3.9 or later
Ollama installed locally
Git
Clone

bash
git clone https://github.com/shreyd04/Local_AI_Assistant.git
cd Local_AI_Assistant
Create a virtual environment

bash
python3 -m venv .venv
source .venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
The project dependencies are exactly those in requirements.txt:

Text
fastapi
uvicorn[standard]
ollama
pydantic
psutil
httpx
pytest
Ollama prerequisite

Install Ollama using the official installer instructions at:

Text
https://ollama.ai
Then pull the default model used in the project:

bash
ollama pull mistral:7b-instruct-v0.3-q5_K_M
13. Running the Application

Start Ollama
bash
ollama serve
Start the backend
bash
source .venv/bin/activate
cd Local_AI_Assistant
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
Test the backend
bash
curl http://127.0.0.1:8000/health
Start the frontend
bash
python -m http.server 5500 --directory frontend
Open the browser
Text
http://127.0.0.1:5500
Try a prompt
Example prompts:

“Explain what an API is.”
“What is a database index and why is it useful?”
“Compare a process and a thread in operating systems.”
Toggle Structured JSON mode
14. Benchmark Reproduction

These commands are the ones implemented in the project by script names under app/:

Baseline benchmark

bash
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

bash
python -m app.cold_warm_benchmark
What it measures:

cold start after unloading the model
warm measurements after warm-up
TTFT, latency, throughput, memory
Output:

cold_warm_benchmark.csv
Temperature benchmark

bash
python -m app.temperature_benchmark
What it measures:

temperature = 0.0 and 0.7
prompt-by-prompt repeated runs
TTFT, latency, throughput, output token count
Output:

temperature_results.csv
Structured output benchmark

bash
python -m app.structured_benchmark
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

<img width="1440" height="825" alt="Screenshot 2026-09-30 at 5 19 02 PM" src="https://github.com/user-attachments/assets/02180f22-e690-4ead-8fe7-41eff4ee2b2e" />
Chat Interface

<img width="1440" height="858" alt="Screenshot 2026-09-30 at 5 27 49 PM" src="https://github.com/user-attachments/assets/b2f06ab9-1bf5-43c8-bfef-b70dd608fb92" />
Structured Output

<img width="252" height="818" alt="Screenshot 2026-09-30 at 8 33 22 PM" src="https://github.com/user-attachments/assets/de15a5a0-5069-4467-8e62-6f85226ad349" />
Benchmark Results

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
