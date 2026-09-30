# Local AI Assistant — Frontend

A lightweight, dependency-free frontend for the Local AI Assistant project.

## Requirements

- FastAPI backend running at `http://127.0.0.1:8000`
- Ollama running locally
- The backend exposing:
  - `GET /health`
  - `POST /chat`
  - `POST /structured-chat`

## Run locally

From the project root:

```bash
python -m http.server 5500 --directory frontend
```

Open:

```text
http://127.0.0.1:5500
```

The frontend calls the FastAPI backend at:

```text
http://127.0.0.1:8000
```

## Features

- ChatGPT-style conversational interface
- Backend connection indicator
- Temperature control
- Standard response mode
- Structured JSON mode
- Pydantic-validated structured response display
- TTFT, total latency, throughput and output-token metrics
- Responsive layout
- No frontend npm dependencies
