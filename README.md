# Resource & GPU Utilization Dashboard

A lightweight, real-time web dashboard for monitoring CPU load, memory usage, and NVIDIA GPU metrics (utilization, memory, temperature, clock speed, thermal throttling).

## Features

- **CPU**: Overall and per-core utilization with live trend chart
- **Memory**: Usage percentage, used/total GB with live trend chart
- **GPU** (per card):
  - Utilization % and Memory % gauges + trend charts
  - Temperature in °C
  - Current graphics clock frequency (MHz)
  - Power draw (W), with the enforced power limit in the label when available (`nvmlDeviceGetPowerUsage` / `nvidia-smi power.draw`)
  - Thermal throttle indicators (`hw_thermal_slowdown`, `sw_thermal_slowdown`) — pulsing red when active
- **Local LLM (llama.cpp / vLLM)** — generation & prompt tokens/s, requests in flight, KV cache / context, latency (TTFT/ITL/E2E for vLLM), spec-decode accept, prefix-cache hit, loaded model name. Auto-detects the backend (vLLM wins on tie); pin with `--llm-backend`
- **Real-time updates** via Server-Sent Events (SSE) — no page refreshes needed

## Prerequisites

- Python 3.10+
- NVIDIA GPU drivers with `nvidia-smi` available (for GPU metrics; dashboard works without GPUs)
- (Optional) llama.cpp `llama-server` started with `--metrics` for the Local LLM section; `/props` and `/slots` are served by default

## Installation

```bash
cd resource_dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: pynvml (recommended for GPU metrics)

The `nvidia-ml-py` package is included in `requirements.txt`. If installation fails due to missing NVIDIA development headers, the dashboard will automatically fall back to parsing `nvidia-smi` CLI output.

## Running

```bash
python app.py                          # binds 0.0.0.0:5000
python app.py --host 127.0.0.1 --port 8080
python app.py --llama-url http://192.168.1.50:5001   # point at your llama-server
python app.py --vllm-url http://192.168.1.50:9998   # point at your vLLM server
python app.py --llm-backend vllm                     # pin to vLLM (skip auto-detect)
python app.py --no-llama                             # hide the LLM section entirely
```

Then open **http://localhost:5000** (or your chosen host/port) in your browser.

### Local LLM section (llama.cpp / vLLM)

The dashboard auto-detects which LLM backend is running. In `auto` mode (default), it probes vLLM first (`--vllm-url`, default `http://127.0.0.1:9998`), then llama.cpp (`--llama-url`, default `http://127.0.0.1:5001`). If both are up, vLLM wins. Use `--llm-backend llama` or `--llm-backend vllm` to pin.

**llama.cpp** (when `llama-server` is reachable with `--metrics`):
- Gen/prompt tokens/s from per-slot live decode counters in `GET /slots`
- The prompt rate counts **cached + uncached** prompt tokens; the gauge label shows the last prompt's size and cache-hit percentage
- Per-slot context usage bars, peak ctx, spec-decode accept rate
- The model name and slot count come from `GET /props` (cached 30 s)

**vLLM** (when `/metrics` responds with `vllm:*` metrics):
- Gen/prompt tokens/s from Prometheus counter deltas (rate window 0.5–5 s)
- KV cache % (per-engine max), Running/Waiting (N/M), E2E latency (lifetime mean)
- TTFT / ITL (windowed mean, omitted when idle), prefix-cache hit %, spec-decode accept
- Config: KV cache tokens, num GPU blocks, block size, cache dtype, GPU memory utilization
- Stale-hold: last data held for 15 s after a failed scrape

The card shows **unreachable** when the URL is wrong, unroutable, or the server is not running. If a scrape fails, the last data is held for up to 15 s and shown as **stale**. Pass `--no-llama` to disable the section entirely (no scraping, card hidden); it is enabled by default.

### One-shot API (for testing / scripting)

```bash
curl http://127.0.0.1:5000/api/metrics | python -m json.tool
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Architecture

- **Backend**: Flask with a background `MetricsCollector` thread polling every 1 second
- **Streaming**: Server-Sent Events (SSE) — native browser `EventSource`, auto-reconnects on disconnect
- **Frontend**: Single HTML page with Chart.js for live-updating line charts and CSS gauges
- **GPU monitoring**: `pynvml` (NVIDIA Management Library Python bindings) with automatic fallback to `nvidia-smi --query-gpu=... --format=csv` parsing

## Project Structure

```
resource_dashboard/
├── app.py                  # Flask app + metrics collector + GPU monitors
├── templates/
│   └── index.html          # Dashboard UI with Chart.js
├── tests/
│   └── test_metrics.py     # Unit tests for metrics collectors
├── requirements.txt        # Python dependencies
└── README.md
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `nvidia-ml-py` install fails | Dashboard falls back to CLI parsing automatically. To fix: install NVIDIA driver development headers (`nvidia-driver-dev` on Ubuntu). |
| No GPU data shown | Verify `nvidia-smi` works in terminal. If it errors, check NVIDIA drivers are installed and running. |
| LLM card shows "unreachable" | Check `--llama-url` / `--vllm-url` is reachable from the dashboard host (`curl <url>/metrics`), that the server is running with metrics enabled, and that any firewall allows the port. |
| Charts not updating | Check browser console for SSE connection errors. The green "Live" dot in the top-right corner indicates a healthy connection. |
