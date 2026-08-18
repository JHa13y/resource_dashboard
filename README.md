# Resource & GPU Utilization Dashboard

A lightweight, real-time web dashboard for monitoring CPU load, memory usage, and NVIDIA GPU metrics (utilization, memory, temperature, clock speed, thermal throttling).

## Features

- **CPU**: Overall and per-core utilization with live trend chart
- **Memory**: Usage percentage, used/total GB with live trend chart
- **GPU** (per card):
  - Utilization % and Memory % gauges + trend charts
  - Temperature in °C
  - Current graphics clock frequency (MHz)
  - Thermal throttle indicators (`hw_thermal_slowdown`, `sw_thermal_slowdown`) — pulsing red when active
- **llama.cpp / local LLM** (when a `llama-server` is reachable): generation & prompt tokens/s, request queue (processing/deferred), context high-water mark, loaded model name, and per-slot context usage bars
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
```

Then open **http://localhost:5000** (or your chosen host/port) in your browser.

### llama.cpp section

`--llama-url` (default `http://127.0.0.1:5001`) is the base URL of your `llama-server`. The default is a local-dev convenience only — in production the dashboard usually runs on a **different host** than the llama-server, so pass the remote URL explicitly (e.g. `--llama-url http://<llama-host>:5001`). The card shows **unreachable** when the URL is wrong, unroutable, or the server was not started with `--metrics`. If a scrape fails (e.g. a timeout under heavy server load), the last data is held for up to 15 s and shown as **stale**; the card only turns "unreachable" after that.

Notes:
- Gen/prompt tokens/s are computed by the dashboard from the per-slot live decode counters in `GET /slots` (the `/metrics` counters only advance on request completion on some builds, e.g. LM Studio's); token totals are cumulative since server start.
- The prompt rate counts **cached + uncached** prompt tokens (matches the server's own prompt t/s and the CLI, which include KV-cache hits); the gauge label shows the last prompt's size and cache-hit percentage (e.g. `… · last 74.0k tok · 99.8% cached`), falling back to the `prompt_tokens_cached_total` rate on servers without `/slots` data.
- The model name and slot count come from `GET /props` (cached 30 s); per-slot context usage (`used / n_ctx`) comes from `GET /slots` and updates every second.

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
| LLM card shows "unreachable" | Check `--llama-url` is reachable from the dashboard host (`curl <url>/metrics`), that `llama-server` was started with `--metrics`, and that any firewall allows the port. |
| Charts not updating | Check browser console for SSE connection errors. The green "Live" dot in the top-right corner indicates a healthy connection. |
