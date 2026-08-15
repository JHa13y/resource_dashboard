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
- **Real-time updates** via Server-Sent Events (SSE) — no page refreshes needed

## Prerequisites

- Python 3.10+
- NVIDIA GPU drivers with `nvidia-smi` available (for GPU metrics; dashboard works without GPUs)

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
```

Then open **http://localhost:5000** (or your chosen host/port) in your browser.

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
| Charts not updating | Check browser console for SSE connection errors. The green "Live" dot in the top-right corner indicates a healthy connection. |
