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

### JetPack / embedded (Orin, Thor)

The dashboard auto-detects JetPack boards and switches to a sysfs-based GPU collector, because NVML is unsupported on Orin (mostly "Not Supported" on Thor) and the Orin `nvidia-smi` stub reports plausible-looking junk (0 % utilization, N/A memory).

**Detection** (checked once at first GPU collection; no configuration needed), in order:

1. Device-tree `model` string contains both "NVIDIA" and one of `jetson` / `orin` / `thor` (case-insensitive — the NVIDIA co-occurrence keeps custom aarch64 boards from matching).
2. Device-tree `compatible` string contains `nvidia,tegra` (the stable SoC discriminator; covers vendor boards and model-string gaps).
3. `/etc/nv_tegra_release` first line matches `R\d+`.
4. The kernel release string contains `tegra` (visible inside containers where the device-tree is hidden).

**Where each field comes from on JetPack:**

| Field | Source |
|-------|--------|
| Utilization % | `/sys/class/devfreq/<nvgpu>/device/load` (Orin and any SoC with an nvgpu sysfs); on Thor (no `load` file) from `nvidia-smi --query-gpu=utilization.gpu` |
| Memory (unified) | `/proc/meminfo` — `MemTotal` / `MemAvailable` of the single DRAM pool shared by CPU, OS and GPU |
| Temperature | `/sys/class/thermal/thermal_zone*/temp` of the zone whose `type` is `gpu-thermal` (matched by type, never by index; `−256000` "no sensor" readings are skipped) |
| Graphics clock | devfreq node `cur_freq` (Orin: e.g. `17000000.gpu`; Thor: `gpu-gpc-0`) |
| Name | SoC (`NVIDIA Thor` / `NVIDIA Orin`), falling back to the device-tree model string, then `NVIDIA Jetson` |

**Intentionally absent on JetPack** (the UI renders `–`): power draw / power limit, thermal-throttle badges, PCIe bus id, board vendor. There is no reliable root-free source for rail power without a persistent `tegrastats` process (deferred to a future version); NVML power is NotSupported on Jetson.

**Unified memory semantics.** JetPack has one memory pool: the GPU card's memory gauge (labeled **"Unified mem %"**, tooltip "Shared system memory (CPU+OS+GPU)") shows `MemTotal`/`MemAvailable` of that pool, so it overlaps with the system memory card. Model weights and the vLLM KV cache are CUDA allocations in this pool — starting/stopping a vLLM server moves the gauge by roughly (weights + preallocated KV cache). The pool therefore looks "partly full" even with no GPU load; that is the OS working set, not the GPU.

**Behavior notes:**

- The nvgpu devfreq node appears a few seconds after boot. Until it does, utilization/clock show `–` (never a false 0) and start filling in automatically — no restart needed.
- On Thor the utilization read shells out to `nvidia-smi` with a 2 s timeout. The collector runs sequentially, so a wedged `nvidia-smi` can stall the whole snapshot (CPU/memory/LLM included) by up to 2 s per tick; the SSE queue keeps serving the last good payload in the meantime. Same failure class as the existing desktop `nvidia-smi` path.
- Containers (e.g. CUDA images): the device-tree is usually hidden, so detection falls back to `nv_tegra_release` / the `-tegra` kernel string, and the devfreq/thermal sysfs may be absent unless the host's `/sys` is mounted. The card then degrades to whatever fields it can source (often just the unified-memory bar) rather than failing.

**On-device smoke script** — run on the JetPack host to field-verify what the dashboard will see:

```bash
#!/bin/sh
# jetpack_gpu_smoke.sh — print the exact inputs the dashboard parses
echo "--- /proc/device-tree/model:"
cat /proc/device-tree/model 2>/dev/null | tr '\0' ' '; echo
echo "--- /proc/device-tree/compatible:"
cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' '; echo
echo "--- /etc/nv_tegra_release:"
head -1 /etc/nv_tegra_release 2>/dev/null || echo "(absent)"
echo "--- devfreq nodes:"
ls /sys/class/devfreq/ 2>/dev/null || echo "(absent)"
for d in /sys/class/devfreq/*gpu*; do
    [ -d "$d" ] || continue
    echo "  $d: cur_freq=$(cat $d/cur_freq 2>/dev/null) load=$(cat $d/device/load 2>/dev/null || echo absent)"
done
echo "--- gpu-thermal zones:"
for z in /sys/class/thermal/thermal_zone*; do
    [ -f "$z/type" ] || continue
    [ "$(cat $z/type)" = "gpu-thermal" ] && echo "  $z: $(cat $z/temp) mC"
done
echo "--- nvidia-smi utilization:"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi unavailable)"
echo "--- meminfo (unified pool):"
grep -E 'MemTotal|MemAvailable' /proc/meminfo
```

## Running

```bash
python app.py                          # binds 0.0.0.0:5000
python app.py --host 127.0.0.1 --port 8080
python app.py --llama-url http://192.168.1.50:5001   # point at your llama-server
python app.py --vllm-url http://192.168.1.50:8000   # point at your vLLM server
python app.py --llm-backend vllm                     # pin to vLLM (skip auto-detect)
python app.py --no-llama                             # hide the LLM section entirely
```

Then open **http://localhost:5000** (or your chosen host/port) in your browser.

### Local LLM section (llama.cpp / vLLM)

The dashboard auto-detects which LLM backend is running. In `auto` mode (default), it probes vLLM first (`--vllm-url`, default `http://127.0.0.1:8000`), then llama.cpp (`--llama-url`, default `http://127.0.0.1:5001`). If both are up, vLLM wins (a one-time log warning tells you how to pin the other). Use `--llm-backend llama` or `--llm-backend vllm` to pin.

**llama.cpp** (when `llama-server` is reachable with `--metrics`):
- Gen/prompt tokens/s from per-slot live decode counters in `GET /slots`
- Prompt tokens/s gauge: mean over the last 5 completed requests of **uncached** prompt tokens / prefill duration — the same speed llama-server logs ("prompt eval time"); cache hits do no prefill work, so fully-cached requests don't skew it. Falls back to the per-scrape live rate until the first request completes. The gauge label shows the last prompt's size and cache-hit percentage
- Per-slot context usage bars, peak ctx, spec-decode accept rate
- The model name and slot count come from `GET /props` (cached 30 s)

**vLLM** (when `/metrics` responds with `vllm:*` metrics):
- Gen/prompt tokens/s from Prometheus counter deltas (rate window 0.5–5 s); the prompt gauge shows the mean over the last 5 completed-request windows (Δuncached prompt tokens / ΣTTFT) instead of the raw delta
- KV cache % (per-engine max), Running/Waiting (N/M), E2E latency (lifetime mean)
- TTFT / ITL (windowed mean, omitted when idle), prefix-cache hit %, spec-decode accept
- Config: KV cache tokens, num GPU blocks, block size, cache dtype, GPU memory utilization
- Stale-hold: last data held for 15 s after a failed scrape
- vLLM `uptime_s` is `time.time()` minus `process_start_time_seconds` — a cross-host clock comparison, so it assumes roughly NTP-synchronized clocks (clamped to 0); the llama.cpp state-file uptime is single-host and immune

The card shows **unreachable** when the URL is wrong, unroutable, or the server is not running. If a scrape fails, the last data is held for up to 15 s and shown as **stale**. Pass `--no-llama` to disable the section entirely (no scraping, card hidden); it is enabled by default.

### One-shot API (for testing / scripting)

```bash
curl http://127.0.0.1:5000/api/metrics | python -m json.tool
```

**Breaking change:** the per-snapshot key `llama` was replaced by `llm` (with a `backend: "llama" | "vllm"` discriminator) and `llama_enabled` by `llm_enabled`; the older output shape is gone.

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
| No GPU data shown | Verify `nvidia-smi` works in terminal. If it errors, check NVIDIA drivers are installed and running. On JetPack boards, run the on-device smoke script in the JetPack section to check the sysfs inputs. |
| JetPack GPU shows `–` for utilization | The nvgpu devfreq node appears seconds after boot and fills in automatically; on Thor, verify `nvidia-smi --query-gpu=utilization.gpu` returns a number (not `N/A`). |
| LLM card shows "unreachable" | Check `--llama-url` / `--vllm-url` is reachable from the dashboard host (`curl <url>/metrics`), that the server is running with metrics enabled, and that any firewall allows the port. |
| Charts not updating | Check browser console for SSE connection errors. The green "Live" dot in the top-right corner indicates a healthy connection. |
