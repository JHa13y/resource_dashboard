# Resource & GPU Utilization Dashboard

A lightweight, real-time web dashboard for monitoring CPU load, memory usage, and GPU metrics (utilization, memory, temperature, clock speed, power) for NVIDIA, AMD, and JetPack boards.

## Features

- **CPU**: Overall and per-core utilization with live trend chart
- **Memory**: Usage percentage, used/total GB with live trend chart
- **GPU** (per card): Utilization % and Memory % gauges + trend charts, temperature, clock, power draw, thermal-throttle badges. Sources: **NVIDIA** via `pynvml` → `nvidia-smi`; **AMD** via `amd-smi` → `rocm-smi` → sysfs (see below); **JetPack** via sysfs. AMD throttle badges are present but always report *no throttle* (the tools don't expose a comparable signal); AMD cards show no clock/power on the sysfs-only path.
- **Local LLM (llama.cpp / vLLM)** — generation & prompt tokens/s, requests in flight, KV cache / context, latency (TTFT/ITL/E2E for vLLM), spec-decode accept, prefix-cache hit, loaded model name. Auto-detects the backend (vLLM wins on tie); pin with `--llm-backend`
- **Real-time updates** via Server-Sent Events (SSE) — no page refreshes needed

## Prerequisites

- Python 3.10+
- For GPU metrics: NVIDIA drivers with `nvidia-smi`, and/or AMD ROCm (`amd-smi` ships with ROCm 6.x+; `rocm-smi` and plain sysfs also work). The dashboard runs fine with no GPUs.
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
python app.py --litellm-url http://192.168.1.50:4000  # point at your LiteLLM proxy
python app.py --llm-backend vllm                     # pin to vLLM (skip auto-detect)
python app.py --llm-backend litellm                  # pin to LiteLLM (skip auto-detect)
python app.py --no-llama                             # hide the LLM section entirely
python app.py --upstream athena=http://192.168.1.50:5001   # merge another dashboard
python app.py --config /path/to/config.yaml          # explicit config file
```

Then open **http://localhost:5000** (or your chosen host/port) in your browser.

### Configuration (config.yaml)

All CLI options can also be set in a YAML config file. A `config.yaml` in the
working directory is **auto-detected**; pass `--config PATH` for an explicit
file. Precedence: **CLI flags > config.yaml > built-in defaults**.

```yaml
# see config.example.yaml for the full annotated example
port: 5001
poll_interval: 1.0
llm_backend: litellm            # auto | llama | vllm | litellm
litellm_url: http://127.0.0.1:4000
litellm_token: sk-...
llama_enabled: true            # false == --no-llama (the flag wins when passed)
upstreams:
  - name: athena
    url: http://athena.lungfish-smelt.ts.net:5001
  - name: modelbox
    url: http://192.168.5.207:5001
```

Unknown keys are warned about and ignored; invalid values (wrong type, unknown
backend, non-positive interval) exit with an error. An explicitly passed
`--config PATH` that doesn't exist exits with an error; the auto-detected
`config.yaml` is simply skipped when absent.

### Multi-system consolidation

One dashboard can aggregate several others into a single view.

**Relay model**: every machine runs the same app (collecting its own local
metrics — local GPUs, LLM, etc.), and the main dashboard fetches
`GET /api/metrics` from each upstream instance every second and merges them.
No new software, no SSH, all collection stays local where the data is.

```bash
# on the main machine (config.yaml or CLI):
python app.py \
  --upstream athena=http://athena.lungfish-smelt.ts.net:5001 \
  --upstream modelbox=http://192.168.5.207:5001
```

The UI adds a tab row: **Fleet** (default) + one tab per system, each with a
status dot (green up / yellow stale / red down). The Fleet view shows a
summary strip (per-system CPU / mem / GPU / LLM tokens/s) and a fleet
**tokens/s chart**: total = filled area, each system = non-filled line over
it. Per-system tabs show the full detail view (GPU cards, LLM section, CPU/mem
charts) fed from that system's own data; trend buffers are kept per system so switching
tabs restores history instead of resetting it.

Backend details:
- One daemon thread per upstream (independent 1 s cadence; a hung upstream
  doesn't delay the others; 2 s HTTP timeout)
- Per-upstream stale-hold: last good metrics held for 15 s, then the system
  shows as **down** with null metrics
- Upstream payloads are sanitized to a strict key set (an upstream's own
  `systems` array is never nested; unknown app-level keys are dropped)
- The snapshot gains a `systems` array: `[{name: "local", status, metrics},
  ...]` — the local entry is first and its metrics match the top-level keys
  exactly. All existing top-level keys are unchanged (older UIs keep working)
- Upstream CLI entries (`--upstream name=url`) are merged with config entries
  by name: CLI wins on duplicates

Limitations (v1): upstream auth is not supported (trusted-network assumption —
put the upstream instances on a private network); CPU/mem/GPU trend charts reset on tab
switch (per-system buffers for those are a v2 candidate); version skew —
upstreams should run the same or an older build (older builds without
`systems` still work, they just appear as a single local entry; older
single-threaded builds need the Upstream deployment note below).

**Upstream deployment note:** upstream instances must run a build with the
threaded web server (this build or newer). Older single-threaded builds
queue or return `503` for `/api/metrics` whenever a browser holds the
long-lived SSE `/events` stream, so an aggregating dashboard will show
such an upstream as
**stale** (last values held for 15 s) or **down** until it is redeployed.

### Local LLM section (llama.cpp / vLLM / LiteLLM)

The dashboard auto-detects which LLM backend is running. In `auto` mode (default), it probes vLLM first (`--vllm-url`, default `http://127.0.0.1:5001`), then the LiteLLM proxy (`--litellm-url`, default `http://127.0.0.1:4000`), then llama.cpp (`--llama-url`, default `http://127.0.0.1:5001`). If several are up, vLLM wins, then LiteLLM (a one-time log warning tells you how to pin the other). Use `--llm-backend llama`, `--llm-backend vllm` or `--llm-backend litellm` to pin.

**llama.cpp** (when `llama-server` is reachable with `--metrics`):
- Gen/prompt tokens/s from per-slot live decode counters in `GET /slots`
- Prompt tokens/s gauge: mean over the last 5 completed requests of **uncached** prompt tokens / prefill duration — the same speed llama-server logs ("prompt eval time"); cache hits do no prefill work, so fully-cached requests don't skew it. Falls back to the per-scrape live rate until the first request completes. The gauge label shows the last prompt's size and cache-hit percentage
- Per-slot context usage bars, peak ctx, spec-decode accept rate
- The model name and slot count come from `GET /props` (cached 30 s)

**vLLM** (when `/metrics` responds with `vllm:*` metrics):
- Gen/prompt tokens/s from Prometheus counter deltas (rate window 0.5–15 s); the prompt gauge shows the mean over the last 5 completed-request windows (Δuncached prompt tokens / ΣTTFT) instead of the raw delta
- KV cache % (per-engine max), Running/Waiting (N/M), E2E latency (lifetime mean)
- TTFT / ITL (windowed mean, omitted when idle), prefix-cache hit %, spec-decode accept
- Config: KV cache tokens, num GPU blocks, block size, cache dtype, GPU memory utilization
- Stale-hold: last data held for 15 s after a failed scrape
- vLLM `uptime_s` is `time.time()` minus `process_start_time_seconds` — a cross-host clock comparison, so it assumes roughly NTP-synchronized clocks (clamped to 0); the llama.cpp state-file uptime is single-host and immune

**LiteLLM** (when the proxy `/metrics` responds with `litellm_*` metrics — the proxy needs the `prometheus` callback in `litellm_settings`): 
- Gen/prompt tokens/s from `litellm_output/input_tokens_metric` counter deltas (rate window 0.5–15 s; the last rate is held across longer gaps, e.g. right after an outage, so the value never flashes "–"); the LLM scrape runs on its own 1 s cadence, decoupled from the main collection cycle; when 2+ models are active, per-model gen tokens/s lines are drawn over the total area in the trend chart (single-model keeps the plain area)
- Requests in flight (pod queue depth), requests completed/failed (proxy counters)
- TTFT (windowed mean, streaming requests only), ITL (per-output-token gauge, max across deployments), E2E + upstream API latency (lifetime means)
- Provider-side prompt-cache hit %, total spend (USD, in the usage summary line)
- The model shown is the `requested_model` label (client-facing alias), first seen series
- Stale-hold: last data held for 15 s after a failed scrape; `--litellm-token` sends a `Bearer` header when `require_auth_for_metrics_endpoint` is set

The card shows **unreachable** when the URL is wrong, unroutable, or the server is not running. If a scrape fails, the last data is held for up to 15 s and shown as **stale**. Pass `--no-llama` to disable the section entirely (no scraping, card hidden); it is enabled by default.

### AMD GPU support (amd-smi / rocm-smi / sysfs)

The AMD chain is probed automatically alongside NVIDIA (no configuration needed) — **mixed-vendor boxes (NVIDIA + AMD) are supported**: both card sets are shown, with `index` renumbered 0..n-1 across vendors and a `vendor` tag (`nvidia`/`amd`) on each card (AMD titles are prefixed `AMD ·`). On NVIDIA-only boxes the AMD probe is negative-cached for 30 s so the `amd-smi` subprocess isn't spawned every second, but a freshly loaded `amdgpu` driver is still picked up. Evidence and exact output formats: `docs/research/amd-smi.md`.

| Tier | Tool | Invocation | Notes |
|------|------|------------|-------|
| 1 (preferred) | `amd-smi` (ROCm 6.x+) | `amd-smi monitor --power-usage --temperature --gfx --vram-usage --json` | JSON is the stable contract; VRAM reported in **MB**; `hotspot_temperature` preferred with `memory_temperature` fallback; `gfx` = utilization %, `gfx_clock` = MHz, `power_usage` = W |
| 2 (legacy) | `rocm-smi` (ROCm 5.x) | `rocm-smi --showuse --showmeminfo vram --showtemperatures --csv` | Columns matched by **header name**, never position; VRAM reported in **BYTES**; `GPU use (%)`, `VRAM Total/Used Memory (B)`, first `Temperature (Sensor…)` column, `…Graphics Package Power (W)` |
| 3 (last resort) | sysfs | `/sys/class/drm/cardN/device/{gpu_busy_percent,mem_info_vram_used,mem_info_vram_total,temperature1}` | No clock/power; amdgpu discriminator = co-presence of all three attributes; milli-°C temperature |

Card name / board vendor come from `lspci` (via the sysfs `device` → PCI BDF symlink). `N/A` is treated as *absent*, never 0. All three tiers emit the same per-GPU dict shape the frontend expects, so missing fields simply render as `–`.

### One-shot API (for testing / scripting)

```bash
curl http://127.0.0.1:5000/api/metrics | python -m json.tool
```

**Breaking change:** the per-snapshot key `llama` was replaced by `llm` (with a `backend: "llama" | "vllm" | "litellm"` discriminator) and `llama_enabled` by `llm_enabled`; the older output shape is gone.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Architecture

- **Backend**: Flask with a background `MetricsCollector` thread polling every 1 second
- **Streaming**: Server-Sent Events (SSE) — native browser `EventSource`, auto-reconnects on disconnect
- **Frontend**: Single HTML page with Chart.js for live-updating line charts and CSS gauges
- **GPU monitoring**: NVIDIA via `pynvml` with `nvidia-smi --query-gpu=... --format=csv` fallback; AMD via `amd-smi monitor --json` → `rocm-smi --csv` → sysfs (see the AMD section above)

## Project Structure

```
resource_dashboard/
├── app.py                  # Flask app + metrics collector + GPU monitors
├── config.example.yaml     # Annotated config example (copy to config.yaml)
├── templates/
│   └── index.html          # Dashboard UI with Chart.js
├── tests/
│   └── test_metrics.py     # Unit tests for metrics collectors
├── requirements.txt        # Python dependencies (flask, psutil, nvidia-ml-py, PyYAML)
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
