# Troubleshooting

## Symptoms

| Symptom | Cause / fix |
|---|---|
| `nvidia-ml-py` install fails | Harmless: the dashboard falls back to parsing `nvidia-smi` CLI output automatically. To fix the root cause, install the NVIDIA driver development headers (`nvidia-driver-dev` on Ubuntu). |
| No GPU data shown | Check `nvidia-smi` in a terminal on the dashboard host. If it errors, the drivers are missing or not loaded. On JetPack, run the smoke script in [gpu-detection.md](gpu-detection.md#on-device-smoke-script) to check the sysfs inputs. |
| JetPack shows `–` for utilization | Expected for a few seconds after boot: the nvgpu devfreq node appears late and fills in automatically. On Thor, verify `nvidia-smi --query-gpu=utilization.gpu` returns a number rather than `N/A`. |
| AMD card missing clock/power | You are on the sysfs tier (tier 3) — install/enable `amd-smi` (ships with ROCm 6.x+) or `rocm-smi`. See the tier table in [gpu-detection.md](gpu-detection.md#amd--amd-smi--rocm-smi--sysfs). |
| AMD temperature reads `–` | `edge` is `N/A` on many datacenter cards; the collector prefers `hotspot`. If both are `N/A`, the driver is newer than the installed AMD-SMI release's `gpu_metrics` layout — update `amd-smi-lib`. |
| A GPU shows `N/A` fields that should have values | `N/A` is treated as **unknown**, never 0 — the UI renders `–`. That is intentional: a fabricated 0 is worse than a gap. |
| LLM card shows **unreachable** | Wrong/unroutable URL or the server isn't exporting metrics: `curl <url>/metrics` from the dashboard host, confirm the server runs with metrics enabled (llama.cpp needs `--metrics`; LiteLLM needs the `prometheus` callback), and check the firewall. See [llm-backends.md](llm-backends.md). |
| LLM card shows **stale** | A scrape failed within the last 15 s; values are held while the scraper retries. Beyond 15 s the target reports `down`. |
| LiteLLM metrics are empty | The proxy needs the `prometheus` callback in `litellm_settings`, and if `require_auth_for_metrics_endpoint` is set you must pass `--litellm-token` / `RD_LITELLM_TOKEN`. |
| Auto-detect picked the wrong backend | More than one backend answered. Probe order is vLLM → LiteLLM → Strata → llama.cpp; pin with `--llm-backend`. A one-time log warning names the runner-up. |
| `vllm` uptime looks wrong / negative-ish | vLLM `uptime_s` compares two hosts' clocks (`process_start_time_seconds` vs local `time.time()`, clamped to 0). Sync both hosts with NTP. |
| Charts stop updating | Look for SSE errors in the browser console. The green **Live** dot (top-right) means the stream is healthy; if it's gone, check the proxy buffering `/events` (`X-Accel-Buffering: no` is sent, but an intermediary may still buffer). |
| A fleet system is permanently **stale** or **down** | That upstream likely runs a single-threaded server: it returns `503`/queues `/api/metrics` while a browser holds `/events`. Redeploy it on a threaded server — see [deployment.md](deployment.md#production-wsgi-serving). |
| A fleet system disappeared | After 30 consecutive failures its fetch thread exits (`upstream_removed` on `/health`); the interval doubles per failure up to 60 s first. Fix reachability and restart the dashboard. |
| Startup exits with `llms: configured; drop the legacy LLM settings …` | `llms:` and a legacy key (`llm_backend`, `llama_url`, `vllm_url`, `litellm_url`, `litellm_token`, `strata_url`) were both set. `llms:` replaces them — remove the legacy keys. |
| Startup exits on a config value | Validation is strict: wrong type, unknown backend, non-positive `poll_interval`, bad target name, duplicate name, or two `primary:` entries all fail loudly. Unknown *keys* are only warned about. |
| `/api/metrics` returns `503 {"error": "no data yet"}` | Normal before the first collection cycle completes. Persistent 503 means the collector never completed a cycle — check `/health`. |

## Reading the logs

- A persistently failing target logs its exception **once**, then is suppressed for a 60 s
  cooldown (at most one `still failing` warning per cooldown); the recovery is logged once.
  So a silent log usually means "still failing", not "fixed" — confirm with `/health`.
- GPU identity / `lspci` work happens off the snapshot critical path; the CPU/memory/LLM
  numbers keep flowing while a slow tool resolves.
- On Thor a wedged `nvidia-smi` can stall one snapshot by up to its 2 s timeout; SSE keeps
  serving the last good payload meanwhile.

## Self-observability

```bash
curl -s http://127.0.0.1:5000/health | python -m json.tool
```

`status: "degraded"` plus a `last_errors` entry names the target that is failing (LLM target
name, upstream name, or `"main"` for the collection cycle).
`last_cycle_duration_s` tells you whether the main loop is overrunning `poll_interval`.
Field-by-field detail is in [api.md](api.md#health).

## Reporting a problem

Open an issue with: the dashboard version (`git rev-parse --short HEAD`), OS/distro, the
output of `/health`, which collector you suspect (GPU vendor/tier, LLM backend), and the
relevant `journalctl -u resource-dashboard` lines. Please redact hostnames and tokens.
