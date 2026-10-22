# HTTP API

All endpoints are read-only `GET`. There is no authentication — see
[deployment.md](deployment.md#security-posture--read-first).

| Endpoint | Purpose |
|---|---|
| `/` | Dashboard HTML (single page, Chart.js). Server-side renders the hostname and the configured default view. |
| `/events` | Server-Sent Events stream of the same JSON payload. |
| `/api/metrics` | One-shot JSON snapshot. `503 {"error": "no data yet"}` before the first collection cycle completes. |
| `/health` | Self-observability snapshot (not metrics). |

```bash
curl http://127.0.0.1:5000/api/metrics | python -m json.tool
curl http://127.0.0.1:5000/health | python -m json.tool
```

## `/events` (SSE)

`data: <snapshot json>\n\n` per cycle, with a `: keepalive` comment every 5 s of silence so
proxies and browsers don't time the stream out. `X-Accel-Buffering: no` is set so nginx
doesn't buffer.

Each client gets its own subscription buffer backed by a last-snapshot + sequence hub, so a
blocked or slow client cannot starve the others or `/api/metrics`. The browser uses native
`EventSource`, which auto-reconnects; the green **Live** dot in the top-right corner
indicates a healthy stream.

## `/api/metrics` payload

```jsonc
{
  "timestamp": 1760000000.0,          // unix seconds of this snapshot
  "cpu": 12.4,                        // overall %
  "cpu_per_core": [8.1, 20.3, …],     // per-core %
  "memory": { "percent": 61.2, "total_gb": 31.3, "used_gb": 19.2, "available_gb": 12.1 },
  "swap": { "used_gb": 0.4, "total_gb": 8.0, "pct": 5.0,
            "in_kb_s": 0.0, "out_kb_s": 0.0 },  // null when the host has no swap;
                                                // rates omitted on the first scrape
  "host_health": { "psi_mem_avg10": 0.0,  // null without /proc/pressure/memory
                   "d_state_procs": 0 },
  "gpus": [ { "index": 0, "name": "NVIDIA GeForce RTX 3090 (Gigabyte)", "vendor": "nvidia",
              "utilization_pct": 42, "memory_pct": 61.2,
              "memory_used_gb": 11.6, "memory_total_gb": 23.6,
              "temperature_c": 55, "graphics_clock_mhz": 1800,
              "power_w": 120.4, "power_limit_w": 300.0,
              "throttle": { "hw_thermal_slowdown": false, "sw_thermal_slowdown": false },
              "pci_bus_id": "0000:2b:00.0", "board_vendor": "Gigabyte" } ],
              // unsupported/unknown fields are absent, never 0 — the UI renders –
  "llm_enabled": true,
  "llms": [ { "name": "local", "backend": "strata", "status": "up", "…": "…" } ],
  "llm": { "name": "local", "backend": "strata", "status": "up", "…": "…" },
  "systems": [ { "name": "local", "status": "up", "metrics": { "cpu": 12.4, "memory": {…},
                                                           "gpus": […], "llms": […] } },
               { "name": "node-a", "status": "up", "metrics": { "cpu": 3.1, "memory": {…},
                                                          "gpus": […], "llms": […] } } ]
}
```

Notes:

- **`llms`** — one entry per configured target, primary first, `status` ∈ `up` / `stale` /
  `down`. Unreachable targets are present as `{name, backend, status: "down"}` placeholders,
  never missing.
- **`llm`** — the **primary target's entry** (or `null` when it is unreachable). It is *not*
  an aggregate; fleet consumers sum `llms`.
- **`systems`** — the local entry first, then one per upstream. The local entry's
  `metrics` is a **slim fleet projection** (`cpu`, `memory`, per-GPU utilization, LLM gen
  tok/s) — the local detail tab reads the top-level keys instead, so the payload never
  re-embeds the whole local snapshot. Upstream entries carry the sanitized projection of
  that system's payload; an upstream's own `systems` array is never nested.
- All pre-existing top-level keys are unchanged, so older UIs keep working against newer
  backends.
- GPU field names are the same across the NVIDIA, AMD and JetPack collectors
  (`index`, `utilization_pct`, `memory_pct`, `memory_used_gb`/`memory_total_gb`,
  `temperature_c`, `graphics_clock_mhz`, `power_w`/`power_limit_w`, `throttle`, `name`,
  `board_vendor`, `pci_bus_id`, `vendor`); fields a tier cannot source are omitted (rendered
  `–`), never reported as `0`. Power keys are omitted entirely when a device or VM rejects
  the power query, so old and new clients degrade the same way.

### Payload-shape history

**Breaking change (already shipped):** the per-snapshot key `llama` was replaced by `llm`
(with a `backend: "llama" | "vllm" | "litellm" | "strata"` discriminator), and
`llama_enabled` by `llm_enabled`. The older output shape is gone.

## `/health`

```jsonc
{
  "status": "ok",                     // "degraded" when any target has a recent error
  "poll_interval": 1.0,               // configured main-cycle interval (s)
  "last_cycle_duration_s": 0.043,     // wall clock of the last _collect(); null before first
  "last_errors": {                    // per-target last error; absent = no recent error
    "local": "ConnectionRefusedError(…, 'Connection refused')",
    "node-a": "down (http://node-a.example.com:5001 returned None)",
    "main": "…"
  },
  "upstream_removed": 0               // upstreams dropped after N consecutive failures
}
```

Keys in `last_errors` are each LLM target's name, each upstream's name, and `"main"` for the
main collection cycle. A value is the most recent failure — the exception string, or
`"down (…returned None)"` for a stale-hold expiry.

`/health` never consumes or mutates the metrics snapshot, so it stays readable even when
collection is wedged.
