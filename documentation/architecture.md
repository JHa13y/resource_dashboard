# Architecture

## Shape

- **Backend** — Flask app plus a background `MetricsCollector` thread that polls every
  `poll_interval` seconds (default 1 s) and publishes one snapshot.
- **Streaming** — Server-Sent Events over native browser `EventSource` (auto-reconnect).
- **Frontend** — one HTML page (`templates/index.html`) with Chart.js for live line charts
  and CSS gauges. Chart.js is vendored in `static/chart.umd.min.js` (MIT) — no CDN, so the
  dashboard works on an air-gapped LAN.
- **Collection** — `collectors/gpu.py` exposes `collect_gpu_metrics()` and runs the NVIDIA /
  AMD / JetPack chains; `collectors/llm.py` exposes `collect_llama_metrics()`,
  `collect_vllm_metrics()`, `collect_litellm_metrics()` and `collect_strata_metrics()`, all
  sharing per-target state and stale-hold through `BaseLLMCollector` in
  `collectors/base.py`; `app.py` owns settings, the loops, the SSE transport and the fleet
  layer.

## Threads and cadence

| Thread | Cadence | Notes |
|---|---|---|
| Main collector | fixed rate, `next += poll_interval` | Drift-corrected (not `sleep(interval)` after work), with a drift log line when a cycle overruns. GPU identity/`lspci` work is kept off the snapshot critical path. |
| LLM scraper | its own 1 s cadence | Decoupled so the main cycle cannot widen or narrow a backend's rate window; per-target scrape in parallel. |
| One per upstream | 1 s, independent | 2 s HTTP timeout; a hung upstream never delays peers or the local cycle. |

The fixed-rate loop matters because every backend derives tokens/s from **counter deltas
over a 0.5–15 s window**: if the scrape cadence drifted with request load, the computed
rates would drift with it.

## Broadcast hub

`/events` used to be a single-consumer queue: one blocked browser could starve the others
and `/api/metrics`. It is now a hub — last-snapshot + monotonic sequence + per-client
subscription buffers. A new subscriber is primed with the last snapshot; a slow client
loses frames from *its own* buffer instead of blocking anyone else. `stop()` wakes and joins
all upstream threads on shutdown.

## Diagnosability

- **Stale-hold**: a failed scrape holds the last good values for 15 s (`status: "stale"`),
  then the target reports `down` with nulls.
- **Busy-latch**: a busy LLM sample produced during an over-long main cycle is latched and
  published on the next idle tick, so a short "generating" state is never lost.
- **Rate windows**: `0.5 ≤ elapsed ≤ 15 s`; a longer gap holds the last computed rate rather
  than reporting a spike or `–`.
- **Log rate limiting**: a persistently failing target logs its exception once, then is
  suppressed for a 60 s cooldown (one `still failing` warning per cooldown), and a recovery
  is logged once — instead of the old per-cycle `log.exception` flood.
- **`/health`** exposes the last cycle duration and per-target last errors so a wedge is
  visible without tailing logs.
- **Strata slot diagnosability**: `slots_busy`, `last_busy_ts`, `state_seen`,
  `slots_state_mismatch` distinguish a stuck slot from an idle engine.

## Fleet (multi-system) model

**Relay model:** every machine runs the same app and collects its **own** local metrics; the
aggregating dashboard fetches `GET /api/metrics` from each upstream once per second and
merges. No agents, no SSH, no new software — collection stays local where the data is.

The UI adds a tab row (**Fleet** + one tab per system) with a status dot per system (green
up / yellow stale / red down). The Fleet view shows a summary strip (per-system CPU / mem /
GPU / LLM tokens/s) and a fleet **tokens/s chart** (total as a filled area, each system a
non-filled line over it). Per-system tabs render the full detail view from that system's own
data, and trend buffers are kept **per system**, so switching tabs restores history instead
of resetting it.

To keep the payload small, `systems[].metrics` is a **slim projection** — just what the fleet
strip needs — and the local entry's data is not re-embedded (the local detail tab reads the
top-level keys).

## Startup and serving

`app.py` is an app factory with a **lazy WSGI bootstrap**: importing `app` does not start
threads; the collector starts on first use, and settings are applied regardless of serve
mode. `python -m app` runs the explicit bootstrap (`main()`), which resolves
env > CLI > config and starts the collector on the resolved settings. `app:wsgi` applies the
same resolved settings at import time, so a WSGI server (e.g. `waitress-serve app:wsgi`)
never runs on built-in defaults.

Deps are pinned in `requirements.txt`; the app is served by `waitress` in production and must
be threaded for fleet use (see [deployment.md](deployment.md#production-wsgi-serving)).

## Known limitations (v1)

- No upstream authentication — trusted-network assumption.
- CPU/mem/GPU trend charts reset on tab switch (per-system buffers for those are a v2
  candidate); LLM per-target history is preserved.
- Version skew: upstreams should run the same or an older build. Older builds without
  `systems` still work but appear as a single local entry; older *single-threaded* builds
  need the deployment note in [deployment.md](deployment.md#production-wsgi-serving).
- AMD throttle badges always report *no throttle* and the sysfs tier has no clock/power
  (see [gpu-detection.md](gpu-detection.md#amd--amd-smi--rocm-smi--sysfs)).
- JetPack exposes no rail power without a persistent `tegrastats` process (deferred).
