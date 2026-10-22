# LLM backends

The dashboard auto-detects which inference backend is running. In `auto` mode it probes
**vLLM → LiteLLM → Strata → llama.cpp**; if several answer, the first in that order wins
and a one-time log warning tells you how to pin the other. Pin explicitly with
`--llm-backend {llama|vllm|litellm|strata}`.

A card shows **unreachable** when the URL is wrong, unroutable, or the server is down. If a
scrape fails, the last data is held for up to **15 s** and shown as **stale**. `--no-llama`
disables the section entirely (no scraping, card hidden); it is enabled by default.

Every backend exposes generation and prompt tokens/s from counter deltas over a 0.5–15 s
rate window: a rate older than the window is dropped, while the last good rate is held
across longer gaps (e.g. right after an outage) so the value never flashes `–`.

## llama.cpp (`llama-server --metrics`)

`/props` and `/slots` are served by default; `--metrics` is required for the token counters.

- Gen/prompt tokens/s from per-slot live decode counters in `GET /slots`.
- **Prompt tokens/s gauge**: mean over the last 5 completed requests of **uncached** prompt
  tokens / prefill duration — the same speed llama-server logs as "prompt eval time". Cache
  hits do no prefill work, so fully-cached requests don't skew it. Falls back to the
  per-scrape live rate until the first request completes. The gauge label shows the last
  prompt's size and cache-hit percentage.
- Per-slot context usage bars, peak context, spec-decode accept rate.
- Model name and slot count come from `GET /props` (cached 30 s).
- Uptime is derived from a local per-target usage-state file — single-host, so it is immune
  to clock skew (unlike vLLM's, see below).

## vLLM

Detected when `/metrics` responds with `vllm:*` metrics.

- Gen/prompt tokens/s from Prometheus counter deltas; the prompt gauge is the mean over the
  last 5 completed-request windows (Δuncached prompt tokens / ΣTTFT) rather than a raw delta.
- KV cache % (per-engine max), Running/Waiting (N/M), E2E latency (lifetime mean).
- TTFT / ITL (windowed means, omitted when idle), prefix-cache hit %, spec-decode accept.
- Config line: KV cache tokens, num GPU blocks, block size, cache dtype, GPU memory utilization.
- Stale-hold: last data held 15 s after a failed scrape.
- `uptime_s` is `time.time()` minus `process_start_time_seconds` — a **cross-host clock
  comparison**, so it assumes roughly NTP-synchronized clocks (clamped to 0).

## LiteLLM proxy

Detected when the proxy `/metrics` responds with `litellm_*` metrics. The proxy needs the
`prometheus` callback in `litellm_settings`.

- Gen/prompt tokens/s from `litellm_output/input_tokens_metric` counter deltas. The LLM
  scrape runs on its own 1 s cadence, decoupled from the main collection cycle. When 2+
  models are active, per-model gen tokens/s lines are drawn over the total area in the
  trend chart (single model keeps the plain area).
- Requests in flight (pod queue depth), requests completed/failed (proxy counters).
- TTFT (windowed mean, streaming requests only), ITL (per-output-token gauge, max across
  deployments), E2E + upstream API latency (lifetime means).
- Provider-side prompt-cache hit %, total spend (USD, in the usage summary line).
- The model shown is the `requested_model` label (the client-facing alias), first-seen series.
- `--litellm-token` (or `RD_LITELLM_TOKEN`) sends a `Bearer` header when the proxy sets
  `require_auth_for_metrics_endpoint`.
- Stale-hold: last data held 15 s after a failed scrape.

Docs: [docs.litellm.ai — proxy/prometheus](https://docs.litellm.ai/docs/proxy/prometheus).

## Strata

Detected when `/metrics` responds with the Strata JSON document (`engine` + `live` objects).
Default port 8080. Strata's `/metrics` is **JSON, not Prometheus text**.

- Gen/prompt tokens/s from `totals.output_tokens` / `totals.prompt_tokens` counter deltas.
- Running / queued+waiting batch requests, live tok/s and engine state
  (idle / reading / generating) from `live`.
- Per-slot context bars from `live.slots` (idle slots show the KV prefix they hold),
  free/total batch slots, spec-decode (MTP) accept rate from `totals.drafts_*`.
- KV prefix reuse % (`totals.reused` ⊂ `totals.prompt_tokens`, nested like vLLM's cached accounting).
- TTFT / E2E / ITL as windowed means over newly finished entries of the `requests` ring
  buffer (`prompt_ms` / `duration_s` / `decode_tok_s`).
- Usage summary: total/gen tokens, prompt-cached %, uptime from `totals.since`.
- Slot diagnosability exposes `state_seen` / `slots_busy` / `last_busy_ts` /
  `slots_state_mismatch` so a stuck slot is distinguishable from an idle engine.

Strata is a separate project: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).
