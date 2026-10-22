# Configuration

Precedence is always **environment > CLI flags > `config.yaml` > built-in defaults**
(the one exception being `RD_LITELLM_TOKEN`, which outranks everything for that single
key — see [Environment variables](#environment-variables)).

## Config file discovery

| How | Behaviour |
|---|---|
| `config.yaml` in the working directory | **Auto-detected.** Missing is fine — defaults are used. |
| `--config PATH` | Explicit. If the path does not exist the process **exits with an error**. |

Unknown keys are warned about and ignored. Invalid values (wrong type, unknown backend,
non-positive interval) cause a startup error rather than a silent fallback.

Copy [`config.example.yaml`](../config.example.yaml) to `config.yaml` to start.

## Keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `host` | str | `0.0.0.0` | Bind address. `0.0.0.0` exposes the dashboard to the network (see the security note in [deployment.md](deployment.md)). |
| `port` | int | `5000` | Listen port. |
| `poll_interval` | float | `1.0` | Seconds between collection cycles. Must be `> 0`. |
| `llm_backend` | str | `auto` | `auto` \| `llama` \| `vllm` \| `litellm` \| `strata`. Auto-detect probes vLLM → LiteLLM → Strata → llama.cpp. |
| `llama_url` | str | `http://127.0.0.1:5001` | llama.cpp `llama-server` base URL. |
| `vllm_url` | str | `http://127.0.0.1:5001` | vLLM base URL (`/metrics`). |
| `litellm_url` | str | `http://127.0.0.1:4000` | LiteLLM proxy base URL. |
| `litellm_token` | str | — | Bearer token for a proxy with `require_auth_for_metrics_endpoint`. |
| `strata_url` | str | `http://127.0.0.1:8080` | Strata inference engine (JSON `/metrics`). |
| `llama_enabled` | bool | `true` | `false` is equivalent to `--no-llama` (the CLI flag wins when passed). |
| `llms` | list | — | Multiple inference targets. **Replaces** all single-backend keys above — see [Multiple LLM servers](#multiple-llm-servers-llms). |
| `default_view` | str | `local` | System tab shown on load: `local` \| `fleet`. |
| `upstreams` | list | — | Other dashboards to aggregate — see [Upstream dashboards](#upstream-dashboards-upstreams). |

## Environment variables

| Variable | Effect |
|---|---|
| `RD_LITELLM_TOKEN` | When set and non-empty, overrides **both** `--litellm-token` and `litellm_token`, so the token stays out of config files and process lists. An empty value is treated as unset. |

## CLI reference

```bash
python app.py [options]
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Explicit config file (exits if absent). |
| `--host ADDR` | Bind address (default `0.0.0.0`). |
| `--port N` | Port (default `5000`). |
| `--poll-interval SEC` | Collection interval in seconds (default `1.0`). |
| `--llama-url URL` | llama.cpp server URL. |
| `--vllm-url URL` | vLLM server URL. |
| `--litellm-url URL` | LiteLLM proxy URL. |
| `--litellm-token TOKEN` | Bearer token for the LiteLLM metrics endpoint (prefer `RD_LITELLM_TOKEN`). |
| `--strata-url URL` | Strata engine URL. |
| `--llm-backend {auto,llama,vllm,litellm,strata}` | Pin the backend, skip auto-detect. |
| `--default-view {local,fleet}` | Tab shown on page load. |
| `--no-llama` | Disable the LLM section entirely (no scraping, card hidden). |
| `--upstream name=url` | Aggregate another dashboard; repeatable. Merged with `upstreams:` by name, CLI wins on duplicates. |

Examples:

```bash
python app.py                                        # 0.0.0.0:5000, auto-detect LLM
python app.py --host 127.0.0.1 --port 8080           # local-only
python app.py --vllm-url http://192.168.1.10:8000    # remote vLLM
python app.py --llm-backend strata                   # pin backend, skip auto-detect
python app.py --poll-interval 2 --no-llama           # slower cycle, no LLM card
python app.py --default-view fleet                   # open on the Fleet tab
python app.py --upstream node-a=http://node-a.example.com:5001 \
              --upstream node-b=http://192.168.1.20:5001
```

## Multiple LLM servers (`llms:`)

Scrape any number of inference servers at once. Each target gets its own card, its own
trend history, and its own stale-hold; targets are scraped in parallel, so one slow or
down server cannot delay the others.

```yaml
llms:
  - name: local           # [A-Za-z0-9._-], unique, max 64 chars
    backend: strata       # llama | vllm | litellm | strata (no auto per target)
    url: http://127.0.0.1:8080
  - name: gpu-1
    backend: vllm
    url: http://192.168.1.30:8000
  - name: proxy
    backend: litellm
    url: http://127.0.0.1:4000
    token: sk-...         # optional; Bearer header (litellm only)
    primary: true         # optional; at most one (default: first entry)
```

Rules:

- `llms:` **replaces** the single-backend settings. Any legacy key (`llm_backend`,
  `llama_url`, `vllm_url`, `litellm_url`, `litellm_token`, `strata_url`) present
  alongside `llms:` exits at startup with
  `llms: configured; drop the legacy LLM settings (…) or remove llms`. Without `llms:`
  the single-backend mode behaves exactly as before (one implicit target named
  `primary`, auto-detect allowed).
- Validation is **strict**: bad names, unknown/missing backends, empty URLs, duplicate
  names, or two `primary:` entries exit with an error (unlike `upstreams:`, which warns
  and skips).
- Payload: `/api/metrics` gains an `llms` array (primary first), each entry with `name`,
  `backend`, `status` (`up`/`stale`/`down`) plus the backend fields. Unreachable targets
  appear as `{name, backend, status: "down"}` placeholders, never missing. The legacy
  `llm` key is the **primary target's entry** (or `null`), not an aggregate.
- Fleet aggregation sums each system's `llms` entries (`up` + `stale`) for the tokens/s
  strip and chart, so a system draws one line regardless of how many servers it hosts.
- Per-target histories reset independently when a target's backend changes (auto mode
  only; `llms:` targets are pinned).
- **Process attribution**: the `process` field (RSS/CPU of the `llama-server` process) is
  attached only when the dashboard's *sole* target is a llama server on the dashboard's
  own host. With several targets it is omitted rather than pointing at the wrong server.
- `--no-llama` / `llm_enabled: false` disable all targets at once (`llms: []` in the payload).
- Per-target llama.cpp usage-state files are named `.llama_usage_state.<name>.json`
  (the implicit single target keeps `.llama_usage_state.json`). These files are local
  state and are gitignored.

## Upstream dashboards (`upstreams:`)

```yaml
upstreams:
  - name: node-a
    url: http://node-a.example.com:5001
  - name: node-b
    url: http://192.168.1.20:5001
```

- One daemon thread per upstream, independent 1 s cadence, 2 s HTTP timeout — a hung
  upstream never delays the others.
- Per-upstream stale-hold: the last good metrics are held for 15 s, after which the
  system shows as **down** with null metrics.
- Upstream payloads are sanitized to a strict key set; an upstream's own `systems` array
  is never nested and unknown app-level keys are dropped.
- The snapshot gains a `systems` array: `[{name: "local", status, metrics}, …]` with the
  local entry first and its metrics matching the top-level keys exactly. All existing
  top-level keys are unchanged, so older UIs keep working.
- A failing upstream's fetch interval doubles per consecutive failure (capped at 60 s), and
  after **30 consecutive failures** its thread exits and the name is counted in
  `upstream_removed` on `/health`. A success resets the backoff to `poll_interval`.
- **Auth is not supported** — trusted-network assumption. Put upstream instances on a
  private network.
