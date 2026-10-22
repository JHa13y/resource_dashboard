"""LLM / inference-server collectors (llama.cpp, vLLM, LiteLLM, Strata)
plus the upstream-dashboard collector. Moved verbatim from ``app.py``
(T12 package split); behavior is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import psutil

from collectors.base import BaseLLMCollector

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# llama.cpp (llama-server) Monitoring
# ---------------------------------------------------------------------------

# Base URL of the local llama-server (overridable via --llama-url).
DEFAULT_LLAMA_URL = "http://127.0.0.1:5001"

# Base URL of the vLLM server (overridable via --vllm-url).
DEFAULT_VLLM_URL = "http://127.0.0.1:5001"

# Base URL of the LiteLLM proxy (overridable via --litellm-url).
DEFAULT_LITELLM_URL = "http://127.0.0.1:4000"

# Base URL of the Strata inference engine (overridable via --strata-url).
# Strata serves its web UI + JSON /metrics on port 8080 by default.
DEFAULT_STRATA_URL = "http://127.0.0.1:8080"

# How often (seconds) to re-fetch /props; model info only changes on reload.
# All module-level scrape state below is keyed by LLM *target name*
# ("primary" for the implicit single-backend target), so several servers of
# the same backend type scrape independently. Each key is touched by exactly
# one scrape thread (mirrors the _upstream_last_good discipline).
_LLAMA_PROPS_TTL = 30.0
_llama_props_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# Previous successful scrape's cumulative counters + timestamp, used to
# compute tokens/s from per-scrape counter deltas. The windowed
# ``*_tokens_seconds`` gauges read 0 on some builds (e.g. the llama.cpp
# bundled with LM Studio) while the cumulative counters increment correctly.
# Last successful snapshot (monotonic ts + dict). When a scrape fails
# (e.g. the 2 s timeout under heavy llama-server load) the UI keeps showing
# this data, flagged "stale", for up to this many seconds so a single bad
# tick doesn't flicker the card to "unreachable".
_LLAMA_STALE_MAX_AGE_S = 15.0
_llama_collector = BaseLLMCollector(_LLAMA_STALE_MAX_AGE_S)
# Backward-compatible aliases (tests reference these names directly).
_llama_prev = _llama_collector._prev
_llama_last_good = _llama_collector._last_good
_llama_req_rates = _llama_collector._req_rates

# Per-slot decode counters from the previous successful scrape, used for
# LIVE tokens/s: on some builds (e.g. the llama.cpp bundled with LM Studio)
# the /metrics counters only advance when a request COMPLETES, so counter
# deltas read 0 during generation while the per-slot n_decoded /
# n_prompt_tokens_processed update per token.
_llama_slot_prev: dict[str, dict[str, Any]] = {}

# (monotonic ts, n_decode_total) pairs from the last successful scrapes.
# n_decode_total ticks per llama_decode() call; frozen across 10 consecutive
# successful scrapes while a slot is mid-generation means the server is
# hung (stuck decode / GPU wedged). Long prefills call no llama_decode, so
# slots with decoded == 0 are excluded (see the stall check below).
_LLAMA_DECODE_HISTORY_LEN = 10
_llama_decode_history: dict[str, list[tuple[float, float]]] = {}

# Per-completed-request prompt-rate samples (tokens/s), kept for BOTH
# llama and vLLM: the per-scrape delta spikes for one tick when a large
# prompt prefills and then reads 0. The gauge reports the mean of the last
# N completed requests instead, falling back to the per-scrape live rate
# until the first measurable request completes.
# A prefill shorter than this cannot be measured at 1 s scrape cadence
# (it finishes between two scrapes); such requests are skipped.
_LLAMA_MIN_PREFILL_S = 0.25

# Local llama-server process handle (T5, keyed per target in T8): a
# per-target retained handle so cpu_percent(None) measures against the
# previous tick. A dead handle (NoSuchProcess) is dropped and
# re-identified on the next scrape — a model reload spawns a new PID.
# Keyed by state_key so parallel llama targets do not clobber each
# other's retained handle (T8: the old single unkeyed global raced
# across concurrent targets).
_llama_proc: dict[str, psutil.Process | None] = {}

# (dashboard key, Prometheus metric) pairs — verified against llama.cpp
# ``to_metrics()`` (tools/server/server-task.cpp). Missing metrics are simply
# omitted (older builds expose a shorter set).
_LLAMA_METRIC_MAP = (
    ("gen_tokens_per_s", "llamacpp:predicted_tokens_seconds"),
    ("prompt_tokens_per_s", "llamacpp:prompt_tokens_seconds"),
    ("gen_tokens_total", "llamacpp:tokens_predicted_total"),
    ("prompt_tokens_total", "llamacpp:prompt_tokens_total"),
    ("prompt_tokens_cached_total", "llamacpp:prompt_tokens_cached_total"),
    ("requests_processing", "llamacpp:requests_processing"),
    ("requests_deferred", "llamacpp:requests_deferred"),
    ("context_tokens_max", "llamacpp:n_tokens_max"),
    ("spec_decode_draft_tokens_total",
     "llamacpp:spec_decode_num_draft_tokens_total"),
    ("spec_decode_accepted_tokens_total",
     "llamacpp:spec_decode_num_accepted_tokens_total"),
    ("prompt_seconds_total", "llamacpp:prompt_seconds_total"),
    ("tokens_predicted_seconds_total",
     "llamacpp:tokens_predicted_seconds_total"),
    ("n_decode_total", "llamacpp:n_decode_total"),
)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse plain Prometheus text into ``{metric_name: value}``.

    Only unlabeled ``llamacpp:*`` series are kept; ``# HELP``/``# TYPE``
    comment lines, labeled series (``{...}``) and malformed lines are
    skipped. Never raises.
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            metrics[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return metrics


def _llama_get_json(base_url: str, path: str, timeout: float = 2.0) -> Any:
    """GET a llama-server JSON endpoint; ``None`` on any failure."""
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + path, timeout=timeout
        ) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001 – collector must never raise
        return None


def _llama_health_detail(body: str) -> str | None:
    """Short human-readable detail from a /health body.

    Parsed generically (JSON ``status`` field, else trimmed text) — never
    matched against a specific build's exact wording.
    """
    body = (body or "").strip()
    if not body:
        return None
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            s = data.get("status")
            if isinstance(s, str) and s.strip():
                return s.strip()[:60]
    except Exception:  # noqa: BLE001 – not JSON; fall back to raw text
        pass
    text = " ".join(body.split())
    return text[:60] or None


def _llama_health(base_url: str) -> dict[str, Any] | None:
    """GET /health (2 s timeout); ``None`` on any failure.

    llama-server answers HTTP 200 once it can serve and HTTP 503 while a
    model is still loading, so this distinguishes "loading (be patient)"
    from "down". Returns ``{"state": "ok", ...}`` on 200 and
    ``{"state": "loading", "detail": <body>}`` on 503. Never raises.
    """
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/health", timeout=2
        ) as r:
            return {"state": "ok", "detail": _llama_health_detail(
                r.read().decode(errors="replace")
            )}
    except urllib.error.HTTPError as e:
        if e.code == 503:
            try:
                body = e.read().decode(errors="replace")
            except Exception:  # noqa: BLE001
                body = ""
            return {"state": "loading", "detail": _llama_health_detail(body)}
        return None
    except Exception:  # noqa: BLE001 – collector must never raise
        return None


def _llama_find_process(state_key: str = "primary") -> psutil.Process | None:
    """Find the local llama-server process for one scrape target. Never
    raises.

    Name match first (the usual build layout is
    ``/opt/llama.cpp/build/bin/llama-server``, psutil name
    ``llama-server``), then a cmdline-argument fallback; multiple matches
    resolve to the largest RSS. Returns ``None`` when no local process
    exists (remote/dev mode). A retained live handle is reused so
    ``cpu_percent(None)`` keeps measuring; it was primed (first real
    value on the scrape after identification). Handles are keyed by
    ``state_key`` so parallel llama targets do not clobber each other's
    retained handle (T8).
    """
    global _llama_proc
    retained = _llama_proc.get(state_key)
    if retained is not None:
        try:
            retained.memory_info().rss
            return retained
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            _llama_proc[state_key] = None
    try:
        by_name: list[psutil.Process] = []
        by_cmdline: list[psutil.Process] = []
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            try:
                if p.info["name"] == "llama-server":
                    by_name.append(p)
                    continue
                if any("llama-server" in arg
                       for arg in (p.info["cmdline"] or [])):
                    by_cmdline.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue
        matches = by_name or by_cmdline
        if not matches:
            return None

        def _rss(p: psutil.Process) -> int:
            mi = p.info["memory_info"]
            return mi.rss if mi is not None else 0

        proc = max(matches, key=_rss)
        proc.cpu_percent(None)  # prime; first real value on next scrape
        _llama_proc[state_key] = proc
        return proc
    except Exception:  # noqa: BLE001 – collector must never raise
        return None


def _llama_props(base_url: str,
                 state_key: str = "primary") -> dict[str, Any] | None:
    """Fetch /props with a 30 s cache (model info only changes on reload)."""
    now = time.monotonic()
    cache = _llama_props_cache.get(state_key)
    if cache is not None:
        ts, data = cache
        if now - ts < _LLAMA_PROPS_TTL:
            return data
    data = _llama_get_json(base_url, "/props")
    if isinstance(data, dict):
        _llama_props_cache[state_key] = (now, data)
    return data


# ---------------------------------------------------------------------------
# Cumulative usage state (persistent server uptime, T3)
# ---------------------------------------------------------------------------

def _llama_usage_state_path(state_key: str = "primary") -> str:
    """State-file location; RD_LLAMA_USAGE_STATE overrides the default.

    Named targets (multi-LLM) get a per-target filename next to the
    default one: ``.llama_usage_state.<name>.json``. The env override is
    process-wide, so it can only ever address the default target.
    """
    # The default state file lives at the project root (the parent of the
    # collectors/ package), matching the pre-T12 location in app.py.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if state_key == "primary":
        return os.environ.get("RD_LLAMA_USAGE_STATE") or os.path.join(
            _project_root, ".llama_usage_state.json"
        )
    base = os.environ.get("RD_LLAMA_USAGE_STATE")
    base_dir = (os.path.dirname(os.path.abspath(base)) if base
                else _project_root)
    return os.path.join(base_dir, f".llama_usage_state.{state_key}.json")


def _llama_load_usage_state(path: str) -> dict[str, float] | None:
    """Load ``{first_seen, last_reset}`` epoch stamps.

    ``None`` when the file is missing or corrupt (caller reinitializes).
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return {
            "first_seen": float(data["first_seen"]),
            "last_reset": float(data["last_reset"]),
        }
    except Exception:  # noqa: BLE001 – missing/corrupt → reinitialize
        return None


def _llama_save_usage_state(path: str, state: dict[str, float]) -> None:
    """Atomic write (tmp + rename); never raises (uptime is best-effort)."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def _parse_llama_slots(data: Any) -> list[dict[str, Any]]:
    """Normalise the /slots array into per-slot context-usage dicts.

    Context usage follows the llama.cpp README formula
    (prompt + cache + predicted tokens over n_ctx); the /slots payload
    carries ``n_prompt_tokens`` (prompt + cache) and
    ``next_token.n_decoded`` (predicted so far).
    """
    if not isinstance(data, list):
        return []
    slots: list[dict[str, Any]] = []
    for slot in data:
        if not isinstance(slot, dict):
            continue
        try:
            n_ctx = int(slot.get("n_ctx", 0) or 0)
            prompt = int(slot.get("n_prompt_tokens", 0) or 0)
            next_tok = slot.get("next_token")
            # Tokens left to generate; None when the slot is idle or the
            # field is absent (idle is marked n_remain = -1).
            remain = None
            if isinstance(next_tok, dict):
                decoded = int(next_tok.get("n_decoded", 0) or 0)
                r = next_tok.get("n_remain")
                if r is not None and int(r) >= 0:
                    remain = int(r)
            elif isinstance(next_tok, list):
                # Some builds (e.g. LM Studio's) emit a LIST of per-token
                # dicts; total their n_decoded / n_remain values.
                decoded = 0
                remain_sum = None
                for item in next_tok:
                    if not isinstance(item, dict):
                        continue
                    decoded += int(item.get("n_decoded", 0) or 0)
                    r = item.get("n_remain")
                    if r is None:
                        continue
                    rv = int(r)
                    if rv < 0:
                        remain_sum = None  # idle marker wins
                    elif remain_sum is not None:
                        remain_sum += rv
                    else:
                        remain_sum = rv
                remain = remain_sum
            else:
                decoded = 0
            used = prompt + decoded
            slots.append({
                "id": slot.get("id"),
                "n_ctx": n_ctx,
                "context_used": used,
                "context_pct": round(100.0 * used / n_ctx, 1) if n_ctx else 0.0,
                "is_processing": bool(slot.get("is_processing", False)),
                "decoded": decoded,
                "remain": remain,
                "prompt_processed": int(
                    slot.get("n_prompt_tokens_processed", 0) or 0
                ),
                # KV-cache hits for this task (set in one go at task start,
                # stable afterwards) and the task's total prompt size.
                "cached": int(slot.get("n_prompt_tokens_cache", 0) or 0),
                "prompt_total": prompt,
                # Changes per new task on the slot; used for new-task
                # detection (last-prompt cache stats).
                "id_task": slot.get("id_task"),
            })
        except (TypeError, ValueError):
            continue
    return slots


def collect_llama_metrics(
    base_url: str, state_key: str = "primary"
) -> dict[str, Any] | None:
    """Scrape llama-server ``/metrics`` + ``/props`` + ``/slots``.

    ``state_key`` names the scrape target: all per-scrape state (rate
    windows, stale-hold, slot deltas, props cache, usage-state file) is
    kept per key so several llama servers scrape independently.

    Returns ``None`` when the server is unreachable or not started with
    ``--metrics`` (UI shows "unreachable"). Never raises.

    ``gen_tokens_per_s`` / ``prompt_tokens_per_s`` are computed from the
    per-scrape deltas of the cumulative token counters; the server's
    windowed gauges are only a fallback for the first scrape (they read
    0 permanently on some builds, e.g. LM Studio's bundled llama.cpp).

    ``prompt_tokens_per_s`` is the mean of the last LLM_REQ_RATE_WINDOW
    completed-request rates (UNCACHED prompt tokens / prefill duration,
    tracked per-slot — the speed llama-server itself logs) when any
    exist; else the raw per-scrape delta, which spikes for one tick on a
    large prefill and reads 0 afterwards.

    On a failed scrape the last successful snapshot is returned with
    ``status: "stale"`` for up to ``_LLAMA_STALE_MAX_AGE_S`` seconds so a
    transient timeout doesn't flicker the UI to "unreachable".
    """

    def _stale_or_none() -> dict[str, Any] | None:
        """Hold last-good data during a transient scrape failure."""
        held_good = _llama_collector._last_good.get(state_key)
        if held_good is not None:
            ts, held = held_good
            if time.monotonic() - ts <= _LLAMA_STALE_MAX_AGE_S:
                out = dict(held)
                out["status"] = "stale"
                return out
        return None

    # /health first: while a model is loading the server answers 503 on
    # every endpoint (including /metrics), so the loading state is only
    # observable via /health. On 200 the /metrics up/stale/unreachable
    # logic below is unchanged.
    health = _llama_health(base_url)
    if health is not None and health["state"] == "loading":
        out_loading: dict[str, Any] = {"status": "loading"}
        if health.get("detail"):
            out_loading["health_detail"] = health["detail"]
        return out_loading

    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/metrics", timeout=2
        ) as r:
            body = r.read().decode()
    except Exception:  # noqa: BLE001 – collector must never raise
        return _stale_or_none()

    # Some 2025 builds JSON-escape the whole body (fixed upstream 2025-11-28,
    # PR #17386): unwrap if it's a JSON string.
    if body.lstrip().startswith('"'):
        try:
            body = json.loads(body)
        except Exception:  # noqa: BLE001
            return _stale_or_none()

    raw = _parse_prometheus(body)
    if not raw:
        return _stale_or_none()  # endpoint up but no llamacpp: metrics

    out: dict[str, Any] = {"status": "up"}
    for key, metric in _LLAMA_METRIC_MAP:
        if metric in raw:
            out[key] = round(raw[metric], 1)

    # tokens/s from counter deltas (see _llama_prev).
    gen_total = raw.get("llamacpp:tokens_predicted_total")
    prompt_total = raw.get("llamacpp:prompt_tokens_total")
    cached_total = raw.get("llamacpp:prompt_tokens_cached_total")
    now = time.monotonic()
    prev = _llama_collector.get_prev(state_key)
    rates = _llama_collector.get_rates(state_key)
    elapsed = now - prev["ts"] if prev is not None else 0.0
    # A counter decrease means the server restarted/reloaded: the
    # per-request prompt-rate history belongs to the old process.
    if prev is not None and (
        (gen_total is not None and gen_total < prev["gen"])
        or (prompt_total is not None and prompt_total < prev["prompt"])
    ):
        rates.clear()
    if (
        prev is not None
        and gen_total is not None
        and prompt_total is not None
        and elapsed > 0
    ):
        # A negative delta means the counters were reset (server restart);
        # report 0 rather than a nonsense rate.
        out["gen_tokens_per_s"] = round(
            max(0.0, gen_total - prev["gen"]) / elapsed, 1
        )
        out["prompt_tokens_per_s"] = round(
            max(0.0, prompt_total - prev["prompt"]) / elapsed, 1
        )
        # Older builds may not expose the cached counter: only emit the
        # rate when both scrapes have it.
        if cached_total is not None and prev.get("cached") is not None:
            out["prompt_cached_tokens_per_s"] = round(
                max(0.0, cached_total - prev["cached"]) / elapsed, 1
            )
    else:
        # First scrape (or a counter missing): fall back to the server's
        # windowed gauge if present, else 0.0.
        out.setdefault("gen_tokens_per_s", 0.0)
        out.setdefault("prompt_tokens_per_s", 0.0)
    if gen_total is not None and prompt_total is not None:
        new_prev: dict[str, Any] = {
            "ts": now,
            "gen": gen_total,
            "prompt": prompt_total,
        }
        if cached_total is not None:
            new_prev["cached"] = cached_total
        _llama_collector.set_prev(state_key, new_prev)

    props = _llama_props(base_url, state_key)
    if isinstance(props, dict):
        model_path = props.get("model_path")
        if isinstance(model_path, str) and model_path:
            out["model"] = os.path.basename(model_path)
            out["model_path"] = model_path
        total_slots = props.get("total_slots")
        if isinstance(total_slots, int):
            out["total_slots"] = total_slots
        out["sleeping"] = bool(props.get("is_sleeping", False))
        build_info = props.get("build_info")
        if isinstance(build_info, str) and build_info:
            out["build_info"] = build_info

    # Local llama-server process (co-located deployments only): None when
    # the dashboard runs on a different host — the UI hides the card
    # silently, no red state. Keyed by state_key so parallel llama
    # targets retain their own handle (T8).
    proc = _llama_find_process(state_key)
    process_info: dict[str, Any] | None = None
    if proc is not None:
        try:
            process_info = {
                "rss_gb": round(proc.memory_info().rss / 2**30, 1),
                "cpu_pct": round(proc.cpu_percent(None), 1),
                "threads": proc.num_threads(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            process_info = None  # died mid-scrape; re-identify next tick
    out["process"] = process_info

    # Model file size on THIS host (co-located only; the path comes from
    # /props and points at the llama host's filesystem).
    model_path = out.get("model_path")
    if isinstance(model_path, str) and model_path:
        try:
            out["model_size_gb"] = round(
                os.path.getsize(model_path) / 2**30, 1
            )
        except OSError:  # file not local (remote mode) or removed
            pass

    # Free slots: total (props) minus in-flight requests (/metrics).
    # Omitted when either side is unknown (older builds).
    total_slots = out.get("total_slots")
    requests_processing = out.get("requests_processing")
    if isinstance(total_slots, int) and requests_processing is not None:
        out["free_slots"] = max(0, total_slots - int(requests_processing))

    # Lifetime spec-decode accept rate. Key omitted when spec decode is off
    # (counters absent) or no drafts yet (draft == 0). Lifetime only: on the
    # reference build these counters tick per completion step, so a windowed
    # ratio would flicker between 0 and bursty values.
    spec_draft = out.get("spec_decode_draft_tokens_total")
    spec_accepted = out.get("spec_decode_accepted_tokens_total")
    if spec_draft is not None and spec_accepted is not None and spec_draft > 0:
        out["spec_accept_pct"] = round(100.0 * spec_accepted / spec_draft, 1)

    # Cumulative usage summary + persistent server uptime. Uptime survives
    # dashboard deploys (state file) and is re-baselined only when the
    # llama-server itself restarts/reloads, detected as a cumulative
    # counter decrease versus the previous successful scrape.
    gen_seconds_total = out.get("tokens_predicted_seconds_total")
    if (
        gen_total is not None
        or prompt_total is not None
        or cached_total is not None
    ):
        usage: dict[str, Any] = {
            "total_tokens": int(round(
                (prompt_total or 0) + (cached_total or 0) + (gen_total or 0)
            )),
            "gen_tokens": int(round(gen_total or 0)),
            # Average t/s WHILE GENERATING (idle time excluded by
            # construction: the server only accumulates predicted seconds
            # while producing tokens).
            "gen_avg_tps": (
                round(gen_total / gen_seconds_total, 1)
                if gen_total is not None and gen_seconds_total else None
            ),
            "prompt_cached_pct": (
                round(100.0 * cached_total / (cached_total + prompt_total), 1)
                if cached_total is not None and prompt_total is not None
                and cached_total + prompt_total > 0
                else None
            ),
        }
        now_wall = time.time()
        state_path = _llama_usage_state_path(state_key)
        state = _llama_load_usage_state(state_path)
        if state is None:
            state = {"first_seen": now_wall, "last_reset": now_wall}
            _llama_save_usage_state(state_path, state)
        elif prev is not None and (
            (gen_total is not None and prev.get("gen") is not None
             and gen_total < prev["gen"])
            or (prompt_total is not None and prev.get("prompt") is not None
                and prompt_total < prev["prompt"])
            or (cached_total is not None and prev.get("cached") is not None
                and cached_total < prev["cached"])
        ):
            # Server restart/reload: re-baseline uptime, keep first_seen.
            state["last_reset"] = now_wall
            _llama_save_usage_state(state_path, state)
        usage["uptime_s"] = max(
            0, int(now_wall - max(state["first_seen"], state["last_reset"]))
        )
        out["usage"] = usage

    slots = _parse_llama_slots(_llama_get_json(base_url, "/slots"))
    out["slots"] = slots

    # Decode-stall flag: n_decode_total is the only live /metrics counter
    # (it ticks per llama_decode() call). Frozen across 10 consecutive
    # successful scrapes while a slot is mid-generation (decoded > 0 —
    # long prefills call no llama_decode and must not trigger this) means
    # the server stopped advancing: hang / GPU wedged.
    decode_total = raw.get("llamacpp:n_decode_total")
    decode_hist = _llama_decode_history.setdefault(state_key, [])
    if decode_total is not None:
        decode_hist.append((now, decode_total))
        if len(decode_hist) > _LLAMA_DECODE_HISTORY_LEN:
            decode_hist.pop(0)
    if len(decode_hist) >= _LLAMA_DECODE_HISTORY_LEN:
        if len({v for _, v in decode_hist}) == 1:
            for slot in slots:
                if slot["is_processing"] and slot["decoded"] > 0:
                    out["stalled"] = True
                    break

    # Live tokens/s from per-slot decode counters (see _llama_slot_prev):
    # they update per token, unlike the /metrics counters on some builds.
    prev_slot = _llama_slot_prev.get(state_key)
    prompt_last = prev_slot.get("prompt_last") if prev_slot is not None else None
    if slots and prev_slot is not None:
        slot_elapsed = now - prev_slot["ts"]
        if slot_elapsed > 0:
            d_decoded = 0
            d_prompt = 0
            matched = False
            for slot in slots:
                p = prev_slot["slots"].get(slot["id"])
                if p is None:
                    continue
                matched = True
                # Per-task counters reset when a new request takes the slot:
                # clamp negative deltas instead of reporting a nonsense rate.
                d_decoded += max(0, slot["decoded"] - p["decoded"])
                # Count cached + uncached prompt tokens: the server's own
                # prompt t/s includes cache hits, and a chat workload is
                # mostly cache hits (the uncached tail prefills in <1 s and
                # is aliased away by 1 s scrapes otherwise).
                d_prompt += (
                    max(0, slot["prompt_processed"] - p["prompt"])
                    + max(0, slot["cached"] - p["cached"])
                )
            if matched:
                out["gen_tokens_per_s"] = round(d_decoded / slot_elapsed, 1)
                out["prompt_tokens_per_s"] = round(d_prompt / slot_elapsed, 1)
        # New-task detection: id_task changed (or first sight of a slot)
        # with a non-empty prompt -> record that prompt's size and cache
        # share. Most recent wins; the value persists across scrapes.
        for slot in slots:
            task_id = slot["id_task"]
            if task_id is None or slot["prompt_total"] <= 0:
                continue
            p = prev_slot["slots"].get(slot["id"])
            if p is None or p.get("id_task") != task_id:
                prompt_last = {
                    "total": slot["prompt_total"],
                    "cached": slot["cached"],
                    "pct_cached": round(
                        100.0 * slot["cached"] / slot["prompt_total"], 1
                    ),
                }
    if prompt_last is not None:
        out["prompt_last"] = prompt_last

    # Per-request prompt rate (see LLM_REQ_RATE_WINDOW): a task completes
    # when (a) processed + cached tokens reach the task's prompt size
    # (n_prompt_tokens = prompt + cache), or (b) the slot goes idle
    # (is_processing false) while still holding the task — the universal
    # fallback, needed because some builds (LM Studio's llama.cpp) never
    # advance n_prompt_tokens_processed. The rate is prompt size /
    # (first-sight .. completion) wall time; prefills shorter than
    # _LLAMA_MIN_PREFILL_S finish between scrapes and are skipped.
    new_slot_state: dict[Any, dict[str, Any]] = {}
    for s in slots:
        p = prev_slot["slots"].get(s["id"]) if prev_slot is not None else None
        if p is not None and p.get("id_task") == s["id_task"]:
            entry = {"task_ts": p.get("task_ts", now),
                     "prefill_done": p.get("prefill_done", False)}
        else:
            # Already finished (or generation already running, or the slot
            # is idle showing an old task) at first sight: cannot be
            # timed, no sample.
            entry = {"task_ts": now, "prefill_done":
                     not s["is_processing"]
                     or s["decoded"] > 0
                     or (s["prompt_total"] > 0
                         and s["prompt_processed"] + s["cached"]
                         >= s["prompt_total"])}
        new_slot_state[s["id"]] = entry
    if prev_slot is not None:
        for s in slots:
            p = prev_slot["slots"].get(s["id"])
            if p is None or p.get("id_task") != s["id_task"]:
                continue  # new task (or first sight): no rate this scrape
            if p.get("prefill_done"):
                continue
            total = s["prompt_total"]
            if total <= 0:
                continue
            # Prefill complete when generation has started (n_decoded > 0)
            # or the slot went idle with the task still held. The
            # processed/cached counters must NOT be used here: LM Studio's
            # build sets n_prompt_tokens_cache one scrape after task start
            # and streams n_prompt_tokens live during prefill, so
            # processed+cached >= total fires ~1 s after start and the
            # sample degenerates to prompt-size / 1 s (verified against a
            # live /slots trace on 100.68.63.37).
            if s["decoded"] <= 0 and s["is_processing"]:
                continue  # prefill still in flight
            task_ts = p.get("task_ts")
            if task_ts is not None:
                duration = now - task_ts
                # Sample the UNCACHED prompt rate (n_prompt_tokens_processed
                # / wall time): this matches the speed llama-server itself
                # logs ("prompt eval time = X ms / Y tokens"). Using the
                # prompt total would count KV-cache hits as prefill work
                # and inflate the rate to ~prompt size on mostly-cached
                # requests (agentic workloads). Fully-cached requests do
                # no prefill work: no sample.
                if (
                    duration >= _LLAMA_MIN_PREFILL_S
                    and s["prompt_processed"] > 0
                ):
                    rates.append(
                        s["prompt_processed"] / duration
                    )
            new_slot_state[s["id"]]["prefill_done"] = True

    _llama_slot_prev[state_key] = {
        "ts": now,
        "slots": {
            s["id"]: {
                "decoded": s["decoded"],
                "prompt": s["prompt_processed"],
                "cached": s["cached"],
                "id_task": s["id_task"],
                **new_slot_state[s["id"]],
            }
            for s in slots
        },
        "prompt_last": prompt_last,
    }

    # Gauge value: mean of the last 5 completed-request prompt rates when
    # available; otherwise the per-scrape live rate computed above.
    if rates:
        out["prompt_tokens_per_s"] = round(
            sum(rates) / len(rates), 1
        )

    _llama_collector.record_good(state_key, out)
    return out


# ---------------------------------------------------------------------------
# vLLM Monitoring
# ---------------------------------------------------------------------------

# Stale-hold: last successful vLLM scrape (monotonic ts + dict).
_VLLM_STALE_MAX_AGE_S = 15.0

# Rate window (all LLM backends): the gap between two successful
# scrapes must be within 0.5 s (avoid dividing tiny deltas by tiny
# times) and 15 s (== the stale-hold window). A longer gap means the
# scrape was down; the LiteLLM collector holds the last computed rate
# instead of omitting it, so the UI never flashes "–" on recovery.
_RATE_MAX_GAP_S = 15.0

_VLLM_STALE_MAX_AGE_S = 15.0
_vllm_collector = BaseLLMCollector(_VLLM_STALE_MAX_AGE_S)
# Backward-compatible aliases (tests reference these names directly).
_vllm_prev = _vllm_collector._prev
_vllm_last_good = _vllm_collector._last_good
_vllm_req_rates = _vllm_collector._req_rates

# Previous successful scrape's cumulative counters + histogram sums, used
# to compute rates and windowed means from per-scrape deltas.

# 30 s cache for the parsed cache_config_info dict (static during runtime).
_VLLM_CONFIG_TTL = 30.0
_vllm_config_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# Warn-once guard for LLM auto mode: the first time vLLM comes up, tell the
# operator it is the active backend and how to pin the choice.
# Approximation (accepted): we do NOT probe the llama server just to verify
# it is also reachable, so this may fire when only vLLM is up.
_llm_tie_warned = False


def _parse_prometheus_labeled(text: str, prefix: str) -> dict[str, dict]:
    """Parse Prometheus text into
    ``{metric_name: {"series": {labelset: float}, "labels": {name: set(values)}}}``.

    Keeps only series with the given *prefix* (e.g. ``"vllm:"``) plus the
    companion unprefixed ``process_start_time_seconds``. Labeled and
    unlabeled series are both captured. ``# HELP``/``# TYPE`` comment
    lines and malformed lines are skipped. Never raises.
    """
    result: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Determine metric name, labels string, and value
        if "{" in line:
            brace_start = line.index("{")
            metric_name = line[:brace_start].strip()
            brace_end = line.index("}")
            labels_str = line[brace_start + 1:brace_end]
            rest = line[brace_end + 1:].strip()
        else:
            parts0 = line.split()
            if not parts0:
                continue
            metric_name = parts0[0]
            labels_str = ""
            rest = " ".join(parts0[1:])

        # Check prefix or companion allowlist
        if not metric_name.startswith(prefix) and metric_name != "process_start_time_seconds":
            continue

        # Parse value
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except (ValueError, IndexError):
            continue

        # Parse labels into a sorted tuple of (key, value) pairs
        if labels_str:
            label_pairs: list[tuple[str, str]] = []
            for pair in labels_str.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"')
                label_pairs.append((k, v))
            label_pairs.sort()
            labelset = tuple(label_pairs)
            label_names: dict[str, set[str]] = {}
            for k, v in label_pairs:
                label_names.setdefault(k, set()).add(v)
        else:
            labelset = ()
            label_names = {}

        # Accumulate into result
        if metric_name not in result:
            result[metric_name] = {"series": {}, "labels": {}}
        result[metric_name]["series"][labelset] = value
        for k, vals in label_names.items():
            if k in result[metric_name]["labels"]:
                result[metric_name]["labels"][k].update(vals)
            else:
                result[metric_name]["labels"][k] = vals

    return result


def _vllm_get_entry(metrics: dict[str, dict], *candidates: str) -> dict[str, dict] | None:
    """Return the first matching entry by full metric name (with ``vllm:`` prefix)."""
    for name in candidates:
        if name in metrics:
            return metrics[name]
    return None


def _vllm_sum(entry: dict[str, dict] | None) -> float | None:
    """Sum all series values for a metric entry. None if absent or empty."""
    if entry is None or not entry.get("series"):
        return None
    return sum(entry["series"].values())


def _vllm_max(entry: dict[str, dict] | None) -> float | None:
    """Max of all series values for a metric entry. None if absent or empty."""
    if entry is None or not entry.get("series"):
        return None
    return max(entry["series"].values())


def _parse_vllm_config(entry: dict[str, dict]) -> dict[str, Any]:
    """Parse the ``cache_config_info`` label dict into a config summary.

    The row with value 1.0 carries all config as labels. Values that are
    the literal string ``"None"`` or cannot be converted are omitted.
    Never raises.
    """
    cfg_labels: dict[str, str] = {}
    for labelset, val in entry.get("series", {}).items():
        if val == 1.0:
            cfg_labels = dict(labelset)
            break

    if not cfg_labels:
        return {}

    config: dict[str, Any] = {}
    for f in ("kv_cache_size_tokens", "num_gpu_blocks", "block_size"):
        v = cfg_labels.get(f)
        if v is not None:
            try:
                config[f] = int(v)
            except (ValueError, TypeError):
                pass

    v = cfg_labels.get("gpu_memory_utilization")
    if v is not None:
        try:
            config["gpu_memory_utilization"] = float(v)
        except (ValueError, TypeError):
            pass

    for f in ("cache_dtype", "enable_prefix_caching"):
        v = cfg_labels.get(f)
        if v is not None and v != "None":
            config[f] = v

    return config


def collect_vllm_metrics(
    base_url: str, state_key: str = "primary"
) -> dict[str, Any] | None:
    """Scrape vLLM ``/metrics`` and build the dashboard snapshot.

    ``state_key`` names the scrape target; all per-scrape state is kept
    per key so several vLLM servers scrape independently.

    Returns ``None`` when the server is unreachable, or the body contains
    no ``vllm:`` series. Never raises.

    On a failed scrape the last successful snapshot is returned with
    ``status: "stale"`` for up to ``_VLLM_STALE_MAX_AGE_S`` seconds so a
    transient timeout doesn't flicker the UI to "unreachable".

    Counter semantics (confirmed against vLLM docs/source): prompt_tokens_cached_total
    is a SUBSET of prompt_tokens_total (nested, not disjoint). Usage total is
    therefore ``prompt_total + gen_total`` (NOT + cached).
    """

    def _stale_or_none() -> dict[str, Any] | None:
        held_good = _vllm_collector._last_good.get(state_key)
        if held_good is not None:
            ts, held = held_good
            if time.monotonic() - ts <= _VLLM_STALE_MAX_AGE_S:
                out = dict(held)
                out["status"] = "stale"
                return out
        return None

    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/metrics", timeout=2
        ) as r:
            body = r.read().decode()
    except Exception:  # noqa: BLE001 – collector must never raise
        return _stale_or_none()

    # Deliberate deviation from the strict R2 wording: when /metrics is
    # reachable but carries no vllm: series, return the <=15 s stale hold
    # instead of strict None (mirrors the llama convention; better hysteresis).
    metrics = _parse_prometheus_labeled(body, "vllm:")
    if not any(k.startswith("vllm:") for k in metrics):
        return _stale_or_none()  # endpoint up but no vllm: metrics

    out: dict[str, Any] = {"status": "up"}
    now = time.monotonic()
    prev = _vllm_collector.get_prev(state_key)
    rates = _vllm_collector.get_rates(state_key)
    elapsed = now - prev["ts"] if prev is not None else 0.0

    # --- Model name (from any series' model_name label; extracted every cycle, no TTL cache) ---
    model_name = None
    for entry in metrics.values():
        if "model_name" in entry.get("labels", {}):
            vals = entry["labels"]["model_name"]
            if vals:
                model_name = next(iter(vals))
                break
    if model_name:
        out["model"] = model_name

    # --- Gauges ---
    # requests_processing (sum across engines)
    val = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:num_requests_running"))
    if val is not None:
        out["requests_processing"] = round(val, 1)

    # requests_waiting (sum across engines)
    val = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:num_requests_waiting"))
    if val is not None:
        out["requests_waiting"] = round(val, 1)

    # kv_cache_usage_pct (max across engines; 0..1 → ×100; legacy: gpu_cache_usage_perc)
    val = _vllm_max(_vllm_get_entry(metrics,
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc"))
    if val is not None:
        out["kv_cache_usage_pct"] = round(val * 100, 1)

    # --- Engine sleep state (one-hot label parse) ---
    sleep_entry = _vllm_get_entry(metrics, "vllm:engine_sleep_state")
    if sleep_entry is not None and sleep_entry.get("series"):
        sleeping = False
        for labelset, v in sleep_entry["series"].items():
            labels = dict(labelset)
            ss = labels.get("sleep_state", "")
            if ss in ("weights_offloaded", "discard_all") and v == 1.0:
                sleeping = True
                break
            if ss == "awake" and v == 1.0:
                sleeping = False
                break
        out["sleeping"] = sleeping

    # --- Counters (cumulative) ---
    gen_total = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:generation_tokens_total",
        "vllm:num_generation_tokens"))
    prompt_total = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:prompt_tokens_total",
        "vllm:num_prompt_tokens"))
    cached_total = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:prompt_tokens_cached_total"))
    preemptions = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:num_preemptions_total"))
    req_success = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:request_success_total"))
    spec_draft = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:spec_decode_num_draft_tokens_total"))
    spec_accepted = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:spec_decode_num_accepted_tokens_total"))

    if gen_total is not None:
        out["gen_tokens_total"] = round(gen_total, 1)
    if prompt_total is not None:
        out["prompt_tokens_total"] = round(prompt_total, 1)
    if cached_total is not None:
        out["prompt_tokens_cached_total"] = round(cached_total, 1)
    if preemptions is not None:
        out["preemptions_total"] = round(preemptions, 1)
    if req_success is not None:
        out["requests_completed_total"] = round(req_success, 1)
    if spec_draft is not None:
        out["spec_decode_draft_tokens_total"] = round(spec_draft, 1)
    if spec_accepted is not None:
        out["spec_decode_accepted_tokens_total"] = round(spec_accepted, 1)

    # --- Rates (Δ counters / elapsed; window guards) ---
    rate_ok = prev is not None and 0.5 <= elapsed <= _RATE_MAX_GAP_S
    # Each rate is computed only when the previous scrape has that counter
    # (a version rename mid-flight may lack it): omit the rate key otherwise.
    if rate_ok and gen_total is not None and prev.get("gen") is not None:
        if gen_total < prev["gen"]:
            out["gen_tokens_per_s"] = 0.0  # counter reset
        else:
            out["gen_tokens_per_s"] = round(
                max(0.0, gen_total - prev["gen"]) / elapsed, 1)
    if rate_ok and prompt_total is not None and prev.get("prompt") is not None:
        if prompt_total < prev["prompt"]:
            out["prompt_tokens_per_s"] = 0.0  # counter reset
            rates.clear()  # restart: history belongs to old proc
        else:
            out["prompt_tokens_per_s"] = round(
                max(0.0, prompt_total - prev["prompt"]) / elapsed, 1)

    # --- Prefix cache hit rate (lifetime ratio; omit when queries == 0) ---
    pc_queries = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries_total"))
    pc_hits = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits_total"))
    if pc_queries is not None and pc_hits is not None and pc_queries > 0:
        out["prefix_cache_hit_pct"] = round(100.0 * pc_hits / pc_queries, 1)

    # --- Spec decode accept rate (lifetime ratio; omit when draft == 0) ---
    if spec_draft is not None and spec_accepted is not None and spec_draft > 0:
        out["spec_accept_pct"] = round(100.0 * spec_accepted / spec_draft, 1)

    # --- Histograms ---
    # TTFT: windowed mean (Δsum/Δcount); omit when Δcount ≤ 0; NO lifetime fallback
    ttft_sum = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:time_to_first_token_seconds_sum"))
    ttft_count = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:time_to_first_token_seconds_count"))
    if rate_ok and ttft_sum is not None and ttft_count is not None:
        d_count = ttft_count - prev.get("ttft_count", 0)  # type: ignore[union-attr]
        if d_count > 0:
            d_sum = max(0.0, ttft_sum - prev.get("ttft_sum", 0))  # type: ignore[union-attr]
            out["ttft_seconds"] = round(d_sum / d_count, 1)

    # ITL: windowed mean × 1000; same omit rule (legacy: time_per_output_token_seconds)
    itl_sum = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:inter_token_latency_seconds_sum",
        "vllm:time_per_output_token_seconds_sum"))
    itl_count = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:inter_token_latency_seconds_count",
        "vllm:time_per_output_token_seconds_count"))
    if rate_ok and itl_sum is not None and itl_count is not None:
        d_count = itl_count - prev.get("itl_count", 0)  # type: ignore[union-attr]
        if d_count > 0:
            d_sum = max(0.0, itl_sum - prev.get("itl_sum", 0))  # type: ignore[union-attr]
            out["itl_ms"] = round(d_sum / d_count * 1000, 1)

    # E2E latency: lifetime mean (sum/count); summary stat
    e2e_sum = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:e2e_request_latency_seconds_sum"))
    e2e_count = _vllm_sum(_vllm_get_entry(metrics,
        "vllm:e2e_request_latency_seconds_count"))
    if e2e_sum is not None and e2e_count is not None and e2e_count > 0:
        out["e2e_latency_seconds"] = round(e2e_sum / e2e_count, 1)

    # --- Prompt rate per completed request (see LLM_REQ_RATE_WINDOW) ---
    # vLLM exposes no per-request prompt size/duration, so each scrape in
    # which requests completed contributes one window sample:
    # Δprompt_tokens / ΣTTFT (TTFT ≈ prefill + queue time). The gauge
    # reports the mean of the last 5 samples, falling back to the raw
    # per-scrape delta until the first completed request is observed.
    if (rate_ok and req_success is not None
            and prev.get("req_success") is not None
            and ttft_sum is not None and prev.get("ttft_sum") is not None
            and prompt_total is not None and prev.get("prompt") is not None):
        d_req = req_success - prev["req_success"]
        if d_req > 0:
            d_ttft = max(0.0, ttft_sum - prev["ttft_sum"])
            if d_ttft > 0:
                # Uncached tokens only (cached is a SUBSET of prompt
                # total): mirrors the llama per-request semantics.
                d_prompt = max(0.0, prompt_total - prev["prompt"])
                if (cached_total is not None
                        and prev.get("cached") is not None):
                    d_prompt = max(
                        0.0, d_prompt - (cached_total - prev["cached"])
                    )
                if d_prompt > 0:
                    rates.append(d_prompt / d_ttft)
    if rates:
        out["prompt_tokens_per_s"] = round(
            sum(rates) / len(rates), 1
        )

    # --- Config (30 s cache) ---
    cfg_entry = _vllm_get_entry(metrics, "vllm:cache_config_info")
    if cfg_entry is not None and cfg_entry.get("series"):
        now_ts = time.monotonic()
        cache = _vllm_config_cache.get(state_key)
        if cache is not None:
            cache_ts, cached_cfg = cache
            if now_ts - cache_ts < _VLLM_CONFIG_TTL:
                out["config"] = cached_cfg
            else:
                out["config"] = _parse_vllm_config(cfg_entry)
                _vllm_config_cache[state_key] = (now_ts, out["config"])
        else:
            out["config"] = _parse_vllm_config(cfg_entry)
            _vllm_config_cache[state_key] = (now_ts, out["config"])

    # --- Usage (derived; vLLM counters are NESTED: cached ⊂ prompt) ---
    if gen_total is not None or prompt_total is not None:
        usage: dict[str, Any] = {}
        total = 0
        if prompt_total is not None:
            total += int(round(prompt_total))
        if gen_total is not None:
            total += int(round(gen_total))
        usage["total_tokens"] = total
        if gen_total is not None:
            usage["gen_tokens"] = int(round(gen_total))
        # gen_avg_tps omitted for vLLM (no gen-seconds counter)
        if (cached_total is not None and prompt_total is not None
                and prompt_total > 0):
            usage["prompt_cached_pct"] = round(
                100.0 * cached_total / prompt_total, 1)
        # uptime from process_start_time_seconds (cross-host clock, clamp ≥ 0)
        proc_start = metrics.get("process_start_time_seconds")
        if proc_start is not None and proc_start.get("series"):
            start_val = next(iter(proc_start["series"].values()))
            usage["uptime_s"] = max(0, int(time.time() - start_val))
        out["usage"] = usage

    # --- Update previous state ---
    new_prev: dict[str, float] = {"ts": now}
    if gen_total is not None:
        new_prev["gen"] = gen_total
    if prompt_total is not None:
        new_prev["prompt"] = prompt_total
    if cached_total is not None:
        new_prev["cached"] = cached_total
    if ttft_sum is not None:
        new_prev["ttft_sum"] = ttft_sum
    if ttft_count is not None:
        new_prev["ttft_count"] = ttft_count
    if req_success is not None:
        new_prev["req_success"] = req_success
    if itl_sum is not None:
        new_prev["itl_sum"] = itl_sum
    if itl_count is not None:
        new_prev["itl_count"] = itl_count
    _vllm_collector.set_prev(state_key, new_prev)

    _vllm_collector.record_good(state_key, out)
    return out


# ---------------------------------------------------------------------------
# LiteLLM Monitoring
# ---------------------------------------------------------------------------

# Stale-hold: last successful LiteLLM scrape (monotonic ts + dict).
_LITELLM_STALE_MAX_AGE_S = 15.0
_litellm_collector = BaseLLMCollector(_LITELLM_STALE_MAX_AGE_S)
# Backward-compatible aliases (tests reference these names directly).
_litellm_prev = _litellm_collector._prev
_litellm_last_good = _litellm_collector._last_good

# Previous successful scrape's cumulative counters + histogram sums, used
# to compute rates and windowed means from per-scrape deltas.

# Warn-once guard for LLM auto mode: the first time LiteLLM wins, tell the
# operator it is the active backend and how to pin the choice.
_litellm_tie_warned = False


def collect_litellm_metrics(
    base_url: str,
    token: str | None = None,
    state_key: str = "primary",
) -> dict[str, Any] | None:
    """Scrape a LiteLLM proxy ``/metrics`` and build the dashboard snapshot.

    Returns ``None`` when the server is unreachable, or the body carries
    no ``litellm_`` series. Never raises. Mirrors the vLLM collector
    conventions: 15 s stale-hold on failure, Δ-counter rates with a
    0.5–5 s scrape window, windowed means for histograms.

    ``token`` adds an ``Authorization: Bearer`` header when the proxy
    requires auth for its metrics endpoint
    (``require_auth_for_metrics_endpoint``).

    Metric names verified against the LiteLLM proxy prometheus callback
    (docs.litellm.ai/docs/proxy/prometheus). Counters are exposed with the
    prometheus_client ``_total`` suffix; the bare name is tried as a
    fallback for custom/older builds. Token counters are DISJOINT
    (input + output), unlike vLLM's nested cached ⊂ prompt accounting.

    ``gen_tokens_per_s_by_model``: per-model gen tokens/s grouped by the
    client-facing alias (requested_model → litellm_model_name → model);
    multiple deployments of one model are summed. Used by the UI's
    per-model chart lines (shown only when 2+ models are present).

    ``state_key`` names the scrape target; per-scrape state is keyed.
    """

    def _stale_or_none() -> dict[str, Any] | None:
        held_good = _litellm_collector._last_good.get(state_key)
        if held_good is not None:
            ts, held = held_good
            if time.monotonic() - ts <= _LITELLM_STALE_MAX_AGE_S:
                out = dict(held)
                out["status"] = "stale"
                return out
        return None

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/metrics", headers=headers)
        with urllib.request.urlopen(req, timeout=2) as r:
            body = r.read().decode()
    except Exception:  # noqa: BLE001 – collector must never raise
        return _stale_or_none()

    # Endpoint up but no litellm_ series (e.g. a different service, or the
    # prometheus callback not configured): keep the <=15 s stale hold.
    metrics = _parse_prometheus_labeled(body, "litellm_")
    if not any(k.startswith("litellm_") for k in metrics):
        return _stale_or_none()

    out: dict[str, Any] = {"status": "up"}
    now = time.monotonic()
    prev = _litellm_collector.get_prev(state_key)
    elapsed = now - prev["ts"] if prev is not None else 0.0

    # --- Model name: requested_model is the client-facing alias; the
    # literal "None" label means unresolved in LiteLLM — skip it. ---
    model_name = None
    for label in ("requested_model", "litellm_model_name", "model"):
        vals: set[str] | None = None
        for entry in metrics.values():
            v = entry.get("labels", {}).get(label)
            if v:
                vals = v if vals is None else vals | v
        if vals:
            candidates = sorted(v for v in vals if v and v != "None")
            if candidates:
                model_name = candidates[0]
                break
    if model_name:
        out["model"] = model_name

    # --- Gauges ---
    # requests in flight across workers (pod-queue-depth gauge)
    val = _vllm_sum(_vllm_get_entry(metrics, "litellm_in_flight_requests"))
    if val is not None:
        out["requests_processing"] = round(val, 1)

    # --- Counters (cumulative; input/output are DISJOINT) ---
    gen_total = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_output_tokens_metric_total",
        "litellm_output_tokens_metric"))
    prompt_total = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_input_tokens_metric_total",
        "litellm_input_tokens_metric"))
    cached_total = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_input_cached_tokens_metric_total",
        "litellm_input_cached_tokens_metric"))
    req_total = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_proxy_total_requests_metric_total",
        "litellm_proxy_total_requests_metric"))
    req_failed = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_proxy_failed_requests_metric_total",
        "litellm_proxy_failed_requests_metric"))
    spend = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_spend_metric_total",
        "litellm_spend_metric"))

    if gen_total is not None:
        out["gen_tokens_total"] = round(gen_total, 1)
    if prompt_total is not None:
        out["prompt_tokens_total"] = round(prompt_total, 1)
    if req_total is not None:
        out["requests_completed_total"] = round(req_total, 1)
    if req_failed is not None:
        out["requests_failed_total"] = round(req_failed, 1)
    if spend is not None:
        out["spend_usd"] = round(spend, 4)

    # --- Per-model gen totals (grouped by client-facing alias) ---
    # Grouped by the client-facing alias (requested_model →
    # litellm_model_name → model); several deployments of the same model
    # (different model_id) are summed. The literal "None" label means
    # unresolved in LiteLLM — skip it.
    gen_by_model: dict[str, float] = {}
    entry = _vllm_get_entry(metrics,
        "litellm_output_tokens_metric_total",
        "litellm_output_tokens_metric")
    if entry is not None:
        for labelset, val in entry["series"].items():
            labels = dict(labelset)
            name = None
            for lab in ("requested_model", "litellm_model_name", "model"):
                v = labels.get(lab)
                if v and v != "None":
                    name = v
                    break
            if name is not None:
                gen_by_model[name] = gen_by_model.get(name, 0.0) + val

    # --- Rates (Δ counters / elapsed; window guards) ---
    # The gap between two successful scrapes must be within
    # 0.5–_RATE_MAX_GAP_S (== the 15 s stale-hold window). A longer
    # gap means the scrape was down (e.g. right after an outage): the
    # last computed rate is held instead of omitted, so the UI never
    # flashes "–" on recovery.
    rate_ok = prev is not None and 0.5 <= elapsed <= _RATE_MAX_GAP_S
    if rate_ok:
        gen_rate: float | None = None
        prompt_rate: float | None = None
        if gen_total is not None and prev.get("gen") is not None:
            if gen_total < prev["gen"]:
                gen_rate = 0.0  # counter reset
            else:
                gen_rate = round(
                    max(0.0, gen_total - prev["gen"]) / elapsed, 1)
        if prompt_total is not None and prev.get("prompt") is not None:
            if prompt_total < prev["prompt"]:
                prompt_rate = 0.0  # counter reset
            else:
                prompt_rate = round(
                    max(0.0, prompt_total - prev["prompt"]) / elapsed, 1)
        prev_by_model = prev.get("gen_by_model") or {}
        per_model_rates: dict[str, float] = {}
        for name, total in gen_by_model.items():
            p = prev_by_model.get(name)
            if p is None:
                continue  # new model: no baseline yet
            if total < p:
                per_model_rates[name] = 0.0  # counter reset
            else:
                per_model_rates[name] = round(
                    max(0.0, total - p) / elapsed, 1)
        last_rate: dict[str, Any] = {
            "gen": gen_rate, "prompt": prompt_rate,
            "per_model": per_model_rates or None,
        }
    else:
        last_rate = (prev or {}).get("last_rate")

    if last_rate is not None:
        if last_rate["gen"] is not None:
            out["gen_tokens_per_s"] = last_rate["gen"]
        if last_rate["prompt"] is not None:
            out["prompt_tokens_per_s"] = last_rate["prompt"]
        if last_rate.get("per_model"):
            out["gen_tokens_per_s_by_model"] = last_rate["per_model"]

    # --- Prefix (provider-side prompt) cache hit rate (lifetime ratio) ---
    if (cached_total is not None and prompt_total is not None
            and prompt_total > 0):
        out["prefix_cache_hit_pct"] = round(
            100.0 * cached_total / prompt_total, 1)

    # --- Latency ---
    # TTFT: windowed mean (Δsum/Δcount); streaming requests only; omit when
    # Δcount <= 0 (no streaming traffic in the window).
    ttft_sum = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_llm_api_time_to_first_token_metric_sum"))
    ttft_count = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_llm_api_time_to_first_token_metric_count"))
    if rate_ok and ttft_sum is not None and ttft_count is not None:
        d_count = ttft_count - prev.get("ttft_count", 0)  # type: ignore[union-attr]
        if d_count > 0:
            d_sum = max(0.0, ttft_sum - prev.get("ttft_sum", 0))  # type: ignore[union-attr]
            out["ttft_seconds"] = round(d_sum / d_count, 1)

    # ITL: per-output-token latency gauge (seconds) → ms; max across
    # deployments/keys.
    val = _vllm_max(_vllm_get_entry(metrics,
        "litellm_deployment_latency_per_output_token"))
    if val is not None:
        out["itl_ms"] = round(val * 1000, 1)

    # E2E (proxy arrival → end of processing): lifetime mean (sum/count).
    e2e_sum = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_request_total_latency_metric_sum"))
    e2e_count = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_request_total_latency_metric_count"))
    if e2e_sum is not None and e2e_count is not None and e2e_count > 0:
        out["e2e_latency_seconds"] = round(e2e_sum / e2e_count, 1)

    # Upstream LLM API call latency: lifetime mean (excludes proxy auth +
    # queue overhead).
    api_sum = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_llm_api_latency_metric_sum"))
    api_count = _vllm_sum(_vllm_get_entry(metrics,
        "litellm_llm_api_latency_metric_count"))
    if api_sum is not None and api_count is not None and api_count > 0:
        out["llm_api_latency_seconds"] = round(api_sum / api_count, 1)

    # --- Usage (derived) ---
    if gen_total is not None or prompt_total is not None:
        usage: dict[str, Any] = {}
        total = 0
        if prompt_total is not None:
            total += int(round(prompt_total))
        if gen_total is not None:
            total += int(round(gen_total))
        usage["total_tokens"] = total
        if gen_total is not None:
            usage["gen_tokens"] = int(round(gen_total))
        if (cached_total is not None and prompt_total is not None
                and prompt_total > 0):
            usage["prompt_cached_pct"] = round(
                100.0 * cached_total / prompt_total, 1)
        if spend is not None:
            usage["total_spend_usd"] = round(spend, 4)
        # uptime from process_start_time_seconds (cross-host clock, clamp ≥ 0)
        proc_start = metrics.get("process_start_time_seconds")
        if proc_start is not None and proc_start.get("series"):
            start_val = next(iter(proc_start["series"].values()))
            usage["uptime_s"] = max(0, int(time.time() - start_val))
        out["usage"] = usage

    # --- Update previous state ---
    new_prev: dict[str, Any] = {"ts": now}
    for key, val in (("gen", gen_total), ("prompt", prompt_total),
                     ("ttft_sum", ttft_sum), ("ttft_count", ttft_count),
                     ("e2e_sum", e2e_sum), ("e2e_count", e2e_count)):
        if val is not None:
            new_prev[key] = val
    if gen_by_model:
        new_prev["gen_by_model"] = gen_by_model
    if last_rate is not None:
        new_prev["last_rate"] = last_rate
    _litellm_collector.set_prev(state_key, new_prev)

    _litellm_collector.record_good(state_key, out)
    return out


# ---------------------------------------------------------------------------
# Strata Monitoring (https://github.com/Niko1221/Strata)
# ---------------------------------------------------------------------------

# Stale-hold: last successful Strata scrape (monotonic ts + dict).
_STRATA_STALE_MAX_AGE_S = 15.0
_strata_collector = BaseLLMCollector(_STRATA_STALE_MAX_AGE_S)
# Backward-compatible aliases (tests reference these names directly).
_strata_prev = _strata_collector._prev
_strata_last_good = _strata_collector._last_good

# Previous successful scrape's cumulative totals + request-ring watermark,
# used to compute rates and windowed means from deltas.

# Warn-once guard for LLM auto mode: the first time Strata wins, tell the
# operator it is the active backend and how to pin the choice.
_strata_tie_warned = False


def _strata_num(val: Any) -> float | None:
    """Coerce a Strata JSON number (or None) to float; None otherwise."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _parse_strata_slots(data: Any,
                        n_ctx: float | None) -> list[dict[str, Any]]:
    """Normalise Strata's ``live.slots`` into the dashboard slot shape.

    A Strata slot is ``{slot, state: "idle"|"reading"|"decoding",
    held_tokens}`` when idle (holding the KV prefix for a next turn) or
    ``{slot, state, prompt_tokens, generated, elapsed_s, tok_s}`` while
    active. The UI slot renderer consumes the llama.cpp fields
    (id, context_used, context_pct, n_ctx, is_processing, decoded,
    prompt_total).
    """
    slots: list[dict[str, Any]] = []
    if not isinstance(data, list):
        return slots
    for entry in data:
        if not isinstance(entry, dict):
            continue
        state_str = str(entry.get("state") or "idle")
        prompt = _strata_num(entry.get("prompt_tokens")) or 0.0
        generated = _strata_num(entry.get("generated")) or 0.0
        elapsed_s = _strata_num(entry.get("elapsed_s")) or 0.0
        held = _strata_num(entry.get("held_tokens")) or 0.0
        # Cheap guard: a slot is processing if its state string is not
        # idle, OR it has generated tokens, OR time has elapsed (catches
        # short requests whose state string lags the actual activity).
        active = (state_str != "idle") or (generated > 0) or (elapsed_s > 0)
        used = (prompt + generated) if active else held
        slot: dict[str, Any] = {
            "id": entry.get("slot", len(slots)),
            "context_used": int(used),
            "n_ctx": int(n_ctx) if n_ctx else 0,
            "context_pct": round(100.0 * used / n_ctx, 1) if n_ctx else 0,
            "is_processing": active,
            # Idle slots render as "holding N tok" from prompt_total.
            "prompt_total": int(prompt or held),
        }
        if active:
            slot["decoded"] = int(generated)
        slots.append(slot)
    return slots


def collect_strata_metrics(
    base_url: str, state_key: str = "primary"
) -> dict[str, Any] | None:
    """Scrape a Strata inference engine ``/metrics`` and build the snapshot.

    Strata serves a JSON document (not Prometheus text): ``engine``
    (model/version/context), ``live`` (state, queue, batch slots, live
    tok/s), ``requests`` (last finished requests, newest first), and
    ``totals`` (cumulative counters since server start). A reachable
    endpoint without the ``engine``/``live`` shape (a different service,
    or non-JSON) keeps the <=15 s stale hold, mirroring the
    vLLM/LiteLLM collectors. Never raises.

    Counter semantics: ``totals.reused`` is a SUBSET of
    ``totals.prompt_tokens`` (nested KV prefix reuse, vLLM-style
    accounting), so the usage total is ``prompt_total + gen_total`` (NOT
    + reused). Restart resets the counters -> one 0.0 rate tick.

    Latency lines come from the request ring buffer: requests whose
    ``time`` (epoch s) exceeds the previous scrape's watermark contribute
    windowed means (TTFT = prompt_ms, E2E = duration_s, ITL ~
    1000/decode_tok_s). Omitted on the first scrape.

    ``state_key`` names the scrape target; per-scrape state is keyed.
    """

    def _stale_or_none() -> dict[str, Any] | None:
        held_good = _strata_collector._last_good.get(state_key)
        if held_good is not None:
            ts, held = held_good
            if time.monotonic() - ts <= _STRATA_STALE_MAX_AGE_S:
                out = dict(held)
                out["status"] = "stale"
                return out
        return None

    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/metrics", timeout=2
        ) as r:
            body = r.read().decode()
        data = json.loads(body)
    except Exception:  # noqa: BLE001 – collector must never raise
        return _stale_or_none()
    if (not isinstance(data, dict)
            or not isinstance(data.get("engine"), dict)
            or not isinstance(data.get("live"), dict)):
        return _stale_or_none()  # endpoint up but not a Strata server

    engine = data["engine"]
    live = data["live"]
    totals = (data["totals"] if isinstance(data.get("totals"), dict)
              else {})

    out: dict[str, Any] = {"status": "up"}
    now = time.monotonic()
    prev = _strata_collector.get_prev(state_key)
    elapsed = now - prev["ts"] if prev is not None else 0.0

    # --- Model + engine facts ---
    if engine.get("model"):
        out["model"] = engine["model"]
    n_ctx = (_strata_num(engine.get("context"))
             or _strata_num(engine.get("max_context")))
    config: dict[str, Any] = {}
    for key in ("version", "kv", "images"):
        if engine.get(key) is not None:
            config[key] = engine[key]
    if n_ctx:
        config["context"] = int(n_ctx)

    # --- Live gauges ---
    state = str(live.get("state") or "idle")
    out["state"] = state
    parallel = _strata_num(live.get("parallel"))
    running = _strata_num(live.get("running"))
    if running is None:
        running = 1.0 if state in ("reading", "generating") else 0.0
    out["requests_processing"] = round(running, 1)
    queued = _strata_num(live.get("queued")) or 0.0
    waiting = _strata_num(live.get("waiting")) or 0.0
    out["requests_waiting"] = round(queued + waiting, 1)
    tok_s = _strata_num(live.get("tok_s"))
    if tok_s is None:
        tok_s = _strata_num(live.get("tok_s_mean"))
    if tok_s is not None:
        out["live_tok_s"] = round(tok_s, 1)
    if parallel is not None and parallel >= 1:
        out["free_slots"] = max(0, int(parallel) - int(running))
        out["total_slots"] = int(parallel)
        config["parallel"] = int(parallel)
    slots = _parse_strata_slots(live.get("slots"), n_ctx)
    # Always emit the slot list (even empty) so the empty case is
    # explicit / diagnosable (a silently-dropped empty list hid "no
    # slots" from the UI).
    out["slots"] = slots

    # --- Slot diagnosability (T6): busy count from the cheap guard,
    # engine/slot cross-check, last-busy timestamp, last-seen state. ---
    slots_busy = sum(1 for s in slots if s.get("is_processing"))
    any_active = slots_busy > 0
    # Cross-check: slot-level activity vs engine-level running count.
    out["slots_busy"] = slots_busy
    out["slots_state_mismatch"] = any_active != (running > 0)
    # last_busy_ts: epoch ts of the last time a slot was observed busy
    # (persists across scrapes via _strata_prev; a diagnostic for "idle
    # while generating" — the engine is idle now but a slot was busy at
    # this ts).
    prev_last_busy_ts = prev.get("last_busy_ts") if prev else None
    last_busy_ts = time.time() if any_active else prev_last_busy_ts
    if last_busy_ts is not None:
        out["last_busy_ts"] = last_busy_ts
    # state_seen: the last observed NON-IDLE raw state (persists across
    # scrapes); a diagnostic for "idle while generating" (the state was
    # seen generating but the current scrape is idle).
    prev_state_seen = prev.get("state_seen") if prev else None
    state_seen = state if state != "idle" else prev_state_seen
    if state_seen is not None:
        out["state_seen"] = state_seen

    # --- Cumulative totals (counters since server start) ---
    gen_total = _strata_num(totals.get("output_tokens"))
    prompt_total = _strata_num(totals.get("prompt_tokens"))
    reused_total = _strata_num(totals.get("reused"))
    req_total = _strata_num(totals.get("requests"))
    draft_total = _strata_num(totals.get("drafts_offered"))
    accept_total = _strata_num(totals.get("drafts_accepted"))

    if gen_total is not None:
        out["gen_tokens_total"] = round(gen_total, 1)
    if prompt_total is not None:
        out["prompt_tokens_total"] = round(prompt_total, 1)
    if reused_total is not None:
        out["prompt_tokens_cached_total"] = round(reused_total, 1)
    if req_total is not None:
        out["requests_completed_total"] = round(req_total, 1)
    if draft_total is not None:
        out["spec_decode_draft_tokens_total"] = round(draft_total, 1)
    if accept_total is not None:
        out["spec_decode_accepted_tokens_total"] = round(accept_total, 1)

    # --- Rates (Δ counters / elapsed; window guards; last rate held) ---
    # Same convention as LiteLLM: the gap must be within
    # 0.5–_RATE_MAX_GAP_S; a longer gap holds the last computed rate.
    rate_ok = prev is not None and 0.5 <= elapsed <= _RATE_MAX_GAP_S
    if rate_ok:
        gen_rate: float | None = None
        prompt_rate: float | None = None
        if gen_total is not None and prev.get("gen") is not None:
            gen_rate = 0.0 if gen_total < prev["gen"] else round(
                max(0.0, gen_total - prev["gen"]) / elapsed, 1)
        if prompt_total is not None and prev.get("prompt") is not None:
            prompt_rate = 0.0 if prompt_total < prev["prompt"] else round(
                max(0.0, prompt_total - prev["prompt"]) / elapsed, 1)
        last_rate: dict[str, Any] = {"gen": gen_rate, "prompt": prompt_rate}
    else:
        last_rate = (prev or {}).get("last_rate")
    if last_rate is not None and last_rate["gen"] is not None:
        out["gen_tokens_per_s"] = last_rate["gen"]
    # NOTE: the prompt counter-delta is only a FALLBACK (before the first
    # completion the ring is empty and the live prefill below is absent);
    # the last-5-requests mean below overrides it.

    # Strata's totals.* counters only advance when a request COMPLETES
    # (the engine updates them on the DONE path), so counter-delta rates
    # spike once at completion and sit frozen during the whole decode.
    # The engine exposes live rates separately: `live.tok_s` is the
    # WINDOWED aggregate (sum of every running request's rate over the
    # last 2 s - the same figure the engine's own Monitor tab shows),
    # and `live.prefill_tok_s_mean` the prefill speed. Prefer those
    # while the engine is busy so the gauges move in real time.
    # NOTE: `live.slots[].tok_s` is each slot's WHOLE-REQUEST mean
    # (generated / since-first-token, the same formula as tok_s_mean),
    # NOT a live rate - summing them over-reports (a long-running or
    # slowing request's mean outruns its windowed rate), so they must
    # not be used here.
    if state == "generating":
        live_gen = (_strata_num(live.get("tok_s"))
                    or _strata_num(live.get("tok_s_mean")) or 0.0)
        if live_gen > 0:
            out["gen_tokens_per_s"] = round(live_gen, 1)
    # --- Prompt prefill rate: the last 5 completed requests, vLLM-style ---
    # vLLM's prompt gauge is a windowed rate (Δprompt / ΣTTFT) that stays
    # current on every tick; Strata's equivalent is the same Σ/Σ aggregate
    # over the ring buffer (the server returns it newest-first, so the
    # first 5 entries are the most recent completions), which does not
    # spike at completion and freeze during the decode. Each request
    # contributes its FRESH (uncached) tokens over the read time - the
    # engine's own definition (prompt_per_second = (prompt_tokens -
    # cache_n) / prompt_ms): a mostly-cached prompt's 20k tokens in 1 ms
    # is a cache lookup, not a 20M tok/s prefill. The Σ/Σ weighting
    # (not a per-request mean) keeps such expert-cache / KV-cache spikes
    # from dominating: a 2459-token read in 0.2 ms sits next to the
    # 21k-token read in 17.6 s that the work actually took.
    ring = data.get("requests")
    ring = ring if isinstance(ring, list) else []
    pf_fresh = 0.0
    pf_ms = 0.0
    for r in ring[:5]:
        if not isinstance(r, dict):
            continue
        pt = _strata_num(r.get("prompt_tokens"))
        pms = _strata_num(r.get("prompt_ms"))
        fresh = None if pt is None else pt - (_strata_num(r.get("reused")) or 0.0)
        if (fresh is not None and pms is not None
                and fresh > 0 and pms > 0):
            pf_fresh += fresh
            pf_ms += pms
    if pf_ms > 0:
        out["prompt_tokens_per_s"] = round(pf_fresh / (pf_ms / 1000.0), 1)
    elif state == "reading":
        # Active prefill, no completed requests yet: the live prefill
        # speed (the only source while the first prompt is being read).
        prefill = _strata_num(live.get("prefill_tok_s_mean"))
        if prefill is not None and prefill > 0:
            out["prompt_tokens_per_s"] = round(prefill, 1)
    if ("prompt_tokens_per_s" not in out and last_rate is not None
            and last_rate["prompt"] is not None):
        out["prompt_tokens_per_s"] = last_rate["prompt"]

    # --- KV prefix reuse + spec decode accept (lifetime ratios) ---
    if (reused_total is not None and prompt_total is not None
            and prompt_total > 0):
        out["prefix_cache_hit_pct"] = round(
            100.0 * reused_total / prompt_total, 1)
    if (draft_total is not None and accept_total is not None
            and draft_total > 0):
        out["spec_accept_pct"] = round(100.0 * accept_total / draft_total,
                                       1)

    # --- Request ring buffer: windowed means over NEW completions ---
    hist = data.get("requests")
    hist = hist if isinstance(hist, list) else []
    seen_times = [t for t in (_strata_num(r.get("time"))
                              for r in hist if isinstance(r, dict))
                  if t is not None]
    watermark = prev.get("req_watermark", 0.0) if prev else 0.0
    fresh = [r for r in hist if isinstance(r, dict)
             and (_strata_num(r.get("time")) or 0.0) > watermark]
    if fresh and prev is not None:
        def _mean(vals: list[float], scale: float = 1.0) -> float | None:
            return round(sum(vals) / len(vals) * scale, 1) if vals else None
        ttft = _mean([v for v in (_strata_num(r.get("prompt_ms"))
                                  for r in fresh) if v is not None], 0.001)
        e2e = _mean([v for v in (_strata_num(r.get("duration_s"))
                                 for r in fresh) if v is not None])
        itl = _mean([1000.0 / v for v in
                     (_strata_num(r.get("decode_tok_s")) for r in fresh)
                     if v is not None and v > 0])
        if ttft is not None:
            out["ttft_seconds"] = ttft
        if e2e is not None:
            out["e2e_latency_seconds"] = e2e
        if itl is not None:
            out["itl_ms"] = itl
        # Completion-tick rate smoothing: totals.* jump in one go for
        # the finished request(s), so the 1-tick Δ/elapsed rate would
        # flash a fake spike (whole-request tokens per second). The ring
        # entries carry the true speeds — decode_tok_s per request and
        # prompt tokens per prompt_ms (aggregated) — so use them instead
        # while the engine is not mid-batch (busy ticks already got the
        # live-rate override above, which stays the better source then).
        if state not in ("generating", "reading"):
            decode = _mean([v for v in
                            (_strata_num(r.get("decode_tok_s"))
                             for r in fresh) if v is not None and v > 0])
            if decode is not None:
                out["gen_tokens_per_s"] = decode
    req_watermark = max(seen_times) if seen_times else watermark

    # --- Usage (derived; Strata totals are NESTED: reused ⊂ prompt) ---
    if gen_total is not None or prompt_total is not None:
        usage: dict[str, Any] = {}
        total = 0
        if prompt_total is not None:
            total += int(round(prompt_total))
        if gen_total is not None:
            total += int(round(gen_total))
        usage["total_tokens"] = total
        if gen_total is not None:
            usage["gen_tokens"] = int(round(gen_total))
        if (reused_total is not None and prompt_total is not None
                and prompt_total > 0):
            usage["prompt_cached_pct"] = round(
                100.0 * reused_total / prompt_total, 1)
        since = _strata_num(totals.get("since"))
        if since is not None:
            usage["uptime_s"] = max(0, int(time.time() - since))
        out["usage"] = usage

    if config:
        out["config"] = config

    # --- Update previous state ---
    new_prev: dict[str, Any] = {"ts": now, "req_watermark": req_watermark}
    if gen_total is not None:
        new_prev["gen"] = gen_total
    if prompt_total is not None:
        new_prev["prompt"] = prompt_total
    if last_rate is not None:
        new_prev["last_rate"] = last_rate
    # Persist the last-busy ts + last-seen state across scrapes (the
    # computed values already carry the previous value when idle).
    if last_busy_ts is not None:
        new_prev["last_busy_ts"] = last_busy_ts
    if state_seen is not None:
        new_prev["state_seen"] = state_seen
    _strata_collector.set_prev(state_key, new_prev)

    _strata_collector.record_good(state_key, out)
    return out


# ---------------------------------------------------------------------------
# Upstream dashboard aggregation (multi-system consolidation)
# ---------------------------------------------------------------------------

# 15 s stale-hold per upstream (mirrors _LITELLM_STALE_MAX_AGE_S): a
# failed fetch keeps the last good entry flagged "stale" for 15 s, then
# the entry drops to "down".
_UPSTREAM_STALE_MAX_AGE_S = 15.0
# A failing upstream backs off exponentially (double per failure) up to
# this cap so a dead peer is not hammered every cycle (T8).
_UPSTREAM_BACKOFF_CAP_S = 60.0
# After this many consecutive failures an upstream is removed (the fetch
# loop stops hammering it; the UI keeps showing "down" via the
# last-good/placeholder path) (T8).
_UPSTREAM_REMOVAL_AFTER_FAILURES = 30

# Strict sanitizer key list: only per-system keys are forwarded from an
# upstream snapshot. `systems` is explicitly excluded (no recursion) and
# any other app-level key is dropped (a newer upstream's extra keys are
# silently ignored; an older upstream without `systems` works fine).
_UPSTREAM_KEYS = ("timestamp", "cpu", "cpu_per_core", "memory", "gpus",
                 "swap", "host_health", "llm_enabled", "llm", "llms")

# Keys the UI consumes for a PEER system's detail tab + fleet strip
# (a subset of _UPSTREAM_KEYS, so unknown keys are still dropped by the
# sanitizer). `timestamp`, `cpu_per_core`, and legacy `llm` are dropped:
# the UI never reads them from a peer's metrics. Full `gpus`/`llms` are
# kept (the peer detail tab renders them; a peer has no top-level copy).
_PEER_PROJECTION_KEYS = ("cpu", "memory", "gpus", "llms", "llm_enabled",
                       "swap", "host_health")

# Last good entry per upstream name: {name: (monotonic ts, entry dict)}.
# Each name is touched by exactly one fetch thread -> no locking needed.
_upstream_collector = BaseLLMCollector(_UPSTREAM_STALE_MAX_AGE_S)
# Backward-compatible alias (tests reference this name directly).
_upstream_last_good = _upstream_collector._last_good


def collect_upstream_metrics(url: str, name: str) -> dict[str, Any] | None:
    """Fetch one upstream dashboard's ``/api/metrics`` and sanitize it
    into a per-system entry.

    Returns ``{"name", "status": "up"|"stale", "metrics"}`` or ``None``
    (never fetched yet, or the 15 s stale-hold expired). Never raises.
    Mirrors the LLM collector conventions: 2 s timeout, per-name 15 s
    stale-hold, strict sanitizer (no ``systems`` recursion).
    """
    def _stale_or_none() -> dict[str, Any] | None:
        held = _upstream_collector._last_good.get(name)
        if held is not None:
            ts, entry = held
            if time.monotonic() - ts <= _UPSTREAM_STALE_MAX_AGE_S:
                out = dict(entry)
                out["status"] = "stale"
                return out
        return None

    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/metrics",
                                    timeout=2) as r:
            payload = json.loads(r.read().decode())
    except Exception:  # noqa: BLE001 - collector must never raise
        return _stale_or_none()

    if not isinstance(payload, dict):
        return _stale_or_none()

    # Sanitize (drop unknown keys) then project to what the UI consumes
    # for a peer's detail tab + fleet strip. `_PEER_PROJECTION_KEYS` is a
    # subset of `_UPSTREAM_KEYS`, so unknown keys are still dropped.
    metrics = {k: payload[k] for k in _PEER_PROJECTION_KEYS if k in payload}
    out = {"name": name, "status": "up", "metrics": metrics}
    _upstream_collector.record_good(name, out)
    return out
