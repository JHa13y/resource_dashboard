#!/usr/bin/env python3
"""Resource & GPU Utilization Dashboard.

A lightweight Flask-based web dashboard that streams real-time CPU, memory,
and GPU metrics (NVIDIA pynvml/nvidia-smi, AMD amd-smi/rocm-smi/sysfs,
JetPack sysfs) via Server-Sent Events (SSE).

Usage:
    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator, NamedTuple

import psutil
from flask import Flask, Response, render_template

import collectors.llm as llm
from collectors.gpu import collect_gpu_metrics
from collectors.llm import (
    DEFAULT_LITELLM_URL,
    DEFAULT_LLAMA_URL,
    DEFAULT_STRATA_URL,
    DEFAULT_VLLM_URL,
    _UPSTREAM_BACKOFF_CAP_S,
    _UPSTREAM_REMOVAL_AFTER_FAILURES,
    collect_litellm_metrics,
    collect_llama_metrics,
    collect_strata_metrics,
    collect_upstream_metrics,
    collect_vllm_metrics,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Configuration (config.yaml + CLI resolution)
# ---------------------------------------------------------------------------

# Built-in defaults (used when neither CLI nor config provides a value).
_CONFIG_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 5000,
    "poll_interval": 1.0,
    "llm_backend": "auto",
    "llama_enabled": True,
    "llama_url": DEFAULT_LLAMA_URL,
    "vllm_url": DEFAULT_VLLM_URL,
    "litellm_url": DEFAULT_LITELLM_URL,
    "litellm_token": None,
    "strata_url": DEFAULT_STRATA_URL,
    "default_view": "local",
    "llms": None,
}
_CONFIG_BACKENDS = ("auto", "llama", "vllm", "litellm", "strata")
_CONFIG_VIEWS = ("local", "fleet")

# Backend set for named multi-LLM targets (no auto: auto is probe-chain
# sugar for the implicit single "primary" target only).
_CONFIG_TARGET_BACKENDS = ("llama", "vllm", "litellm", "strata")
_LLM_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}$")

# Legacy single-backend settings keys; `llms` conflicts with any of them
# (checked by RAW presence — config key present / CLI arg not None — not
# via pick(), which cannot distinguish unset from defaulted).
_LLM_LEGACY_KEYS = ("llm_backend", "llama_url", "vllm_url", "litellm_url",
                    "litellm_token", "strata_url")


class LLMTarget(NamedTuple):
    """One LLM scrape target (multi-LLM).

    ``backend`` may be ``"auto"`` only for the implicit legacy target
    ``"primary"``, which probes the per-type collector URLs; named
    targets from ``llms:`` are always pinned. ``url`` is unused for the
    auto target. ``state_key`` for the scrape-state dicts is ``name``.
    """

    name: str
    backend: str
    url: str | None
    token: str | None = None
_CONFIG_KNOWN_KEYS = frozenset(set(_CONFIG_DEFAULTS) | {"upstreams"})


def _validate_llms(entries: Any, fail) -> list[dict[str, Any]]:
    """Validate + sanitize an ``llms:`` config block (multi-LLM targets).

    Strict by design (hard exit via ``fail``): a mis-scoped scrape target
    would silently produce wrong data, unlike an unreachable upstream
    which self-reports ``down`` in the UI (hence upstreams stay warn+skip).
    """
    if not isinstance(entries, list) or not entries:
        fail("llms must be a non-empty list of {name, backend, url} "
             "entries (use llama_enabled: false to disable scraping)")
    seen: set[str] = set()
    primaries = 0
    clean: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"llms[{i}] must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not _LLM_NAME_RE.fullmatch(name):
            fail(f"llms[{i}].name must match [A-Za-z0-9._-]{{1,64}}")
        if name in seen:
            fail(f"llms[{i}].name {name!r} duplicated")
        seen.add(name)
        backend = entry.get("backend")
        if backend not in _CONFIG_TARGET_BACKENDS:
            fail(f"llms[{i}].backend must be one of "
                 f"{', '.join(_CONFIG_TARGET_BACKENDS)}")
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            fail(f"llms[{i}].url must be a non-empty string")
        token = entry.get("token")
        if token is not None and not isinstance(token, str):
            fail(f"llms[{i}].token must be a string")
        if token is not None and backend != "litellm":
            log.warning("llms[%d].token ignored (only litellm uses a "
                        "Bearer token)", i)
            token = None
        primary = entry.get("primary", False)
        if not isinstance(primary, bool):
            fail(f"llms[{i}].primary must be a boolean")
        if primary:
            primaries += 1
            if primaries > 1:
                fail("llms: at most one entry may set primary: true")
        clean.append({"name": name, "backend": backend, "url": url,
                      "token": token, "primary": primary})
    urls = [e["url"] for e in clean]
    for u in sorted({u for u in urls if urls.count(u) > 1}):
        log.warning("llms: url %s is configured for more than one target", u)
    return clean


def load_config(path: str | None = None,
                explicit: bool = False) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    * auto-detected (``path is None``): ``./config.yaml`` is used only
      when present; absent -> ``{}`` (silent).
    * explicit ``--config PATH`` absent -> log error + ``sys.exit(1)``
      (user intent - fail fast).
    * malformed YAML / wrong top-level type -> log error + exit 1.
    * unknown keys -> warn + ignore (forward compatibility).
    * invalid known keys -> log error + exit 1.
    """
    if path is None:
        if not os.path.isfile("config.yaml"):
            return {}
        path = "config.yaml"
    if not os.path.isfile(path):
        if explicit:
            log.error("Config file not found: %s", path)
            sys.exit(1)
        return {}
    try:
        import yaml
    except ImportError:
        if explicit:
            log.error("PyYAML is required to read %s "
                     "(pip install PyYAML)", path)
            sys.exit(1)
        log.warning("PyYAML not installed - ignoring config file %s",
                    path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        log.error("Malformed YAML in %s: %s", path, exc)
        sys.exit(1)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        log.error("Config file %s must be a YAML mapping", path)
        sys.exit(1)
    for key in data:
        if key not in _CONFIG_KNOWN_KEYS:
            log.warning("Unknown config key %r in %s - ignoring", key, path)

    def _fail(msg: str) -> None:
        log.error("Invalid config in %s: %s", path, msg)
        sys.exit(1)

    if "port" in data and (
            isinstance(data["port"], bool)
            or not isinstance(data["port"], int)):
        _fail("port must be an integer")
    if "poll_interval" in data:
        pi = data["poll_interval"]
        if (isinstance(pi, bool) or not isinstance(pi, (int, float))
                or pi <= 0):
            _fail("poll_interval must be a number > 0")
    if "llm_backend" in data and data["llm_backend"] not in _CONFIG_BACKENDS:
        _fail(f"llm_backend must be one of {', '.join(_CONFIG_BACKENDS)}")
    if ("default_view" in data
            and data["default_view"] not in _CONFIG_VIEWS):
        _fail(f"default_view must be one of {', '.join(_CONFIG_VIEWS)}")
    if "llms" in data:
        data["llms"] = _validate_llms(data["llms"], _fail)
    for key in ("host", "llama_url", "vllm_url", "litellm_url"):
        if key in data and not isinstance(data[key], str):
            _fail(f"{key} must be a string")
    if "llama_enabled" in data and not isinstance(
            data["llama_enabled"], bool):
        _fail("llama_enabled must be true/false")
    if "litellm_token" in data and data["litellm_token"] is not None \
            and not isinstance(data["litellm_token"], str):
        _fail("litellm_token must be a string or null")
    if "upstreams" in data and not isinstance(data["upstreams"], list):
        _fail("upstreams must be a list of {name, url} entries")
    return data


def merge_upstreams(config_upstreams: list | None,
                    cli_upstreams: list[str] | None) -> list[dict[str, str]]:
    """Merge config + CLI upstream entries.

    Malformed entries (missing/empty ``name`` or ``url``) -> warn + skip;
    duplicate names within one source -> warn, last wins; a CLI entry
    overrides a config entry with the same name. Order: config order,
    then CLI-only names in CLI order.
    """
    def _clean(entries: list | None,
               source: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for e in entries or []:
            name = e.get("name") if isinstance(e, dict) else None
            url = e.get("url") if isinstance(e, dict) else None
            if (not isinstance(name, str) or not name
                    or not isinstance(url, str) or not url):
                log.warning("Skipping malformed upstream entry %r (%s)",
                            e, source)
                continue
            if name in out:
                log.warning("Duplicate upstream name %r (%s) - last wins",
                            name, source)
            out[name] = {"name": name, "url": url.rstrip("/")}
        return out

    cfg = _clean(config_upstreams, "config")
    cli: dict[str, dict[str, str]] = {}
    cli_order: list[str] = []
    for spec in cli_upstreams or []:
        if "=" not in spec:
            log.warning("Skipping malformed --upstream %r "
                       "(expected name=url)", spec)
            continue
        name, url = spec.split("=", 1)
        name, url = name.strip(), url.strip()
        if not name or not url:
            log.warning("Skipping malformed --upstream %r "
                       "(empty name or url)", spec)
            continue
        if name in cli:
            log.warning("Duplicate --upstream name %r - last wins", name)
        cli[name] = {"name": name, "url": url.rstrip("/")}
        if name not in cli_order:
            cli_order.append(name)

    merged = {**cfg, **cli}
    seen: list[str] = []
    for n in list(cfg.keys()) + cli_order:
        if n not in seen:
            seen.append(n)
    return [merged[n] for n in seen]


def resolve_settings(args: argparse.Namespace,
                    config: dict[str, Any]) -> dict[str, Any]:
    """Resolve CLI > config > built-in defaults into a settings dict.

    ``litellm_token`` additionally honors ``RD_LITELLM_TOKEN``: when the
    env var is set and non-empty it wins over both the CLI flag and
    the config value.
    """
    def pick(key: str, cli_value: Any) -> Any:
        return cli_value if cli_value is not None else config.get(
            key, _CONFIG_DEFAULTS[key])

    def pick_token(key: str, cli_value: Any) -> Any:
        env_token = os.environ.get("RD_LITELLM_TOKEN")
        return env_token if env_token else pick(key, cli_value)

    llama_enabled = config.get("llama_enabled", True)
    if args.no_llama:
        llama_enabled = False  # flag wins when present

    llms_cfg = config.get("llms")
    if llms_cfg is not None:
        clash = {k for k in _LLM_LEGACY_KEYS
                 if getattr(args, k, None) is not None}
        clash |= {k for k in _LLM_LEGACY_KEYS if k in config}
        if clash:
            log.error("llms: configured; drop the legacy LLM settings "
                      "(%s) or remove llms", ", ".join(sorted(clash)))
            sys.exit(1)
        targets = [LLMTarget(e["name"], e["backend"], e["url"], e["token"])
                   for e in llms_cfg]
        primary_name = next((e["name"] for e in llms_cfg if e["primary"]),
                            None)
        if primary_name:
            # stable sort: the primary entry leads the list (its snapshot
            # becomes the legacy `llm` alias)
            targets.sort(key=lambda t: t.name != primary_name)
    else:
        backend = pick("llm_backend", args.llm_backend)
        url_map = {"llama": pick("llama_url", args.llama_url),
                   "vllm": pick("vllm_url", args.vllm_url),
                   "litellm": pick("litellm_url", args.litellm_url),
                   "strata": pick("strata_url", args.strata_url)}
        targets = [LLMTarget("primary", backend, url_map.get(backend),
                             pick_token("litellm_token", args.litellm_token))]

    return {
        "host": pick("host", args.host),
        "port": pick("port", args.port),
        "poll_interval": pick("poll_interval", args.poll_interval),
        "llama_url": pick("llama_url", args.llama_url),
        "vllm_url": pick("vllm_url", args.vllm_url),
        "litellm_url": pick("litellm_url", args.litellm_url),
        "litellm_token": pick_token("litellm_token", args.litellm_token),
        "strata_url": pick("strata_url", args.strata_url),
        "llm_backend": pick("llm_backend", args.llm_backend),
        "default_view": pick("default_view", args.default_view),
        "llama_enabled": llama_enabled,
        "llm_targets": targets,
        "upstreams": merge_upstreams(config.get("upstreams"),
                                    args.upstream),
    }


# ---------------------------------------------------------------------------
# Swap usage + I/O rate (T6)
# ---------------------------------------------------------------------------

# Previous swap I/O counters (cumulative bytes) + monotonic ts, used to
# compute swap in/out rates from per-scrape deltas.
_swap_prev: dict[str, float] | None = None


def collect_swap_metrics() -> dict[str, Any] | None:
    """Collect swap usage and I/O rates. Never raises.

    Returns ``None`` only when the host has no swap (total == 0) — the
    sole platform guard; psutil's sin/sout counters are present on both
    Linux and macOS. ``in_kb_s`` / ``out_kb_s`` are per-scrape deltas of
    the cumulative sin/sout counters (omitted on the first scrape, when
    there is no previous reading) and clamped to >= 0 so a counter reset
    (swapon/off, reboot) never yields a negative rate.
    """
    global _swap_prev
    try:
        sw = psutil.swap_memory()
    except Exception:  # noqa: BLE001 – collector must never raise
        return None
    if not sw.total:
        return None
    out: dict[str, Any] = {
        "used_gb": round(sw.used / 2**30, 1),
        "total_gb": round(sw.total / 2**30, 1),
        "pct": round(sw.percent, 1),
    }
    now = time.monotonic()
    prev = _swap_prev
    if prev is not None:
        elapsed = now - prev["ts"]
        if elapsed > 0:
            out["in_kb_s"] = round(
                max(0.0, sw.sin - prev["sin"]) / elapsed / 1024.0, 1
            )
            out["out_kb_s"] = round(
                max(0.0, sw.sout - prev["sout"]) / elapsed / 1024.0, 1
            )
    _swap_prev = {"ts": now, "sin": sw.sin, "sout": sw.sout}
    return out


# ---------------------------------------------------------------------------
# Host health: PSI memory pressure + D-state processes (T7)
# ---------------------------------------------------------------------------

# Linux Pressure Stall Information file for memory. Absent on macOS and on
# Linux kernels without CONFIG_PSI — the metric is simply None there.
_PSI_MEMORY_PATH = "/proc/pressure/memory"


def _read_psi_mem_avg10(path: str = _PSI_MEMORY_PATH) -> float | None:
    """Parse the ``some avg10=`` value (percent) from a PSI pressure file.

    The file format is::

        some avg10=0.05 avg60=0.02 avg300=0.01 total=123456789
        full avg10=0.00 avg60=0.00 avg300=0.00 total=0

    Only the ``some`` line's avg10 is used (10 s window, any stall). Returns
    ``None`` when the file is missing (non-Linux host) or unparseable.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:  # no /proc (macOS), PSI disabled, or permission error
        return None
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "some":
            continue
        for tok in parts[1:]:
            if tok.startswith("avg10="):
                try:
                    return round(float(tok[len("avg10="):]), 2)
                except ValueError:
                    return None
    return None


def collect_host_health_metrics(
    psi_mem_path: str = _PSI_MEMORY_PATH,
) -> dict[str, Any]:
    """Collect host health: PSI memory pressure + D-state process count.

    Never raises. ``psi_mem_avg10`` is None on hosts without
    ``/proc/pressure/memory`` (macOS, kernels without CONFIG_PSI); the
    D-state count degrades to 0 if the process table cannot be read.
    ``psi_mem_path`` is injectable for tests.
    """
    try:
        d_state = sum(
            1
            for p in psutil.process_iter(["status"])
            if p.info.get("status") == psutil.STATUS_DISK_SLEEP
        )
    except Exception:  # noqa: BLE001 – collector must never raise
        d_state = 0
    return {
        "psi_mem_avg10": _read_psi_mem_avg10(psi_mem_path),
        "d_state_procs": d_state,
    }


# ---------------------------------------------------------------------------
# Metrics Collector Thread
# ---------------------------------------------------------------------------

def _url_host_is_local(url: str | None) -> bool:
    """True when a target URL points at this host (process-card gate)."""
    if not url:
        return False
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    return (host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
            or host == platform.node() or host.endswith(".local"))


def _llm_entries_busy(entries: list[dict[str, Any]] | None) -> bool:
    """True when any non-down LLM entry shows activity (a busy tick).

    Backend-agnostic: a target is busy when it reports a running request
    count, a generating/reading engine state, or a processing slot. Used
    by the LLM thread's busy-latch so a short request / busy tick the
    slower main loop misses is not discarded.
    """
    for e in entries or []:
        if not isinstance(e, dict) or e.get("status") == "down":
            continue
        if (e.get("requests_processing") or 0) > 0:
            return True
        if e.get("state") in ("reading", "generating"):
            return True
        for s in e.get("slots") or []:
            if isinstance(s, dict) and s.get("is_processing"):
                return True
    return False


class MetricsCollector(threading.Thread):
    """Background thread that polls system + GPU metrics and broadcasts each
    JSON snapshot to every subscriber, plus a cached last-snapshot for
    one-shot readers (``/api/metrics``).

    A per-client bounded buffer carries the fan-out; when one is full the
    *oldest* entry is dropped so a blocked/slow client cannot grow unbounded
    or starve the other clients or the producer.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        llama_url: str = DEFAULT_LLAMA_URL,
        llama_enabled: bool = True,
        vllm_url: str = DEFAULT_VLLM_URL,
        llm_backend: str = "auto",
        litellm_url: str = DEFAULT_LITELLM_URL,
        litellm_token: str | None = None,
        strata_url: str = DEFAULT_STRATA_URL,
        llm_targets: list[LLMTarget] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.llama_url = llama_url
        self.llama_enabled = llama_enabled
        self.vllm_url = vllm_url
        self.llm_backend = llm_backend
        self.litellm_url = litellm_url
        self.litellm_token = litellm_token
        self.strata_url = strata_url
        # Multi-LLM: ordered LLMTarget list; None = implicit single
        # "primary" target built from the legacy per-type attributes
        # above (rebuilt per tick, so __main__'s attribute writes and an
        # atomic llm_targets rebind both take effect on the next scrape).
        self.llm_targets = llm_targets
        # Broadcast hub: the last serialized snapshot + a monotonic
        # sequence, fanned out to every subscriber. Replaces the old
        # single-consumer queue.Queue — each snapshot is now broadcast
        # (not distributed to one consumer), a blocked/slow client cannot
        # starve the others or the producer, and /api/metrics reads the
        # cached snapshot without consuming it.
        self._hub = threading.Condition()
        self._last_payload: str | None = None
        self._sequence = 0
        self._subscribers: list[queue.Queue[str]] = []
        self.running = True
        # The LLM scrape runs on its own 1 s cadence, decoupled from
        # the main cycle (which includes GPU/health probes that can
        # make the cycle exceed the rate window). This keeps the
        # token-rate window ~poll_interval even when the main cycle
        # is slow.
        self._llm_latest: list[dict[str, Any]] | None = None
        # Busy-latch: the most recent BUSY LLM sample, retained so a slow
        # main loop does not discard a short request / busy tick it
        # missed. `_llm_busy_latch_id` increments per new busy sample;
        # `_llm_busy_latch_published_id` is the id the main loop last
        # published (so a busy sample is published once, not repeatedly).
        self._llm_busy_latch: list[dict[str, Any]] | None = None
        self._llm_busy_latch_id = 0
        self._llm_busy_latch_published_id = 0
        self._llm_ready = threading.Event()
        self._llm_stop = threading.Event()
        self._llm_thread = threading.Thread(target=self._llm_loop,
                                           daemon=True,
                                           name=f"rd-llm-{id(self):x}")
        # Scrape threads of this instance carry the same prefix so tests
        # can attribute HTTP calls to a specific collector (the module
        # global starts at import and outlives any single test).
        self._llm_thread_prefix = f"rd-llm-{id(self):x}"
        # Upstream dashboard aggregation: one fetch thread per upstream
        # (started from __main__ via start_upstreams; the collector is
        # already running by then). _upstream_latest is initialized here
        # because _collect() runs from import time, before start_upstreams.
        self._upstream_latest: dict[str, dict[str, Any] | None] = {}
        self._upstream_names: list[str] = []
        self._upstream_stop = threading.Event()
        self._upstream_threads: list[threading.Thread] = []
        # Names removed after N consecutive failures (T8): the fetch loop
        # stops hammering them; the UI keeps showing "down" via the
        # last-good/placeholder path.
        self._upstream_removed: set[str] = set()
        # Self-observability (T7): the wall-clock duration of the last main
        # collection cycle, and a per-target last-error map (keyed by each
        # LLM target's name, each upstream's name, and "main" for the main
        # metrics collection). A target with no recent error is absent.
        # `_last_error_log_ts` drives the log rate limiting (a persistently
        # failing target logs the exception once, then is suppressed for
        # `_error_cooldown_s` rather than flooding the log every cycle).
        self._last_cycle_duration: float | None = None
        self._last_errors: dict[str, str] = {}
        self._last_error_log_ts: dict[str, float] = {}
        self._error_cooldown_s: float = 60.0

        # Seed psutil (first cpu_percent call returns 0.0 and must be discarded)
        psutil.cpu_percent()

    # -- public API ----------------------------------------------------------

    def get_latest(self, timeout: float | None = None) -> str | None:
        """Return the most recent metrics JSON string (non-destructive: reads
        the cached snapshot, does not consume it). Blocks up to *timeout*
        waiting for the first snapshot if none exists yet."""
        with self._hub:
            if self._last_payload is None and timeout is not None:
                self._hub.wait(timeout)
            return self._last_payload

    # -- internal ------------------------------------------------------------

    def publish(self, payload: str) -> None:
        """Publish one serialized snapshot to the hub (broadcast to every
        subscriber; the producer never blocks on a slow consumer)."""
        with self._hub:
            self._last_payload = payload
            self._sequence += 1
            self._hub.notify_all()
            subscribers = list(self._subscribers)
        for q in subscribers:
            # Per-client bounded buffer with drop-oldest: a blocked/slow
            # client cannot grow unbounded or starve the producer.
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # dropped for this slow client only

    def subscribe(self, maxsize: int = 30) -> queue.Queue[str]:
        """Return a per-client bounded queue that receives every future
        snapshot (and the current one immediately, if any — a mid-stream
        subscriber does not wait a cycle)."""
        q: queue.Queue[str] = queue.Queue(maxsize=maxsize)
        with self._hub:
            last = self._last_payload
            self._subscribers.append(q)
            # Seed under the lock (atomic with the append) so a concurrent
            # publish() cannot interleave and put a newer snapshot ahead of
            # the seeded (older) one — that would break frame ordering.
            if last is not None:
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(last)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        """Detach a per-client buffer (e.g. when its SSE client leaves)."""
        with self._hub:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    # -- self-observability (T7) ------------------------------------------------

    def _record_error(self, key: str, error: str, *,
                     exc: Exception | None = None,
                     log_msg: str | None = None) -> None:
        """Record a per-target last error and log it, rate-limited.

        A persistently failing target would otherwise log an exception
        every cycle (the every-cycle ``log.exception`` flood). Instead:
        the first failure logs the exception (with traceback); failures
        within ``_error_cooldown_s`` are suppressed, and after
        the cooldown a single ``still failing`` warning is logged (at
        most once per cooldown). The error string is always recorded in
        ``_last_errors`` regardless of the log suppression, so ``/health``
        always reflects the latest failure.
        """
        self._last_errors[key] = error
        now = time.monotonic()
        last_log = self._last_error_log_ts.get(key)
        if last_log is not None and (now - last_log) < self._error_cooldown_s:
            return  # suppressed (within the cooldown)
        self._last_error_log_ts[key] = now
        msg = log_msg or f"Error: {key}"
        if last_log is not None:
            log.warning("%s — still failing, suppressing for %.0fs",
                        msg, self._error_cooldown_s)
        elif exc is not None:
            log.exception(msg)
        else:
            log.warning(msg)

    def _clear_error(self, key: str, *, log_msg: str | None = None) -> None:
        """Clear a per-target last error (the target recovered). Log the
        recovery once if it was previously failing."""
        if key in self._last_errors:
            log.info("%s recovered", log_msg or key)
            del self._last_errors[key]
            self._last_error_log_ts.pop(key, None)

    def health(self) -> dict[str, Any]:
        """Self-observability snapshot for ``GET /health``: the last main
        cycle duration, the per-target last-error map, and a few sensible
        extras. A target with no recent error is absent from
        ``last_errors``."""
        return {
            "status": "ok" if not self._last_errors else "degraded",
            "poll_interval": self.poll_interval,
            "last_cycle_duration_s": self._last_cycle_duration,
            "last_errors": dict(self._last_errors),
            "upstream_removed": len(self._upstream_removed),
        }

    def _llm_loop(self) -> None:
        """LLM scrape on its own cadence (decoupled from the main
        cycle, which includes GPU/health probes). Keeps the token-rate
        window ~poll_interval even when the main cycle is slow.

        Multi-LLM: every configured target is scraped per tick.
        Several targets run concurrently so one slow target (llama
        worst case is three 2 s-timeout calls) cannot stretch the other
        targets' 0.5-15 s rate windows or the 15 s stale-holds."""
        while self.running:
            try:
                if self.llama_enabled:
                    targets = self._targets_snapshot()
                    if len(targets) <= 1:
                        self._llm_latest = (
                            [self._collect_one(targets[0])]
                            if targets else [])
                    else:
                        with ThreadPoolExecutor(
                                max_workers=len(targets),
                                thread_name_prefix=self._llm_thread_prefix,
                        ) as ex:
                            self._llm_latest = list(
                                ex.map(self._collect_one, targets))
                else:
                    self._llm_latest = None
            except Exception as e:  # noqa: BLE001
                # Batch-level (not per-target) failure: rate-limit the log
                # so a persistently failing LLM loop does not flood the log
                # every cycle (T7).
                self._record_error("llm", repr(e), exc=e,
                                  log_msg="Error collecting LLM metrics")
            # Busy-latch: retain the most recent BUSY sample so a slow
            # main loop does not discard a short request / busy tick it
            # missed (an idle sample does NOT clear the latch).
            if _llm_entries_busy(self._llm_latest):
                self._llm_busy_latch = list(self._llm_latest or [])
                self._llm_busy_latch_id += 1
            self._llm_ready.set()
            self._llm_stop.wait(self.poll_interval)

    def _targets_snapshot(self) -> list[LLMTarget]:
        """Current target list (rebuilt per tick so a mid-run rebind of
        ``llm_targets`` — or of the legacy attributes before any rebind —
        takes effect on the next scrape)."""
        targets = self.llm_targets
        if targets is None:
            url = {"llama": self.llama_url, "vllm": self.vllm_url,
                   "litellm": self.litellm_url,
                   "strata": self.strata_url}.get(self.llm_backend)
            targets = [LLMTarget("primary", self.llm_backend, url,
                                 self.litellm_token)]
        return list(targets)

    def _collect_one(self, target: LLMTarget) -> dict[str, Any]:
        """Scrape a single target into its snapshot entry.

        Failed scrape (None past the stale-hold) still yields an entry:
        ``{name, backend, status: "down"}`` — names stay stable in the UI
        instead of vanishing, and history can gap-tick.
        """
        if target.backend == "auto":
            data = self._collect_llm(state_key=target.name)
        elif target.backend == "llama":
            data = llm.collect_llama_metrics(target.url,
                                         state_key=target.name)
            if data is not None:
                data["backend"] = "llama"
        elif target.backend == "vllm":
            data = llm.collect_vllm_metrics(target.url, state_key=target.name)
            if data is not None:
                data["backend"] = "vllm"
        elif target.backend == "litellm":
            data = llm.collect_litellm_metrics(target.url, target.token,
                                           state_key=target.name)
            if data is not None:
                data["backend"] = "litellm"
        elif target.backend == "strata":
            data = llm.collect_strata_metrics(target.url,
                                          state_key=target.name)
            if data is not None:
                data["backend"] = "strata"
        else:
            data = None
        if data is None:
            # Per-target failure (the scrape returned None past the
            # stale-hold): record it in the per-target last-error map and log
            # it rate-limited so a persistently-down target does not flood
            # the log every cycle (T7).
            self._record_error(target.name, "down (scrape returned None)",
                              log_msg=f"LLM target {target.name} down")
            return {"name": target.name, "backend": target.backend,
                    "status": "down"}
        # Success: clear any prior last error for this target (recovery is
        # logged once by _clear_error) (T7).
        self._clear_error(target.name,
                          log_msg=f"LLM target {target.name}")
        # llama process-card attribution: the local biggest-RSS
        # llama-server can only be tied to a llama target when the URL is
        # local AND that target is the dashboard's only llama server
        # (behavior change vs. the single-backend builds; see README).
        if data.get("backend") == "llama":
            url = (target.url if target.backend == "llama"
                   else self.llama_url)
            llama_count = sum(1 for t in self._targets_snapshot()
                              if t.backend == "llama")
            if target.backend == "auto":
                llama_count += 1  # the auto winner is a llama server too
            if llama_count > 1 or not _url_host_is_local(url):
                data.pop("process", None)
        data["name"] = target.name
        return data

    def stop(self, timeout: float = 2.0) -> None:
        """Shut all loops down promptly (the LLM thread and the upstream
        fetch threads otherwise sleep out their whole poll interval before
        noticing ``running`` flipped) — used by tests and shutdown paths.
        Sets ``_upstream_stop`` (wakes the ``_upstream_stop.wait()`` in
        ``_upstream_loop``) and joins the upstream threads, so shutdown is
        prompt and clean rather than leaving daemon fetch threads running
        (T8)."""
        self.running = False
        self._llm_stop.set()
        self._upstream_stop.set()
        if self._llm_thread.is_alive():
            self._llm_thread.join(timeout=timeout)
        for t in self._upstream_threads:
            if t.is_alive():
                t.join(timeout=timeout)

    def start_upstreams(self, entries: list[dict[str, str]]) -> None:
        """Spawn one daemon fetch thread per configured upstream (a no-op
        for an empty list; names already started are skipped). Called
        from __main__ after config resolution; the collector is already
        running."""
        for entry in entries:
            name, url = entry["name"], entry["url"]
            if name in self._upstream_names:
                continue
            t = threading.Thread(target=self._upstream_loop,
                                args=(name, url), daemon=True)
            self._upstream_threads.append(t)
            self._upstream_names.append(name)
            t.start()

    def _upstream_loop(self, name: str, url: str) -> None:
        """Fetch one upstream's /api/metrics on the main poll cadence.
        No ready-wait: upstreams may be slow to come up; the UI shows
        "down" until the first fetch lands. A failing upstream (a
        ``None`` return — the 15 s stale-hold expired) backs off
        exponentially (double per failure, capped at
        ``_UPSTREAM_BACKOFF_CAP_S``) so a dead peer is not hammered every
        cycle; a success (an ``up`` or ``stale`` entry) resets to the poll
        cadence. After ``_UPSTREAM_REMOVAL_AFTER_FAILURES`` consecutive
        failures the upstream is removed (the loop stops fetching it; the
        UI keeps showing "down" via the last-good/placeholder path) (T8)."""
        backoff = self.poll_interval
        failures = 0
        while self.running:
            upstream_exc: Exception | None = None
            try:
                out = llm.collect_upstream_metrics(url, name)
            except Exception as e:  # noqa: BLE001
                upstream_exc = e
                out = None
            # Always store the result (a None marks the upstream "down").
            self._upstream_latest[name] = out
            if out is None:
                # A failure (exception or down): record the most specific
                # error and log it rate-limited so a persistently failing
                # upstream does not flood the log every cycle (T7).
                if upstream_exc is not None:
                    self._record_error(name, repr(upstream_exc),
                                      exc=upstream_exc,
                                      log_msg=f"Error collecting upstream {name}")
                else:
                    self._record_error(name, "down (fetch returned None)",
                                      log_msg=f"Upstream {name} down")
                failures += 1
                if failures >= _UPSTREAM_REMOVAL_AFTER_FAILURES:
                    log.warning("Upstream %s removed after %d consecutive "
                               "failures",
                               name, failures)
                    self._upstream_removed.add(name)
                    break
                backoff = min(backoff * 2, _UPSTREAM_BACKOFF_CAP_S)
            else:
                # Success: clear any prior last error for this upstream
                # (recovery is logged once by _clear_error) (T7).
                self._clear_error(name, log_msg=f"Upstream {name}")
                failures = 0
                backoff = self.poll_interval
            self._upstream_stop.wait(backoff)

    def _collect_llm(self, state_key: str = "primary") -> dict[str, Any] | None:
        """Auto-mode LLM backend selection over the legacy per-type URLs
        (re-evaluated every cycle; used for the implicit "auto" target).

        auto semantics: try vLLM first, then the LiteLLM proxy, then
        Strata, fall through to llama.cpp. The per-collector 15 s stale-hold IS the
        hysteresis: a <15 s blip keeps the current backend flagged
        stale; a real outage falls through to the next backend, and
        recovery flips back on the next cycle. A pinned backend skips
        the other branches entirely.
        """
        if not self.llama_enabled:
            return None
        if self.llm_backend in ("auto", "vllm"):
            data = llm.collect_vllm_metrics(self.vllm_url, state_key=state_key)
            if data is not None:
                data["backend"] = "vllm"
                # Warn once: vLLM is silently preferred in auto mode. We
                # do not probe the llama server here (keeps the cycle cheap),
                # so this fires the first time vLLM is active in auto mode
                # even if llama is down - the accepted approximation.
                if self.llm_backend == "auto" and not llm._llm_tie_warned:
                    llm._llm_tie_warned = True
                    log.warning(
                        "LLM auto mode: using vLLM at %s. If llama.cpp is "
                        "also reachable and you want it instead, pin "
                        "--llm-backend llama (logged once).",
                        self.vllm_url)
                return data
        if self.llm_backend in ("auto", "litellm"):
            data = llm.collect_litellm_metrics(self.litellm_url,
                                           self.litellm_token,
                                           state_key=state_key)
            if data is not None:
                data["backend"] = "litellm"
                if self.llm_backend == "auto" and not llm._litellm_tie_warned:
                    llm._litellm_tie_warned = True
                    log.warning(
                        "LLM auto mode: using LiteLLM proxy at %s. If "
                        "llama.cpp is also reachable and you want it "
                        "instead, pin --llm-backend llama (logged once).",
                        self.litellm_url)
                return data
        if self.llm_backend in ("auto", "strata"):
            data = llm.collect_strata_metrics(self.strata_url,
                                          state_key=state_key)
            if data is not None:
                data["backend"] = "strata"
                if self.llm_backend == "auto" and not llm._strata_tie_warned:
                    llm._strata_tie_warned = True
                    log.warning(
                        "LLM auto mode: using Strata at %s. If llama.cpp "
                        "is also reachable and you want it instead, pin "
                        "--llm-backend llama (logged once).",
                        self.strata_url)
                return data
        if self.llm_backend in ("auto", "llama"):
            data = llm.collect_llama_metrics(self.llama_url,
                                         state_key=state_key)
            if data is not None:
                data["backend"] = "llama"
                return data
        return None

    def _collect(self) -> dict[str, Any]:
        """Snapshot all metrics once (non-blocking)."""
        cpu_overall = psutil.cpu_percent(percpu=False)
        cpu_per_core = psutil.cpu_percent(percpu=True)

        vm = psutil.virtual_memory()

        metrics = {
            "timestamp": time.time(),
            "cpu": round(cpu_overall, 1),
            "cpu_per_core": [round(c, 1) for c in cpu_per_core],
            "memory": {
                "percent": round(vm.percent, 1),
                "total_gb": round(vm.total / (1024**3), 1),
                "used_gb": round(vm.used / (1024**3), 1),
                "available_gb": round(vm.available / (1024**3), 1),
            },
            "gpus": collect_gpu_metrics(),
            "swap": collect_swap_metrics(),
            "host_health": collect_host_health_metrics(),
            # Stable signal for the frontend: the LLM section is on by
            # default; --no-llama skips scraping entirely and hides the card.
            "llm_enabled": self.llama_enabled,
        }
        # Multi-LLM: ordered entries, one per configured target (down
        # placeholders included). The legacy `llm` key is the PRIMARY
        # (first) target's entry — None when it is unreachable, exactly
        # the old single-backend contract. It is NOT an aggregate across
        # targets: fleet consumers sum `llms`.
        entries = ((self._llm_latest or []) if self.llama_enabled else [])
        # Busy-latch: if the current sample is idle but a newer busy
        # sample was latched (a busy tick the slow main loop missed),
        # publish the latched busy sample so the busy state is not lost.
        # If the current sample is itself busy, it is published directly
        # and the (redundant) latch watermark is advanced so the latched
        # sample is not re-published after the current goes idle.
        if self.llama_enabled and self._llm_busy_latch is not None:
            if _llm_entries_busy(entries):
                self._llm_busy_latch_published_id = self._llm_busy_latch_id
            elif self._llm_busy_latch_id > self._llm_busy_latch_published_id:
                entries = self._llm_busy_latch
                self._llm_busy_latch_published_id = self._llm_busy_latch_id
        metrics["llms"] = entries
        primary = entries[0] if entries else None
        metrics["llm"] = (primary
                          if primary is not None
                          and primary.get("status") != "down"
                          else None)
        # Multi-system consolidation: the local system first, then one
        # entry per configured upstream. `list()` snapshot: start_upstreams
        # may still be appending.
        #
        # The local system's data is already in the top-level keys, so its
        # systems[].metrics is a SLIM fleet projection (cpu, memory, gpu
        # utilization, llm gen) — just what the fleet strip needs. The local
        # detail tab reads the top-level payload (templates/index.html), so
        # the redundant full re-embedding is gone. Built before
        # metrics["systems"] is assigned, so it can never self-nest `systems`.
        local_projection = {
            "cpu": metrics["cpu"],
            "memory": metrics["memory"],
            "gpus": [{"utilization_pct": g.get("utilization_pct")}
                    for g in metrics.get("gpus", [])],
            "llms": [{"status": e.get("status"),
                     "name": e.get("name"),
                     "gen_tokens_per_s": e.get("gen_tokens_per_s"),
                     "backend": e.get("backend")}
                    for e in metrics.get("llms", [])],
        }
        systems = [{"name": "local", "status": "up",
                   "metrics": local_projection}]
        for name in list(self._upstream_names):
            entry = self._upstream_latest.get(name)
            systems.append(entry if entry is not None else
                           {"name": name, "status": "down",
                            "metrics": None})
        metrics["systems"] = systems
        return metrics

    def run(self) -> None:
        self._llm_thread.start()
        # Wait for the first LLM cycle so the first snapshot is
        # complete (matches the previous inline behavior); capped so a
        # hanging probe cannot stall startup forever.
        self._llm_ready.wait(timeout=10)
        # Fixed-rate scheduling: cycle k is due at epoch + k * interval,
        # independent of how long the previous _collect() took (the old
        # sleep-after-collect made period = interval + collect cost).
        next_ts = time.monotonic()
        while self.running:
            slot_start = next_ts
            cycle_start = time.monotonic()
            metrics: dict[str, Any] | None = None
            try:
                metrics = self._collect()
            except Exception as e:  # noqa: BLE001
                # Rate-limit the log so a persistently failing main
                # collection does not flood the log every cycle (T7).
                self._record_error("main", repr(e), exc=e,
                                  log_msg="Error collecting metrics")
            t_end = time.monotonic()
            collect_s = t_end - cycle_start
            # Record the wall-clock duration of the last main collection
            # cycle for /health (T7).
            self._last_cycle_duration = collect_s
            # On success, clear any prior "main" last error (recovery is
            # logged once by _clear_error) (T7).
            if metrics is not None:
                self._clear_error("main", log_msg="main metrics collection")
            # Advance the schedule from the fixed epoch, not from now.
            next_ts += self.poll_interval
            # A cycle that overruns the next slot does not burst-
            # catch-up: missed slots are skipped and the next cycle
            # starts at the next due slot.
            skipped = 0
            while next_ts < t_end:
                next_ts += self.poll_interval
                skipped += 1
            if skipped:
                log.warning(
                    "cycle overran the schedule by %.2fs; skipping %d "
                    "slot(s) (no burst catch-up)",
                    t_end - (slot_start + self.poll_interval), skipped)
            # Per-cycle duration + drift vs schedule: one DEBUG line per
            # cycle (no flood at the default INFO level); a cycle that
            # finishes well past its slot is surfaced at INFO.
            drift = t_end - (slot_start + self.poll_interval)
            log.debug("cycle: collect=%.2fs drift=%+.2fs",
                      collect_s, drift)
            if drift > max(0.1, self.poll_interval * 0.25):
                log.info("cycle late: collect=%.2fs, %.2fs past its "
                         "slot", collect_s, drift)
            if metrics is not None:
                self.publish(json.dumps(metrics))
            time.sleep(max(0.0, next_ts - time.monotonic()))


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__)
# Created (NOT started) at module level; started by apply_settings() so
# the collector runs on the resolved settings, not built-in defaults.
collector = MetricsCollector(poll_interval=1.0)
# True once the settings have been applied (by main() for a direct serve,
# or by the lazy before_request hook for a WSGI/import-based serve).
# Prevents re-resolving/re-applying the settings on every request, and
# keeps the collector from starting at import time (which would do real
# scrape work during tests / before the app is actually served).
_settings_applied = False


# Initial system tab shown on page load ("local" | "fleet"); set from
# settings in apply_settings() before the server starts.
_default_view = "local"


@app.route("/")
def index() -> str:
    """Serve the dashboard HTML page."""
    return render_template("index.html", hostname=platform.node(),
                           default_view=_default_view)


@app.route("/events")
def events() -> Response:
    """SSE endpoint – streams JSON metrics to connected clients. Each client
    gets its own per-client buffer (broadcast), so a blocked client cannot
    starve the others or /api/metrics."""
    def generate() -> Generator[str, None, None]:
        q = collector.subscribe()
        try:
            while True:
                try:
                    data = q.get(timeout=5.0)
                except queue.Empty:
                    # Keepalive comment – prevents proxy / browser timeout
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {data}\n\n"
        finally:
            collector.unsubscribe(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.route("/api/metrics")
def api_metrics() -> Response:
    """One-shot JSON endpoint (useful for testing / non-SSE clients)."""
    data = collector.get_latest(timeout=2.0)
    if data is None:
        return Response(
            json.dumps({"error": "no data yet"}),
            status=503,
            mimetype="application/json",
        )
    return Response(data, mimetype="application/json")


@app.route("/health")
def health() -> Response:
    """Self-observability endpoint (T7): the last main cycle duration, the
    per-target last-error map (a target with no recent error is absent),
    and a few sensible extras. Read-only; does not consume or mutate the
    metrics snapshot."""
    return Response(json.dumps(collector.health()),
                    mimetype="application/json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (shared by main() and the WSGI entry
    point so the no-CLI namespace matches the real defaults)."""
    parser = argparse.ArgumentParser(description="Resource & GPU Utilization Dashboard")
    parser.add_argument("--config", default=None,
                        help="YAML config file (default: ./config.yaml when "
                             "present). CLI arguments override config values, "
                             "which override built-in defaults.")
    parser.add_argument("--host", default=None, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: 5000)")
    parser.add_argument("--poll-interval", type=float, default=None,
                        help="Poll interval in seconds (default: 1.0)")
    parser.add_argument("--llama-url", default=None,
                        help="Base URL of llama.cpp llama-server "
                             "(expects /metrics enabled via --metrics). "
                             "The default http://127.0.0.1:5001 works when the dashboard and "
                             "llama-server run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<llama-host>:5001")
    parser.add_argument("--vllm-url", default=None,
                        help="Base URL of a vLLM server (expects /metrics). "
                             "The default http://127.0.0.1:5001 works when the dashboard and "
                             "vLLM run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<vllm-host>:8000")
    parser.add_argument("--litellm-url", default=None,
                        help="Base URL of a LiteLLM proxy (expects /metrics with the "
                             "prometheus callback enabled). "
                             "The default http://127.0.0.1:4000 works when the dashboard and "
                             "the proxy run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<litellm-host>:4000")
    parser.add_argument("--litellm-token", default=None,
                        help="Bearer token for the LiteLLM /metrics endpoint "
                             "(only needed when require_auth_for_metrics_endpoint is set)")
    parser.add_argument("--strata-url", default=None,
                        help="Base URL of a Strata inference engine "
                             "(expects its JSON /metrics endpoint). "
                             "The default http://127.0.0.1:8080 works when the dashboard and "
                             "Strata run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<strata-host>:8080")
    parser.add_argument("--llm-backend", default=None,
                        choices=["auto", "llama", "vllm", "litellm",
                                 "strata"],
                        help="LLM backend selection (default: auto). "
                             "auto tries vLLM first, then the LiteLLM proxy, "
                             "then Strata, then llama.cpp; pin to 'llama', "
                             "'vllm', 'litellm' or 'strata' to skip the others.")
    parser.add_argument("--default-view", default=None,
                        choices=["local", "fleet"],
                        help="System tab shown on page load (default: local). "
                             "'fleet' opens the multi-system summary view.")
    parser.add_argument("--no-llama", action="store_true",
                        help="disable the LLM section (enabled by default)")
    parser.add_argument("--upstream", action="append", default=[],
                        metavar="NAME=URL",
                        help="Upstream dashboard to consolidate (repeatable; also "
                             "available as config.yaml `upstreams`). Each upstream "
                             "runs this same app, e.g. NAME=http://host:5000")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (defaults to sys.argv[1:]). Passing [] yields the
    no-CLI namespace (all defaults) used by the WSGI entry point."""
    return _build_parser().parse_args(argv)


def apply_settings(settings: dict[str, Any]) -> None:
    """Apply every resolved setting to the module-level collector and start
    it. This is the single place settings are applied, so the collector
    runs on the resolved settings regardless of how the app is served
    (direct `python app.py` or WSGI/import-based). Idempotent: a second
    call re-applies the settings (a no-op) but does not re-start the
    already-running collector."""
    global _default_view, _settings_applied
    collector.poll_interval = settings["poll_interval"]
    collector.llama_url = settings["llama_url"]
    collector.llama_enabled = settings["llama_enabled"]
    collector.vllm_url = settings["vllm_url"]
    collector.litellm_url = settings["litellm_url"]
    collector.litellm_token = settings["litellm_token"]
    collector.strata_url = settings["strata_url"]
    collector.llm_backend = settings["llm_backend"]
    # Multi-LLM: atomic rebind of the target list (single attribute
    # assignment; the scrape loop snapshots it per tick). The legacy
    # attributes above stay meaningful for the implicit auto target.
    collector.llm_targets = settings["llm_targets"]
    collector.start_upstreams(settings["upstreams"])
    _default_view = settings["default_view"]  # module scope; read by index()
    # Guard against double-start: MetricsCollector is a threading.Thread,
    # whose start() raises RuntimeError if already started (e.g. apply_
    # settings called twice). A second call is a no-op.
    try:
        collector.start()
    except RuntimeError:
        pass
    _settings_applied = True


def _ensure_bootstrap() -> None:
    """Lazy bootstrap for a WSGI/import-based serve: on the first request,
    if the settings have not yet been applied (i.e. the app was imported by
    a WSGI server rather than run via main()), resolve the settings from the
    config (no CLI) and apply them (starting the collector on the resolved
    settings). Subsequent calls are a no-op. This keeps the collector from
    starting at import time (which would do real scrape work during tests /
    before the app is actually served)."""
    if _settings_applied:
        return
    config = load_config(None)
    args = _parse_args([])
    settings = resolve_settings(args, config)
    apply_settings(settings)


@app.before_request
def _lazy_bootstrap() -> None:
    _ensure_bootstrap()


def main() -> None:
    """Explicit bootstrap for a direct serve (`python app.py`): resolve the
    settings (CLI > config > defaults), apply them (starting the collector
    on the resolved settings), and run the dev server."""
    args = _parse_args()
    config = load_config(args.config, explicit=args.config is not None)
    settings = resolve_settings(args, config)
    apply_settings(settings)
    llm_status = "disabled" if not settings["llama_enabled"] else (
        "targets=" + ", ".join(
            f"{t.name}:{t.backend}" + (f"@{t.url}" if t.url else "")
            for t in settings["llm_targets"]))
    up_status = (f", upstreams={', '.join(u['name'] for u in settings['upstreams'])}"
                if settings["upstreams"] else "")
    log.info("Starting Resource Dashboard on http://%s:%d (LLM: %s%s)",
             settings["host"], settings["port"], llm_status, up_status)
    # waitress is a production WSGI server (a thread per connection), so a
    # client holding the long-lived SSE /events stream does not block other
    # requests (incl. /api/metrics from an aggregating dashboard).
    from waitress import serve
    serve(app, host=settings["host"], port=settings["port"], threads=20)


def _bootstrap_wsgi() -> Flask:
    """Eager WSGI/import-based bootstrap (optional): resolve the settings
    from the config (no CLI) and apply them (starting the collector on the
    resolved settings). NOT called at import time (see the lazy
    before_request hook above); kept for an eager bootstrap if desired."""
    config = load_config(None)
    args = _parse_args([])
    settings = resolve_settings(args, config)
    apply_settings(settings)
    return app


# WSGI entry point (e.g. `waitress-serve app:wsgi`, T10). `wsgi` is the Flask
# app (a WSGI callable); the settings are applied lazily on the first request
# (via the before_request hook) so the collector does not start at import time.
# When run directly (`python app.py`), `main()` applies the settings explicitly
# before `app.run(...)` and the before_request hook is a no-op.
wsgi = app


if __name__ == "__main__":
    main()
