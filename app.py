#!/usr/bin/env python3
"""Resource & GPU Utilization Dashboard.

A lightweight Flask-based web dashboard that streams real-time CPU, memory,
and NVIDIA GPU metrics via Server-Sent Events (SSE).

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
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Generator

import psutil
from flask import Flask, Response, render_template

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# lspci-based board vendor lookup
# ---------------------------------------------------------------------------

_lspci_cache: dict[str, str] | None = None  # bus_id → vendor name


def _lspci_vendor_map(bus_ids: list[str]) -> dict[str, str]:
    global _lspci_cache
    if _lspci_cache is None:
        _lspci_cache = {}

    mapping: dict[str, str] = {}
    for bus_id in bus_ids:
        # Return cached result immediately if we already have it
        if bus_id in _lspci_cache:
            mapping[bus_id] = _lspci_cache[bus_id]
            continue

        try:
            result = subprocess.run(  # noqa: S603 – static args only
                ["lspci", "-s", bus_id, "-vm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            vendor_name = _parse_lspci_subsystem(result.stdout)
        except FileNotFoundError:
            log.info("lspci not found – board vendor lookup unavailable")
            break
        except subprocess.TimeoutExpired:
            log.warning("lspci timed out for %s", bus_id)
            break

        _lspci_cache[bus_id] = vendor_name
        mapping[bus_id] = vendor_name

    return mapping


def _parse_lspci_subsystem(output: str) -> str:
    """Extract the board vendor name from lspci output.

    Handles both real-world formats::

        # ``lspci -s <bus> -vm`` (machine-readable, tab-separated):
        Slot:\t01:00.0
        Class:\tVGA compatible controller
        Vendor:\tNVIDIA Corporation
        Device:\tGA102 [GeForce RTX 3080]
        SVendor:\tGigabyte Technology Co., Ltd
        SDevice:\tDevice 1234

        # ``lspci -s <bus> -v`` (verbose):
        01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3080]
                Subsystem: Dell Device 5678

    An "SVendor:" line is preferred; the first token after it is returned
    (e.g. "Gigabyte", "Dell"). If no SVendor line exists, falls back to the
    first token after a "Subsystem:" line. Returns an empty string when
    neither line is present so callers can apply their own fallbacks
    (hex id / "unknown").
    """
    for line in output.splitlines():
        if "SVendor:" in line:
            vendor_part = line.split("SVendor:", 1)[1].strip()
            return vendor_part.split()[0] if vendor_part else ""
    for line in output.splitlines():
        if "Subsystem:" in line:
            vendor_part = line.split("Subsystem:", 1)[1].strip()
            return vendor_part.split()[0] if vendor_part else ""
    return ""


# ---------------------------------------------------------------------------
# GPU Monitoring
# ---------------------------------------------------------------------------

def _gpu_pynvml() -> list[dict[str, Any]]:
    """Collect GPU metrics via the pynvml (NVIDIA Management Library) bindings.

    Returns a list of dicts — one per GPU device.  Returns an empty list on
    any error so the caller can fall back to CLI parsing.
    """
    try:
        import pynvml
    except ImportError:
        log.warning("pynvml not installed – falling back to nvidia-smi CLI")
        return []

    gpus: list[dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:  # noqa: BLE001
        log.warning("pynvml initialisation failed (%s) – falling back to CLI", exc)
        return []

    for idx in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_GRAPHICS
            )

            # Identification: name, PCIe bus ID
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = (
                    pci_info.busId.decode()
                    if isinstance(pci_info.busId, bytes)
                    else str(pci_info.busId)
                ).strip("\x00")
            except Exception:
                bus_id = "unknown"

            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
            except Exception:
                name = "Unknown GPU"

            # Board vendor from lspci (cached internally) or fallback to raw hex
            board_vendor = _lspci_vendor_map([bus_id]).get(bus_id)
            if not board_vendor:
                try:
                    board_vendor = f"0x{pci_info.pciSubSystemId & 0xFFFF:04X}"
                except (NameError, AttributeError):
                    board_vendor = "unknown"

            # Throttle-reason bitmask
            throttle_bits = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            hw_thermal = bool(
                throttle_bits & pynvml.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN
            )
            sw_thermal = bool(
                throttle_bits & pynvml.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN
            )

            mem_total = mem.total if mem.total else 1  # guard div-by-zero
            gpu: dict[str, Any] = {
                "index": idx,
                "utilization_pct": util.gpu,
                "memory_used_gb": round(mem.used / (1024**3), 1),
                "memory_total_gb": round(mem.total / (1024**3), 1),
                "memory_pct": round(mem.used / mem_total * 100, 1),
                "temperature_c": temp,
                "graphics_clock_mhz": clock_mhz,
                "throttle": {
                    "hw_thermal_slowdown": hw_thermal,
                    "sw_thermal_slowdown": sw_thermal,
                },
                "name": name,
                "pci_bus_id": bus_id,
                "board_vendor": board_vendor,
            }
            # Optional power fields (mW -> W). Some devices/VMs do not support
            # power queries; omit the key entirely on failure so old and new
            # clients both degrade gracefully.
            try:
                gpu["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:  # noqa: BLE001 – per-field errors are non-fatal
                pass
            try:
                gpu["power_limit_w"] = round(
                    pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1
                )
            except Exception:  # noqa: BLE001 – per-field errors are non-fatal
                pass
            gpus.append(gpu)
        except Exception:  # noqa: BLE001 – per-device errors are non-fatal
            log.debug("Skipping GPU %d (pynvml error)", idx)

    return gpus


def _gpu_cli() -> list[dict[str, Any]]:
    """Collect GPU metrics by parsing ``nvidia-smi`` CSV output.

    Used as a fallback when pynvml is unavailable or fails to initialise.
    """
    gpus: list[dict[str, Any]] = []
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,clocks.current.graphics,"
                "clocks_throttle_reasons.hw_thermal_slowdown,"
                "clocks_throttle_reasons.sw_thermal_slowdown,"
                "pci.bus_id,name,pci.device_id,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        reader = csv.reader(io.StringIO(result.stdout))
        for idx, row in enumerate(reader):
            if len(row) < 12:
                continue
            mem_used_mb = int(row[1])
            mem_total_mb = max(int(row[2]), 1)

            # Board vendor from lspci or fallback to raw hex
            bus_id = row[7].strip()
            board_vendor = _lspci_vendor_map([bus_id]).get(bus_id)
            if not board_vendor:
                raw_device_id = row[9].strip()
                try:
                    board_vendor = f"0x{int(raw_device_id, 16) & 0xFFFF:04X}"
                except ValueError:
                    board_vendor = "unknown"

            # Optional power fields – nvidia-smi reports "[Not Supported]"
            # (or "[N/A]") when the query is unavailable on this GPU.
            power_w: float | None = None
            try:
                power_w = round(float(row[10]), 1)
            except ValueError:
                pass
            power_limit_w: float | None = None
            try:
                power_limit_w = round(float(row[11]), 1)
            except ValueError:
                pass

            gpu: dict[str, Any] = {
                "index": idx,
                "utilization_pct": int(row[0]),
                "memory_used_gb": round(mem_used_mb / 1024, 1),
                "memory_total_gb": round(mem_total_mb / 1024, 1),
                "memory_pct": round(mem_used_mb / mem_total_mb * 100, 1),
                "temperature_c": int(row[3]),
                "graphics_clock_mhz": int(row[4]),
                "throttle": {
                    "hw_thermal_slowdown": row[5].strip().lower() == "active",
                    "sw_thermal_slowdown": row[6].strip().lower() == "active",
                },
                "name": row[8].strip(),
                "pci_bus_id": bus_id,
                "board_vendor": board_vendor,
            }
            if power_w is not None:
                gpu["power_w"] = power_w
            if power_limit_w is not None:
                gpu["power_limit_w"] = power_limit_w
            gpus.append(gpu)
    except FileNotFoundError:
        log.info("nvidia-smi not found – no GPU metrics available")
    except subprocess.TimeoutExpired:
        log.warning("nvidia-smi timed out")

    return gpus


def collect_gpu_metrics() -> list[dict[str, Any]]:
    """Return GPU metrics via pynvml (preferred) or CLI fallback."""
    gpus = _gpu_pynvml()
    if not gpus:
        gpus = _gpu_cli()
    return gpus


# ---------------------------------------------------------------------------
# llama.cpp (llama-server) Monitoring
# ---------------------------------------------------------------------------

# Base URL of the local llama-server (overridable via --llama-url).
DEFAULT_LLAMA_URL = "http://127.0.0.1:5001"

# Base URL of the vLLM server (overridable via --vllm-url).
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"

# How often (seconds) to re-fetch /props; model info only changes on reload.
_LLAMA_PROPS_TTL = 30.0
_llama_props_cache: tuple[float, dict[str, Any]] | None = None

# Previous successful scrape's cumulative counters + timestamp, used to
# compute tokens/s from per-scrape counter deltas. The windowed
# ``*_tokens_seconds`` gauges read 0 on some builds (e.g. the llama.cpp
# bundled with LM Studio) while the cumulative counters increment correctly.
_llama_prev: dict[str, float] | None = None

# Last successful snapshot (monotonic ts + dict). When a scrape fails
# (e.g. the 2 s timeout under heavy llama-server load) the UI keeps showing
# this data, flagged "stale", for up to this many seconds so a single bad
# tick doesn't flicker the card to "unreachable".
_LLAMA_STALE_MAX_AGE_S = 15.0
_llama_last_good: tuple[float, dict[str, Any]] | None = None

# Per-slot decode counters from the previous successful scrape, used for
# LIVE tokens/s: on some builds (e.g. the llama.cpp bundled with LM Studio)
# the /metrics counters only advance when a request COMPLETES, so counter
# deltas read 0 during generation while the per-slot n_decoded /
# n_prompt_tokens_processed update per token.
_llama_slot_prev: dict[str, Any] | None = None

# (monotonic ts, n_decode_total) pairs from the last successful scrapes.
# n_decode_total ticks per llama_decode() call; frozen across 10 consecutive
# successful scrapes while a slot is mid-generation means the server is
# hung (stuck decode / GPU wedged). Long prefills call no llama_decode, so
# slots with decoded == 0 are excluded (see the stall check below).
_LLAMA_DECODE_HISTORY_LEN = 10
_llama_decode_history: list[tuple[float, float]] = []

# Local llama-server process handle (T5): retained across scrapes so
# cpu_percent(None) measures against the previous tick. A dead handle
# (NoSuchProcess) is dropped and re-identified on the next scrape — a
# model reload spawns a new PID.
_llama_proc: psutil.Process | None = None

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


def _llama_find_process() -> psutil.Process | None:
    """Find the local llama-server process. Never raises.

    Name match first (the confirmed host binary is
    ``/home/joshua/llama.cpp/build/bin/llama-server``, psutil name
    ``llama-server``), then a cmdline-argument fallback; multiple matches
    resolve to the largest RSS. Returns ``None`` when no local process
    exists (remote/dev mode). A retained live handle is reused so
    ``cpu_percent(None)`` keeps measuring; it was primed (first real
    value on the scrape after identification).
    """
    global _llama_proc
    if _llama_proc is not None:
        try:
            _llama_proc.memory_info().rss
            return _llama_proc
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            _llama_proc = None
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
        _llama_proc = proc
        return proc
    except Exception:  # noqa: BLE001 – collector must never raise
        return None


def _llama_props(base_url: str) -> dict[str, Any] | None:
    """Fetch /props with a 30 s cache (model info only changes on reload)."""
    global _llama_props_cache
    now = time.monotonic()
    if _llama_props_cache is not None:
        ts, data = _llama_props_cache
        if now - ts < _LLAMA_PROPS_TTL:
            return data
    data = _llama_get_json(base_url, "/props")
    if isinstance(data, dict):
        _llama_props_cache = (now, data)
    return data


# ---------------------------------------------------------------------------
# Cumulative usage state (persistent server uptime, T3)
# ---------------------------------------------------------------------------

def _llama_usage_state_path() -> str:
    """State-file location; RD_LLAMA_USAGE_STATE overrides the default."""
    return os.environ.get("RD_LLAMA_USAGE_STATE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".llama_usage_state.json"
    )


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


def collect_llama_metrics(base_url: str) -> dict[str, Any] | None:
    """Scrape llama-server ``/metrics`` + ``/props`` + ``/slots``.

    Returns ``None`` when the server is unreachable or not started with
    ``--metrics`` (UI shows "unreachable"). Never raises.

    ``gen_tokens_per_s`` / ``prompt_tokens_per_s`` are computed from the
    per-scrape deltas of the cumulative token counters; the server's
    windowed gauges are only a fallback for the first scrape (they read
    0 permanently on some builds, e.g. LM Studio's bundled llama.cpp).

    On a failed scrape the last successful snapshot is returned with
    ``status: "stale"`` for up to ``_LLAMA_STALE_MAX_AGE_S`` seconds so a
    transient timeout doesn't flicker the UI to "unreachable".
    """
    global _llama_prev, _llama_last_good, _llama_slot_prev, _llama_decode_history

    def _stale_or_none() -> dict[str, Any] | None:
        """Hold last-good data during a transient scrape failure."""
        if _llama_last_good is not None:
            ts, held = _llama_last_good
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
    prev = _llama_prev
    elapsed = now - prev["ts"] if prev is not None else 0.0
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
        _llama_prev = {
            "ts": now,
            "gen": gen_total,
            "prompt": prompt_total,
        }
        if cached_total is not None:
            _llama_prev["cached"] = cached_total

    props = _llama_props(base_url)
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
    # silently, no red state.
    proc = _llama_find_process()
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
        state_path = _llama_usage_state_path()
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
    if decode_total is not None:
        _llama_decode_history.append((now, decode_total))
        if len(_llama_decode_history) > _LLAMA_DECODE_HISTORY_LEN:
            _llama_decode_history.pop(0)
    if len(_llama_decode_history) >= _LLAMA_DECODE_HISTORY_LEN:
        if len({v for _, v in _llama_decode_history}) == 1:
            for slot in slots:
                if slot["is_processing"] and slot["decoded"] > 0:
                    out["stalled"] = True
                    break

    # Live tokens/s from per-slot decode counters (see _llama_slot_prev):
    # they update per token, unlike the /metrics counters on some builds.
    prev_slot = _llama_slot_prev
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
    _llama_slot_prev = {
        "ts": now,
        "slots": {
            s["id"]: {
                "decoded": s["decoded"],
                "prompt": s["prompt_processed"],
                "cached": s["cached"],
                "id_task": s["id_task"],
            }
            for s in slots
        },
        "prompt_last": prompt_last,
    }

    _llama_last_good = (now, out)
    return out


# ---------------------------------------------------------------------------
# vLLM Monitoring
# ---------------------------------------------------------------------------

# Stale-hold: last successful vLLM scrape (monotonic ts + dict).
_VLLM_STALE_MAX_AGE_S = 15.0
_vllm_last_good: tuple[float, dict[str, Any]] | None = None

# Previous successful scrape's cumulative counters + histogram sums, used
# to compute rates and windowed means from per-scrape deltas.
_vllm_prev: dict[str, float] | None = None

# 30 s cache for the parsed cache_config_info dict (static during runtime).
_VLLM_CONFIG_TTL = 30.0
_vllm_config_cache: tuple[float, dict[str, Any]] | None = None

# Warn-once guard for LLM auto mode: the first time vLLM comes up, tell the
# operator it is the active backend and how to pin the choice.
# Approximation (accepted): we do NOT probe the llama server just to verify
# it is also reachable, so this may fire when only vLLM is up.
_llm_tie_warned = False


def _parse_prometheus_labeled(text: str, prefix: str) -> dict[str, dict]:
    """Parse Prometheus text into ``{metric_name: {"series": {labelset: float}, "labels": {name: set(values)}}}``.

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


def collect_vllm_metrics(base_url: str) -> dict[str, Any] | None:
    """Scrape vLLM ``/metrics`` and build the dashboard snapshot.

    Returns ``None`` when the server is unreachable, or the body contains
    no ``vllm:`` series. Never raises.

    On a failed scrape the last successful snapshot is returned with
    ``status: "stale"`` for up to ``_VLLM_STALE_MAX_AGE_S`` seconds so a
    transient timeout doesn't flicker the UI to "unreachable".

    Counter semantics (confirmed against vLLM docs/source): prompt_tokens_cached_total
    is a SUBSET of prompt_tokens_total (nested, not disjoint). Usage total is
    therefore ``prompt_total + gen_total`` (NOT + cached).
    """
    global _vllm_last_good, _vllm_prev, _vllm_config_cache

    def _stale_or_none() -> dict[str, Any] | None:
        if _vllm_last_good is not None:
            ts, held = _vllm_last_good
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

    metrics = _parse_prometheus_labeled(body, "vllm:")
    if not any(k.startswith("vllm:") for k in metrics):
        return _stale_or_none()  # endpoint up but no vllm: metrics

    out: dict[str, Any] = {"status": "up"}
    now = time.monotonic()
    prev = _vllm_prev
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
    rate_ok = prev is not None and 0.5 <= elapsed <= 5.0
    if rate_ok and gen_total is not None and prompt_total is not None:
        if gen_total < prev.get("gen", 0):
            out["gen_tokens_per_s"] = 0.0  # counter reset
        else:
            out["gen_tokens_per_s"] = round(
                max(0.0, gen_total - prev["gen"]) / elapsed, 1)
        if prompt_total < prev.get("prompt", 0):
            out["prompt_tokens_per_s"] = 0.0  # counter reset
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

    # --- Config (30 s cache) ---
    cfg_entry = _vllm_get_entry(metrics, "vllm:cache_config_info")
    if cfg_entry is not None and cfg_entry.get("series"):
        now_ts = time.monotonic()
        if _vllm_config_cache is not None:
            cache_ts, cached_cfg = _vllm_config_cache
            if now_ts - cache_ts < _VLLM_CONFIG_TTL:
                out["config"] = cached_cfg
            else:
                out["config"] = _parse_vllm_config(cfg_entry)
                _vllm_config_cache = (now_ts, out["config"])
        else:
            out["config"] = _parse_vllm_config(cfg_entry)
            _vllm_config_cache = (now_ts, out["config"])

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
    if ttft_sum is not None:
        new_prev["ttft_sum"] = ttft_sum
    if ttft_count is not None:
        new_prev["ttft_count"] = ttft_count
    if itl_sum is not None:
        new_prev["itl_sum"] = itl_sum
    if itl_count is not None:
        new_prev["itl_count"] = itl_count
    _vllm_prev = new_prev

    _vllm_last_good = (now, out)
    return out


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

class MetricsCollector(threading.Thread):
    """Background thread that polls system + GPU metrics and pushes JSON to a queue.

    The queue is bounded; when full the *oldest* entry is discarded so consumers
    always see the freshest data.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        llama_url: str = DEFAULT_LLAMA_URL,
        llama_enabled: bool = True,
        vllm_url: str = DEFAULT_VLLM_URL,
        llm_backend: str = "auto",
    ) -> None:
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.llama_url = llama_url
        self.llama_enabled = llama_enabled
        self.vllm_url = vllm_url
        self.llm_backend = llm_backend
        # Bounded queue – maxsize entries; oldest dropped when full.
        self._queue: queue.Queue[str] = queue.Queue(maxsize=30)
        self.running = True

        # Seed psutil (first cpu_percent call returns 0.0 and must be discarded)
        psutil.cpu_percent()

    # -- public API ----------------------------------------------------------

    def get_latest(self, timeout: float | None = None) -> str | None:
        """Return the most recent metrics JSON string (blocking with *timeout*)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- internal ------------------------------------------------------------

    def _drain_old(self) -> None:
        """Discard all but the newest entry so stale data never accumulates."""
        while self._queue.qsize() > 1:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _collect_llm(self) -> dict[str, Any] | None:
        """Stateless LLM backend selection (re-evaluated every cycle).

        auto semantics: try vLLM first, fall through to llama. The
        per-collector 15 s stale-hold IS the hysteresis: a <15 s blip
        keeps the current backend flagged stale; a real outage falls
        through to the other backend, and recovery flips back on the
        next cycle. Pinned backend skips the other branch entirely.
        """
        global _llm_tie_warned
        if not self.llama_enabled:
            return None
        if self.llm_backend in ("auto", "vllm"):
            data = collect_vllm_metrics(self.vllm_url)
            if data is not None:
                data["backend"] = "vllm"
                # Warn once: vLLM is silently preferred in auto mode. We
                # do not probe the llama server here (keeps the cycle cheap),
                # so this fires the first time vLLM is active in auto mode
                # even if llama is down - the accepted approximation.
                if self.llm_backend == "auto" and not _llm_tie_warned:
                    _llm_tie_warned = True
                    log.warning(
                        "LLM auto mode: using vLLM at %s. If llama.cpp is "
                        "also reachable and you want it instead, pin "
                        "--llm-backend llama (logged once).",
                        self.vllm_url)
                return data
        if self.llm_backend in ("auto", "llama"):
            data = collect_llama_metrics(self.llama_url)
            if data is not None:
                data["backend"] = "llama"
                return data
        return None

    def _collect(self) -> dict[str, Any]:
        """Snapshot all metrics once (non-blocking)."""
        cpu_overall = psutil.cpu_percent(percpu=False)
        cpu_per_core = psutil.cpu_percent(percpu=True)

        vm = psutil.virtual_memory()

        return {
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
            "llm": self._collect_llm(),
        }

    def run(self) -> None:
        while self.running:
            try:
                metrics = self._collect()
            except Exception:  # noqa: BLE001
                log.exception("Error collecting metrics")
                time.sleep(self.poll_interval)
                continue

            payload = json.dumps(metrics)
            if self._queue.full():
                self._drain_old()
            self._queue.put_nowait(payload)
            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__)
collector = MetricsCollector(poll_interval=1.0)
collector.start()


@app.route("/")
def index() -> str:
    """Serve the dashboard HTML page."""
    return render_template("index.html")


@app.route("/events")
def events() -> Response:
    """SSE endpoint – streams JSON metrics to connected clients."""
    def generate() -> Generator[str, None, None]:
        while True:
            data = collector.get_latest(timeout=5.0)
            if data is not None:
                yield f"data: {data}\n\n"
            else:
                # Keepalive comment – prevents proxy / browser timeout
                yield ": keepalive\n\n"

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resource & GPU Utilization Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    parser.add_argument("--llama-url", default=DEFAULT_LLAMA_URL,
                        help="Base URL of llama.cpp llama-server (expects /metrics enabled via --metrics). "
                             "The default http://127.0.0.1:5001 works when the dashboard and "
                             "llama-server run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<llama-host>:5001")
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL,
                        help="Base URL of a vLLM server (expects /metrics). "
                             "The default http://127.0.0.1:8000 works when the dashboard and "
                             "vLLM run on the same host; pass a remote URL otherwise, "
                             "e.g. http://<vllm-host>:8000")
    parser.add_argument("--llm-backend", default="auto", choices=["auto", "llama", "vllm"],
                        help="LLM backend selection (default: auto). "
                             "auto tries vLLM first, then llama.cpp; "
                             "pin to 'llama' or 'vllm' to skip the other.")
    parser.add_argument("--no-llama", action="store_true",
                        help="disable the LLM section (enabled by default)")
    args = parser.parse_args()

    collector.llama_url = args.llama_url
    collector.llama_enabled = not args.no_llama
    collector.vllm_url = args.vllm_url
    collector.llm_backend = args.llm_backend

    llm_status = "disabled" if args.no_llama else f"backend={args.llm_backend}, vllm={args.vllm_url}, llama={args.llama_url}"
    log.info("Starting Resource Dashboard on http://%s:%d (LLM: %s)",
             args.host, args.port, llm_status)
    app.run(host=args.host, port=args.port, debug=False)
