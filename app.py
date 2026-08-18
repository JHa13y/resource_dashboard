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
            gpus.append({
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
            })
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
                "pci.bus_id,name,pci.device_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        reader = csv.reader(io.StringIO(result.stdout))
        for idx, row in enumerate(reader):
            if len(row) < 10:
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

            gpus.append({
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
            })
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
            decoded = (
                int(next_tok.get("n_decoded", 0) or 0)
                if isinstance(next_tok, dict)
                else 0
            )
            used = prompt + decoded
            slots.append({
                "id": slot.get("id"),
                "n_ctx": n_ctx,
                "context_used": used,
                "context_pct": round(100.0 * used / n_ctx, 1) if n_ctx else 0.0,
                "is_processing": bool(slot.get("is_processing", False)),
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
    global _llama_prev, _llama_last_good

    def _stale_or_none() -> dict[str, Any] | None:
        """Hold last-good data during a transient scrape failure."""
        if _llama_last_good is not None:
            ts, held = _llama_last_good
            if time.monotonic() - ts <= _LLAMA_STALE_MAX_AGE_S:
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

    out["slots"] = _parse_llama_slots(_llama_get_json(base_url, "/slots"))
    _llama_last_good = (now, out)
    return out


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
    ) -> None:
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.llama_url = llama_url
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
            "llama": collect_llama_metrics(self.llama_url),
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
    args = parser.parse_args()

    collector.llama_url = args.llama_url

    log.info("Starting Resource Dashboard on http://%s:%d (llama-server: %s)",
             args.host, args.port, args.llama_url)
    app.run(host=args.host, port=args.port, debug=False)
