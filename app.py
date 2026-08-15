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
import queue
import subprocess
import threading
import time
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
    """Extract manufacturer name from an lspci Subsystem line.

    Expected format::

        Subsystem: Dell Device 1234

    Returns the first token after "Subsystem:" (e.g. "Dell").
    """
    for line in output.splitlines():
        if "Subsystem:" in line:
            vendor_part = line.split("Subsystem:", 1)[1].strip()
            return vendor_part.split()[0] if vendor_part else "Unknown"
    return "Unknown"


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
# Metrics Collector Thread
# ---------------------------------------------------------------------------

class MetricsCollector(threading.Thread):
    """Background thread that polls system + GPU metrics and pushes JSON to a queue.

    The queue is bounded; when full the *oldest* entry is discarded so consumers
    always see the freshest data.
    """

    def __init__(self, poll_interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
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
    args = parser.parse_args()

    log.info("Starting Resource Dashboard on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)
