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
# Negative cache for failed/timed-out lookups (bus_id → monotonic ts):
# a hung or missing lspci must not stall the snapshot path on every
# poll — the failure is served from this cache and the subprocess is
# re-run only after _LSPICI_NEGATIVE_TTL (same pattern as
# _AMD_NEGATIVE_REPROBE_S).
_lspci_negative: dict[str, float] = {}
_LSPICI_NEGATIVE_TTL = 300.0


def _lspci_vendor_map(bus_ids: list[str]) -> dict[str, str]:
    global _lspci_cache
    if _lspci_cache is None:
        _lspci_cache = {}
    now = time.monotonic()

    mapping: dict[str, str] = {}
    for bus_id in bus_ids:
        # Return cached result immediately if we already have it
        if bus_id in _lspci_cache:
            mapping[bus_id] = _lspci_cache[bus_id]
            continue
        failed_at = _lspci_negative.get(bus_id)
        if failed_at is not None and \
                (now - failed_at) < _LSPICI_NEGATIVE_TTL:
            mapping[bus_id] = ""  # cached failure — no subprocess
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
            log.info("lspci not found – board vendor lookup unavailable "
                     "(negative-cached %.0fs)", _LSPICI_NEGATIVE_TTL)
            _lspci_negative.update({b: now for b in bus_ids})
            break
        except subprocess.TimeoutExpired:
            log.warning("lspci timed out for %s "
                       "(negative-cached %.0fs)",
                       bus_id, _LSPICI_NEGATIVE_TTL)
            _lspci_negative.update({b: now for b in bus_ids})
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
    global _nvidia_cli_missing_logged
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
        if not _nvidia_cli_missing_logged:
            log.info("nvidia-smi not found – no NVIDIA GPU metrics available")
            _nvidia_cli_missing_logged = True
    except subprocess.TimeoutExpired:
        log.warning("nvidia-smi timed out")

    return gpus


# ---------------------------------------------------------------------------
# AMD GPU Monitoring (amd-smi → rocm-smi → sysfs)
# ---------------------------------------------------------------------------
#
# Evidence: docs/research/amd-smi.md. amd-smi ships with ROCm 6.x+ (the
# amd-smi-lib package, /opt/rocm/bin); it is the official replacement for
# rocm-smi. JSON is the most stable machine contract — CSV column order is
# insertion-order and can shift between versions, and the human tables
# change by platform (XCP column, dynamic GTT/VRAM header). amd-smi
# reports VRAM in MB; legacy rocm-smi reports VRAM in BYTES (1024×). "N/A"
# is the universal missing-value token and is treated as absent, never 0.

_AMD_SMI_BIN_CANDIDATES = ("amd-smi", "/opt/rocm/bin/amd-smi")
_ROCM_SMI_BIN_CANDIDATES = ("rocm-smi", "/opt/rocm/bin/rocm-smi")

# Warn-once guard: nvidia-smi absent (e.g. pure AMD box) — the fallback
# chain now continues to the AMD path, so the "no GPU metrics" message
# would otherwise fire every poll.
_nvidia_cli_missing_logged = False

# lspci device-name cache (bus_id → "Radeon RX 7900 XTX" style name).
_lspci_name_cache: dict[str, str] | None = None
# Failed/timed-out GPU-name lookups (bus_id → monotonic ts); see
# _lspci_negative — same TTL, same no-stall guarantee.
_lspci_name_negative: dict[str, float] = {}


def _find_cli_binary(candidates: tuple[str, ...]) -> str | None:
    """Resolve a CLI from PATH candidates then absolute ROCm paths."""
    import shutil

    for cand in candidates:
        if os.path.isabs(cand):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        elif shutil.which(cand) is not None:
            return cand
    return None


def _sysfs_pci_bus_id(card: int | str) -> str | None:
    """PCI BDF (e.g. ``0000:66:00.0``) for a DRM card from the sysfs
    ``device`` symlink. None on non-Linux hosts or any error."""
    try:
        target = os.readlink(f"/sys/class/drm/card{card}/device")
    except OSError:
        return None
    return os.path.basename(target) or None


def _parse_lspci_device(output: str) -> str:
    """Extract the board model name from ``lspci -s <bus> -vm`` output.

    Prefers the bracketed part of the ``Device:`` line (the marketing
    name on AMD cards, e.g. ``Device: Navi 31 [Radeon RX 7900 XTX]`` →
    ``Radeon RX 7900 XTX``); falls back to the raw remainder.
    """
    for line in output.splitlines():
        if line.startswith("Device:"):
            part = line.split("Device:", 1)[1].strip()
            m = re.search(r"\[([^\]]+)\]", part)
            return m.group(1).strip() if m else part
    return ""


def _lspci_device_name_map(bus_ids: list[str]) -> dict[str, str]:
    global _lspci_name_cache
    if _lspci_name_cache is None:
        _lspci_name_cache = {}
    now = time.monotonic()

    mapping: dict[str, str] = {}
    for bus_id in bus_ids:
        if bus_id in _lspci_name_cache:
            mapping[bus_id] = _lspci_name_cache[bus_id]
            continue
        failed_at = _lspci_name_negative.get(bus_id)
        if failed_at is not None and \
                (now - failed_at) < _LSPICI_NEGATIVE_TTL:
            mapping[bus_id] = ""  # cached failure — no subprocess
            continue
        try:
            result = subprocess.run(  # noqa: S603 – static args only
                ["lspci", "-s", bus_id, "-vm"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            name = _parse_lspci_device(result.stdout)
        except FileNotFoundError:
            log.info("lspci not found – GPU name lookup unavailable "
                     "(negative-cached %.0fs)", _LSPICI_NEGATIVE_TTL)
            _lspci_name_negative.update({b: now for b in bus_ids})
            break
        except subprocess.TimeoutExpired:
            log.warning("lspci timed out for %s "
                       "(negative-cached %.0fs)",
                       bus_id, _LSPICI_NEGATIVE_TTL)
            _lspci_name_negative.update({b: now for b in bus_ids})
            break
        _lspci_name_cache[bus_id] = name
        mapping[bus_id] = name

    return mapping


def _amd_unwrap(val: Any) -> float | None:
    """Unwrap an amd-smi JSON leaf: numerics arrive as
    ``{"value": N, "unit": "W"}`` (scalars pass through); ``"N/A"``,
    None and malformed values → None."""
    if isinstance(val, dict):
        val = val.get("value")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _amd_smi_entry_to_gpu(entry: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Map one ``amd-smi monitor --json`` object to the dashboard GPU dict.

    Field map (monitor keys, VRAM in MB): ``gfx`` → utilization %,
    ``vram_used``/``vram_total`` → memory, ``hotspot_temperature`` (fall
    back to ``memory_temperature`` — ``edge`` is N/A on datacenter cards),
    ``gfx_clock`` → MHz, ``power_usage`` → W. Rows with no usable data
    (e.g. secondary XCP rows on partitioned MI300X) are dropped.
    """
    util = _amd_unwrap(entry.get("gfx"))
    vram_used = _amd_unwrap(entry.get("vram_used"))
    vram_total = _amd_unwrap(entry.get("vram_total"))
    temp = _amd_unwrap(entry.get("hotspot_temperature"))
    if temp is None:
        temp = _amd_unwrap(entry.get("memory_temperature"))
    clock = _amd_unwrap(entry.get("gfx_clock"))
    power = _amd_unwrap(entry.get("power_usage"))

    if util is None and vram_used is None:
        return None

    bus_id = _sysfs_pci_bus_id(idx)
    board_vendor = "unknown"
    name = "AMD GPU"
    if bus_id:
        vendor_map = _lspci_vendor_map([bus_id]).get(bus_id)
        if vendor_map:
            board_vendor = vendor_map
        name_map = _lspci_device_name_map([bus_id]).get(bus_id)
        if name_map:
            name = name_map

    gpu: dict[str, Any] = {
        "index": idx,
        "throttle": {
            "hw_thermal_slowdown": False,
            "sw_thermal_slowdown": False,
        },
        "name": name,
        "board_vendor": board_vendor,
    }
    if bus_id:
        gpu["pci_bus_id"] = bus_id
    if util is not None:
        gpu["utilization_pct"] = util
    if vram_used is not None:
        gpu["memory_used_gb"] = round(vram_used / 1024, 1)
    if vram_total is not None:
        gpu["memory_total_gb"] = round(vram_total / 1024, 1)
    if vram_used is not None and vram_total:
        gpu["memory_pct"] = round(vram_used / vram_total * 100, 1)
    if temp is not None:
        gpu["temperature_c"] = round(temp, 1)
    if clock is not None:
        gpu["graphics_clock_mhz"] = round(clock, 1)
    if power is not None:
        gpu["power_w"] = round(power, 1)
    return gpu


def _gpu_amd_smi() -> list[dict[str, Any]]:
    """Collect AMD GPU metrics via ``amd-smi monitor --json`` (ROCm 6.x+).

    One-shot per poll (no watch mode — the dashboard owns the cadence).
    Returns [] when the tool is missing, fails, or no AMD GPU is found;
    never raises.
    """
    binary = _find_cli_binary(_AMD_SMI_BIN_CANDIDATES)
    if binary is None:
        return []
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [binary, "monitor", "--power-usage", "--temperature",
             "--gfx", "--vram-usage", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        log.warning("amd-smi timed out")
        return []
    if result.returncode != 0:
        return []  # no AMD GPUs / driver not loaded

    try:
        payload = json.loads(result.stdout)
    except ValueError:
        log.warning("amd-smi returned unparseable JSON")
        return []
    if not isinstance(payload, list) or not payload:
        return []

    gpus: list[dict[str, Any]] = []
    for pos, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("gpu")
        if not isinstance(idx, int) or isinstance(idx, bool):
            idx = pos
        gpu = _amd_smi_entry_to_gpu(entry, idx)
        if gpu is not None:
            gpus.append(gpu)
    return gpus


def _gpu_rocm_smi() -> list[dict[str, Any]]:
    """Legacy fallback (ROCm 5.x / pre-amd-smi builds): parse
    ``rocm-smi --showuse --showmeminfo vram --showtemperatures --csv``.

    Columns are located by header NAME, never position — the legacy tool
    uses entirely different names and reports VRAM in BYTES (amd-smi
    uses MB). ``device`` values are ``card0``/``card1``… Never raises.
    """
    binary = _find_cli_binary(_ROCM_SMI_BIN_CANDIDATES)
    if binary is None:
        return []
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [binary, "--showuse", "--showmeminfo", "vram",
             "--showtemperatures", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        log.warning("rocm-smi timed out")
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []

    rows = [r for r in csv.reader(io.StringIO(result.stdout))
            if r and not r[0].lstrip().startswith("#")]
    if len(rows) < 2:
        return []
    header = [h.strip() for h in rows[0]]

    def col(pred) -> int | None:
        for i, h in enumerate(header):
            if pred(h):
                return i
        return None

    i_util = col(lambda h: h == "GPU use (%)")
    i_vram_total = col(lambda h: h == "VRAM Total Memory (B)")
    i_vram_used = col(lambda h: h == "VRAM Total Used Memory (B)")
    i_temp = col(lambda h: h.startswith("Temperature (Sensor"))
    i_power = col(lambda h: h.endswith("Graphics Package Power (W)"))

    gpus: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[1:]):
        if not row or not row[0].strip():
            continue

        def fval(i: int | None) -> float | None:
            if i is None or i >= len(row):
                return None
            try:
                return float(row[i])
            except ValueError:
                return None  # "N/A" and friends

        util = fval(i_util)
        vram_used_b = fval(i_vram_used)
        vram_total_b = fval(i_vram_total)
        if util is None and vram_used_b is None:
            continue
        temp = fval(i_temp)
        power = fval(i_power)

        dev = row[0].strip()  # e.g. "card0"
        bus_id = _sysfs_pci_bus_id(dev)
        board_vendor = "unknown"
        name = "AMD GPU"
        if bus_id:
            vendor_map = _lspci_vendor_map([bus_id]).get(bus_id)
            if vendor_map:
                board_vendor = vendor_map
            name_map = _lspci_device_name_map([bus_id]).get(bus_id)
            if name_map:
                name = name_map

        gpu: dict[str, Any] = {
            "index": idx,
            "throttle": {
                "hw_thermal_slowdown": False,
                "sw_thermal_slowdown": False,
            },
            "name": name,
            "board_vendor": board_vendor,
        }
        if bus_id:
            gpu["pci_bus_id"] = bus_id
        if util is not None:
            gpu["utilization_pct"] = util
        if vram_used_b is not None:
            gpu["memory_used_gb"] = round(vram_used_b / 1024**3, 1)
        if vram_total_b is not None:
            gpu["memory_total_gb"] = round(vram_total_b / 1024**3, 1)
        if vram_used_b is not None and vram_total_b:
            gpu["memory_pct"] = round(vram_used_b / vram_total_b * 100, 1)
        if temp is not None:
            gpu["temperature_c"] = round(temp, 1)
        if power is not None:
            gpu["power_w"] = round(power, 1)
        gpus.append(gpu)
    return gpus


def _gpu_amd_sysfs(
    drm_dir: str = "/sys/class/drm",
) -> list[dict[str, Any]]:
    """Last resort when no AMD SMI tool is installed: amdgpu sysfs
    attributes (``gpu_busy_percent``, ``mem_info_vram_used/total`` bytes,
    ``temperature1`` milli-°C). No clock or power data here. The
    amdgpu discriminator is the co-presence of all three attributes
    (i915 has gpu_busy_percent but no VRAM meminfo). Never raises."""
    try:
        cards = sorted(
            d for d in os.listdir(drm_dir)
            if re.fullmatch(r"card\d+", d)
            and os.path.exists(f"{drm_dir}/{d}/device/gpu_busy_percent")
            and os.path.exists(f"{drm_dir}/{d}/device/mem_info_vram_used")
            and os.path.exists(f"{drm_dir}/{d}/device/mem_info_vram_total")
        )
    except OSError:
        return []

    gpus: list[dict[str, Any]] = []
    for card in cards:
        dev = f"{drm_dir}/{card}/device"
        try:
            idx = int(card.removeprefix("card"))
        except ValueError:
            continue
        util = _read_int_file(f"{dev}/gpu_busy_percent")
        vram_used_b = _read_int_file(f"{dev}/mem_info_vram_used")
        vram_total_b = _read_int_file(f"{dev}/mem_info_vram_total")
        temp_milli = _read_int_file(f"{dev}/temperature1")
        if util is None and vram_used_b is None:
            continue

        bus_id = _sysfs_pci_bus_id(card)
        board_vendor = "unknown"
        name = "AMD GPU"
        if bus_id:
            vendor_map = _lspci_vendor_map([bus_id]).get(bus_id)
            if vendor_map:
                board_vendor = vendor_map
            name_map = _lspci_device_name_map([bus_id]).get(bus_id)
            if name_map:
                name = name_map

        gpu: dict[str, Any] = {
            "index": idx,
            "throttle": {
                "hw_thermal_slowdown": False,
                "sw_thermal_slowdown": False,
            },
            "name": name,
            "board_vendor": board_vendor,
        }
        if bus_id:
            gpu["pci_bus_id"] = bus_id
        if util is not None:
            gpu["utilization_pct"] = util
        if vram_used_b is not None:
            gpu["memory_used_gb"] = round(vram_used_b / 1024**3, 1)
        if vram_total_b is not None:
            gpu["memory_total_gb"] = round(vram_total_b / 1024**3, 1)
        if vram_used_b is not None and vram_total_b:
            gpu["memory_pct"] = round(vram_used_b / vram_total_b * 100, 1)
        if temp_milli is not None:
            gpu["temperature_c"] = round(temp_milli / 1000, 1)
        gpus.append(gpu)
    return gpus


# Negative-cache for the AMD probe on mixed/NVIDIA-only boxes: amd-smi is
# a subprocess, so once it reports no AMD GPU we don't re-run it every
# second — but we DO re-probe periodically so a freshly loaded amdgpu
# driver (or hotplug) is picked up without a dashboard restart.
_AMD_NEGATIVE_REPROBE_S = 30.0
_amd_absent_since: float | None = None


def _gpu_amd() -> list[dict[str, Any]]:
    """AMD GPU metrics: amd-smi (preferred) → rocm-smi (legacy) → sysfs.

    When no AMD GPU is found the result is negative-cached for
    ``_AMD_NEGATIVE_REPROBE_S`` seconds (see above). Never raises."""
    global _amd_absent_since
    if _amd_absent_since is not None:
        if time.monotonic() - _amd_absent_since < _AMD_NEGATIVE_REPROBE_S:
            return []
        _amd_absent_since = None  # re-probe due

    gpus = _gpu_amd_smi()
    if not gpus:
        gpus = _gpu_rocm_smi()
    if not gpus:
        gpus = _gpu_amd_sysfs()
    if not gpus:
        _amd_absent_since = time.monotonic()
    else:
        _amd_absent_since = None
    return gpus


def _merge_gpu_lists(nvidia: list[dict[str, Any]],
                     amd: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-vendor GPU lists for mixed-vendor machines.

    Both vendors number their cards from 0 (and the AMD tiers report
    card indices independently), and the frontend keys its DOM and charts
    by ``index`` — so the merged list is renumbered 0..n-1 (NVIDIA cards
    first, in their original order). Each card also gets a ``vendor``
    tag ("nvidia" / "amd") for clarity and future per-vendor rendering.
    """
    merged: list[dict[str, Any]] = []
    for gpu in nvidia:
        gpu = dict(gpu)
        gpu["vendor"] = "nvidia"
        merged.append(gpu)
    offset = (max((g["index"] for g in nvidia), default=-1) + 1) if nvidia else 0
    for gpu in amd:
        gpu = dict(gpu)
        gpu["vendor"] = "amd"
        merged.append(gpu)
    for i, gpu in enumerate(merged):
        gpu["index"] = i
    return merged


# ---------------------------------------------------------------------------
# JetPack (Orin / Thor) platform detection + integrated-GPU collection
# ---------------------------------------------------------------------------
#
# JetPack boards are unified-memory SoCs: NVML is unsupported (Orin) or
# mostly "Not Supported" (Thor), and the Orin nvidia-smi stub reports
# plausible-looking junk (0% utilisation, N/A memory). The honest sources
# there are sysfs (devfreq ``device/load`` / ``cur_freq``, the ``gpu-thermal``
# zone) and /proc/meminfo (the single DRAM pool shared by CPU, OS and GPU —
# vLLM weights + KV cache live there). On Thor only, nvidia-smi is a
# usable utilisation source. See research_jetpack_gpu.md for the evidence.

# Lazy, resettable platform cache — filled on the first
# collect_gpu_metrics() call, never at import (importing app.py must stay
# side-effect-free; the test suite imports it on macOS where /proc is
# absent). Test reset: monkeypatch.setattr(app, "_PLATFORM", None) — and
# likewise "_JETSON_SOC", "_jetson_devfreq_cache", "_jetson_mode_logged".
_PLATFORM: str | None = None      # "desktop" | "jetson"
_JETSON_SOC: str | None = None    # "thor" | "orin" | "unknown"

# Real paths — module constants so tests can point them at temp files
# (same injectable-path seam as collect_host_health_metrics' psi_mem_path).
_DT_MODEL_PATHS = (
    "/proc/device-tree/model",
    "/sys/firmware/devicetree/base/model",
)
_DT_COMPATIBLE_PATHS = (
    "/proc/device-tree/compatible",
    "/sys/firmware/devicetree/base/compatible",
)
_NV_TEGRA_RELEASE_PATH = "/etc/nv_tegra_release"
_DEVFREQ_DIR = "/sys/class/devfreq"
_MEMINFO_PATH = "/proc/meminfo"
_THERMAL_DIR = "/sys/class/thermal"


def _read_dt_string(path: str) -> str:
    """Read a NUL-padded device-tree string file; '' on any error."""
    try:
        with open(path, "rb") as f:
            return f.read().rstrip(b"\x00").decode(errors="replace")
    except OSError:
        return ""


def _read_dt_bytes(path: str) -> bytes:
    """Read a raw device-tree blob (e.g. ``compatible``); b'' on error."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def _soc_from_compatible(compatible: bytes) -> str:
    """Parse the SoC id from a device-tree ``compatible`` blob.

    The NUL-separated list ends with the SoC id (``nvidia,tegra234`` =
    Orin, ``nvidia,tegra264`` = Thor). Returns "thor" / "orin" /
    "unknown" — never raises.
    """
    if b"nvidia,tegra264" in compatible:
        return "thor"
    if b"nvidia,tegra234" in compatible:
        return "orin"
    return "unknown"


def _init_platform() -> str:
    r"""Detect "desktop" vs "jetson" once and cache it in the globals.

    Cascade (each step best-effort; missing/unreadable files are skipped):
    1. device-tree ``model`` must contain BOTH "nvidia" and one of
       "jetson"/"orin"/"thor" (case-insensitive) — the nvidia
       co-occurrence guard keeps custom aarch64 boards out.
    2. device-tree ``compatible`` contains "nvidia,tegra" (the stable SoC
       discriminator; covers vendor boards and model-string gaps).
    3. ``/etc/nv_tegra_release`` first line matches ``R\d+`` (survives
       apt-based L4T updates that can drop the file's location).
    4. "tegra" in the kernel release (visible in containers where the
       device-tree is hidden).
    """
    global _PLATFORM, _JETSON_SOC
    if _PLATFORM is not None:
        return _PLATFORM

    model = ""
    for path in _DT_MODEL_PATHS:
        model = _read_dt_string(path)
        if model:
            break

    compatible = b""
    for path in _DT_COMPATIBLE_PATHS:
        compatible = _read_dt_bytes(path)
        if compatible:
            break

    if (
        model
        and re.search(r"nvidia", model, re.I)
        and re.search(r"jetson|orin|thor", model, re.I)
    ):
        _PLATFORM = "jetson"
    elif b"nvidia,tegra" in compatible:
        _PLATFORM = "jetson"
    else:
        found = False
        try:
            with open(_NV_TEGRA_RELEASE_PATH) as f:
                found = bool(re.search(r"R\d+", f.readline()))
        except OSError:
            pass
        if not found:
            try:
                found = "tegra" in platform.release()
            except Exception:  # noqa: BLE001 – detection must never raise
                pass
        _PLATFORM = "jetson" if found else "desktop"

    _JETSON_SOC = _soc_from_compatible(compatible)
    return _PLATFORM


def _jetson_board_label() -> str:
    """Name for the JetPack GPU card: SoC first, device-tree model next."""
    if _JETSON_SOC == "thor":
        return "NVIDIA Thor"
    if _JETSON_SOC == "orin":
        return "NVIDIA Orin"
    for path in _DT_MODEL_PATHS:
        model = _read_dt_string(path)
        if model:
            return model
    return "NVIDIA Jetson"


_jetson_devfreq_cache: str | None = None  # resolved node name; never caches absence


def _jetson_devfreq_node(devfreq_dir: str) -> str | None:
    r"""Resolve the nvgpu devfreq node name (late-boot safe).

    Strict patterns over a sorted listing: ``^gpu-gpc-\d+$`` (Thor) or a
    name ending ``.gpu`` (Orin, e.g. ``17000000.gpu``). The node appears
    seconds after boot, so the resolved name is cached once found but
    absence is never cached. No locking needed — the collector is
    single-threaded (MetricsCollector.run).
    """
    global _jetson_devfreq_cache
    if _jetson_devfreq_cache is not None:
        return _jetson_devfreq_cache
    try:
        entries = sorted(os.listdir(devfreq_dir))
    except OSError:
        return None
    for name in entries:
        if re.fullmatch(r"gpu-gpc-\d+", name) or name.endswith(".gpu"):
            _jetson_devfreq_cache = name
            return name
    return None


def _read_int_file(path: str) -> int | None:
    """Read a single-integer sysfs file; None on any error."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _jetson_thor_utilization() -> float | None:
    """GPU utilization % from nvidia-smi on Thor (the only SoC that shells out).

    2 s timeout (a wedged CLI stalls the snapshot by at most that long —
    same class as the desktop path, and the SSE queue holds the last
    payload). "N/A" / "[Not Supported]" / non-numeric output → None.
    """
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        return round(float(result.stdout.split(",")[0].strip()), 1)
    except ValueError:
        return None


def _read_unified_mem_gb(meminfo_path: str) -> tuple[float, float] | None:
    """(total, used) GiB of the unified DRAM pool from /proc/meminfo.

    used = MemTotal − MemAvailable — the pool shared by CPU, OS and GPU,
    so CUDA weights and the vLLM KV cache show up here. None when the
    file is missing or incomplete.
    """
    try:
        with open(meminfo_path) as f:
            text = f.read()
    except OSError:
        return None
    kb: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
            try:
                kb[parts[0]] = float(parts[1])  # kB
            except ValueError:
                return None
    if "MemTotal:" not in kb or "MemAvailable:" not in kb:
        return None
    total = kb["MemTotal:"]
    used = max(total - kb["MemAvailable:"], 0.0)
    return total / 1024**2, used / 1024**2


def _jetson_gpu_temp_c(thermal_dir: str) -> float | None:
    """Max °C across thermal zones whose ``type`` file says ``gpu-thermal``.

    Zones are matched by their ``type`` file, never by index (indices
    shift between SoCs/boards); ``temp < 0`` (−256000 = no sensor) is
    skipped. None when no zone qualifies.
    """
    try:
        zones = sorted(os.listdir(thermal_dir))
    except OSError:
        return None
    temps: list[float] = []
    for zone in zones:
        if not zone.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join(thermal_dir, zone, "type")) as f:
                if f.read().strip() != "gpu-thermal":
                    continue
            with open(os.path.join(thermal_dir, zone, "temp")) as f:
                temp = int(f.read().strip())  # millidegrees C
        except (OSError, ValueError):
            continue
        if temp < 0:
            continue
        temps.append(temp / 1000.0)
    if not temps:
        return None
    return round(max(temps), 1)


def collect_jetson_gpu_metrics(
    devfreq_dir: str = _DEVFREQ_DIR,
    meminfo_path: str = _MEMINFO_PATH,
    thermal_dir: str = _THERMAL_DIR,
) -> list[dict[str, Any]]:
    """Collect the single integrated GPU's metrics on JetPack (Orin/Thor).

    pynvml and the desktop nvidia-smi path are NOT used here (NVML is
    unsupported on Orin / mostly NotSupported on Thor; the Orin nvidia-smi
    stub reports plausible junk). Memory is the unified DRAM pool
    (``memory_unified: true``). Utilization is chosen per tick by
    capability: devfreq ``device/load`` (any SoC with an nvgpu sysfs) →
    known-Thor nvidia-smi → omitted (never a false 0). Never raises;
    absent sources are omitted per the house rule. The path arguments are
    injectable for tests (real paths are the module defaults).
    """
    _init_platform()
    gpu: dict[str, Any] = {
        "index": 0,
        "name": _jetson_board_label(),
        "memory_unified": True,
    }

    # Unified memory pool (CPU + OS + CUDA weights + vLLM KV cache).
    mem = _read_unified_mem_gb(meminfo_path)
    if mem is not None:
        total_gb, used_gb = mem
        gpu["memory_total_gb"] = round(total_gb, 1)
        gpu["memory_used_gb"] = round(used_gb, 1)
        if total_gb > 0:
            gpu["memory_pct"] = round(used_gb / total_gb * 100, 1)

    # Utilization — per-tick capability rule (no SoC table).
    node = _jetson_devfreq_node(devfreq_dir)
    util: float | None = None
    if node is not None:
        load = _read_int_file(os.path.join(devfreq_dir, node, "device", "load"))
        if load is not None:
            util = round(load / 10.0, 1)  # load is 0–1000
    if util is None and _JETSON_SOC == "thor":
        util = _jetson_thor_utilization()
    if util is not None:
        gpu["utilization_pct"] = util

    # Graphics clock (devfreq cur_freq, Hz → MHz).
    if node is not None:
        cur = _read_int_file(os.path.join(devfreq_dir, node, "cur_freq"))
        if cur is not None:
            gpu["graphics_clock_mhz"] = round(cur / 1e6, 1)

    # Temperature (gpu-thermal zone).
    temp = _jetson_gpu_temp_c(thermal_dir)
    if temp is not None:
        gpu["temperature_c"] = temp

    return [gpu]


_jetson_mode_logged = False


def collect_gpu_metrics() -> list[dict[str, Any]]:
    """Return GPU metrics via the platform-appropriate collector.

    JetPack (Orin/Thor): sysfs + unified memory — pynvml and the desktop
    nvidia-smi path are skipped entirely. Desktop: pynvml (preferred) or
    nvidia-smi CLI fallback, unchanged.
    """
    global _jetson_mode_logged
    if _init_platform() == "jetson":
        if not _jetson_mode_logged:
            log.info("GPU mode: jetson (SoC: %s)", _JETSON_SOC)
            _jetson_mode_logged = True
        return collect_jetson_gpu_metrics()
    nvidia = _gpu_pynvml()
    if not nvidia:
        nvidia = _gpu_cli()
    # Mixed-vendor boxes (NVIDIA + AMD) are a supported configuration:
    # probe the AMD chain regardless of what NVIDIA returned and merge.
    # On NVIDIA-only boxes the AMD probe is negative-cached (30 s) so the
    # amd-smi subprocess is not spawned every second.
    amd = _gpu_amd()
    return _merge_gpu_lists(nvidia, amd)


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
_llama_prev: dict[str, dict[str, float]] = {}

# Last successful snapshot (monotonic ts + dict). When a scrape fails
# (e.g. the 2 s timeout under heavy llama-server load) the UI keeps showing
# this data, flagged "stale", for up to this many seconds so a single bad
# tick doesn't flicker the card to "unreachable".
_LLAMA_STALE_MAX_AGE_S = 15.0
_llama_last_good: dict[str, tuple[float, dict[str, Any]]] = {}

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
LLM_REQ_RATE_WINDOW = 5
_llama_req_rates: dict[str, deque[float]] = {}
_vllm_req_rates: dict[str, deque[float]] = {}
# A prefill shorter than this cannot be measured at 1 s scrape cadence
# (it finishes between two scrapes); such requests are skipped.
_LLAMA_MIN_PREFILL_S = 0.25

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
    if state_key == "primary":
        return os.environ.get("RD_LLAMA_USAGE_STATE") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".llama_usage_state.json"
        )
    base = os.environ.get("RD_LLAMA_USAGE_STATE")
    base_dir = (os.path.dirname(os.path.abspath(base)) if base
                else os.path.dirname(os.path.abspath(__file__)))
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
        held_good = _llama_last_good.get(state_key)
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
    prev = _llama_prev.get(state_key)
    rates = _llama_req_rates.setdefault(
        state_key, deque(maxlen=LLM_REQ_RATE_WINDOW))
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
        _llama_prev[state_key] = new_prev

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

    _llama_last_good[state_key] = (now, out)
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

_vllm_last_good: dict[str, tuple[float, dict[str, Any]]] = {}

# Previous successful scrape's cumulative counters + histogram sums, used
# to compute rates and windowed means from per-scrape deltas.
_vllm_prev: dict[str, dict[str, Any]] = {}

# 30 s cache for the parsed cache_config_info dict (static during runtime).
_VLLM_CONFIG_TTL = 30.0
_vllm_config_cache: dict[str, tuple[float, dict[str, Any]]] = {}

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
        held_good = _vllm_last_good.get(state_key)
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
    prev = _vllm_prev.get(state_key)
    rates = _vllm_req_rates.setdefault(
        state_key, deque(maxlen=LLM_REQ_RATE_WINDOW))
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
    _vllm_prev[state_key] = new_prev

    _vllm_last_good[state_key] = (now, out)
    return out


# ---------------------------------------------------------------------------
# LiteLLM Monitoring
# ---------------------------------------------------------------------------

# Stale-hold: last successful LiteLLM scrape (monotonic ts + dict).
_LITELLM_STALE_MAX_AGE_S = 15.0
_litellm_last_good: dict[str, tuple[float, dict[str, Any]]] = {}

# Previous successful scrape's cumulative counters + histogram sums, used
# to compute rates and windowed means from per-scrape deltas.
_litellm_prev: dict[str, dict[str, Any]] = {}

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
        held_good = _litellm_last_good.get(state_key)
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
    prev = _litellm_prev.get(state_key)
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
    _litellm_prev[state_key] = new_prev

    _litellm_last_good[state_key] = (now, out)
    return out


# ---------------------------------------------------------------------------
# Strata Monitoring (https://github.com/Niko1221/Strata)
# ---------------------------------------------------------------------------

# Stale-hold: last successful Strata scrape (monotonic ts + dict).
_STRATA_STALE_MAX_AGE_S = 15.0
_strata_last_good: dict[str, tuple[float, dict[str, Any]]] = {}

# Previous successful scrape's cumulative totals + request-ring watermark,
# used to compute rates and windowed means from deltas.
_strata_prev: dict[str, dict[str, Any]] = {}

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
        active = str(entry.get("state") or "idle") != "idle"
        prompt = _strata_num(entry.get("prompt_tokens")) or 0.0
        generated = _strata_num(entry.get("generated")) or 0.0
        held = _strata_num(entry.get("held_tokens")) or 0.0
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
        held_good = _strata_last_good.get(state_key)
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
    prev = _strata_prev.get(state_key)
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
    if slots:
        out["slots"] = slots

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
    _strata_prev[state_key] = new_prev

    _strata_last_good[state_key] = (now, out)
    return out


# ---------------------------------------------------------------------------
# Upstream dashboard aggregation (multi-system consolidation)
# ---------------------------------------------------------------------------

# 15 s stale-hold per upstream (mirrors _LITELLM_STALE_MAX_AGE_S): a
# failed fetch keeps the last good entry flagged "stale" for 15 s, then
# the entry drops to "down".
_UPSTREAM_STALE_MAX_AGE_S = 15.0

# Strict sanitizer key list: only per-system keys are forwarded from an
# upstream snapshot. `systems` is explicitly excluded (no recursion) and
# any other app-level key is dropped (a newer upstream's extra keys are
# silently ignored; an older upstream without `systems` works fine).
_UPSTREAM_KEYS = ("timestamp", "cpu", "cpu_per_core", "memory", "gpus",
                 "swap", "host_health", "llm_enabled", "llm", "llms")

# Last good entry per upstream name: {name: (monotonic ts, entry dict)}.
# Each name is touched by exactly one fetch thread -> no locking needed.
_upstream_last_good: dict[str, tuple[float, dict[str, Any]]] = {}


def collect_upstream_metrics(url: str, name: str) -> dict[str, Any] | None:
    """Fetch one upstream dashboard's ``/api/metrics`` and sanitize it
    into a per-system entry.

    Returns ``{"name", "status": "up"|"stale", "metrics"}`` or ``None``
    (never fetched yet, or the 15 s stale-hold expired). Never raises.
    Mirrors the LLM collector conventions: 2 s timeout, per-name 15 s
    stale-hold, strict sanitizer (no ``systems`` recursion).
    """
    def _stale_or_none() -> dict[str, Any] | None:
        held = _upstream_last_good.get(name)
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

    metrics = {k: payload[k] for k in _UPSTREAM_KEYS if k in payload}
    out = {"name": name, "status": "up", "metrics": metrics}
    _upstream_last_good[name] = (time.monotonic(), out)
    return out


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
        # Bounded queue – maxsize entries; oldest dropped when full.
        self._queue: queue.Queue[str] = queue.Queue(maxsize=30)
        self.running = True
        # The LLM scrape runs on its own 1 s cadence, decoupled from
        # the main cycle (which includes GPU/health probes that can
        # make the cycle exceed the rate window). This keeps the
        # token-rate window ~poll_interval even when the main cycle
        # is slow.
        self._llm_latest: list[dict[str, Any]] | None = None
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
            except Exception:  # noqa: BLE001
                log.exception("Error collecting LLM metrics")
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
            data = collect_llama_metrics(target.url,
                                         state_key=target.name)
            if data is not None:
                data["backend"] = "llama"
        elif target.backend == "vllm":
            data = collect_vllm_metrics(target.url, state_key=target.name)
            if data is not None:
                data["backend"] = "vllm"
        elif target.backend == "litellm":
            data = collect_litellm_metrics(target.url, target.token,
                                           state_key=target.name)
            if data is not None:
                data["backend"] = "litellm"
        elif target.backend == "strata":
            data = collect_strata_metrics(target.url,
                                          state_key=target.name)
            if data is not None:
                data["backend"] = "strata"
        else:
            data = None
        if data is None:
            return {"name": target.name, "backend": target.backend,
                    "status": "down"}
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
        """Shut both loops down promptly (the LLM thread otherwise
        sleeps out its whole poll interval before noticing
        ``running`` flipped) — used by tests and shutdown paths."""
        self.running = False
        self._llm_stop.set()
        if self._llm_thread.is_alive():
            self._llm_thread.join(timeout=timeout)

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
        "down" until the first fetch lands."""
        while self.running:
            try:
                self._upstream_latest[name] = collect_upstream_metrics(
                    url, name)
            except Exception:  # noqa: BLE001
                log.exception("Error collecting upstream %s", name)
            self._upstream_stop.wait(self.poll_interval)

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
        global _llm_tie_warned, _litellm_tie_warned, _strata_tie_warned
        if not self.llama_enabled:
            return None
        if self.llm_backend in ("auto", "vllm"):
            data = collect_vllm_metrics(self.vllm_url, state_key=state_key)
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
        if self.llm_backend in ("auto", "litellm"):
            data = collect_litellm_metrics(self.litellm_url,
                                           self.litellm_token,
                                           state_key=state_key)
            if data is not None:
                data["backend"] = "litellm"
                if self.llm_backend == "auto" and not _litellm_tie_warned:
                    _litellm_tie_warned = True
                    log.warning(
                        "LLM auto mode: using LiteLLM proxy at %s. If "
                        "llama.cpp is also reachable and you want it "
                        "instead, pin --llm-backend llama (logged once).",
                        self.litellm_url)
                return data
        if self.llm_backend in ("auto", "strata"):
            data = collect_strata_metrics(self.strata_url,
                                          state_key=state_key)
            if data is not None:
                data["backend"] = "strata"
                if self.llm_backend == "auto" and not _strata_tie_warned:
                    _strata_tie_warned = True
                    log.warning(
                        "LLM auto mode: using Strata at %s. If llama.cpp "
                        "is also reachable and you want it instead, pin "
                        "--llm-backend llama (logged once).",
                        self.strata_url)
                return data
        if self.llm_backend in ("auto", "llama"):
            data = collect_llama_metrics(self.llama_url,
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
        metrics["llms"] = entries
        primary = entries[0] if entries else None
        metrics["llm"] = (primary
                          if primary is not None
                          and primary.get("status") != "down"
                          else None)
        # Multi-system consolidation: the local system first (same values
        # as the top-level keys), then one entry per configured upstream.
        # `list()` snapshot: start_upstreams may still be appending.
        systems = [{"name": "local", "status": "up",
                   "metrics": dict(metrics)}]
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
            except Exception:  # noqa: BLE001
                log.exception("Error collecting metrics")
            t_end = time.monotonic()
            collect_s = t_end - cycle_start
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
                payload = json.dumps(metrics)
                if self._queue.full():
                    self._drain_old()
                self._queue.put_nowait(payload)
            time.sleep(max(0.0, next_ts - time.monotonic()))


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__)
collector = MetricsCollector(poll_interval=1.0)
collector.start()


# Initial system tab shown on page load ("local" | "fleet"); set from
# settings in __main__ before the server starts.
_default_view = "local"


@app.route("/")
def index() -> str:
    """Serve the dashboard HTML page."""
    return render_template("index.html", hostname=platform.node(),
                           default_view=_default_view)


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
    parser.add_argument("--config", default=None,
                        help="YAML config file (default: ./config.yaml when "
                             "present). CLI arguments override config values, "
                             "which override built-in defaults.")
    parser.add_argument("--host", default=None, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: 5000)")
    parser.add_argument("--poll-interval", type=float, default=None,
                        help="Poll interval in seconds (default: 1.0)")
    parser.add_argument("--llama-url", default=None,
                        help="Base URL of llama.cpp llama-server (expects /metrics enabled via --metrics). "
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
    args = parser.parse_args()

    config = load_config(args.config, explicit=args.config is not None)
    settings = resolve_settings(args, config)

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

    llm_status = "disabled" if not settings["llama_enabled"] else (
        "targets=" + ", ".join(
            f"{t.name}:{t.backend}" + (f"@{t.url}" if t.url else "")
            for t in settings["llm_targets"]))
    up_status = (f", upstreams={', '.join(u['name'] for u in settings['upstreams'])}"
                if settings["upstreams"] else "")
    log.info("Starting Resource Dashboard on http://%s:%d (LLM: %s%s)",
             settings["host"], settings["port"], llm_status, up_status)
    # threaded=True: the dev server is single-threaded by default, so a client
    # holding the long-lived SSE /events stream would make every other request
    # (incl. /api/metrics from an aggregating dashboard) queue or 503.
    app.run(host=settings["host"], port=settings["port"], debug=False, threaded=True)
