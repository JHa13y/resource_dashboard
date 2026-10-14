"""GPU collectors (NVIDIA pynvml/nvidia-smi, AMD amd-smi/rocm-smi/sysfs,
JetPack sysfs). Moved verbatim from ``app.py`` (T12 package split);
behavior is unchanged.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import platform
import re
import subprocess
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PCI vendor-ID → name map (pci.ids-independent board/GPU vendor)
# ---------------------------------------------------------------------------
# The board/GPU vendor is resolved from the *raw* PCI vendor ID — read
# from the sysfs ``vendor`` file, which carries the numeric ID without
# consulting the ``pci.ids`` database. The ID is stable across hardware
# generations, so a small hardcoded map covers the common cases; unknown
# IDs fall back to the raw ID string (still deterministic and
# pci.ids-independent).
_VENDOR_ID_NAMES: dict[int, str] = {
    0x10DE: "NVIDIA",
    0x1002: "AMD",
    0x1028: "Gigabyte",
    0x1043: "Dell",
    0x1458: "MSI",
    0x144D: "ASRock",
    0x103C: "HP",
    0x106B: "Apple",
    0x1569: "Gigabyte",
}


def _bdf_sysfs_form(bdf: str) -> str | None:
    """Normalize a PCI BDF to the 4-digit-domain sysfs form (e.g.
    ``0000:2d:00.0``). Handles the 8-digit-domain form
    (``00000000:2d:00.0``) that occurs in some payloads. None when the
    BDF is malformed or empty."""
    if not bdf:
        return None
    m = re.match(
        r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$",
        bdf.strip())
    if not m:
        return None
    domain, bus, slot, func = m.groups()
    return f"{domain[-4:].lower()}:{bus.lower()}:{slot.lower()}.{func}"


def _sysfs_pci_vendor_id(bdf: str,
                         pci_dir: str = "/sys/bus/pci/devices") -> str | None:
    """Raw PCI vendor ID (e.g. ``0x10de``) for a BDF from the sysfs
    ``vendor`` file — pci.ids-independent (the file carries the numeric
    ID, not the pci.ids-resolved name). None on non-Linux hosts or any
    error (a missing device node is a normal, non-fatal condition)."""
    normalized = _bdf_sysfs_form(bdf)
    if not normalized:
        return None
    try:
        with open(f"{pci_dir}/{normalized}/vendor") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _vendor_id_name(vendor_id: str | None) -> str:
    """Map a raw PCI vendor ID (e.g. ``0x10de``) to a name via the
    hardcoded map; fall back to the raw ID string when the ID is unknown,
    and ``unknown`` when the ID is absent. Never raises."""
    if not vendor_id:
        return "unknown"
    try:
        vid = int(vendor_id, 16)
    except (ValueError, TypeError):
        return vendor_id
    return _VENDOR_ID_NAMES.get(vid, vendor_id)


def _sysfs_pci_vendor_name(bdf: str,
                           pci_dir: str = "/sys/bus/pci/devices") -> str:
    """Board/GPU vendor name for a BDF, resolved from the raw PCI vendor
    ID (the sysfs ``vendor`` file) via a hardcoded map — pci.ids-
    independent. Falls back to the raw ID string when the ID is unknown,
    and ``unknown`` when the ID can't be read. Never raises."""
    return _vendor_id_name(_sysfs_pci_vendor_id(bdf, pci_dir))


# ---------------------------------------------------------------------------
# Board/package vendor (lspci ``SVendor:`` — the physical card maker, e.g.
# "Gigabyte"/"Dell"/"ASRock"). Re-introduced in T13 (T10 removed it) so the
# board/package vendor can be folded into the GPU name. Best-effort: lspci
# may be missing/time out → "" (no parentheses); never raises, never blocks
# the snapshot path unboundedly (5 s timeout + negative cache).
# ---------------------------------------------------------------------------
_lspci_cache: dict[str, str] | None = None  # bus_id → board/package vendor
_lspci_negative: dict[str, float] = {}  # bus_id → failed_at (negative cache)
_LSPICI_NEGATIVE_TTL = 300.0
_lspci_missing_logged = False


def _parse_lspci_subsystem(output: str) -> str:
    """Extract the board/package vendor name from lspci output.

    An "SVendor:" line is preferred; the first token after it is returned
    (e.g. "Gigabyte", "Dell"). If no SVendor line exists, falls back to the
    first token after a "Subsystem:" line. Returns an empty string when
    neither line is present so callers can apply their own fallbacks.
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


def _lspci_vendor_map(bus_ids: list[str]) -> dict[str, str]:
    """Board/package vendor (lspci ``SVendor:``) per BDF. Best-effort with a
    positive cache + a negative cache (``_LSPICI_NEGATIVE_TTL``) so a
    missing/timing-out lspci never blocks the snapshot path unboundedly.
    Returns "" for any BDF whose vendor can't be determined. Never raises."""
    global _lspci_cache, _lspci_missing_logged
    if _lspci_cache is None:
        _lspci_cache = {}
    now = time.monotonic()

    mapping: dict[str, str] = {}
    for bus_id in bus_ids:
        if not bus_id:
            mapping[bus_id] = ""
            continue
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
            if not _lspci_missing_logged:
                log.info("lspci not found – board/package vendor lookup "
                         "unavailable (negative-cached %.0fs)",
                         _LSPICI_NEGATIVE_TTL)
                _lspci_missing_logged = True
            _lspci_negative.update({b: now for b in bus_ids if b})
            break
        except subprocess.TimeoutExpired:
            log.warning("lspci timed out for %s (negative-cached %.0fs)",
                       bus_id, _LSPICI_NEGATIVE_TTL)
            _lspci_negative.update({b: now for b in bus_ids if b})
            break
        _lspci_cache[bus_id] = vendor_name
        mapping[bus_id] = vendor_name
    return mapping


def _fold_board_vendor_into_name(name: str, bus_id: str) -> str:
    """Fold the board/package vendor (lspci ``SVendor:``) into the GPU name
    in parentheses (e.g. ``"NVIDIA RTX 3090"`` → ``"NVIDIA RTX 3090
    (Gigabyte)"``). If the board/package vendor is unknown (empty), the name
    is returned unchanged (no parentheses). Best-effort: never raises."""
    if not bus_id or bus_id == "unknown":
        return name
    board_vendor = _lspci_vendor_map([bus_id]).get(bus_id, "")
    if not board_vendor:
        return name
    # Avoid a redundant "(Gigabyte)" when the name already contains the
    # board/package vendor (case-insensitive).
    if board_vendor.lower() in name.lower():
        return name
    return f"{name} ({board_vendor})"


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

            # Board/GPU vendor from the raw PCI vendor ID (sysfs ``vendor``
            # file) via a hardcoded map — pci.ids-independent.
            board_vendor = _sysfs_pci_vendor_name(bus_id)
            # Fold the board/package vendor (lspci ``SVendor:``) into the
            # name (e.g. "NVIDIA RTX 3090 (Gigabyte)"). T13.
            name = _fold_board_vendor_into_name(name, bus_id)

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

            # Board/GPU vendor from the raw PCI vendor ID (sysfs ``vendor``
            # file) via a hardcoded map — pci.ids-independent.
            bus_id = row[7].strip()
            board_vendor = _sysfs_pci_vendor_name(bus_id)
            # Name + fold the board/package vendor (lspci ``SVendor:``) into
            # it (e.g. "NVIDIA RTX 3090 (Gigabyte)"). T13.
            name = _fold_board_vendor_into_name(row[8].strip(), bus_id)

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
                "name": name,
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
# reports VRAM in MB or GB depending on build (the ``unit`` field is
# respected by ``_amd_vram_gb``); legacy rocm-smi reports VRAM in BYTES
# (1024×). "N/A" is the universal missing-value token and is treated as
# absent, never 0.

_AMD_SMI_BIN_CANDIDATES = ("amd-smi", "/opt/rocm/bin/amd-smi")
_ROCM_SMI_BIN_CANDIDATES = ("rocm-smi", "/opt/rocm/bin/rocm-smi")

# Warn-once guard: nvidia-smi absent (e.g. pure AMD box) — the fallback
# chain now continues to the AMD path, so the "no GPU metrics" message
# would otherwise fire every poll.
_nvidia_cli_missing_logged = False

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


def _sysfs_pci_bus_id(card: int | str,
                      drm_dir: str = "/sys/class/drm") -> str | None:
    """PCI BDF (e.g. ``0000:66:00.0``) for a DRM card from the sysfs
    ``device`` symlink. ``card`` may be a global card number (int) or a
    full ``cardN`` name (str, as reported by rocm-smi). None on
    non-Linux hosts or any error."""
    name = (card if isinstance(card, str) and card.startswith("card")
            else f"card{card}")
    try:
        target = os.readlink(f"{drm_dir}/{name}/device")
    except OSError:
        return None
    return os.path.basename(target) or None


# A GPU name that is actually a PCI BDF/slot, a bare hex id, or a
# ``Device <hex>`` token is implausible — it is the *address* of the
# card, not its name (the lspci first-``Device:``-line bug: the first
# ``Device:`` line is the slot, the product name is on a later line).
# Rejected names are neither used nor positive-cached. The domain is
# 0-8 hex chars (``0000:`` and the 8-wide ``00000000:`` form both occur
# in the payload).
_GPU_NAME_BDF_RE = re.compile(
    r"^[0-9a-fA-F]{0,8}:?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GPU_NAME_HEX_RE = re.compile(r"^(0x)?[0-9a-f]{4,}$", re.IGNORECASE)
_GPU_NAME_DEVICE_HEX_RE = re.compile(r"^Device [0-9a-fx]+$", re.IGNORECASE)


def _is_implausible_gpu_name(name: str) -> bool:
    """True when *name* is a BDF/slot, a bare hex id, or a ``Device
    <hex>`` token — i.e. the card's address, not its name. Such names
    are rejected (neither used nor positive-cached) so a poisoned lspci
    read cannot stick."""
    name = name.strip()
    return (bool(_GPU_NAME_BDF_RE.match(name))
            or bool(_GPU_NAME_HEX_RE.match(name))
            or bool(_GPU_NAME_DEVICE_HEX_RE.match(name)))


# ---------------------------------------------------------------------------
# AMD gpu index → PCI BDF resolution (T3)
#
# amd-smi's ``gpu`` field is a *vendor-local* index (0..n-1 within AMD).
# Passing it as a global ``/sys/class/drm/cardN`` number mis-resolves on
# mixed-vendor boxes: DRM numbers cards across all vendors, so on the
# live box the second R9700 (AMD gpu 1) resolved to the RTX 3090's DRM
# node and reported the NVIDIA card's BDF + board vendor. The BDF must
# come from the tool's own data, in this fallback order:
#   (a) ``amd-smi list --json`` (carries ``bus_id``) — resolved once and
#       cached; failures negative-cached for _AMD_BDF_NEGATIVE_TTL (same
#       no-stall pattern as the T2 lspci caches);
#   (b) else rocm-smi's own ``cardN`` string (its ``device`` column, a
#       global DRM number) → sysfs;
#   (c) else ``/sys/class/drm/card*`` filtered by ``device/vendor ==
#       0x1002`` (AMD only), mapped in card-number order — a vendor-local
#       index is never mistaken for a global ``cardN``.
# ---------------------------------------------------------------------------

_amd_bdf_map: dict[int, str] | None = None  # gpu index → bus_id (positive)
_rocm_cards_cache: list[str] | None = None  # rocm-smi device column, in order
# Failed BDF lookups (source → monotonic ts); a hung/missing tool is not
# re-run on every poll (same TTL as _LSPCI_NEGATIVE_TTL).
_amd_bdf_negative: dict[str, float] = {}
_AMD_BDF_NEGATIVE_TTL = 300.0


def _amd_smi_list_bus_ids() -> dict[int, str] | None:
    """``gpu index → bus_id`` from ``amd-smi list --json`` (the tool's
    own data). Resolved once and cached; failures (and lists with no
    usable bus_id) are negative-cached for _AMD_BDF_NEGATIVE_TTL.
    None on any failure. Never raises."""
    global _amd_bdf_map
    if _amd_bdf_map is not None:
        return _amd_bdf_map
    now = time.monotonic()
    failed_at = _amd_bdf_negative.get("list")
    if failed_at is not None and (now - failed_at) < _AMD_BDF_NEGATIVE_TTL:
        return None
    binary = _find_cli_binary(_AMD_SMI_BIN_CANDIDATES)
    if binary is None:
        _amd_bdf_negative["list"] = now
        return None
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [binary, "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _amd_bdf_negative["list"] = now
        return None
    if result.returncode != 0:
        _amd_bdf_negative["list"] = now
        return None
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        _amd_bdf_negative["list"] = now
        return None
    if not isinstance(payload, list):
        _amd_bdf_negative["list"] = now
        return None
    mapping: dict[int, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("gpu")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        # ``bdf`` is the current amd-smi field; ``bus_id`` is the legacy
        # name (older builds) — try both.
        bus = entry.get("bdf") or entry.get("bus_id")
        if isinstance(bus, str) and bus.strip():
            mapping[idx] = bus.strip()
    if not mapping:
        _amd_bdf_negative["list"] = now
        return None
    _amd_bdf_map = mapping
    return mapping


def _rocm_smi_card_names() -> list[str] | None:
    """rocm-smi's own ``cardN`` device strings in device order (its
    ``device`` column, e.g. ``["card2", "card3"]``) — global DRM
    numbers, never vendor-local indices. Resolved once and cached;
    failures negative-cached for _AMD_BDF_NEGATIVE_TTL. None on any
    failure. Never raises."""
    global _rocm_cards_cache
    if _rocm_cards_cache is not None:
        return _rocm_cards_cache
    now = time.monotonic()
    failed_at = _amd_bdf_negative.get("rocm")
    if failed_at is not None and (now - failed_at) < _AMD_BDF_NEGATIVE_TTL:
        return None
    binary = _find_cli_binary(_ROCM_SMI_BIN_CANDIDATES)
    if binary is None:
        _amd_bdf_negative["rocm"] = now
        return None
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [binary, "--showuse", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _amd_bdf_negative["rocm"] = now
        return None
    if result.returncode != 0 or not result.stdout.strip():
        _amd_bdf_negative["rocm"] = now
        return None
    rows = [r for r in csv.reader(io.StringIO(result.stdout))
            if r and not r[0].lstrip().startswith("#")]
    names = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    if not names:
        _amd_bdf_negative["rocm"] = now
        return None
    _rocm_cards_cache = names
    return names


# ---------------------------------------------------------------------------
# AMD product/asic (market) name — the tool's own identity, preferred
# over lspci so the label is not hostage to pci.ids freshness (the R9700
# is new hardware). A static device fact: resolved once via ``amd-smi
# static --asic --json`` and cached; failures negative-cached for
# _AMD_BDF_NEGATIVE_TTL (same no-stall pattern as the T2/T3 caches).
# ---------------------------------------------------------------------------

# Fields that may carry the market name in ``amd-smi static --asic
# --json``; the first plausible (non-implausible) string value wins.
_AMD_PRODUCT_NAME_FIELDS = ("asic", "market_name", "name",
                           "product_name", "device_name")

_amd_product_names: dict[int, str] = {}  # gpu index → market name (static)
_amd_product_names_ts: float | None = None  # last (failed) attempt, monotonic


def _amd_smi_product_names(binary: str) -> dict[int, str]:
    """``{vendor-local gpu index: market name}`` from ``amd-smi static
    --asic --json``. The tool's own product/asic name — preferred over
    lspci. {} on any failure (the caller falls back to lspci, then
    "AMD GPU"). Never raises."""
    try:
        result = subprocess.run(  # noqa: S603 – static args only
            [binary, "static", "--asic", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        return {}
    # Accept both a bare list and the ``{"gpu_data": [...]}`` wrapper
    # (amd-smi versions differ).
    if isinstance(payload, dict):
        payload = payload.get("gpu_data")
    if not isinstance(payload, list):
        return {}
    names: dict[int, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("gpu")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        # Market name: nested ``asic.market_name`` (current amd-smi) first,
        # then the top-level field list (older builds). Every candidate is
        # implausibility-checked.
        candidates: list[Any] = []
        asic = entry.get("asic")
        if isinstance(asic, dict):
            candidates.append(asic.get("market_name"))
        candidates.extend(entry.get(f) for f in _AMD_PRODUCT_NAME_FIELDS)
        for val in candidates:
            if (isinstance(val, str) and val.strip()
                    and not _is_implausible_gpu_name(val)):
                names[idx] = val.strip()
                break
    return names


def _amd_product_name(idx: int) -> str | None:
    """The tool's own product/asic name for a vendor-local gpu index,
    cached (a static device fact). None when unavailable (the caller
    falls back to lspci, then "AMD GPU"). A failed/empty lookup is
    negative-cached for _AMD_BDF_NEGATIVE_TTL (no per-poll subprocess).
    Never raises."""
    global _amd_product_names, _amd_product_names_ts
    if _amd_product_names:
        return _amd_product_names.get(idx)
    now = time.monotonic()
    if _amd_product_names_ts is not None and \
            (now - _amd_product_names_ts) < _AMD_BDF_NEGATIVE_TTL:
        return None
    binary = _find_cli_binary(_AMD_SMI_BIN_CANDIDATES)
    if binary is None:
        _amd_product_names_ts = now
        return None
    names = _amd_smi_product_names(binary)
    if names:
        _amd_product_names = names
    else:
        _amd_product_names_ts = now
    return names.get(idx)


def _sysfs_amd_card_bdfs(drm_dir: str = "/sys/class/drm") -> list[tuple[str, str]]:
    """``(cardN name, BDF)`` for every DRM card whose ``device/vendor``
    is ``0x1002`` (AMD), sorted by global card number. Mapping a
    vendor-local AMD index onto this filtered list is what keeps it from
    being mistaken for a global ``cardN``. [] on non-Linux or error."""
    try:
        entries = os.listdir(drm_dir)
    except OSError:
        return []
    out: list[tuple[str, str]] = []
    for name in entries:
        if not re.fullmatch(r"card\d+", name):
            continue
        try:
            with open(f"{drm_dir}/{name}/device/vendor") as f:
                vendor = int(f.read().strip(), 16)
        except (OSError, ValueError):
            continue
        if vendor != 0x1002:
            continue
        bdf = _sysfs_pci_bus_id(name, drm_dir)
        if bdf:
            out.append((name, bdf))
    out.sort(key=lambda t: int(t[0].removeprefix("card")))
    return out


def _amd_gpu_bus_id(gpu_idx: int,
                    drm_dir: str = "/sys/class/drm") -> str | None:
    """Resolve an amd-smi vendor-local ``gpu`` index to a PCI BDF via
    the fallback chain in the header above (a → b → c). A vendor-local
    index is never passed as a global ``cardN``. None when no source can
    resolve it (non-Linux hosts no-op cleanly). Never raises."""
    mapping = _amd_smi_list_bus_ids()
    if mapping is not None and gpu_idx in mapping:
        return mapping[gpu_idx]
    cards = _rocm_smi_card_names()
    if cards is not None and 0 <= gpu_idx < len(cards):
        bdf = _sysfs_pci_bus_id(cards[gpu_idx])
        if bdf:
            return bdf
    amd_cards = _sysfs_amd_card_bdfs(drm_dir)
    if 0 <= gpu_idx < len(amd_cards):
        return amd_cards[gpu_idx][1]
    return None


def _amd_unwrap(val: Any) -> float | None:
    """Unwrap an amd-smi JSON leaf: numerics arrive as
    ``{"value": N, "unit": "W"}`` (scalars pass through); ``"N/A"``,
    None and malformed values → None."""
    if isinstance(val, dict):
        val = val.get("value")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _amd_vram_gb(val: Any) -> float | None:
    """Unwrap an amd-smi vram leaf and return GiB, respecting the unit.
    ``{"value": N, "unit": "GB"}`` → N (already GiB); ``"MB"`` → N/1024;
    a bare scalar or an absent/unknown unit → N/1024 (legacy behaviour,
    preserves older MB-reporting builds). ``"N/A"``/None/malformed → None."""
    if isinstance(val, dict):
        value = val.get("value")
        unit = val.get("unit")
        unit = unit.upper() if isinstance(unit, str) else ""
    else:
        value, unit = val, ""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if unit == "GB":
        return value
    # "MB", absent, or unknown unit → legacy (assume MB → GiB).
    return value / 1024


def _amd_smi_entry_to_gpu(entry: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Map one ``amd-smi monitor --json`` object to the dashboard GPU dict.

    Field map (monitor keys, VRAM unit-aware GB/MB): ``gfx`` → utilization %,
    ``vram_used``/``vram_total`` → memory in GiB (unit-aware via
    ``_amd_vram_gb``), ``hotspot_temperature`` (fall
    back to ``memory_temperature`` — ``edge`` is N/A on datacenter cards),
    ``gfx_clk`` (fallback ``gfx_clock``) → MHz, ``power_usage`` → W. Rows with
    no usable data
    (e.g. secondary XCP rows on partitioned MI300X) are dropped.
    """
    util = _amd_unwrap(entry.get("gfx"))
    vram_used = _amd_vram_gb(entry.get("vram_used"))
    vram_total = _amd_vram_gb(entry.get("vram_total"))
    temp = _amd_unwrap(entry.get("hotspot_temperature"))
    if temp is None:
        temp = _amd_unwrap(entry.get("memory_temperature"))
    # ``gfx_clk`` is the current amd-smi field; ``gfx_clock`` is the legacy
    # name — try both.
    clock_raw = entry.get("gfx_clk")
    if clock_raw is None:
        clock_raw = entry.get("gfx_clock")
    clock = _amd_unwrap(clock_raw)
    power = _amd_unwrap(entry.get("power_usage"))

    if util is None and vram_used is None:
        return None

    # The vendor-local ``gpu`` index is resolved to a BDF from the tool's
    # own data (T3) — never passed as a global ``/sys/class/drm/cardN``.
    bus_id = _amd_gpu_bus_id(idx)
    # Board/GPU vendor from the raw PCI vendor ID (sysfs ``vendor`` file)
    # via a hardcoded map — pci.ids-independent.
    board_vendor = _sysfs_pci_vendor_name(bus_id) if bus_id else "unknown"
    # Name: rely on the tool's own product/asic (market) name — a static
    # device fact — so the label is not hostage to pci.ids freshness (the
    # R9700 is new hardware). Fall back to "AMD GPU".
    name = "AMD GPU"
    tool_name = _amd_product_name(idx)
    if tool_name:
        name = tool_name
    # Fold the board/package vendor (lspci ``SVendor:``) into the name
    # (e.g. "Radeon RX 7900 XTX (ASRock)"). T13.
    name = _fold_board_vendor_into_name(name, bus_id)

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
    # ``_amd_vram_gb`` already returns GiB (unit-aware) — no /1024 here.
    if vram_used is not None:
        gpu["memory_used_gb"] = round(vram_used, 1)
    if vram_total is not None:
        gpu["memory_total_gb"] = round(vram_total, 1)
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
        # Board/GPU vendor from the raw PCI vendor ID (sysfs ``vendor``
        # file) via a hardcoded map — pci.ids-independent.
        board_vendor = _sysfs_pci_vendor_name(bus_id) if bus_id else "unknown"
        # No tool product name on the rocm path — "AMD GPU".
        name = "AMD GPU"
        # Fold the board/package vendor (lspci ``SVendor:``) into the name.
        # T13.
        name = _fold_board_vendor_into_name(name, bus_id)

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
        # Board/GPU vendor from the raw PCI vendor ID (sysfs ``vendor``
        # file) via a hardcoded map — pci.ids-independent.
        board_vendor = _sysfs_pci_vendor_name(bus_id) if bus_id else "unknown"
        # No tool product name on the sysfs path — "AMD GPU".
        name = "AMD GPU"
        # Fold the board/package vendor (lspci ``SVendor:``) into the name.
        # T13.
        name = _fold_board_vendor_into_name(name, bus_id)

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


def _normalize_bdf(bdf: str) -> str:
    """Canonical form for BDF comparison: 4-digit zero-padded domain,
    lowercase, zero-padded bus/device — ``00000000:61:00.0`` and
    ``0000:61:00.0`` are the same device (the domain width is
    inconsistent across vendors in the payload)."""
    m = re.fullmatch(
        r"([0-9a-fA-F]{1,8}):([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,3})\.([0-9a-fA-F])",
        bdf.strip())
    if not m:
        return bdf.strip().lower()
    return (f"{int(m.group(1), 16):04x}:{m.group(2).lower().zfill(2)}:"
            f"{m.group(3).lower().zfill(2)}.{m.group(4).lower()}")


def _merge_gpu_lists(nvidia: list[dict[str, Any]],
                     amd: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-vendor GPU lists for mixed-vendor machines.

    Both vendors number their cards from 0 (and the AMD tiers report
    card indices independently), and the frontend keys its DOM and charts
    by ``index`` — so the merged list is renumbered 0..n-1 (NVIDIA cards
    first, in their original order). Each card also gets a ``vendor``
    tag ("nvidia" / "amd") for clarity and future per-vendor rendering.

    Colliding-BDF guard: no two entries in one payload may claim the same
    (normalized) PCI BDF — a collision means a mis-resolved card (e.g. an AMD
    card mapped onto an NVIDIA DRM node). It is flagged (a warning names both
    cards) and the duplicate is dropped; two cards on one device are
    never silently rendered.
    """
    merged: list[dict[str, Any]] = []
    for gpu in nvidia:
        gpu = dict(gpu)
        gpu["vendor"] = "nvidia"
        merged.append(gpu)
    for gpu in amd:
        gpu = dict(gpu)
        gpu["vendor"] = "amd"
        merged.append(gpu)
    seen: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    for gpu in merged:
        bdf = gpu.get("pci_bus_id")
        if not bdf:
            kept.append(gpu)
            continue
        norm = _normalize_bdf(str(bdf))
        if norm in seen:
            other = seen[norm]
            log.warning(
                "GPU BDF collision: %s (vendor=%s) and %s (vendor=%s) "
                "both claim %s — dropping the duplicate",
                gpu.get("name"), gpu.get("vendor"),
                other.get("name"), other.get("vendor"), bdf)
            continue
        seen[norm] = gpu
        kept.append(gpu)
    for i, gpu in enumerate(kept):
        gpu["index"] = i
    return kept


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
