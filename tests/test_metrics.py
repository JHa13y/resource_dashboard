"""Unit tests for the resource dashboard metrics collectors.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import pytest


# ---------------------------------------------------------------------------
# GPU CLI parser tests
# ---------------------------------------------------------------------------

def _fake_nvidia_smi(output: str) -> None:
    """Side-effect to make subprocess.run return *output*."""
    mock = MagicMock()
    mock.stdout = output
    raise SystemExit(mock)  # never reached – used as patch return_value


@pytest.fixture(autouse=True)
def no_real_nvidia():
    """Ensure real nvidia-smi / pynvml are never called during tests."""
    with (
        patch("app._gpu_pynvml", return_value=[]),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="")  # default: no GPUs
        yield mock_run


def test_gpu_cli_parses_single_gpu(no_real_nvidia):
    from app import _gpu_cli

    no_real_nvidia.return_value = MagicMock(
        stdout="85, 4096, 8192, 72, 1410, Active, Not Active, 0000:01:00.0, NVIDIA A100-SXM4-80GB, 0x1EB810DE, 42.5, 400.00\n",
    )
    result = _gpu_cli()

    assert len(result) == 1
    gpu = result[0]
    assert gpu["index"] == 0
    assert gpu["utilization_pct"] == 85
    assert gpu["memory_used_gb"] == 4.0
    assert gpu["memory_total_gb"] == 8.0
    assert gpu["memory_pct"] == pytest.approx(50.0)
    assert gpu["temperature_c"] == 72
    assert gpu["graphics_clock_mhz"] == 1410
    assert gpu["throttle"]["hw_thermal_slowdown"] is True
    assert gpu["throttle"]["sw_thermal_slowdown"] is False
    assert gpu["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpu["pci_bus_id"] == "0000:01:00.0"
    # lspci not available in test → falls back to raw hex from device_id
    assert gpu["board_vendor"] is not None
    assert gpu["power_w"] == 42.5
    assert gpu["power_limit_w"] == 400.0


def test_gpu_cli_parses_multiple_gpus(no_real_nvidia):
    from app import _gpu_cli

    no_real_nvidia.return_value = MagicMock(
        stdout=(
            "30, 2048, 8192, 55, 1200, Not Active, Not Active, 0000:01:00.0, NVIDIA A100-SXM4-80GB, 0x1EB810DE, 120.3, 400.00\n"
            "92, 7000, 8192, 88, 1650, Active, Active, 0000:02:00.0, GeForce RTX 4090, 0x268410DE, [Not Supported], [Not Supported]\n"
        ),
    )
    result = _gpu_cli()

    assert len(result) == 2
    assert result[0]["utilization_pct"] == 30
    assert result[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert result[0]["pci_bus_id"] == "0000:01:00.0"
    assert result[1]["throttle"]["hw_thermal_slowdown"] is True
    assert result[1]["throttle"]["sw_thermal_slowdown"] is True
    assert result[1]["name"] == "GeForce RTX 4090"
    assert result[1]["pci_bus_id"] == "0000:02:00.0"
    assert result[1]["board_vendor"] is not None
    assert result[0]["power_w"] == 120.3
    assert result[0]["power_limit_w"] == 400.0
    # "[Not Supported]" power columns → keys omitted entirely
    assert "power_w" not in result[1]
    assert "power_limit_w" not in result[1]


def test_gpu_cli_empty_output(no_real_nvidia):
    from app import _gpu_cli

    no_real_nvidia.return_value = MagicMock(stdout="")
    result = _gpu_cli()
    assert result == []


def test_gpu_cli_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        from app import _gpu_cli
        result = _gpu_cli()
        assert result == []


# ---------------------------------------------------------------------------
# PCI vendor-ID → name map tests (pci.ids-independent board/GPU vendor)
# ---------------------------------------------------------------------------

def test_vendor_id_name_maps_known_ids():
    from app import _vendor_id_name

    # Known GPU vendor IDs map to their names.
    assert _vendor_id_name("0x10de") == "NVIDIA"
    assert _vendor_id_name("0x1002") == "AMD"
    # Known board vendor IDs map to their names.
    assert _vendor_id_name("0x1028") == "Gigabyte"
    assert _vendor_id_name("0x1043") == "Dell"
    assert _vendor_id_name("0x1458") == "MSI"
    assert _vendor_id_name("0x144d") == "ASRock"
    # Case-insensitive (sysfs reports lowercase hex).
    assert _vendor_id_name("0x10DE") == "NVIDIA"
    # Absent / empty ID → "unknown".
    assert _vendor_id_name(None) == "unknown"
    assert _vendor_id_name("") == "unknown"
    # Unknown ID → the raw ID string (deterministic, pci.ids-independent).
    assert _vendor_id_name("0x9999") == "0x9999"


def test_sysfs_pci_vendor_name_missing_device():
    """A BDF with no sysfs device node (non-Linux host, or a device that
    isn't present) resolves to ``unknown`` — the sysfs read fails cleanly
    and the fallback kicks in. Never raises."""
    from app import _sysfs_pci_vendor_name

    # A BDF that doesn't correspond to a real device node.
    assert _sysfs_pci_vendor_name("0000:ff:ff.0") == "unknown"
    # A malformed BDF also resolves to ``unknown``.
    assert _sysfs_pci_vendor_name("not-a-bdf") == "unknown"
    assert _sysfs_pci_vendor_name("") == "unknown"


# ---------------------------------------------------------------------------
# pynvml tests (mocked)
# ---------------------------------------------------------------------------

def test_gpu_pynvml_basic():
    mock_nvml = MagicMock()
    mock_nvml.nvmlInit.return_value = 0
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=73)
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=4_000_000_000, total=8_000_000_000)
    mock_nvml.nvmlDeviceGetTemperature.return_value = 67
    mock_nvml.nvmlDeviceGetClockInfo.return_value = 1500
    mock_nvml.nvmlDeviceGetCurrentClocksThrottleReasons.return_value = 0x40
    mock_nvml.NVML_CLOCKS_THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 0x40
    mock_nvml.NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN = 0x80
    mock_nvml.nvmlDeviceGetPciInfo.return_value = MagicMock(
        busId=b"0000:01:00.0\x00",
        pciDeviceId=0x1EB810DE,
        pciSubSystemId=0x12A210DE,
    )
    mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA A100-SXM4-80GB"

    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        if "app" in sys.modules:
            del sys.modules["app"]
        import app as app_mod
        # Mock the sysfs vendor-ID read so the board vendor is resolved
        # from the hardcoded map (pci.ids-independent).
        with patch.object(app_mod, "_sysfs_pci_vendor_id",
                         return_value="0x10de"):
            result = app_mod._gpu_pynvml()

    assert len(result) == 1
    gpu = result[0]
    assert gpu["utilization_pct"] == 73
    assert gpu["memory_used_gb"] == pytest.approx(3.7, rel=0.05)
    assert gpu["temperature_c"] == 67
    assert gpu["graphics_clock_mhz"] == 1500
    assert gpu["throttle"]["hw_thermal_slowdown"] is True
    assert gpu["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpu["pci_bus_id"] == "0000:01:00.0"
    assert gpu["board_vendor"] == "NVIDIA"


def test_gpu_pynvml_not_installed():
    from app import _gpu_pynvml

    with patch.dict("sys.modules", {"pynvml": None}):
        result = _gpu_pynvml()
        assert result == []


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------

def test_collector_produces_data():
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.2)
    collector.start()
    try:
        # Wait for at least one cycle
        data = collector.get_latest(timeout=3.0)
        assert data is not None
        metrics = json.loads(data)
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "gpus" in metrics
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_collector_queue_bounded():
    """A slow subscriber's per-client buffer is bounded: when full, the
    *oldest* snapshot is dropped (drop-oldest) so a blocked client cannot
    grow unbounded or starve the producer."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    # Shrink the per-client buffer for a fast test.
    q = collector.subscribe(maxsize=3)
    collector.start()
    try:
        time.sleep(0.6)  # let it publish ~6 snapshots (buffer max is 3)
        # Drain the subscriber's bounded buffer.
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
        assert len(items) <= 3
        # Drop-oldest: the retained items are the most recent, in order.
        if len(items) >= 2:
            ts = [json.loads(x)["timestamp"] for x in items]
            assert ts == sorted(ts)
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_collector_llm_disabled_performs_no_llm_calls():
    """llama_enabled=False (--no-llama): no LLM HTTP calls, flag in snapshot."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    # The module-global collector (started at import) keeps probing in
    # the background; count only calls made by THIS collector's scrape
    # threads (named per instance in MetricsCollector.__init__).
    own_thread = collector._llm_thread.name
    own_prefix = collector._llm_thread_prefix
    own_calls = []

    def fake_urlopen(*args, **kwargs):
        t = threading.current_thread().name
        if t == own_thread or t.startswith(own_prefix):
            own_calls.append(t)
        raise ConnectionRefusedError("LLM must not be scraped")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["llm_enabled"] is False
            assert metrics.get("llm") is None
            assert metrics.get("llms") == []
        finally:
            collector.stop()
            collector.join(timeout=2)
        assert not own_calls


def test_collector_llm_enabled_by_default():
    """Default (no flag): llama_enabled True and llama scraped as before."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    assert collector.llama_enabled is True
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    })):
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["llm_enabled"] is True
            assert metrics["llm"]["status"] == "up"
            assert metrics["llm"]["model"] == "x.gguf"
        finally:
            collector.stop()
            collector.join(timeout=2)


def test_collector_llm_thread_populates_latest():
    """The LLM scrape runs on its own thread; _llm_latest is populated
    and the first main snapshot is complete (run() waits for the first
    LLM cycle)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    })):
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["llm"]["status"] == "up"
            assert collector._llm_latest is not None
            assert collector._llm_latest[0]["status"] == "up"
        finally:
            collector.stop()
            collector.join(timeout=2)


def test_two_subscribers_receive_same_snapshot():
    """Two /events clients both receive the SAME snapshot (broadcast, not
    distribution): the old single-consumer queue gave each client a
    disjoint frame."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    q1 = collector.subscribe()
    q2 = collector.subscribe()
    collector.start()
    try:
        # Both subscribers receive the first published snapshot.
        p1 = q1.get(timeout=5.0)
        p2 = q2.get(timeout=5.0)
        assert p1 is not None and p2 is not None
        m1 = json.loads(p1)
        m2 = json.loads(p2)
        # Identical payload (same timestamp + content) = broadcast.
        assert m1["timestamp"] == m2["timestamp"]
        assert m1 == m2
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_api_metrics_200_while_subscriber_attached(monkeypatch):
    """/api/metrics returns 200 (the cached snapshot) while a subscriber is
    attached — the old destructive get_latest() let a blocked SSE client
    win the only frame, so /api/metrics timed out (503)."""
    import app as app_mod
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start()
    try:
        # A snapshot exists.
        assert collector.get_latest(timeout=5.0) is not None
        # Attach a subscriber (a /events client).
        q = collector.subscribe()
        try:
            monkeypatch.setattr(app_mod, "collector", collector)
            resp = app_mod.app.test_client().get("/api/metrics")
            assert resp.status_code == 200
            body = resp.get_json()
            assert "cpu" in body and "systems" in body
        finally:
            collector.unsubscribe(q)
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_midstream_subscriber_gets_current_snapshot_without_waiting():
    """A client subscribing mid-stream gets the current snapshot
    immediately (seeded on subscribe), without waiting a full cycle."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.2)
    collector.llama_enabled = False
    collector.start()
    try:
        # Wait until a snapshot exists (mid-stream = after production began).
        assert collector.get_latest(timeout=5.0) is not None
        # A new client subscribes now.
        q = collector.subscribe()
        # It must get the current snapshot well under one 0.2 s cycle.
        t0 = time.monotonic()
        data = q.get(timeout=0.1)  # 0.1 s < 0.2 s cycle
        elapsed = time.monotonic() - t0
        assert data is not None
        m = json.loads(data)
        assert "cpu" in m
        # Delivered without waiting a cycle.
        assert elapsed < 0.19
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_systems_metrics_no_self_nesting():
    """T5: the local system's systems[].metrics is a slim fleet projection,
    not the full top-level re-embedding. The absolute invariant holds:
    `systems` must not self-nest inside systems[0].metrics. The pre-T5
    "equals the top-level minus systems" equality is intentionally relaxed
    (the projection is a subset); the local detail data is available at
    the top level."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start()
    try:
        data = collector.get_latest(timeout=5.0)
        assert data is not None
        m = json.loads(data)
        # Top-level keys are frozen.
        top = {"timestamp", "cpu", "cpu_per_core", "memory", "gpus",
               "swap", "host_health", "llm_enabled", "llm", "llms",
               "systems"}
        assert set(m.keys()) == top
        # systems[0] is the local entry.
        local = m["systems"][0]
        assert local["name"] == "local"
        # Slim fleet projection: just what the fleet strip needs.
        assert set(local["metrics"].keys()) == {"cpu", "memory", "gpus",
                                              "llms"}
        # The self-nesting trap: `systems` must NOT appear inside
        # systems[0].metrics.
        assert "systems" not in local["metrics"]
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_fleet_projection_local_not_reembedded():
    """T5: the local system's systems[].metrics is a slim fleet projection,
    not the full top-level re-embedding. The heavy/redundant fields are
    dropped; the local detail data is available at the top level."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start()
    try:
        data = collector.get_latest(timeout=5.0)
        assert data is not None
        m = json.loads(data)
        local = m["systems"][0]
        lm = local["metrics"]
        # Slim projection: just what the fleet strip needs.
        assert set(lm.keys()) == {"cpu", "memory", "gpus", "llms"}
        # The heavy/redundant fields are NOT re-embedded.
        for k in ("cpu_per_core", "swap", "host_health", "timestamp",
                  "llm"):
            assert k not in lm
        # The local detail data is available at the top level.
        for k in ("cpu", "memory", "gpus", "swap", "host_health",
                  "llm_enabled", "llms"):
            assert k in m
        # gpus in the projection are slimmed to utilization_pct only.
        for g in lm["gpus"]:
            assert set(g.keys()) == {"utilization_pct"}
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_fleet_projection_local_llms_carry_name():
    """T5 CHANGES r1: the local projection's llms entries carry `name`
    (the UI keys the per-target trend buffers by name; a nameless entry
    would key the buffer by JS `undefined` and leave the local LLM card
    chart permanently empty)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = True
    # Populate _llm_latest the way the LLM thread would (this test calls
    # _collect() directly without starting the threads).
    collector._llm_latest = [
        {"name": "primary", "status": "up", "backend": "llama",
         "gen_tokens_per_s": 100.0},
        {"name": "second", "status": "up", "backend": "strata",
         "gen_tokens_per_s": 50.0},
    ]
    with patch("app.collect_gpu_metrics", return_value=[]), \
         patch("app.collect_swap_metrics", return_value=None), \
         patch("app.collect_host_health_metrics",
               return_value={"psi_mem_avg10": None, "d_state_procs": 0}):
        snapshot = collector._collect()
    local = snapshot["systems"][0]
    lm = local["metrics"]
    # The local projection's llms entries carry name (so the UI keys the
    # per-target trend buffers by the real target name, not `undefined`).
    assert len(lm["llms"]) == 2
    assert [e["name"] for e in lm["llms"]] == ["primary", "second"]
    # name is the only addition to the slim llms projection.
    for e in lm["llms"]:
        assert set(e.keys()) == {"status", "name", "gen_tokens_per_s",
                                 "backend"}


def test_fleet_projection_peer_full():
    """T5: a peer system's systems[].metrics is a full projection
    sufficient for the detail tab (cpu, memory, gpus, llms, llm_enabled,
    swap, host_health), dropping the fields the UI never reads from a
    peer (timestamp, cpu_per_core, legacy llm). A peer's own `systems`
    is dropped by the sanitizer (not in the projection key set)."""
    import app

    fake_payload = {
        "timestamp": 1234.5,
        "cpu": 10.0,
        "cpu_per_core": [1.0, 2.0],
        "memory": {"percent": 50.0, "total_gb": 16.0,
                   "used_gb": 8.0, "available_gb": 8.0},
        "gpus": [{"index": 0, "utilization_pct": 5.0, "name": "fake"}],
        "swap": {"used_gb": 0.0, "total_gb": 4.0, "pct": 0.0},
        "host_health": {"d_state_procs": 0},
        "llm_enabled": False,
        "llm": None,
        "llms": [],
        "systems": [{"name": "should-not-appear"}],
    }

    class FakeResponse:
        def read(self):
            return json.dumps(fake_payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        entry = app.collect_upstream_metrics("http://fake:5000", "fakepeer")
    assert entry is not None
    assert entry["name"] == "fakepeer"
    assert entry["status"] == "up"
    pm = entry["metrics"]
    # Full projection for the detail tab.
    for k in ("cpu", "memory", "gpus", "llms", "llm_enabled", "swap",
             "host_health"):
        assert k in pm
    # Dropped fields the UI never reads from a peer.
    for k in ("timestamp", "cpu_per_core", "llm"):
        assert k not in pm
    # No self-nesting: the peer's own `systems` is dropped by the
    # sanitizer (not in the projection key set).
    assert "systems" not in pm
    # Full gpus/llms are kept (the peer detail tab renders them).
    assert pm["gpus"] == fake_payload["gpus"]
    assert pm["llms"] == fake_payload["llms"]


def test_fleet_projection_no_self_nesting():
    """T5: for both the local system and a peer, systems[].metrics does
    not contain `systems` (the section-2 trap, asserted in the
    projection context)."""
    import app
    from app import MetricsCollector

    # Local system: the projection is built before metrics["systems"]
    # is assigned, so it can never self-nest.
    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start()
    try:
        data = collector.get_latest(timeout=5.0)
        assert data is not None
        m = json.loads(data)
        for s in m["systems"]:
            assert "systems" not in s["metrics"]
    finally:
        collector.stop()
        collector.join(timeout=2)

    # Peer system: the sanitizer drops the peer's own `systems`.
    fake_payload = {"cpu": 1.0, "memory": {"percent": 1.0},
                   "gpus": [], "llms": [], "llm_enabled": False,
                   "swap": None, "host_health": None,
                   "systems": [{"name": "nested"}]}

    class FakeResponse:
        def read(self):
            return json.dumps(fake_payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        entry = app.collect_upstream_metrics("http://fake:5000", "fakepeer")
    assert entry is not None
    assert "systems" not in entry["metrics"]


# ---------------------------------------------------------------------------
# collect_gpu_metrics integration (prefers pynvml, falls back to CLI)
# ---------------------------------------------------------------------------

def test_collect_gpu_falls_back_to_cli():
    from app import collect_gpu_metrics

    with (
        patch("app._gpu_pynvml", return_value=[]),
        patch("app._gpu_cli") as mock_cli,
    ):
        mock_cli.return_value = [{"index": 0}]
        result = collect_gpu_metrics()
        assert len(result) == 1
        mock_cli.assert_called_once()


def test_collect_gpu_prefers_pynvml():
    from app import collect_gpu_metrics

    with (
        patch("app._gpu_pynvml") as mock_nv,
        patch("app._gpu_cli") as mock_cli,
    ):
        mock_nv.return_value = [{"index": 0}]
        result = collect_gpu_metrics()
        assert len(result) == 1
        mock_cli.assert_not_called()


# ---------------------------------------------------------------------------
# llama.cpp collector tests
# ---------------------------------------------------------------------------

LLAMA_METRICS_SAMPLE = (
    "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed\n"
    "# TYPE llamacpp:tokens_predicted_total counter\n"
    "llamacpp:tokens_predicted_total 123456\n"
    "llamacpp:predicted_tokens_seconds 52.94\n"
    "llamacpp:prompt_tokens_seconds 32.3\n"
    "llamacpp:requests_processing 1\n"
    "llamacpp:requests_deferred 0\n"
    "llamacpp:n_tokens_max 4096\n"
    "llamacpp:prompt_tokens_cached_total 2048\n"
    'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 5\n'
)


@pytest.fixture(autouse=True)
def no_real_llama():
    """Ensure llama-server HTTP endpoints are never actually reached."""
    with patch(
        "app.urllib.request.urlopen",
        side_effect=ConnectionRefusedError("no llama-server in tests"),
    ):
        yield


@pytest.fixture(autouse=True)
def reset_llama_props_cache():
    """Reset the module-level /props cache around every test."""
    import app as app_mod

    original = app_mod._llama_props_cache
    app_mod._llama_props_cache = {}
    yield
    app_mod._llama_props_cache = original


@pytest.fixture(autouse=True)
def reset_llama_prev():
    """Reset module-level llama scrape history around every test.

    Multi-LLM: all scrape-state containers are target-name-keyed dicts;
    swapping in fresh dicts is the reset."""
    import app as app_mod

    originals = {
        k: getattr(app_mod, k) for k in (
            "_llama_prev", "_llama_last_good", "_llama_slot_prev",
            "_llama_decode_history", "_llama_req_rates",
        )
    }
    for k in originals:
        setattr(app_mod, k, {})
    yield
    for k, v in originals.items():
        setattr(app_mod, k, v)


@pytest.fixture(autouse=True)
def isolate_llama_usage_state(tmp_path, monkeypatch):
    """Point the T3 usage-state file at a per-test tmp path.

    Without this, every llama scrape in the suite would write the default
    ``<app dir>/.llama_usage_state.json`` into the repo.
    """
    monkeypatch.setenv(
        "RD_LLAMA_USAGE_STATE", str(tmp_path / "llama_usage_state.json")
    )
    yield


@pytest.fixture(autouse=True)
def reset_llama_proc():
    """Reset the T5 process handle; default: no local llama-server.

    ``psutil.process_iter`` is patched to return no processes by default so
    tests never depend on (or slow down on) the host's real process table.
    Tests that need a fake server re-patch it.
    """
    import app as app_mod

    original = app_mod._llama_proc
    app_mod._llama_proc = {}
    with patch("app.psutil.process_iter", return_value=[]):
        yield
    app_mod._llama_proc = original


def _fake_llama_proc(name="llama-server", rss_gb=21.0, cpu=340.0,
                     threads=64, cmdline=None):
    """A MagicMock standing in for the local llama-server process."""
    rss = int(rss_gb * 2**30)
    p = MagicMock()
    p.info = {
        "name": name,
        "cmdline": (cmdline if cmdline is not None
                    else ["/home/joshua/llama.cpp/build/bin/llama-server",
                          "--model", "x.gguf"]),
        "memory_info": MagicMock(rss=rss),
    }
    p.memory_info.return_value = MagicMock(rss=rss)
    p.cpu_percent.return_value = cpu
    p.num_threads.return_value = threads
    return p


class _FakeClock:
    """A manually advanced stand-in for time.monotonic."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _http_response(body: bytes):
    """A context-manager mock standing in for urllib's response object."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def make_llama_server(routes: dict, calls: list | None = None):
    """Build a side_effect for ``app.urllib.request.urlopen``.

    *routes* maps a URL path suffix (e.g. ``"/metrics"``) to either a
    bytes body or an exception instance to raise.
    """
    def fake_urlopen(url, timeout=None):
        if calls is not None:
            calls.append(url)
        for suffix, body in routes.items():
            if url.endswith(suffix):
                if isinstance(body, Exception):
                    raise body
                return _http_response(body)
        raise ConnectionRefusedError(f"no route for {url}")

    return fake_urlopen


def test_collect_llama_ok():
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "up"
    assert result["gen_tokens_per_s"] == 52.9
    assert result["prompt_tokens_per_s"] == 32.3
    assert result["gen_tokens_total"] == 123456.0
    assert result["prompt_tokens_cached_total"] == 2048.0
    assert result["requests_processing"] == 1.0
    assert result["requests_deferred"] == 0.0
    assert result["context_tokens_max"] == 4096.0
    assert result["model"] == "x.gguf"
    assert result["slots"] == []
    # labeled series must be dropped by the parser
    assert "spec_decode" not in json.dumps(result)


def test_collect_llama_unreachable():
    from app import collect_llama_metrics

    # no_real_llama fixture makes urlopen raise ConnectionRefusedError
    assert collect_llama_metrics("http://127.0.0.1:9999") is None


def test_collect_llama_json_escaped_body():
    """2025 builds JSON-escaped the whole /metrics body (PR #17386)."""
    from app import collect_llama_metrics

    escaped = json.dumps(LLAMA_METRICS_SAMPLE)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": escaped.encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["gen_tokens_per_s"] == 52.9
    assert result["gen_tokens_total"] == 123456.0


def test_collect_llama_no_llamacpp_metrics():
    from app import collect_llama_metrics

    for body in ("", "some:other_metric 1\n# HELP foo bar\n"):
        with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": body.encode(),
            "/props": b"{}",
            "/slots": b"[]",
        })):
            assert collect_llama_metrics("http://127.0.0.1:9999") is None


def test_collect_llama_stale_hold():
    """A failed scrape within the hold window returns last data as stale."""
    from app import collect_llama_metrics

    routes = {
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    }
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server(routes)):
        first = collect_llama_metrics("http://127.0.0.1:9999")
    assert first is not None and first["status"] == "up"

    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        held = collect_llama_metrics("http://127.0.0.1:9999")
        held_again = collect_llama_metrics("http://127.0.0.1:9999")
    assert held is not None
    assert held["status"] == "stale"
    assert held["gen_tokens_total"] == 123456.0
    assert held["model"] == "x.gguf"
    # the held dict is copied, so repeated failures still report "stale"
    assert held_again is not None and held_again["status"] == "stale"


def test_collect_llama_stale_expiry():
    """A failed scrape after the 15 s hold window returns None again."""
    import app as app_mod
    from app import collect_llama_metrics

    clock = _FakeClock()
    routes = {
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b"{}",
        "/slots": b"[]",
    }
    with patch.object(app_mod.time, "monotonic", clock), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server(routes)):
        assert collect_llama_metrics("http://127.0.0.1:9999")["status"] == "up"

    clock.advance(16.0)
    with patch.object(app_mod.time, "monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        assert collect_llama_metrics("http://127.0.0.1:9999") is None


def test_collect_llama_recover_after_stale():
    """A successful scrape after a stale period restores status "up"."""
    from app import collect_llama_metrics

    routes = {
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b"{}",
        "/slots": b"[]",
    }
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server(routes)):
        collect_llama_metrics("http://127.0.0.1:9999")
    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        assert collect_llama_metrics("http://127.0.0.1:9999")["status"] == "stale"
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server(routes)):
        assert collect_llama_metrics("http://127.0.0.1:9999")["status"] == "up"


def test_parse_prometheus_skips_labels_and_comments():
    from app import _parse_prometheus

    text = (
        "# HELP llamacpp:tokens_predicted_total desc\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 5\n'
        "llamacpp:tokens_predicted_total 42\n"
    )
    assert _parse_prometheus(text) == {"llamacpp:tokens_predicted_total": 42.0}


def test_collect_llama_props_and_slots():
    from app import collect_llama_metrics

    props = {"model_path": "/models/qwen3.8-q6.gguf", "total_slots": 2,
             "is_sleeping": False}
    slots = [
        {"id": 0, "n_ctx": 8192, "n_prompt_tokens": 4096, "is_processing": True,
         "next_token": {"n_decoded": 512, "n_remain": 10}},
        {"id": 1, "n_ctx": 8192, "n_prompt_tokens": 0, "is_processing": False,
         "next_token": {"n_decoded": 0, "n_remain": 0}},
    ]
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": json.dumps(props).encode(),
        "/slots": json.dumps(slots).encode(),
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["model"] == "qwen3.8-q6.gguf"
    assert result["model_path"] == "/models/qwen3.8-q6.gguf"
    assert result["total_slots"] == 2
    assert result["sleeping"] is False

    s0, s1 = result["slots"]
    assert s0["id"] == 0
    assert s0["context_used"] == 4608
    assert s0["context_pct"] == round(100 * 4608 / 8192, 1)
    assert s0["is_processing"] is True
    assert s0["decoded"] == 512
    # n_prompt_tokens_processed absent in the fixture -> 0
    assert s0["prompt_processed"] == 0
    assert s1["id"] == 1
    assert s1["context_used"] == 0
    assert s1["context_pct"] == 0.0
    assert s1["is_processing"] is False
    # free_slots = total_slots (props: 2) - requests_processing (sample: 1)
    assert result["free_slots"] == 1


def test_parse_llama_slots_next_token_shapes():
    """next_token arrives as a dict (classic builds) or a LIST of dicts
    (e.g. the LM Studio build); both must parse to a decoded total."""
    from app import _parse_llama_slots

    # dict shape
    d = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "n_prompt_tokens_processed": 7,
        "next_token": {"n_decoded": 512},
    }])
    assert d[0]["decoded"] == 512
    assert d[0]["prompt_processed"] == 7
    # new keys: cache hits, total prompt size, task id (absent -> None)
    assert d[0]["cached"] == 0
    assert d[0]["prompt_total"] == 10
    assert d[0]["id_task"] is None

    # list shape: total n_decoded, skip non-dict items
    slots = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "n_prompt_tokens_processed": 3,
        "next_token": [
            {"has_next_token": True, "has_new_line": False,
             "n_remain": 16374, "n_decoded": 10},
            {"n_decoded": 5},
            "garbage",
        ],
    }])
    assert slots[0]["decoded"] == 15
    assert slots[0]["prompt_processed"] == 3
    # context = prompt + decoded
    assert slots[0]["context_used"] == 25


def test_parse_llama_slots_remain():
    """n_remain (tokens left to generate) is summed over the next_token
    items; -1 (idle) or absent -> None."""
    from app import _parse_llama_slots

    # list shape: sum positive n_remain over the items
    slots = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "next_token": [
            {"n_decoded": 10, "n_remain": 16000},
            {"n_decoded": 5, "n_remain": 374},
        ],
    }])
    assert slots[0]["remain"] == 16374
    assert slots[0]["decoded"] == 15

    # dict shape (classic builds)
    d = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "next_token": {"n_decoded": 512, "n_remain": 10},
    }])
    assert d[0]["remain"] == 10

    # idle marker: -1 wins -> None
    i = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "next_token": [{"n_decoded": 0, "n_remain": -1}],
    }])
    assert i[0]["remain"] is None

    # absent next_token / empty list / no n_remain field -> None
    a = _parse_llama_slots([{"id": 0, "n_ctx": 100, "n_prompt_tokens": 10}])
    assert a[0]["remain"] is None
    b = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10, "next_token": [],
    }])
    assert b[0]["remain"] is None
    c = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "next_token": [{"n_decoded": 42}],
    }])
    assert c[0]["remain"] is None
    assert c[0]["decoded"] == 42


def test_collect_llama_free_slots():
    """free_slots = total_slots (props) - requests_processing (metrics);
    the key is omitted when either side is unknown."""
    from app import collect_llama_metrics

    def run(props: str, metrics: str):
        # Bypass the 30 s /props cache so each run sees its own props body.
        import app as app_mod
        app_mod._llama_props_cache = {}
        with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": metrics.encode(),
            "/props": props.encode(),
            "/slots": b"[]",
        })):
            return collect_llama_metrics("http://127.0.0.1:9999")

    # known: 3 slots, 1 processing -> 2 free
    r = run('{"total_slots": 3}', "llamacpp:requests_processing 1\n")
    assert r["free_slots"] == 2

    # props without total_slots -> key omitted
    r = run('{}', "llamacpp:requests_processing 1\n")
    assert "free_slots" not in r

    # metrics without requests_processing -> key omitted
    r = run('{"total_slots": 3}', "llamacpp:n_tokens_max 100\n")
    assert "free_slots" not in r


def test_collect_llama_spec_accept_rate():
    """spec_accept_pct = 100 * accepted / draft (lifetime), rounded to 0.1;
    omitted when spec decode is off (counters absent) or draft == 0."""
    from app import collect_llama_metrics

    def run(metrics: str):
        with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": metrics.encode(),
            "/props": b'{"total_slots": 1}',
            "/slots": b"[]",
        })):
            return collect_llama_metrics("http://127.0.0.1:9999")

    # both counters present (live reference-server values: 6050/7717)
    r = run(
        "llamacpp:spec_decode_num_draft_tokens_total 7717\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 6050\n"
    )
    assert r["spec_decode_draft_tokens_total"] == 7717.0
    assert r["spec_decode_accepted_tokens_total"] == 6050.0
    assert r["spec_accept_pct"] == 78.4

    # draft == 0 -> key omitted (no division by zero)
    r = run(
        "llamacpp:spec_decode_num_draft_tokens_total 0\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 0\n"
    )
    assert "spec_accept_pct" not in r

    # counters absent (spec decode off) -> rate key omitted
    r = run("llamacpp:requests_processing 0\n")
    assert "spec_decode_draft_tokens_total" not in r
    assert "spec_decode_accepted_tokens_total" not in r
    assert "spec_accept_pct" not in r


def test_collect_llama_live_rate_from_slots():
    """Live tokens/s from per-slot decode counters: the /metrics counters
    only advance on request completion on some builds (e.g. LM Studio),
    while n_decoded / n_prompt_tokens_processed update per token."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [{"id": 0, "n_ctx": 8192, "n_prompt_tokens": 100,
                   "n_prompt_tokens_processed": 0, "is_processing": True,
                   "next_token": [{"n_decoded": 0}]}],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        first = collect_llama_metrics("http://127.0.0.1:9999")
        # No previous scrape yet -> counter-delta fallback (0 here)
        assert first["gen_tokens_per_s"] == 0.0

        # 67 tokens decoded + 67 prompt tokens processed over 2 s.
        # Generation has started (n_decoded > 0): prefill is done, so the
        # prompt gauge switches to the per-request UNCACHED rate
        # 67 tok / 2 s (matches llama-server's logged prompt-eval speed).
        clock.advance(2.0)
        state["slots"][0]["n_prompt_tokens_processed"] = 67
        state["slots"][0]["next_token"] = [{"n_decoded": 67}]
        second = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["gen_tokens_per_s"] == 33.5
    assert second["prompt_tokens_per_s"] == 33.5


def test_collect_llama_live_rate_task_reset_clamp():
    """n_decoded / n_prompt_tokens_processed reset per task: a drop between
    scrapes must clamp to 0, never a negative rate."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [{"id": 0, "n_ctx": 8192, "n_prompt_tokens": 100,
                   "n_prompt_tokens_processed": 500, "is_processing": True,
                   "next_token": [{"n_decoded": 875}]}],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        collect_llama_metrics("http://127.0.0.1:9999")

        # A new task took the slot: counters dropped (875 -> 10, 500 -> 12)
        clock.advance(2.0)
        state["slots"][0]["n_prompt_tokens_processed"] = 12
        state["slots"][0]["next_token"] = [{"n_decoded": 10}]
        second = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["gen_tokens_per_s"] == 0.0
    assert second["prompt_tokens_per_s"] == 0.0


def test_collect_llama_live_prompt_rate_includes_cached():
    """The live prompt rate counts cached + uncached prompt tokens, matching
    the server's own prompt t/s (a chat workload is mostly KV-cache hits)."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [{"id": 0, "n_ctx": 8192, "n_prompt_tokens": 800,
                   "n_prompt_tokens_processed": 0,
                   "n_prompt_tokens_cache": 0, "id_task": 100,
                   "is_processing": True, "next_token": [{"n_decoded": 0}]}],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        collect_llama_metrics("http://127.0.0.1:9999")

        # 10 uncached + 790 cached prompt tokens over 2 s
        clock.advance(2.0)
        state["slots"][0]["n_prompt_tokens_processed"] = 10
        state["slots"][0]["n_prompt_tokens_cache"] = 790
        second = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["prompt_tokens_per_s"] == 400.0
    assert second["gen_tokens_per_s"] == 0.0


def test_collect_llama_live_rate_cached_reset_clamp():
    """n_prompt_tokens_cache resets per task like n_decoded: a drop between
    scrapes must clamp to 0, never a negative rate."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [{"id": 0, "n_ctx": 8192, "n_prompt_tokens": 800,
                   "n_prompt_tokens_processed": 100,
                   "n_prompt_tokens_cache": 790, "id_task": 100,
                   "is_processing": True, "next_token": [{"n_decoded": 50}]}],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    def set_task(task, processed, cached, decoded):
        slot = state["slots"][0]
        slot["id_task"] = task
        slot["n_prompt_tokens_processed"] = processed
        slot["n_prompt_tokens_cache"] = cached
        slot["next_token"] = [{"n_decoded": decoded}]

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        collect_llama_metrics("http://127.0.0.1:9999")

        # New task took the slot: cached dropped 790 -> 0 (clamped)
        clock.advance(2.0)
        set_task(101, 0, 0, 0)
        second = collect_llama_metrics("http://127.0.0.1:9999")

        # Same task: cached 0 -> 800, processed 0 -> 20 over 2 s.
        # Generation started (decoded 30 > 0): prefill done. The per-
        # request rate counts UNCACHED tokens only: 20 tok / 2 s = 10 t/s
        # (the 800 cache hits did no prefill work).
        clock.advance(2.0)
        set_task(101, 20, 800, 30)
        third = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["prompt_tokens_per_s"] == 0.0
    assert second["gen_tokens_per_s"] == 0.0
    assert third["prompt_tokens_per_s"] == 10.0
    assert third["gen_tokens_per_s"] == 15.0


def _llama_slot_fixture(task, total, processed, cached, decoded=0,
                        processing=True):
    return {"id": 0, "n_ctx": 8192, "n_prompt_tokens": total,
            "n_prompt_tokens_processed": processed,
            "n_prompt_tokens_cache": cached, "id_task": task,
            "is_processing": processing,
            "next_token": [{"n_decoded": decoded}]}


def test_collect_llama_prompt_rate_averages_completed_requests():
    """prompt tokens/s is the mean of the last 5 completed prefills
    (uncached prompt tokens / prefill duration; the fixtures here are
    fully uncached): a large prefill must not spike for one scrape and
    then drop to 0, and the window must evict the oldest sample after
    5 requests."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    def run_request(task, total, per_step, steps):
        """Run a request whose prefill reaches *total* after *steps*
        scrapes of 1 s; returns the snapshot from the completing scrape.
        Per-request rate is total / steps t/s."""
        state["slots"] = [_llama_slot_fixture(task, total, 0, 0)]
        collect_llama_metrics("http://127.0.0.1:9999")
        r = None
        for i in range(1, steps + 1):
            clock.advance(1.0)
            # On the completing scrape generation has started (decoded > 0):
            # that is what marks the prefill done.
            state["slots"] = [
                _llama_slot_fixture(task, total, min(per_step * i, total), 0,
                                    decoded=1 if i == steps else 0)
            ]
            r = collect_llama_metrics("http://127.0.0.1:9999")
        return r

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        # Req 1: 4000 tok over 4 s -> 1000 t/s
        r = run_request(1, 4000, 1000, 4)
        assert r["prompt_tokens_per_s"] == 1000.0
        # Slot goes idle: the rate HOLDS (mean of 1 sample) instead of
        # dropping to 0 for the rest of the idle period.
        clock.advance(1.0)
        state["slots"] = [
            _llama_slot_fixture(1, 4000, 4000, 0, processing=False)
        ]
        r = collect_llama_metrics("http://127.0.0.1:9999")
        assert r["prompt_tokens_per_s"] == 1000.0

        # Req 2: 8000 tok over 4 s -> 2000 t/s; mean(1000, 2000) = 1500
        r = run_request(2, 8000, 2000, 4)
        assert r["prompt_tokens_per_s"] == 1500.0
        # Reqs 3-5: 300, 50, 1000 t/s
        r = run_request(3, 1200, 300, 4)
        r = run_request(4, 200, 50, 4)
        r = run_request(5, 4000, 1000, 4)
        # mean(1000, 2000, 300, 50, 1000) = 870
        assert r["prompt_tokens_per_s"] == pytest.approx(870.0)
        # Req 6: 2000 t/s; window evicts req 1 -> mean(2000, 300, 50,
        # 1000, 2000) = 1070
        r = run_request(6, 8000, 2000, 4)
        assert r["prompt_tokens_per_s"] == pytest.approx(1070.0)


def test_collect_llama_prompt_rate_skips_unmeasurable_prefills():
    """Prefills that finish between scrapes (fast or fully cached) cannot
    be timed: no sample is recorded and the gauge falls back to the
    per-scrape live rate."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        # Fast prefill: already complete at first sight
        "slots": [_llama_slot_fixture(1, 100, 100, 0)],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        first = collect_llama_metrics("http://127.0.0.1:9999")
        # Fully cached request: complete at first sight, no sample; gauge
        # shows the raw live rate (800 cached tokens / 1 s), not a mean.
        clock.advance(1.0)
        state["slots"] = [_llama_slot_fixture(2, 800, 0, 800)]
        second = collect_llama_metrics("http://127.0.0.1:9999")

    assert first["prompt_tokens_per_s"] == 0.0
    assert second["prompt_tokens_per_s"] == 800.0


def test_collect_llama_prompt_rate_lmstudio_idle_counters():
    """LM Studio's llama.cpp never advances n_prompt_tokens_processed
    (stays 0 even for 12k-token prompts) and resets it when the slot goes
    idle: completion must be detected via n_decoded > 0 (generation start)
    and the sample must not be skipped."""
    from app import collect_llama_metrics

    def lm_slot(task, processing, decoded, processed=0):
        return {"id": 0, "n_ctx": 131072, "n_prompt_tokens": 4000,
                "n_prompt_tokens_processed": processed,
                "n_prompt_tokens_cache": 0,
                "id_task": task, "is_processing": processing,
                "next_token": [{"n_decoded": decoded,
                                "n_remain": -1 if decoded == 0 else 9}]}

    state = {"metrics": ("llamacpp:tokens_predicted_total 0\n"
                         "llamacpp:prompt_tokens_total 0\n"),
             "slots": [lm_slot(1, True, 0)]}

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        collect_llama_metrics("http://127.0.0.1:9999")  # t0: task starts
        # t1: still mid-prefill (decoded 0) -> no sample yet
        clock.advance(1.0)
        state["slots"] = [lm_slot(1, True, 0, processed=2000)]
        r = collect_llama_metrics("http://127.0.0.1:9999")
        # t2: first generated token -> prefill done: 4000 tok / 2 s
        clock.advance(1.0)
        state["slots"] = [lm_slot(1, True, 7, processed=4000)]
        r = collect_llama_metrics("http://127.0.0.1:9999")
        assert r["prompt_tokens_per_s"] == 2000.0
        # t3: slot idle (counters reset to 0) -> rate holds
        clock.advance(1.0)
        state["slots"] = [lm_slot(1, False, 0)]
        r = collect_llama_metrics("http://127.0.0.1:9999")
        assert r["prompt_tokens_per_s"] == 2000.0


def test_collect_llama_prompt_rate_idle_slot_at_first_sight():
    """A slot that is idle at first sight (showing a finished task) must
    not produce a bogus 1-scrape-interval sample."""
    from app import collect_llama_metrics

    state = {"metrics": ("llamacpp:tokens_predicted_total 0\n"
                         "llamacpp:prompt_tokens_total 0\n"),
             "slots": [_llama_slot_fixture(9, 12844, 0, 0, processing=False)]}

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        collect_llama_metrics("http://127.0.0.1:9999")
        clock.advance(1.0)
        r = collect_llama_metrics("http://127.0.0.1:9999")

    # No sample recorded; gauge falls back to the (zero) live rate
    assert r["prompt_tokens_per_s"] == 0.0


def test_collect_llama_prompt_rate_cleared_on_counter_reset():
    """A server restart (cumulative counters decrease) clears the
    per-request history: the gauge falls back to the live rate."""
    import app as app_mod
    from app import collect_llama_metrics

    app_mod._llama_req_rates["primary"] = [1000.0]  # pretend a request completed
    app_mod._llama_prev["primary"] = {"ts": 1000.0, "gen": 1000.0,
                                      "prompt": 1000.0}
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": ("llamacpp:tokens_predicted_total 100\n"
                     "llamacpp:prompt_tokens_total 50\n").encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    # Restart detected (prev 1000 -> now 100): history cleared, no mean
    assert list(app_mod._llama_req_rates["primary"]) == []
    assert result["prompt_tokens_per_s"] == 0.0


def test_collect_llama_prompt_last_new_task():
    """A new id_task on a slot records that prompt's size + cache share;
    the value is absent until a task has been seen, then persists."""
    from app import collect_llama_metrics

    state = {
        "metrics": ("llamacpp:tokens_predicted_total 0\n"
                    "llamacpp:prompt_tokens_total 0\n"),
        "slots": [{"id": 0, "n_ctx": 8192, "n_prompt_tokens": 100,
                   "n_prompt_tokens_processed": 100,
                   "n_prompt_tokens_cache": 0, "id_task": 100,
                   "is_processing": True, "next_token": [{"n_decoded": 0}]}],
    }

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["metrics"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(json.dumps(state["slots"]).encode())

    def set_task(task, total, processed, cached, decoded):
        slot = state["slots"][0]
        slot["id_task"] = task
        slot["n_prompt_tokens"] = total
        slot["n_prompt_tokens_processed"] = processed
        slot["n_prompt_tokens_cache"] = cached
        slot["next_token"] = [{"n_decoded": decoded}]

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.time.monotonic", clock):
        first = collect_llama_metrics("http://127.0.0.1:9999")
        assert "prompt_last" not in first  # no previous scrape to compare

        # Same task continues: still no new task
        clock.advance(1.0)
        set_task(100, 100, 100, 0, 20)
        second = collect_llama_metrics("http://127.0.0.1:9999")
        assert "prompt_last" not in second

        # New task: 800 prompt tokens, 790 of them cache hits
        clock.advance(1.0)
        set_task(101, 800, 10, 790, 0)
        third = collect_llama_metrics("http://127.0.0.1:9999")
        assert third["prompt_last"] == {
            "total": 800, "cached": 790, "pct_cached": 98.8,
        }

        # No new task: the value persists and is re-emitted
        clock.advance(1.0)
        set_task(101, 800, 10, 790, 40)
        fourth = collect_llama_metrics("http://127.0.0.1:9999")
        assert fourth["prompt_last"] == {
            "total": 800, "cached": 790, "pct_cached": 98.8,
        }


def test_collect_llama_slots_failure():
    """/slots down must not break the metrics/props data."""
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf"}',
        "/slots": ConnectionRefusedError("nope"),
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "up"
    assert result["gen_tokens_per_s"] == 52.9
    assert result["model"] == "x.gguf"
    assert result["slots"] == []


def test_collect_llama_props_cached():
    """/props is fetched once per 30 s TTL, then re-fetched when stale."""
    import app as app_mod
    from app import collect_llama_metrics

    calls: list[str] = []
    props_body = b'{"model_path": "/models/x.gguf"}'

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/metrics"):
            return _http_response(b"llamacpp:tokens_predicted_total 1\n")
        if url.endswith("/props"):
            return _http_response(props_body)
        return _http_response(b"[]")  # /slots

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        collect_llama_metrics("http://127.0.0.1:9999")
        collect_llama_metrics("http://127.0.0.1:9999")
    assert calls.count("http://127.0.0.1:9999/props") == 1

    # Age the cache past the 30 s TTL -> must be re-fetched
    app_mod._llama_props_cache["primary"] = (
        time.monotonic() - 31.0, {"model_path": "/models/x.gguf"}
    )
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        collect_llama_metrics("http://127.0.0.1:9999")
    assert calls.count("http://127.0.0.1:9999/props") == 2


def _llama_fake_server_with_body(state: dict):
    """side_effect serving a mutable ``state["body"]`` on /metrics."""
    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics"):
            return _http_response(state["body"].encode())
        if url.endswith("/props"):
            return _http_response(b"{}")
        return _http_response(b"[]")  # /slots

    return fake_urlopen


def test_collect_llama_rate_from_counter_deltas():
    """tokens/s must come from per-scrape counter deltas — the windowed
    gauges read 0 permanently on some builds (e.g. LM Studio's)."""
    from app import collect_llama_metrics

    state = {"body": (
        "llamacpp:tokens_predicted_total 100\n"
        "llamacpp:prompt_tokens_total 50\n"
        "llamacpp:prompt_tokens_cached_total 40\n"
        "llamacpp:predicted_tokens_seconds 0\n"
        "llamacpp:prompt_tokens_seconds 0\n"
    )}
    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=_llama_fake_server_with_body(state)), \
         patch("app.time.monotonic", clock):
        first = collect_llama_metrics("http://127.0.0.1:9999")
        # First scrape has no previous state -> gauge fallback (0.0 here)
        assert first["gen_tokens_per_s"] == 0.0
        assert first["prompt_tokens_per_s"] == 0.0

        state["body"] = (
            "llamacpp:tokens_predicted_total 160\n"
            "llamacpp:prompt_tokens_total 80\n"
            "llamacpp:prompt_tokens_cached_total 70\n"
            "llamacpp:predicted_tokens_seconds 0\n"
            "llamacpp:prompt_tokens_seconds 0\n"
        )
        clock.advance(1.0)
        second = collect_llama_metrics("http://127.0.0.1:9999")

    # 60 gen tokens, 30 new prompt tokens and 30 cached prompt tokens over 1.0 s
    assert second["gen_tokens_per_s"] == 60.0
    assert second["prompt_tokens_per_s"] == 30.0
    assert second["prompt_cached_tokens_per_s"] == 30.0


def test_collect_llama_counter_reset():
    """A decreasing counter (server restart) must yield 0, not a negative rate."""
    import app as app_mod
    from app import collect_llama_metrics

    app_mod._llama_prev["primary"] = {
        "ts": time.monotonic(), "gen": 1000.0, "prompt": 1000.0,
    }
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": ("llamacpp:tokens_predicted_total 100\n"
                     "llamacpp:prompt_tokens_total 50\n").encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["gen_tokens_per_s"] == 0.0
    assert result["prompt_tokens_per_s"] == 0.0


def test_collect_llama_first_scrape_fallback_to_gauge():
    """With no previous scrape, rates fall back to the server's windowed
    gauges, or 0.0 when those are absent."""
    from app import collect_llama_metrics

    # LLAMA_METRICS_SAMPLE carries gauges 52.94 / 32.3 (and no
    # prompt_tokens_total, so no previous state is recorded).
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")
    assert result["gen_tokens_per_s"] == 52.9
    assert result["prompt_tokens_per_s"] == 32.3

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": ("llamacpp:tokens_predicted_total 10\n"
                     "llamacpp:prompt_tokens_total 5\n").encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")
    assert result["gen_tokens_per_s"] == 0.0
    assert result["prompt_tokens_per_s"] == 0.0


# ---------------------------------------------------------------------------
# T3: cumulative usage summary + persistent server uptime
# ---------------------------------------------------------------------------

def _usage_metrics_body(predicted=1000, prompt=5000, cached=4000,
                        predicted_s=40.0, prompt_s=10.0,
                        include_cached=True, include_seconds=True):
    lines = [
        f"llamacpp:tokens_predicted_total {predicted}",
        f"llamacpp:prompt_tokens_total {prompt}",
    ]
    if include_cached:
        lines.append(f"llamacpp:prompt_tokens_cached_total {cached}")
    if include_seconds:
        lines.append(f"llamacpp:tokens_predicted_seconds_total {predicted_s}")
        lines.append(f"llamacpp:prompt_seconds_total {prompt_s}")
    return ("\n".join(lines) + "\n").encode()


def _usage_routes(predicted=1000, prompt=5000, cached=4000,
                  predicted_s=40.0, prompt_s=10.0, **kw):
    return {
        "/metrics": _usage_metrics_body(
            predicted, prompt, cached, predicted_s, prompt_s, **kw
        ),
        "/props": b"{}",
        "/slots": b"[]",
    }


def _usage_state_file() -> Path:
    """The per-test state file installed by isolate_llama_usage_state."""
    return Path(os.environ["RD_LLAMA_USAGE_STATE"])


def test_llama_usage_summary_math_and_state_roundtrip(tmp_path):
    from app import collect_llama_metrics, _llama_load_usage_state

    # The autouse fixture points RD_LLAMA_USAGE_STATE at this test's tmp
    # dir – which also exercises the env override (the default lives in
    # the app dir).
    state_file = _usage_state_file()
    assert state_file.parent == tmp_path

    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(_usage_routes())):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    usage = result["usage"]
    assert usage["total_tokens"] == 1000 + 5000 + 4000
    assert usage["gen_tokens"] == 1000
    # 1000 tokens / 40.0 s of generation time = 25.0 t/s avg (idle excluded)
    assert usage["gen_avg_tps"] == 25.0
    assert usage["prompt_cached_pct"] == 44.4  # 4000 / (4000 + 5000)
    assert usage["uptime_s"] >= 0
    # State persisted (survives dashboard restarts/deployments)
    state = _llama_load_usage_state(str(state_file))
    assert state is not None
    assert state["first_seen"] == state["last_reset"]
    # Second scrape: state re-read from file, no re-init
    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(_usage_routes())):
        result2 = collect_llama_metrics("http://127.0.0.1:9999")
    assert result2["usage"]["uptime_s"] >= usage["uptime_s"]


def test_llama_usage_restart_detection_rebaselines_uptime():
    """A cumulative counter decrease (server restart/reload) re-baselines
    last_reset while keeping first_seen."""
    from app import collect_llama_metrics, _llama_load_usage_state

    state_file = _usage_state_file()
    old = time.time() - 100_000
    state_file.write_text(json.dumps({"first_seen": old, "last_reset": old}))

    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(
                   _usage_routes(predicted=5000, prompt=9000, cached=8000)
               )):
        first = collect_llama_metrics("http://127.0.0.1:9999")
    assert first["usage"]["uptime_s"] >= 99_000  # long-running server

    # Server restarts/reloads: cumulative counters drop.
    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(
                   _usage_routes(predicted=10, prompt=100, cached=50)
               )):
        second = collect_llama_metrics("http://127.0.0.1:9999")
    assert second["usage"]["uptime_s"] < 5  # re-baselined at restart

    state = _llama_load_usage_state(str(state_file))
    assert state is not None
    assert state["last_reset"] > state["first_seen"]


def test_llama_usage_corrupt_state_file_reinitializes():
    from app import collect_llama_metrics, _llama_load_usage_state

    state_file = _usage_state_file()
    state_file.write_text("not-json{{{")

    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(_usage_routes())):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["usage"]["uptime_s"] >= 0
    # Corrupt file was rewritten with valid stamps
    state = _llama_load_usage_state(str(state_file))
    assert state is not None
    assert state["first_seen"] == state["last_reset"]


def test_llama_usage_missing_counters_yield_none():
    """No generation seconds yet → gen_avg_tps None; no cached counter →
    prompt_cached_pct None and total excludes it."""
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(
                   _usage_routes(predicted_s=0.0)
               )):
        result = collect_llama_metrics("http://127.0.0.1:9999")
    assert result["usage"]["gen_avg_tps"] is None

    with patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(
                   _usage_routes(include_cached=False)
               )):
        result = collect_llama_metrics("http://127.0.0.1:9999")
    assert result["usage"]["prompt_cached_pct"] is None
    assert result["usage"]["total_tokens"] == 1000 + 5000


def test_llama_usage_state_path_default_and_env_override(tmp_path, monkeypatch):
    import app as app_mod

    monkeypatch.delenv("RD_LLAMA_USAGE_STATE", raising=False)
    default = app_mod._llama_usage_state_path()
    assert default.endswith(".llama_usage_state.json")
    assert os.path.dirname(default) == os.path.dirname(
        os.path.abspath(app_mod.__file__)
    )
    override = str(tmp_path / "custom_state.json")
    monkeypatch.setenv("RD_LLAMA_USAGE_STATE", override)
    assert app_mod._llama_usage_state_path() == override


# ---------------------------------------------------------------------------
# T4: /health loading status, build_info, decode-stall flag
# ---------------------------------------------------------------------------

def _http_error_503(body: bytes) -> urllib.error.HTTPError:
    """A urllib HTTPError(503) carrying *body*, as llama-server sends it
    while a model is loading."""
    return urllib.error.HTTPError(
        "http://127.0.0.1:9999/health", 503, "Service Unavailable",
        {}, io.BytesIO(body),
    )


def test_collect_llama_health_loading():
    """/health 503 (model loading) -> status "loading"; /metrics is not
    even consulted while the model loads."""
    from app import collect_llama_metrics

    calls: list = []
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/health": _http_error_503(b'{"status":"loading model"}'),
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
    }, calls=calls)):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "loading"
    # Body parsed generically (JSON "status" field), not matched exactly.
    assert result["health_detail"] == "loading model"
    # Early return: only /health was fetched.
    assert all("/health" in c for c in calls)
    assert result.get("gen_tokens_per_s") is None


def test_collect_llama_health_loading_plain_text_body():
    """A non-JSON 503 body is still surfaced (trimmed raw text)."""
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/health": _http_error_503(b"  Loading model \n"),
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["status"] == "loading"
    assert result["health_detail"] == "Loading model"


def test_collect_llama_health_ok_keeps_metrics_logic():
    """/health 200 -> normal /metrics flow, status stays "up"."""
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/health": b'{"status":"ok"}',
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "up"
    assert "health_detail" not in result


def test_collect_llama_health_unreachable_keeps_metrics_logic():
    """The /health fetch itself failing (timeout/down) must not disturb
    the existing up/stale/unreachable /metrics logic."""
    from app import collect_llama_metrics

    # no /health route -> health fetch raises -> /metrics decides
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "up"


def test_collect_llama_build_info():
    """/props build_info passes through; omitted when absent."""
    import app as app_mod
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": (b'{"model_path": "/models/x.gguf", '
                   b'"build_info": "b10450-ece963f41"}'),
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["build_info"] == "b10450-ece963f41"

    # Drop the 30 s /props cache so the second scrape re-fetches props.
    app_mod._llama_props_cache = {}
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf"}',
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert "build_info" not in result


# --- decode-stall flag ------------------------------------------------------

def _decode_metrics_body(n_decode: int) -> bytes:
    return (
        f"llamacpp:n_decode_total {n_decode}\n"
        f"llamacpp:tokens_predicted_total 100\n"
    ).encode()


# Slot mid-generation: is_processing with tokens already decoded.
_ACTIVE_SLOT = (b'[{"id": 0, "n_ctx": 1000, "n_prompt_tokens": 10, '
                b'"is_processing": true, '
                b'"next_token": {"n_decoded": 5, "n_remain": 50}}]')
# Slot in a long prefill: is_processing but nothing decoded yet (no
# llama_decode calls -> n_decode_total legitimately frozen).
_PREFILL_SLOT = (b'[{"id": 0, "n_ctx": 1000, "n_prompt_tokens": 500, '
                 b'"is_processing": true, '
                 b'"next_token": {"n_decoded": 0, "n_remain": -1}}]')


def _seed_decode_history(app_mod, value: float, n: int = 9):
    app_mod._llama_decode_history = {
        "primary": [(1000.0 + i, value) for i in range(n)]
    }


def test_collect_llama_stall_flag_when_decode_frozen():
    """n_decode_total frozen across 10 consecutive successful scrapes
    while a slot is mid-generation -> stalled."""
    import app as app_mod
    from app import collect_llama_metrics

    _seed_decode_history(app_mod, 5000.0)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": _decode_metrics_body(5000),
        "/props": b"{}",
        "/slots": _ACTIVE_SLOT,
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["stalled"] is True
    assert len(app_mod._llama_decode_history["primary"]) == 10


def test_collect_llama_no_stall_during_prefill():
    """Frozen counter while a slot is still in prefill (decoded == 0) is
    NOT a stall: long prefills call no llama_decode."""
    import app as app_mod
    from app import collect_llama_metrics

    _seed_decode_history(app_mod, 5000.0)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": _decode_metrics_body(5000),
        "/props": b"{}",
        "/slots": _PREFILL_SLOT,
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert "stalled" not in result


def test_collect_llama_no_stall_when_decode_advances():
    """Any movement of n_decode_total in the 10-scrape window clears the
    stall condition."""
    import app as app_mod
    from app import collect_llama_metrics

    _seed_decode_history(app_mod, 5000.0)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": _decode_metrics_body(5001),
        "/props": b"{}",
        "/slots": _ACTIVE_SLOT,
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert "stalled" not in result


def test_collect_llama_no_stall_before_history_full():
    """The flag needs 10 consecutive scrapes; a fresh history never fires
    (avoids a false flag right after startup)."""
    from app import collect_llama_metrics

    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": _decode_metrics_body(5000),
        "/props": b"{}",
        "/slots": _ACTIVE_SLOT,
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert "stalled" not in result


def test_collect_llama_stall_history_window_bounded():
    """History never grows past 10 entries."""
    import app as app_mod
    from app import collect_llama_metrics

    _seed_decode_history(app_mod, 5000.0, n=10)
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": _decode_metrics_body(5001),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        collect_llama_metrics("http://127.0.0.1:9999")

    assert len(app_mod._llama_decode_history["primary"]) == 10


# ---------------------------------------------------------------------------
# T5: llama-server process card
# ---------------------------------------------------------------------------

def test_llama_process_name_match():
    """psutil name match is identified; model_size_gb guarded (no file)."""
    import app as app_mod
    from app import collect_llama_metrics

    proc = _fake_llama_proc(rss_gb=21.0, cpu=340.0, threads=64)
    with patch("app.psutil.process_iter", return_value=[proc]), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": LLAMA_METRICS_SAMPLE.encode(),
            "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
            "/slots": b"[]",
        })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"] == {"rss_gb": 21.0, "cpu_pct": 340.0,
                                 "threads": 64}
    assert app_mod._llama_proc["primary"] is proc
    proc.cpu_percent.assert_called_with(None)  # primed on identification
    # /models/x.gguf does not exist on this host -> key omitted
    assert "model_size_gb" not in result


def test_llama_process_multiple_matches_largest_rss_wins():
    import app as app_mod
    from app import collect_llama_metrics

    small = _fake_llama_proc(rss_gb=5.0)
    big = _fake_llama_proc(rss_gb=30.0)
    with patch("app.psutil.process_iter", return_value=[small, big]), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": LLAMA_METRICS_SAMPLE.encode(),
            "/props": b"{}",
            "/slots": b"[]",
        })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"]["rss_gb"] == 30.0
    assert app_mod._llama_proc["primary"] is big
    small.cpu_percent.assert_not_called()


def test_llama_process_cmdline_fallback():
    """No exact name match: a cmdline argument containing llama-server."""
    import app as app_mod
    from app import collect_llama_metrics

    other = _fake_llama_proc(name="python3", rss_gb=1.0,
                             cmdline=["python3", "-c", "pass"])
    server = _fake_llama_proc(
        name="llama-server-bin", rss_gb=21.0,
        cmdline=["/opt/llama/build/bin/llama-server", "--model", "x.gguf"],
    )
    with patch("app.psutil.process_iter", return_value=[other, server]), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": LLAMA_METRICS_SAMPLE.encode(),
            "/props": b"{}",
            "/slots": b"[]",
        })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"]["rss_gb"] == 21.0
    assert app_mod._llama_proc["primary"] is server


def test_llama_process_reidentify_after_death():
    """A dead retained handle (model reload = new PID) is re-found."""
    import psutil
    import app as app_mod
    from app import collect_llama_metrics

    dead = _fake_llama_proc(rss_gb=21.0)
    dead.memory_info.side_effect = psutil.NoSuchProcess(0, "")
    fresh = _fake_llama_proc(rss_gb=22.0)
    with patch("app.psutil.process_iter", return_value=[fresh]), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": LLAMA_METRICS_SAMPLE.encode(),
            "/props": b"{}",
            "/slots": b"[]",
        })):
        app_mod._llama_proc["primary"] = dead
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"]["rss_gb"] == 22.0
    assert app_mod._llama_proc["primary"] is fresh


def test_llama_process_none_when_absent():
    """No local llama-server (remote/dev mode) -> process null, no red."""
    from app import collect_llama_metrics

    # reset_llama_proc fixture: process_iter yields no processes
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": LLAMA_METRICS_SAMPLE.encode(),
        "/props": b'{"model_path": "/models/x.gguf", "total_slots": 1}',
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"] is None


def test_llama_process_model_size(tmp_path):
    """model_size_gb from os.path.getsize on the co-located host."""
    from app import collect_llama_metrics

    model = tmp_path / "x.gguf"
    with open(model, "wb") as f:  # sparse: 19.8 GB without writing it
        f.seek(int(19.8 * 2**30))
        f.write(b"\0")
    with patch("app.psutil.process_iter",
               return_value=[_fake_llama_proc()]), \
         patch("app.urllib.request.urlopen", side_effect=make_llama_server({
            "/metrics": LLAMA_METRICS_SAMPLE.encode(),
            "/props": json.dumps(
                {"model_path": str(model), "total_slots": 1}).encode(),
            "/slots": b"[]",
        })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["model_size_gb"] == 19.8


# ---------------------------------------------------------------------------
# T6: swap usage + swap I/O rate card
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_swap_prev():
    """Reset the module-level swap I/O history around every test."""
    import app as app_mod

    original = app_mod._swap_prev
    app_mod._swap_prev = None
    yield
    app_mod._swap_prev = original


def _fake_swap(used_gb=1.5, total_gb=4.0, percent=37.5, sin=0, sout=0):
    """A MagicMock standing in for psutil's swap_memory result."""
    return MagicMock(
        used=int(used_gb * 2**30),
        total=int(total_gb * 2**30),
        percent=percent,
        sin=sin,
        sout=sout,
    )


def test_swap_basic_fields_and_first_scrape():
    """Usage fields map straight through; rate keys are omitted on the
    first scrape (no previous counters) and the counters are retained."""
    import app as app_mod
    from app import collect_swap_metrics

    with patch("app.psutil.swap_memory",
               return_value=_fake_swap(sin=1024, sout=2048)):
        result = collect_swap_metrics()

    assert result is not None
    assert result["used_gb"] == 1.5
    assert result["total_gb"] == 4.0
    assert result["pct"] == 37.5
    assert "in_kb_s" not in result
    assert "out_kb_s" not in result
    assert app_mod._swap_prev == {"ts": app_mod._swap_prev["ts"],
                                   "sin": 1024, "sout": 2048}


def test_swap_rates_from_counter_deltas():
    """in/out kB/s = cumulative delta / elapsed, over a 2 s scrape gap."""
    from app import collect_swap_metrics

    clock = _FakeClock()
    with patch("app.psutil.swap_memory", side_effect=[
        _fake_swap(sin=0, sout=0),
        _fake_swap(sin=1_048_576, sout=524_288),  # +1 MiB in, +0.5 MiB out
    ]), patch("app.time.monotonic", clock):
        first = collect_swap_metrics()
        assert "in_kb_s" not in first
        clock.advance(2.0)
        second = collect_swap_metrics()

    assert second["in_kb_s"] == 512.0   # 1 MiB / 2 s
    assert second["out_kb_s"] == 256.0  # 0.5 MiB / 2 s
    assert second["used_gb"] == 1.5
    assert second["pct"] == 37.5


def test_swap_rate_clamped_on_counter_reset():
    """A counter decrease (swapon/off, reboot) clamps the rate to 0."""
    from app import collect_swap_metrics

    clock = _FakeClock()
    with patch("app.psutil.swap_memory", side_effect=[
        _fake_swap(sin=10_000_000, sout=10_000_000),
        _fake_swap(sin=100, sout=0),
    ]), patch("app.time.monotonic", clock):
        collect_swap_metrics()
        clock.advance(1.0)
        second = collect_swap_metrics()

    assert second["in_kb_s"] == 0.0
    assert second["out_kb_s"] == 0.0


def test_swap_none_when_total_zero():
    """No swap on the host (total == 0) -> None; no history retained."""
    import app as app_mod
    from app import collect_swap_metrics

    with patch("app.psutil.swap_memory",
               return_value=_fake_swap(used_gb=0.0, total_gb=0.0,
                                       percent=0.0)):
        result = collect_swap_metrics()

    assert result is None
    assert app_mod._swap_prev is None


def test_swap_collector_error_returns_none():
    """psutil.swap_memory raising (host without counters) -> None."""
    import app as app_mod
    from app import collect_swap_metrics

    with patch("app.psutil.swap_memory", side_effect=OSError("no swap")):
        result = collect_swap_metrics()

    assert result is None
    assert app_mod._swap_prev is None


def test_collector_snapshot_includes_swap():
    """The collector's per-tick snapshot carries the top-level swap key."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    with patch("app.psutil.swap_memory",
               return_value=_fake_swap(sin=1024, sout=2048)):
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["swap"]["used_gb"] == 1.5
            assert metrics["swap"]["total_gb"] == 4.0
            assert metrics["swap"]["pct"] == 37.5
            assert "in_kb_s" not in metrics["swap"]
        finally:
            collector.stop()
            collector.join(timeout=2)


# ---------------------------------------------------------------------------
# T7: host health card (PSI memory pressure, D-state processes)
# ---------------------------------------------------------------------------

_PSI_MEMORY_BODY = (
    "some avg10=0.05 avg60=0.02 avg300=0.01 total=123456789\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
)


def _fake_status_proc(status: str) -> MagicMock:
    """A MagicMock process exposing only the 'status' info field."""
    p = MagicMock()
    p.info = {"status": status}
    return p


def test_host_health_psi_parse_and_d_state_count(tmp_path):
    """PSI 'some avg10=' is parsed from a real file; D-state procs are
    counted via the (fake) process table."""
    from app import collect_host_health_metrics

    psi_file = tmp_path / "memory"
    psi_file.write_text(_PSI_MEMORY_BODY)
    procs = [
        _fake_status_proc(psutil.STATUS_DISK_SLEEP),
        _fake_status_proc(psutil.STATUS_DISK_SLEEP),
        _fake_status_proc(psutil.STATUS_RUNNING),
        _fake_status_proc(psutil.STATUS_SLEEPING),
        _fake_status_proc(psutil.STATUS_ZOMBIE),
    ]
    with patch("app.psutil.process_iter", return_value=procs):
        result = collect_host_health_metrics(psi_mem_path=str(psi_file))

    assert result["psi_mem_avg10"] == 0.05
    assert result["d_state_procs"] == 2


def test_host_health_psi_missing_file_returns_none(tmp_path):
    """No /proc/pressure/memory (macOS) -> psi None; D-state still counted."""
    from app import collect_host_health_metrics

    procs = [_fake_status_proc(psutil.STATUS_DISK_SLEEP)]
    with patch("app.psutil.process_iter", return_value=procs):
        result = collect_host_health_metrics(
            psi_mem_path=str(tmp_path / "does-not-exist")
        )

    assert result["psi_mem_avg10"] is None
    assert result["d_state_procs"] == 1


def test_host_health_psi_unparseable_returns_none(tmp_path):
    """A 'some' line with a malformed avg10, or no 'some' line at all,
    both yield None (the 'full' line's avg10 must not be used)."""
    from app import collect_host_health_metrics

    bad = tmp_path / "bad"
    bad.write_text("some avg10=garbage avg60=0.02 total=1\nfull avg10=9.99 total=0\n")
    with patch("app.psutil.process_iter", return_value=[]):
        assert (
            collect_host_health_metrics(psi_mem_path=str(bad))["psi_mem_avg10"]
            is None
        )

    no_some = tmp_path / "no_some"
    no_some.write_text("full avg10=9.99 avg60=0.1 avg300=0.0 total=1\n")
    with patch("app.psutil.process_iter", return_value=[]):
        assert (
            collect_host_health_metrics(psi_mem_path=str(no_some))["psi_mem_avg10"]
            is None
        )


def test_host_health_process_iter_error_degrades_to_zero(tmp_path):
    """A process-table failure degrades D-state to 0 (never raises)."""
    from app import collect_host_health_metrics

    with patch("app.psutil.process_iter", side_effect=OSError("boom")):
        result = collect_host_health_metrics(psi_mem_path=str(tmp_path / "x"))

    assert result["d_state_procs"] == 0
    assert result["psi_mem_avg10"] is None


def test_collector_snapshot_includes_host_health():
    """The collector's per-tick snapshot carries the top-level key with the
    spec'd shape (psi float|None, d_state int)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    procs = [
        _fake_status_proc(psutil.STATUS_DISK_SLEEP),
        _fake_status_proc(psutil.STATUS_RUNNING),
    ]
    with patch("app.psutil.process_iter", return_value=procs):
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
        finally:
            collector.stop()
            collector.join(timeout=2)

    hh = metrics["host_health"]
    assert hh["d_state_procs"] == 1
    assert hh["psi_mem_avg10"] is None or isinstance(hh["psi_mem_avg10"], float)


# ---------------------------------------------------------------------------
# vLLM collector tests
# ---------------------------------------------------------------------------

# Trimmed fixture mirroring probe_vllm_metrics.txt (single engine "0",
# model "Qwen3.8_NVFP"). Only the metrics the collector reads are included.
VLLM_METRICS_SAMPLE = (
    '# HELP vllm:num_requests_running Number of requests in model execution batches.\n'
    '# TYPE vllm:num_requests_running gauge\n'
    'vllm:num_requests_running{engine="0",model_name="Qwen3.8_NVFP"} 7.0\n'
    'vllm:num_requests_waiting{engine="0",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n'
    '# TYPE vllm:kv_cache_usage_perc gauge\n'
    'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.8_NVFP"} 0.09575835475578409\n'
    '# HELP vllm:engine_sleep_state Engine sleep state.\n'
    '# TYPE vllm:engine_sleep_state gauge\n'
    'vllm:engine_sleep_state{engine="0",model_name="Qwen3.8_NVFP",sleep_state="awake"} 1.0\n'
    'vllm:engine_sleep_state{engine="0",model_name="Qwen3.8_NVFP",sleep_state="weights_offloaded"} 0.0\n'
    'vllm:engine_sleep_state{engine="0",model_name="Qwen3.8_NVFP",sleep_state="discard_all"} 0.0\n'
    '# HELP vllm:generation_tokens_total Number of generation tokens processed.\n'
    '# TYPE vllm:generation_tokens_total counter\n'
    'vllm:generation_tokens_total{engine="0",model_name="Qwen3.8_NVFP"} 85077.0\n'
    '# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n'
    '# TYPE vllm:prompt_tokens_total counter\n'
    'vllm:prompt_tokens_total{engine="0",model_name="Qwen3.8_NVFP"} 1722879.0\n'
    '# HELP vllm:prompt_tokens_cached_total Number of cached prompt tokens (local + external).\n'
    '# TYPE vllm:prompt_tokens_cached_total counter\n'
    'vllm:prompt_tokens_cached_total{engine="0",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:prefix_cache_queries_total Prefix cache queries.\n'
    '# TYPE vllm:prefix_cache_queries_total counter\n'
    'vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:prefix_cache_hits_total Prefix cache hits.\n'
    '# TYPE vllm:prefix_cache_hits_total counter\n'
    'vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:num_preemptions_total Cumulative number of preemption.\n'
    '# TYPE vllm:num_preemptions_total counter\n'
    'vllm:num_preemptions_total{engine="0",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:request_success_total Count of successfully processed requests.\n'
    '# TYPE vllm:request_success_total counter\n'
    'vllm:request_success_total{engine="0",finished_reason="stop",model_name="Qwen3.8_NVFP"} 91.0\n'
    'vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen3.8_NVFP"} 0.0\n'
    'vllm:request_success_total{engine="0",finished_reason="abort",model_name="Qwen3.8_NVFP"} 0.0\n'
    'vllm:request_success_total{engine="0",finished_reason="error",model_name="Qwen3.8_NVFP"} 0.0\n'
    'vllm:request_success_total{engine="0",finished_reason="repetition",model_name="Qwen3.8_NVFP"} 0.0\n'
    '# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.\n'
    '# TYPE vllm:spec_decode_num_draft_tokens_total counter\n'
    'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen3.8_NVFP"} 62928.0\n'
    '# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.\n'
    '# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n'
    'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen3.8_NVFP"} 53617.0\n'
    '# HELP vllm:time_to_first_token_seconds Histogram of TTFT.\n'
    '# TYPE vllm:time_to_first_token_seconds histogram\n'
    'vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen3.8_NVFP"} 98.0\n'
    'vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen3.8_NVFP"} 1631.9596149921417\n'
    '# HELP vllm:inter_token_latency_seconds Histogram of ITL.\n'
    '# TYPE vllm:inter_token_latency_seconds histogram\n'
    'vllm:inter_token_latency_seconds_count{engine="0",model_name="Qwen3.8_NVFP"} 31464.0\n'
    'vllm:inter_token_latency_seconds_sum{engine="0",model_name="Qwen3.8_NVFP"} 8031.58660459629\n'
    '# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency.\n'
    '# TYPE vllm:e2e_request_latency_seconds histogram\n'
    'vllm:e2e_request_latency_seconds_count{engine="0",model_name="Qwen3.8_NVFP"} 91.0\n'
    'vllm:e2e_request_latency_seconds_sum{engine="0",model_name="Qwen3.8_NVFP"} 8604.979218482971\n'
    '# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n'
    '# TYPE vllm:cache_config_info gauge\n'
    'vllm:cache_config_info{block_size="1600",cache_dtype="fp8",enable_prefix_caching="False",engine="0",gpu_memory_utilization="0.88",kv_cache_size_tokens="2242627",num_gpu_blocks="1557"} 1.0\n'
    '# HELP process_start_time_seconds Start time of the process since unix epoch.\n'
    '# TYPE process_start_time_seconds gauge\n'
    'process_start_time_seconds 1.78725719225e+09\n'
)


@pytest.fixture(autouse=True)
def reset_vllm_state():
    """Reset module-level vLLM scrape history around every test."""
    import app as app_mod

    originals = {
        k: getattr(app_mod, k) for k in (
            "_vllm_prev", "_vllm_last_good", "_vllm_config_cache",
            "_vllm_req_rates",
        )
    }
    for k in originals:
        setattr(app_mod, k, {})
    yield
    for k, v in originals.items():
        setattr(app_mod, k, v)


def make_vllm_server(body: bytes, calls: list | None = None):
    """Build a side_effect for ``app.urllib.request.urlopen`` that serves
    a vLLM /metrics endpoint. Unknown URL suffixes raise
    ConnectionRefusedError (auto-mode tests exploit this for "other backend
    down")."""
    def fake_urlopen(url, timeout=None):
        if calls is not None:
            calls.append(url)
        if url.endswith("/metrics"):
            return _http_response(body)
        raise ConnectionRefusedError(f"no route for {url}")
    return fake_urlopen


# --- _parse_prometheus_labeled tests ----------------------------------------

def test_parse_prometheus_labeled_basic():
    from app import _parse_prometheus_labeled

    text = (
        '# HELP vllm:num_requests_running desc\n'
        '# TYPE vllm:num_requests_running gauge\n'
        'vllm:num_requests_running{engine="0",model_name="Model"} 7.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="Model"} 2.0\n'
        'process_start_time_seconds 1234.5\n'
        'python_info{implementation="CPython"} 1.0\n'
    )
    result = _parse_prometheus_labeled(text, "vllm:")

    assert "vllm:num_requests_running" in result
    assert "vllm:num_requests_waiting" in result
    assert "process_start_time_seconds" in result
    # Unprefixed non-companion metric must NOT be captured
    assert "python_info" not in result

    entry = result["vllm:num_requests_running"]
    # Key is a tuple of (k, v) pairs, sorted by label name
    key = (("engine", "0"), ("model_name", "Model"))
    assert entry["series"][key] == 7.0
    assert entry["labels"]["engine"] == {"0"}
    assert entry["labels"]["model_name"] == {"Model"}

    # process_start_time_seconds: unlabeled
    pst = result["process_start_time_seconds"]
    assert pst["series"][()] == 1234.5
    assert pst["labels"] == {}


def test_parse_prometheus_labeled_multi_engine():
    from app import _parse_prometheus_labeled

    text = (
        'vllm:kv_cache_usage_perc{engine="0",model_name="M"} 0.5\n'
        'vllm:kv_cache_usage_perc{engine="1",model_name="M"} 0.3\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 3.0\n'
        'vllm:num_requests_running{engine="1",model_name="M"} 5.0\n'
    )
    result = _parse_prometheus_labeled(text, "vllm:")

    # Sum across engines
    from app import _vllm_sum
    running = _vllm_sum(result.get("vllm:num_requests_running"))
    assert running == 8.0

    # Max across engines
    from app import _vllm_max
    kv = _vllm_max(result.get("vllm:kv_cache_usage_perc"))
    assert kv == 0.5


def test_parse_prometheus_labeled_garbage_tolerant():
    from app import _parse_prometheus_labeled

    text = (
        "# comment\n"
        "\n"
        "malformed line without value\n"
        "vllm:bad{engine=0} not_a_number\n"
        "vllm:good{engine=\"0\"} 42.0\n"
    )
    result = _parse_prometheus_labeled(text, "vllm:")
    assert "vllm:good" in result
    assert "vllm:bad" not in result


# --- collect_vllm_metrics tests ----------------------------------------------

def test_collect_vllm_happy_path():
    from app import collect_vllm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(VLLM_METRICS_SAMPLE.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["status"] == "up"
    assert result["model"] == "Qwen3.8_NVFP"
    assert result["requests_processing"] == 7.0
    assert result["requests_waiting"] == 0.0
    assert result["kv_cache_usage_pct"] == round(0.09575835475578409 * 100, 1)  # 9.6
    assert result["gen_tokens_total"] == 85077.0
    assert result["prompt_tokens_total"] == 1722879.0
    assert result["requests_completed_total"] == 91.0  # sum across finished_reason
    assert result["preemptions_total"] == 0.0
    assert result["sleeping"] is False
    # E2E latency: lifetime mean
    assert result["e2e_latency_seconds"] == round(8604.979218482971 / 91.0, 1)
    # Spec decode accept rate
    assert result["spec_accept_pct"] == round(100.0 * 53617 / 62928, 1)
    # Config parsed
    assert result["config"]["kv_cache_size_tokens"] == 2242627
    assert result["config"]["num_gpu_blocks"] == 1557
    assert result["config"]["block_size"] == 1600
    assert result["config"]["cache_dtype"] == "fp8"
    assert result["config"]["gpu_memory_utilization"] == 0.88
    # Usage (nested counters: total = prompt + gen, NOT + cached)
    assert result["usage"]["total_tokens"] == 1722879 + 85077
    assert result["usage"]["gen_tokens"] == 85077
    assert result["usage"]["uptime_s"] >= 0


def test_collect_vllm_no_vllm_metrics_returns_none():
    from app import collect_vllm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(b"some:other_metric 1.0\n")):
        assert collect_vllm_metrics("http://127.0.0.1:9999") is None


def test_collect_vllm_unreachable():
    from app import collect_vllm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        assert collect_vllm_metrics("http://127.0.0.1:9999") is None


def test_collect_vllm_stale_hold():
    """A failed scrape within the hold window returns last data as stale."""
    from app import collect_vllm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(VLLM_METRICS_SAMPLE.encode())):
        first = collect_vllm_metrics("http://127.0.0.1:9999")
    assert first is not None and first["status"] == "up"

    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        held = collect_vllm_metrics("http://127.0.0.1:9999")
    assert held is not None
    assert held["status"] == "stale"
    assert held["model"] == "Qwen3.8_NVFP"


def test_collect_vllm_stale_expiry():
    """A failed scrape after the 15 s hold window returns None."""
    import app as app_mod
    from app import collect_vllm_metrics

    clock = _FakeClock()
    with patch.object(app_mod.time, "monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(VLLM_METRICS_SAMPLE.encode())):
        assert collect_vllm_metrics("http://127.0.0.1:9999")["status"] == "up"

    clock.advance(16.0)
    with patch.object(app_mod.time, "monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        assert collect_vllm_metrics("http://127.0.0.1:9999") is None


def test_collect_vllm_rates_across_two_scrapes():
    """gen/prompt tokens/s from counter deltas over a 1 s gap."""
    from app import collect_vllm_metrics

    body1 = (
        'vllm:generation_tokens_total{engine="0",model_name="M"} 100.0\n'
        'vllm:prompt_tokens_total{engine="0",model_name="M"} 500.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    body2 = (
        'vllm:generation_tokens_total{engine="0",model_name="M"} 160.0\n'
        'vllm:prompt_tokens_total{engine="0",model_name="M"} 580.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        first = collect_vllm_metrics("http://127.0.0.1:9999")
    assert first is not None
    assert "gen_tokens_per_s" not in first  # first scrape: no prev

    clock.advance(1.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body2.encode())), \
         patch("app.time.monotonic", clock):
        second = collect_vllm_metrics("http://127.0.0.1:9999")

    assert second["gen_tokens_per_s"] == 60.0  # 60 tokens / 1 s
    assert second["prompt_tokens_per_s"] == 80.0  # 80 tokens / 1 s


def test_collect_vllm_prompt_rate_averages_completed_requests():
    """prompt tokens/s is the mean of the last 5 completed-request windows
    (Δprompt tokens / ΣTTFT), not the spiky per-scrape delta. TTFT ≈
    prefill + queue time, so a window's sample is that window's effective
    prompt rate."""
    from app import collect_vllm_metrics

    def body(gen, prompt, req, ttft_sum, ttft_count):
        return (
            f'vllm:generation_tokens_total{{engine="0",model_name="M"}} {gen}\n'
            f'vllm:prompt_tokens_total{{engine="0",model_name="M"}} {prompt}\n'
            f'vllm:request_success_total{{engine="0",model_name="M"}} {req}\n'
            f'vllm:time_to_first_token_seconds_sum{{engine="0",model_name="M"}} {ttft_sum}\n'
            f'vllm:time_to_first_token_seconds_count{{engine="0",model_name="M"}} {ttft_count}\n'
        )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body(0, 0, 0, 0.0, 0).encode())), \
         patch("app.time.monotonic", clock):
        first = collect_vllm_metrics("http://127.0.0.1:9999")
    # First scrape: no prev -> rate omitted
    assert "prompt_tokens_per_s" not in first

    # Window 1: 3 requests, 600 prompt tokens, ΣTTFT 0.6 s -> 1000 t/s
    # (raw delta would read 600 — the spike-then-0 pattern).
    clock.advance(1.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(
                   body(10, 600, 3, 0.6, 3).encode())), \
         patch("app.time.monotonic", clock):
        second = collect_vllm_metrics("http://127.0.0.1:9999")
    assert second["prompt_tokens_per_s"] == 1000.0

    # Window 2: 2 requests, 700 tokens, ΣTTFT 1.4 s -> 500 t/s;
    # gauge = mean(1000, 500) = 750 (raw delta would read 700).
    clock.advance(1.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(
                   body(30, 1300, 5, 2.0, 5).encode())), \
         patch("app.time.monotonic", clock):
        third = collect_vllm_metrics("http://127.0.0.1:9999")
    assert third["prompt_tokens_per_s"] == 750.0


def test_collect_vllm_counter_reset_rebaselines():
    """A decreasing counter (server restart) yields 0 rate."""
    import app as app_mod
    from app import collect_vllm_metrics

    app_mod._vllm_prev["primary"] = {"ts": 1000.0, "gen": 10000.0,
                                     "prompt": 10000.0}
    body = (
        'vllm:generation_tokens_total{engine="0",model_name="M"} 100.0\n'
        'vllm:prompt_tokens_total{engine="0",model_name="M"} 200.0\n'
    )
    clock = _FakeClock(start=1001.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())), \
         patch("app.time.monotonic", clock):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result is not None
    assert result["gen_tokens_per_s"] == 0.0
    assert result["prompt_tokens_per_s"] == 0.0


def test_collect_vllm_rate_window_too_short():
    """elapsed < 0.5 s: rate is omitted."""
    from app import collect_vllm_metrics

    body1 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 100.0\n'
    body2 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 160.0\n'

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        collect_vllm_metrics("http://127.0.0.1:9999")

    clock.advance(0.2)  # < 0.5 s
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body2.encode())), \
         patch("app.time.monotonic", clock):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "gen_tokens_per_s" not in result


def test_collect_vllm_rate_window_too_long():
    """elapsed > 15 s: rate is omitted (gauges still present)."""
    from app import collect_vllm_metrics

    body1 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 100.0\n'
    body2 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 160.0\n'

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        collect_vllm_metrics("http://127.0.0.1:9999")

    clock.advance(16.0)  # > 15 s
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body2.encode())), \
         patch("app.time.monotonic", clock):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "gen_tokens_per_s" not in result
    assert result["gen_tokens_total"] == 160.0  # gauge still present


def test_collect_vllm_prefix_cache_omitted_at_zero():
    """prefix_cache_hit_pct omitted when queries == 0."""
    from app import collect_vllm_metrics

    body = (
        'vllm:prefix_cache_queries_total{engine="0",model_name="M"} 0.0\n'
        'vllm:prefix_cache_hits_total{engine="0",model_name="M"} 0.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "prefix_cache_hit_pct" not in result


def test_collect_vllm_prefix_cache_populated():
    """prefix_cache_hit_pct = 100 * hits / queries when queries > 0."""
    from app import collect_vllm_metrics

    body = (
        'vllm:prefix_cache_queries_total{engine="0",model_name="M"} 1000.0\n'
        'vllm:prefix_cache_hits_total{engine="0",model_name="M"} 250.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result["prefix_cache_hit_pct"] == 25.0


def test_collect_vllm_ttft_windowed_mean():
    """TTFT windowed mean = Δsum/Δcount; omitted when Δcount ≤ 0."""
    from app import collect_vllm_metrics

    body1 = (
        'vllm:time_to_first_token_seconds_sum{engine="0",model_name="M"} 100.0\n'
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="M"} 10.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    body2 = (
        'vllm:time_to_first_token_seconds_sum{engine="0",model_name="M"} 200.0\n'
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="M"} 15.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        first = collect_vllm_metrics("http://127.0.0.1:9999")
    assert "ttft_seconds" not in first  # no prev

    clock.advance(1.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body2.encode())), \
         patch("app.time.monotonic", clock):
        second = collect_vllm_metrics("http://127.0.0.1:9999")

    # Δsum = 100, Δcount = 5 → mean = 20.0 s
    assert second["ttft_seconds"] == 20.0


def test_collect_vllm_ttft_no_lifetime_fallback():
    """No TTFT on first scrape (no lifetime fallback)."""
    from app import collect_vllm_metrics

    body = (
        'vllm:time_to_first_token_seconds_sum{engine="0",model_name="M"} 100.0\n'
        'vllm:time_to_first_token_seconds_count{engine="0",model_name="M"} 10.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "ttft_seconds" not in result


def test_collect_vllm_itl_windowed_mean():
    """ITL windowed mean × 1000 → ms; omitted when Δcount ≤ 0."""
    from app import collect_vllm_metrics

    body1 = (
        'vllm:inter_token_latency_seconds_sum{engine="0",model_name="M"} 1000.0\n'
        'vllm:inter_token_latency_seconds_count{engine="0",model_name="M"} 100.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    body2 = (
        'vllm:inter_token_latency_seconds_sum{engine="0",model_name="M"} 3000.0\n'
        'vllm:inter_token_latency_seconds_count{engine="0",model_name="M"} 200.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        collect_vllm_metrics("http://127.0.0.1:9999")

    clock.advance(1.0)
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body2.encode())), \
         patch("app.time.monotonic", clock):
        second = collect_vllm_metrics("http://127.0.0.1:9999")

    # Δsum = 2000, Δcount = 100 → mean = 20.0 s → 20000 ms
    assert second["itl_ms"] == 20000.0


def test_collect_vllm_sleep_state_omitted_when_absent():
    """engine_sleep_state absent → sleeping key omitted."""
    from app import collect_vllm_metrics

    body = 'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "sleeping" not in result


def test_collect_vllm_sleep_state_weights_offloaded():
    """engine_sleep_state with weights_offloaded=1 → sleeping=True."""
    from app import collect_vllm_metrics

    body = (
        'vllm:num_requests_running{engine="0",model_name="M"} 0.0\n'
        'vllm:engine_sleep_state{engine="0",model_name="M",sleep_state="awake"} 0.0\n'
        'vllm:engine_sleep_state{engine="0",model_name="M",sleep_state="weights_offloaded"} 1.0\n'
        'vllm:engine_sleep_state{engine="0",model_name="M",sleep_state="discard_all"} 0.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result["sleeping"] is True


def test_collect_vllm_spec_accept_omitted_when_draft_zero():
    """spec_accept_pct omitted when draft == 0."""
    from app import collect_vllm_metrics

    body = (
        'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="M"} 0.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="M"} 0.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert "spec_accept_pct" not in result


def test_collect_vllm_config_none_values_omitted():
    """cache_config_info label values that are the literal string "None"
    are omitted from the config dict."""
    import app as app_mod
    from app import collect_vllm_metrics

    body = (
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
        'vllm:cache_config_info{block_size="1600",cache_dtype="fp8",'
        'enable_prefix_caching="False",engine="0",'
        'gpu_memory_utilization="None",kv_cache_size_tokens="1000",'
        'num_gpu_blocks="100"} 1.0\n'
    )
    app_mod._vllm_config_cache = {}  # bypass 30s cache
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result["config"]["block_size"] == 1600
    assert result["config"]["num_gpu_blocks"] == 100
    assert result["config"]["kv_cache_size_tokens"] == 1000
    assert result["config"]["cache_dtype"] == "fp8"
    assert "gpu_memory_utilization" not in result["config"]


def test_collect_vllm_usage_nested_counters():
    """vLLM usage: total = prompt + gen (nested: cached ⊂ prompt)."""
    from app import collect_vllm_metrics

    body = (
        'vllm:prompt_tokens_total{engine="0",model_name="M"} 10000.0\n'
        'vllm:generation_tokens_total{engine="0",model_name="M"} 5000.0\n'
        'vllm:prompt_tokens_cached_total{engine="0",model_name="M"} 3000.0\n'
        'process_start_time_seconds 1000000.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())), \
         patch("app.time.time", return_value=1001000.0):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    usage = result["usage"]
    assert usage["total_tokens"] == 15000  # prompt + gen (NOT + cached)
    assert usage["gen_tokens"] == 5000
    assert usage["prompt_cached_pct"] == 30.0  # 3000/10000
    assert usage["uptime_s"] == 1000


def test_collect_vllm_version_skew_aliases():
    """Legacy metric names are used when current names are absent."""
    from app import collect_vllm_metrics

    body = (
        'vllm:gpu_cache_usage_perc{engine="0",model_name="M"} 0.5\n'
        'vllm:num_prompt_tokens{engine="0",model_name="M"} 1000.0\n'
        'vllm:num_generation_tokens{engine="0",model_name="M"} 500.0\n'
        'vllm:num_requests_running{engine="0",model_name="M"} 1.0\n'
    )
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body.encode())):
        result = collect_vllm_metrics("http://127.0.0.1:9999")

    assert result["kv_cache_usage_pct"] == 50.0  # 0.5 × 100
    assert result["prompt_tokens_total"] == 1000.0
    assert result["gen_tokens_total"] == 500.0


# --- _collect_llm fallback matrix (oracle R2) --------------------------------

def test_collect_llm_both_up_vllm_wins():
    """auto mode: both vLLM and llama up → vLLM wins, llama not scraped."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")

    vllm_body = VLLM_METRICS_SAMPLE.encode()
    llama_body = LLAMA_METRICS_SAMPLE.encode()
    calls: list[str] = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/metrics"):
            if "9998" in url:
                return _http_response(vllm_body)
            return _http_response(llama_body)
        if url.endswith("/props"):
            return _http_response(b'{"model_path": "/models/x.gguf", "total_slots": 1}')
        if url.endswith("/slots"):
            return _http_response(b"[]")
        if url.endswith("/health"):
            return _http_response(b'{"status":"ok"}')
        raise ConnectionRefusedError(f"no route for {url}")

    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "vllm"
    assert data["model"] == "Qwen3.8_NVFP"
    # llama was NOT scraped (only vllm /metrics called)
    assert not any("9999" in c for c in calls)


def test_collect_llm_only_vllm_up():
    """auto mode: only vLLM up → vLLM returned."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    vllm_body = VLLM_METRICS_SAMPLE.encode()

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics") and "9998" in url:
            return _http_response(vllm_body)
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "vllm"


def test_collect_llm_only_llama_up():
    """auto mode: vLLM down, llama up → llama returned."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    llama_body = LLAMA_METRICS_SAMPLE.encode()

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics") and "9999" in url:
            return _http_response(llama_body)
        if url.endswith("/props") and "9999" in url:
            return _http_response(b'{"model_path": "/models/x.gguf", "total_slots": 1}')
        if url.endswith("/slots") and "9999" in url:
            return _http_response(b"[]")
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "llama"
    assert data["model"] == "x.gguf"


def test_collect_llm_both_down_returns_none():
    """auto mode: both down → None."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("both down")):
        data = collector._collect_llm()

    assert data is None


def test_collect_llm_vllm_stale_stays_vllm():
    """vLLM within stale-hold returns non-None → stays vLLM (stale)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    vllm_body = VLLM_METRICS_SAMPLE.encode()

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics") and "9998" in url:
            return _http_response(vllm_body)
        raise ConnectionRefusedError("down")

    # First: vLLM up
    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()
    assert data["backend"] == "vllm" and data["status"] == "up"

    # Second: vLLM down, within stale-hold → still vLLM (stale)
    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        data = collector._collect_llm()
    assert data is not None
    assert data["backend"] == "vllm"
    assert data["status"] == "stale"


def test_collect_llm_vllm_recovers_returns_to_vllm():
    """vLLM down >15 s → llama; vLLM recovers → selection flips back to vLLM."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    vllm_body = VLLM_METRICS_SAMPLE.encode()
    llama_body = LLAMA_METRICS_SAMPLE.encode()

    def up_vllm_down_llama(url, timeout=None):
        if url.endswith("/metrics") and "9998" in url:
            return _http_response(vllm_body)
        raise ConnectionRefusedError("down")

    def down_vllm_up_llama(url, timeout=None):
        if "9998" in url:
            raise ConnectionRefusedError("vLLM down")
        if url.endswith("/metrics"):
            return _http_response(llama_body)
        if url.endswith("/props"):
            return _http_response(
                b'{"model_path": "/models/x.gguf", "total_slots": 1}')
        if url.endswith("/slots"):
            return _http_response(b"[]")
        raise ConnectionRefusedError(f"no route for {url}")

    clock = _FakeClock()
    # 1. vLLM up
    with patch("app.urllib.request.urlopen", side_effect=up_vllm_down_llama), \
         patch("app.time.monotonic", clock):
        data = collector._collect_llm()
    assert data is not None
    assert data["backend"] == "vllm"

    # 2. vLLM down beyond the 15 s stale-hold → fall through to llama
    clock.advance(16.0)
    with patch("app.urllib.request.urlopen", side_effect=down_vllm_up_llama), \
         patch("app.time.monotonic", clock):
        data = collector._collect_llm()
    assert data is not None
    assert data["backend"] == "llama"

    # 3. vLLM recovers → selection flips back to vLLM
    with patch("app.urllib.request.urlopen", side_effect=up_vllm_down_llama), \
         patch("app.time.monotonic", clock):
        data = collector._collect_llm()
    assert data is not None
    assert data["backend"] == "vllm"


def test_collect_llm_pinned_llama_ignores_vllm():
    """Pinned --llm-backend llama: vLLM URL is never scraped."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="llama")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    llama_body = LLAMA_METRICS_SAMPLE.encode()
    calls: list[str] = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/metrics") and "9999" in url:
            return _http_response(llama_body)
        if url.endswith("/props") and "9999" in url:
            return _http_response(b'{"model_path": "/models/x.gguf", "total_slots": 1}')
        if url.endswith("/slots") and "9999" in url:
            return _http_response(b"[]")
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "llama"
    # vLLM was NOT scraped
    assert not any("9998" in c for c in calls)


def test_collect_llm_pinned_vllm_ignores_llama():
    """Pinned --llm-backend vllm: llama URL is never scraped."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="vllm")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.llama_url = "http://127.0.0.1:9999"

    vllm_body = VLLM_METRICS_SAMPLE.encode()
    calls: list[str] = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/metrics") and "9998" in url:
            return _http_response(vllm_body)
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "vllm"
    # llama was NOT scraped
    assert not any("9999" in c for c in calls)


def test_collect_llm_disabled_returns_none():
    """--no-llama: _collect_llm returns None without any scraping."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)

    with patch("app.urllib.request.urlopen") as fake_urlopen:
        fake_urlopen.side_effect = ConnectionRefusedError("must not be called")
        data = collector._collect_llm()

    assert data is None
    assert fake_urlopen.call_count == 0


# --- Snapshot schema tests ---------------------------------------------------

def test_snapshot_schema_llm_key_singular():
    """Exactly one backend in llm; no llama key; llm_enabled present."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="vllm")
    collector.vllm_url = "http://127.0.0.1:9998"

    vllm_body = VLLM_METRICS_SAMPLE.encode()

    def fake_urlopen(url, timeout=None):
        if url.endswith("/metrics") and "9998" in url:
            return _http_response(vllm_body)
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("app.collect_gpu_metrics", return_value=[]), \
         patch("app.collect_swap_metrics", return_value=None), \
         patch("app.collect_host_health_metrics", return_value={"psi_mem_avg10": None, "d_state_procs": 0}):
        # Populate _llm_latest the way the LLM thread would (this test
        # calls _collect() directly without starting the threads).
        collector._llm_latest = [
            collector._collect_one(collector._targets_snapshot()[0])]
        snapshot = collector._collect()

    assert "llm" in snapshot
    assert "llama" not in snapshot
    assert "llm_enabled" in snapshot
    assert snapshot["llm"]["backend"] == "vllm"
    assert "backend" in snapshot["llm"]


# ---------------------------------------------------------------------------
# JetPack platform detection + integrated-GPU collection
# ---------------------------------------------------------------------------

_MEMINFO_16GB = (
    "MemTotal:        16777216 kB\n"
    "MemFree:         2000000 kB\n"
    "MemAvailable:     8388608 kB\n"
)


@pytest.fixture
def platform_env(tmp_path, monkeypatch):
    """Point all platform-detection inputs at an empty tmp tree, reset the
    lazy platform globals, and pin a deterministic non-tegra kernel."""
    import app

    monkeypatch.setattr(app, "_PLATFORM", None)
    monkeypatch.setattr(app, "_JETSON_SOC", None)
    dt = tmp_path / "dt"
    dt.mkdir()
    monkeypatch.setattr(app, "_DT_MODEL_PATHS", (str(dt / "model"),))
    monkeypatch.setattr(app, "_DT_COMPATIBLE_PATHS", (str(dt / "compatible"),))
    monkeypatch.setattr(app, "_NV_TEGRA_RELEASE_PATH", str(tmp_path / "nv_tegra_release"))
    monkeypatch.setattr("platform.release", lambda: "6.8.0-test")
    return SimpleNamespace(
        model=dt / "model",
        compatible=dt / "compatible",
        nv_tegra_release=tmp_path / "nv_tegra_release",
    )


@pytest.fixture
def jetson_env(tmp_path, monkeypatch):
    """Reset jetson globals to a known state; provide empty fake sysfs dirs
    and a 16 GiB unified-pool meminfo (8 GiB available → 50 % used)."""
    import app

    monkeypatch.setattr(app, "_PLATFORM", "jetson")
    monkeypatch.setattr(app, "_JETSON_SOC", "orin")
    monkeypatch.setattr(app, "_jetson_devfreq_cache", None)
    monkeypatch.setattr(app, "_jetson_mode_logged", True)  # keep logs quiet
    devfreq = tmp_path / "devfreq"
    devfreq.mkdir()
    thermal = tmp_path / "thermal"
    thermal.mkdir()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_MEMINFO_16GB)

    def args():
        return dict(
            devfreq_dir=str(devfreq),
            meminfo_path=str(meminfo),
            thermal_dir=str(thermal),
        )

    return SimpleNamespace(devfreq=devfreq, thermal=thermal, meminfo=meminfo, args=args)


def _add_gpu_node(env, name="17000000.gpu", load=None, cur_freq=None):
    node = env.devfreq / name
    node.mkdir()
    if cur_freq is not None:
        (node / "cur_freq").write_text(f"{cur_freq}\n")
    if load is not None:
        (node / "device").mkdir()
        (node / "device" / "load").write_text(f"{load}\n")
    return node


def _add_zone(env, idx, ztype, temp):
    zone = env.thermal / f"thermal_zone{idx}"
    zone.mkdir()
    (zone / "type").write_text(f"{ztype}\n")
    (zone / "temp").write_text(f"{temp}\n")
    return zone


# -- platform detection ------------------------------------------------------


def test_platform_desktop_no_signals(platform_env):
    import app

    assert app._init_platform() == "desktop"
    assert app._JETSON_SOC == "unknown"


def test_platform_model_jetson(platform_env):
    import app

    platform_env.model.write_bytes(b"NVIDIA Jetson AGX Orin Developer Kit\x00")
    assert app._init_platform() == "jetson"


def test_platform_model_orin_nx_without_jetson_word(platform_env):
    """Vendor-board model string without the literal "Jetson" (oracle m5)."""
    import app

    platform_env.model.write_bytes(b"NVIDIA Orin NX Developer Kit\x00")
    assert app._init_platform() == "jetson"


def test_platform_model_thor_without_nvidia_is_desktop(platform_env):
    """Custom board with "thor" in the model but no nvidia co-occurrence →
    desktop (oracle m1 false-positive guard)."""
    import app

    platform_env.model.write_bytes(b"Acme Thor Developer Board\x00")
    assert app._init_platform() == "desktop"


def test_platform_compatible_only(platform_env):
    import app

    platform_env.compatible.write_bytes(
        b"nvidia,p4071-0000+p3834-0008\x00nvidia,tegra264\x00"
    )
    assert app._init_platform() == "jetson"
    assert app._JETSON_SOC == "thor"


def test_platform_nv_tegra_release_fallback(platform_env):
    import app

    platform_env.nv_tegra_release.write_text(
        "# R38 (release), REVISION: 2.1, GCID: 1234567, EABI: aarch64\n"
    )
    assert app._init_platform() == "jetson"
    assert app._JETSON_SOC == "unknown"


def test_platform_tegra_kernel_fallback(platform_env, monkeypatch):
    import app

    monkeypatch.setattr("platform.release", lambda: "6.8.12-1021-tegra")
    assert app._init_platform() == "jetson"


def test_platform_unreadable_model_file_skipped(platform_env):
    """An unreadable model path (a directory) is an OSError → skipped."""
    import app

    platform_env.model.mkdir()  # open() raises IsADirectoryError
    assert app._init_platform() == "desktop"


def test_platform_result_cached(platform_env):
    """Detection runs once; later file changes do not re-detect."""
    import app

    assert app._init_platform() == "desktop"
    platform_env.model.write_bytes(b"NVIDIA Jetson AGX Orin Developer Kit\x00")
    assert app._init_platform() == "desktop"
    assert app._JETSON_SOC == "unknown"


# -- SoC parse + board label ---------------------------------------------------


def test_soc_from_compatible_tegra264_is_thor():
    from app import _soc_from_compatible

    assert _soc_from_compatible(b"nvidia,p4071+p3834\x00nvidia,tegra264\x00") == "thor"


def test_soc_from_compatible_tegra234_is_orin():
    from app import _soc_from_compatible

    assert _soc_from_compatible(b"nvidia,orin-nx-nv\x00nvidia,tegra234\x00") == "orin"


def test_soc_from_compatible_garbage_is_unknown():
    from app import _soc_from_compatible

    assert _soc_from_compatible(b"nvidia,tegra210\x00") == "unknown"
    assert _soc_from_compatible(b"") == "unknown"


def test_board_label_soc_names(platform_env, monkeypatch):
    import app

    monkeypatch.setattr(app, "_JETSON_SOC", "thor")
    assert app._jetson_board_label() == "NVIDIA Thor"
    monkeypatch.setattr(app, "_JETSON_SOC", "orin")
    assert app._jetson_board_label() == "NVIDIA Orin"
    monkeypatch.setattr(app, "_JETSON_SOC", "unknown")
    platform_env.model.write_bytes(b"NVIDIA Jetson AGX Thor Developer Kit\x00")
    assert app._jetson_board_label() == "NVIDIA Jetson AGX Thor Developer Kit"
    platform_env.model.write_bytes(b"")
    assert app._jetson_board_label() == "NVIDIA Jetson"


# -- jetson collector ----------------------------------------------------------


def test_jetson_orin_happy_path(jetson_env):
    from app import collect_jetson_gpu_metrics

    _add_gpu_node(jetson_env, load=420, cur_freq=1_365_000_000)
    _add_zone(jetson_env, 1, "gpu-thermal", 54_500)
    result = collect_jetson_gpu_metrics(**jetson_env.args())

    assert len(result) == 1
    gpu = result[0]
    assert gpu["index"] == 0
    assert gpu["name"] == "NVIDIA Orin"
    assert gpu["memory_unified"] is True
    assert gpu["utilization_pct"] == 42.0
    assert gpu["memory_total_gb"] == 16.0
    assert gpu["memory_used_gb"] == 8.0
    assert gpu["memory_pct"] == 50.0
    assert gpu["temperature_c"] == 54.5
    assert gpu["graphics_clock_mhz"] == 1365.0
    # Intentionally absent on JetPack (renderer-tolerated)
    for key in ("pci_bus_id", "board_vendor", "throttle", "power_w",
                "power_limit_w"):
        assert key not in gpu


def test_jetson_util_prefers_load_file_over_smi(jetson_env, no_real_nvidia,
                                                monkeypatch):
    """Capability rule: a devfreq node with device/load wins even on Thor."""
    import app

    monkeypatch.setattr(app, "_JETSON_SOC", "thor")
    _add_gpu_node(jetson_env, name="gpu-gpc-0", load=330, cur_freq=1_500_000_000)
    no_real_nvidia.return_value = MagicMock(stdout="99\n")
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert gpu["utilization_pct"] == 33.0
    assert gpu["name"] == "NVIDIA Thor"
    no_real_nvidia.assert_not_called()


def test_jetson_thor_util_from_nvidia_smi(jetson_env, no_real_nvidia,
                                          monkeypatch):
    """Thor without device/load falls back to nvidia-smi (tolerant parse)."""
    import app

    monkeypatch.setattr(app, "_JETSON_SOC", "thor")
    _add_gpu_node(jetson_env, name="gpu-gpc-0", cur_freq=1_500_000_000)
    no_real_nvidia.return_value = MagicMock(stdout="73\n")
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert gpu["utilization_pct"] == 73.0
    assert gpu["graphics_clock_mhz"] == 1500.0
    no_real_nvidia.assert_called_once()


def test_jetson_thor_nvidia_smi_na_omits_util(jetson_env, no_real_nvidia,
                                              monkeypatch):
    import app

    monkeypatch.setattr(app, "_JETSON_SOC", "thor")
    _add_gpu_node(jetson_env, name="gpu-gpc-0", cur_freq=1_500_000_000)
    _add_zone(jetson_env, 1, "gpu-thermal", 49_000)
    no_real_nvidia.return_value = MagicMock(stdout="N/A\n")
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert "utilization_pct" not in gpu
    # Everything else still present
    assert gpu["graphics_clock_mhz"] == 1500.0
    assert gpu["temperature_c"] == 49.0
    assert gpu["memory_pct"] == 50.0


def test_jetson_orin_no_node_omits_util_not_zero(jetson_env, no_real_nvidia):
    """Late-boot Orin (no devfreq node yet): util omitted, never 0, never CLI."""
    import app

    _add_zone(jetson_env, 1, "gpu-thermal", 45_000)
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert "utilization_pct" not in gpu
    assert "graphics_clock_mhz" not in gpu
    assert gpu["temperature_c"] == 45.0
    no_real_nvidia.assert_not_called()


def test_jetson_devfreq_late_boot_not_cached_absent(jetson_env):
    """Node absent on tick 1, present on tick 2 → picked up (no cached
    absence)."""
    import app

    first = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]
    assert "utilization_pct" not in first

    _add_gpu_node(jetson_env, load=500, cur_freq=1_000_000_000)
    second = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]
    assert second["utilization_pct"] == 50.0
    assert second["graphics_clock_mhz"] == 1000.0
    assert app._jetson_devfreq_cache == "17000000.gpu"


def test_jetson_devfreq_strict_pattern(jetson_env):
    """Non-matching devfreq names are ignored (no loose 'contains gpu')."""
    import app

    _add_gpu_node(jetson_env, name="foo-gpu-bar", cur_freq=1_000_000_000)
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert "graphics_clock_mhz" not in gpu
    assert "utilization_pct" not in gpu
    assert app._jetson_devfreq_cache is None


def test_jetson_thermal_max_and_filtering(jetson_env):
    import app

    _add_zone(jetson_env, 0, "tj-thermal", 90_000)      # wrong type
    _add_zone(jetson_env, 1, "gpu-thermal", -256_000)   # no sensor
    _add_zone(jetson_env, 2, "gpu-thermal", 50_000)
    _add_zone(jetson_env, 3, "gpu-thermal", 61_000)
    _add_zone(jetson_env, 4, "cpu-thermal", 99_000)     # wrong type
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert gpu["temperature_c"] == 61.0


def test_jetson_thermal_zone_missing_temp_skipped(jetson_env):
    import app

    zone = jetson_env.thermal / "thermal_zone5"
    zone.mkdir()
    (zone / "type").write_text("gpu-thermal\n")  # no temp file
    _add_zone(jetson_env, 6, "gpu-thermal", 44_000)
    gpu = app.collect_jetson_gpu_metrics(**jetson_env.args())[0]

    assert gpu["temperature_c"] == 44.0


def test_jetson_meminfo_missing_omits_mem_keys(jetson_env):
    import app

    gpu = app.collect_jetson_gpu_metrics(
        devfreq_dir=str(jetson_env.devfreq),
        meminfo_path=str(jetson_env.meminfo.parent / "no_meminfo"),
        thermal_dir=str(jetson_env.thermal),
    )[0]

    for key in ("memory_total_gb", "memory_used_gb", "memory_pct"):
        assert key not in gpu
    assert gpu["memory_unified"] is True


def test_jetson_all_absent_never_raises(jetson_env):
    import app

    gpu = app.collect_jetson_gpu_metrics(
        devfreq_dir=str(jetson_env.devfreq),  # empty dir
        meminfo_path=str(jetson_env.meminfo.parent / "no_meminfo"),
        thermal_dir=str(jetson_env.thermal),  # empty dir
    )[0]

    assert gpu == {"index": 0, "name": "NVIDIA Orin", "memory_unified": True}


# -- collect_gpu_metrics gate --------------------------------------------------


def test_gpu_metrics_gate_jetson_skips_desktop_paths(jetson_env, no_real_nvidia,
                                                    monkeypatch):
    """Platform "jetson" ⇒ _gpu_pynvml and _gpu_cli are never called."""
    import app

    calls = []
    monkeypatch.setattr(app, "_gpu_pynvml",
                        lambda: calls.append("pynvml") or [{"index": 9}])
    monkeypatch.setattr(app, "_gpu_cli",
                        lambda: calls.append("cli") or [{"index": 9}])
    # The no-arg call inside collect_gpu_metrics() binds the path defaults at
    # def-time, so re-pointing the module constants is inert - patch the
    # function's __defaults__ instead (keeps the test hermetic: the fake
    # meminfo below is what feeds the assertions, not the host's /sys).
    monkeypatch.setattr(
        app.collect_jetson_gpu_metrics, "__defaults__",
        (str(jetson_env.devfreq), str(jetson_env.meminfo),
         str(jetson_env.thermal)))
    result = app.collect_gpu_metrics()

    assert calls == []
    no_real_nvidia.assert_not_called()
    assert len(result) == 1
    assert result[0]["name"] == "NVIDIA Orin"
    assert result[0]["memory_unified"] is True
    # Fake 16 GiB pool, 8 GiB available -> proves the fixture paths are live.
    assert result[0]["memory_pct"] == 50.0


def test_gpu_metrics_gate_desktop_unchanged(platform_env, no_real_nvidia,
                                            monkeypatch):
    import app

    assert app._init_platform() == "desktop"
    monkeypatch.setattr(app, "_gpu_pynvml", lambda: [{"index": 0, "name": "fake"}])
    result = app.collect_gpu_metrics()

    # Desktop path now tags each card with its vendor
    assert result == [{"index": 0, "name": "fake", "vendor": "nvidia"}]
    no_real_nvidia.assert_not_called()


def test_gpu_metrics_jetson_mode_logged_once(jetson_env, monkeypatch, caplog):
    import app

    monkeypatch.setattr(app, "_jetson_mode_logged", False)
    with caplog.at_level(logging.INFO):
        app.collect_gpu_metrics()
        app.collect_gpu_metrics()

    msgs = [r.getMessage() for r in caplog.records]
    assert sum("GPU mode: jetson (SoC: orin)" in m for m in msgs) == 1


# ---------------------------------------------------------------------------
# LiteLLM collector tests
# ---------------------------------------------------------------------------

# Trimmed fixture mirroring a real LiteLLM proxy /metrics body (prometheus
# callback enabled). Counters carry the prometheus_client `_total` suffix;
# two label series exercise the sum-across-deployments path.
LITELLM_METRICS_SAMPLE = (
    '# HELP litellm_in_flight_requests Number of HTTP requests currently in-flight.\n'
    '# TYPE litellm_in_flight_requests gauge\n'
    'litellm_in_flight_requests 3.0\n'
    '# HELP litellm_output_tokens_metric Total output tokens.\n'
    '# TYPE litellm_output_tokens_metric counter\n'
    'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12000.0\n'
    'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="llama-3.1-8b"} 345.0\n'
    '# HELP litellm_input_tokens_metric Total input tokens.\n'
    '# TYPE litellm_input_tokens_metric counter\n'
    'litellm_input_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 98765.0\n'
    '# HELP litellm_input_cached_tokens_metric Provider-reported cached input tokens.\n'
    '# TYPE litellm_input_cached_tokens_metric counter\n'
    'litellm_input_cached_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 9876.5\n'
    '# HELP litellm_proxy_total_requests_metric Total requests made to the proxy.\n'
    '# TYPE litellm_proxy_total_requests_metric counter\n'
    'litellm_proxy_total_requests_metric_total{requested_model="llama-3.1-8b",status_code="200"} 42.0\n'
    '# HELP litellm_proxy_failed_requests_metric Failed responses from the proxy.\n'
    '# TYPE litellm_proxy_failed_requests_metric counter\n'
    'litellm_proxy_failed_requests_metric_total{exception_class="Timeout",requested_model="llama-3.1-8b"} 2.0\n'
    '# HELP litellm_spend_metric Total spend.\n'
    '# TYPE litellm_spend_metric counter\n'
    'litellm_spend_metric_total{requested_model="llama-3.1-8b"} 0.123456\n'
    '# HELP litellm_llm_api_time_to_first_token_metric Time to first token (streaming).\n'
    '# TYPE litellm_llm_api_time_to_first_token_metric histogram\n'
    'litellm_llm_api_time_to_first_token_metric_sum{requested_model="llama-3.1-8b"} 12.5\n'
    'litellm_llm_api_time_to_first_token_metric_count{requested_model="llama-3.1-8b"} 10.0\n'
    '# HELP litellm_deployment_latency_per_output_token Latency per output token.\n'
    '# TYPE litellm_deployment_latency_per_output_token gauge\n'
    'litellm_deployment_latency_per_output_token{litellm_model_name="llama-3.1-8b",model_id="m1"} 0.031\n'
    'litellm_deployment_latency_per_output_token{litellm_model_name="llama-3.1-8b",model_id="m2"} 0.022\n'
    '# HELP litellm_request_total_latency_metric End-to-end request latency.\n'
    '# TYPE litellm_request_total_latency_metric histogram\n'
    'litellm_request_total_latency_metric_sum{requested_model="llama-3.1-8b"} 55.0\n'
    'litellm_request_total_latency_metric_count{requested_model="llama-3.1-8b"} 40.0\n'
    '# HELP litellm_llm_api_latency_metric LLM API call latency.\n'
    '# TYPE litellm_llm_api_latency_metric histogram\n'
    'litellm_llm_api_latency_metric_sum{requested_model="llama-3.1-8b"} 50.0\n'
    'litellm_llm_api_latency_metric_count{requested_model="llama-3.1-8b"} 40.0\n'
    '# HELP process_start_time_seconds Start time of the process since unix epoch.\n'
    '# TYPE process_start_time_seconds gauge\n'
    'process_start_time_seconds 1.78725719225e+09\n'
)


@pytest.fixture(autouse=True)
def reset_litellm_state():
    """Reset module-level LiteLLM scrape history around every test."""
    import app as app_mod

    original_prev = app_mod._litellm_prev
    original_last_good = app_mod._litellm_last_good
    app_mod._litellm_prev = {}
    app_mod._litellm_last_good = {}
    yield
    app_mod._litellm_prev = original_prev
    app_mod._litellm_last_good = original_last_good


def make_litellm_server(body: bytes, calls: list | None = None):
    """Side_effect for ``app.urllib.request.urlopen`` serving a LiteLLM
    /metrics endpoint. The collector passes a urllib Request object, so
    resolve ``full_url`` and capture headers (token tests)."""
    import urllib.request as _ur

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        if calls is not None:
            headers = dict(req.header_items()) if isinstance(req, _ur.Request) else {}
            calls.append((url, headers))
        if url.endswith("/metrics"):
            return _http_response(body)
        raise ConnectionRefusedError(f"no route for {url}")
    return fake_urlopen


def test_collect_litellm_happy_path():
    from app import collect_litellm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())):
        result = collect_litellm_metrics("http://127.0.0.1:9997")

    assert result is not None
    assert result["status"] == "up"
    assert result["model"] == "llama-3.1-8b"
    assert result["requests_processing"] == 3.0
    # Counters summed across label series
    assert result["gen_tokens_total"] == 12345.0
    assert result["prompt_tokens_total"] == 98765.0
    assert result["requests_completed_total"] == 42.0
    assert result["requests_failed_total"] == 2.0
    assert result["spend_usd"] == 0.1235
    # Provider-side prompt-cache hit rate (lifetime ratio)
    assert result["prefix_cache_hit_pct"] == round(100.0 * 9876.5 / 98765.0, 1)
    # ITL: max across deployments × 1000
    assert result["itl_ms"] == 31.0
    # Lifetime means
    assert result["e2e_latency_seconds"] == round(55.0 / 40.0, 1)  # 1.4
    assert result["llm_api_latency_seconds"] == round(50.0 / 40.0, 1)  # 1.25
    # First scrape: no Δ-window rates yet
    assert "gen_tokens_per_s" not in result
    assert "prompt_tokens_per_s" not in result
    assert "ttft_seconds" not in result
    # Usage (disjoint counters: total = input + output)
    usage = result["usage"]
    assert usage["total_tokens"] == 98765 + 12345
    assert usage["gen_tokens"] == 12345
    assert usage["prompt_cached_pct"] == round(100.0 * 9876.5 / 98765.0, 1)
    assert usage["total_spend_usd"] == 0.1235
    assert usage["uptime_s"] >= 0


def test_collect_litellm_rates_and_ttft_from_deltas():
    """Second scrape within the 0.5–5 s window yields rates + windowed TTFT."""
    from app import collect_litellm_metrics

    body = LITELLM_METRICS_SAMPLE.replace(
        "litellm_output_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 12000.0",
        "litellm_output_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 12250.0",
    ).replace(
        "litellm_input_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 98765.0",
        "litellm_input_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 99765.0",
    ).replace(
        "litellm_llm_api_time_to_first_token_metric_sum{requested_model=\"llama-3.1-8b\"} 12.5",
        "litellm_llm_api_time_to_first_token_metric_sum{requested_model=\"llama-3.1-8b\"} 18.5",
    ).replace(
        "litellm_llm_api_time_to_first_token_metric_count{requested_model=\"llama-3.1-8b\"} 10.0",
        "litellm_llm_api_time_to_first_token_metric_count{requested_model=\"llama-3.1-8b\"} 14.0",
    )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())), \
         patch("app.time.monotonic", clock):
        first = collect_litellm_metrics("http://127.0.0.1:9997")
        assert "gen_tokens_per_s" not in first
        clock.advance(2.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    # gen: (12250+345) - (12000+345) = 250 tok / 2 s
    assert second["gen_tokens_per_s"] == 125.0
    # prompt: 99765 - 98765 = 1000 tok / 2 s
    assert second["prompt_tokens_per_s"] == 500.0
    # TTFT: windowed mean (18.5-12.5)/(14-10) = 1.5 s
    assert second["ttft_seconds"] == 1.5


def test_collect_litellm_counter_reset_clamps_rate():
    from app import collect_litellm_metrics

    reset_body = LITELLM_METRICS_SAMPLE.replace(
        "litellm_output_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 12000.0",
        "litellm_output_tokens_metric_total{api_provider=\"openai\",model_id=\"m1\",requested_model=\"llama-3.1-8b\"} 1.0",
    )

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())), \
         patch("app.time.monotonic", clock):
        collect_litellm_metrics("http://127.0.0.1:9997")
        clock.advance(2.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(reset_body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    assert second["gen_tokens_per_s"] == 0.0  # counter reset → clamped


def _two_model_body() -> str:
    """LiteLLM body with two distinct requested_model aliases.

    The shared fixture labels both deployments as llama-3.1-8b; rename
    m2 so per-model grouping has two distinct models.
    """
    return LITELLM_METRICS_SAMPLE.replace(
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="llama-3.1-8b"} 345.0',
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="gpt-4o-mini"} 345.0',
    )


def test_collect_litellm_per_model_rates():
    """Per-model gen tokens/s from per-model counter deltas (2 models)."""
    from app import collect_litellm_metrics

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(_two_model_body().encode())), \
         patch("app.time.monotonic", clock):
        first = collect_litellm_metrics("http://127.0.0.1:9997")
        assert "gen_tokens_per_s_by_model" not in first  # no baseline yet
        clock.advance(2.0)
        second_body = _two_model_body().replace(
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12000.0',
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12100.0',
        ).replace(
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="gpt-4o-mini"} 345.0',
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="gpt-4o-mini"} 385.0',
        )
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(second_body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    assert second["gen_tokens_per_s_by_model"] == {
        "llama-3.1-8b": 50.0,  # 100 tok / 2 s
        "gpt-4o-mini": 20.0,   # 40 tok / 2 s
    }
    # Aggregate total still reported (sum of per-model deltas)
    assert second["gen_tokens_per_s"] == 70.0


def test_collect_litellm_per_model_counter_reset_clamps():
    """A per-model counter reset clamps only that model's rate to 0."""
    from app import collect_litellm_metrics

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(_two_model_body().encode())), \
         patch("app.time.monotonic", clock):
        collect_litellm_metrics("http://127.0.0.1:9997")
        clock.advance(2.0)
        reset_body = _two_model_body().replace(
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12000.0',
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12200.0',
        ).replace(
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="gpt-4o-mini"} 345.0',
            'litellm_output_tokens_metric_total{api_provider="openai",model_id="m2",requested_model="gpt-4o-mini"} 1.0',
        )
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(reset_body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    pm = second["gen_tokens_per_s_by_model"]
    assert pm["gpt-4o-mini"] == 0.0   # reset → clamped
    assert pm["llama-3.1-8b"] == 100.0  # 200 tok / 2 s, unaffected


def test_collect_litellm_per_model_new_model_omitted():
    """A model with no previous baseline is omitted from per-model rates."""
    from app import collect_litellm_metrics

    only_m1 = "\n".join(
        line for line in LITELLM_METRICS_SAMPLE.splitlines()
        if 'model_id="m2"' not in line
    )
    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(only_m1.encode())), \
         patch("app.time.monotonic", clock):
        collect_litellm_metrics("http://127.0.0.1:9997")
        clock.advance(2.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(_two_model_body().encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    pm = second["gen_tokens_per_s_by_model"]
    assert "gpt-4o-mini" not in pm  # new model: no baseline yet
    assert pm["llama-3.1-8b"] == 0.0  # 12000 - 12000 = 0


def test_collect_litellm_rate_gap_within_widened_window():
    """A 12 s gap (within the 0.5–15 s window) computes the rate over
    the actual gap instead of omitting it."""
    from app import collect_litellm_metrics

    body = LITELLM_METRICS_SAMPLE.replace(
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12000.0',
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12300.0',
    )
    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())), \
         patch("app.time.monotonic", clock):
        collect_litellm_metrics("http://127.0.0.1:9997")
        clock.advance(12.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")

    assert second["gen_tokens_per_s"] == 25.0  # 300 tok / 12 s


def test_collect_litellm_rate_hold_after_long_gap():
    """A gap > 15 s (e.g. right after an outage) holds the last
    computed rate instead of omitting it — the UI never flashes '–'."""
    from app import collect_litellm_metrics

    body = LITELLM_METRICS_SAMPLE.replace(
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12000.0',
        'litellm_output_tokens_metric_total{api_provider="openai",model_id="m1",requested_model="llama-3.1-8b"} 12250.0',
    )
    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())), \
         patch("app.time.monotonic", clock):
        collect_litellm_metrics("http://127.0.0.1:9997")
        clock.advance(2.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(body.encode())):
            second = collect_litellm_metrics("http://127.0.0.1:9997")
        assert second["gen_tokens_per_s"] == 125.0  # 250 tok / 2 s
        # 20 s outage, then recovery with the same (unchanged) body.
        clock.advance(20.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=make_litellm_server(body.encode())):
            third = collect_litellm_metrics("http://127.0.0.1:9997")

    assert third["gen_tokens_per_s"] == 125.0  # held, not omitted
    assert third["gen_tokens_per_s_by_model"]["llama-3.1-8b"] == 125.0


def test_collect_litellm_bearer_token_sent():
    """A configured token is sent as an Authorization: Bearer header."""
    from app import collect_litellm_metrics

    calls: list = []
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode(),
                                               calls=calls)):
        result = collect_litellm_metrics("http://127.0.0.1:9997", token="sk-123")

    assert result is not None
    url, headers = calls[0]
    assert url == "http://127.0.0.1:9997/metrics"
    # urllib capitalizes header names (e.g. "Authorization")
    auth = [v for k, v in headers.items() if k.lower() == "authorization"]
    assert auth == ["Bearer sk-123"]

    # No token → no Authorization header
    calls.clear()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode(),
                                               calls=calls)):
        collect_litellm_metrics("http://127.0.0.1:9997")
    url, headers = calls[0]
    assert all(k.lower() != "authorization" for k in headers)


def test_collect_litellm_no_litellm_metrics_returns_none():
    from app import collect_litellm_metrics

    other_body = b'up 1.0\npython_info{implementation="CPython"} 1.0\n'
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(other_body)):
        result = collect_litellm_metrics("http://127.0.0.1:9997")

    assert result is None


def test_collect_litellm_unreachable():
    from app import collect_litellm_metrics

    with patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        assert collect_litellm_metrics("http://127.0.0.1:9997") is None


def test_collect_litellm_stale_hold_then_expiry():
    from app import collect_litellm_metrics

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_litellm_server(LITELLM_METRICS_SAMPLE.encode())), \
         patch("app.time.monotonic", clock):
        good = collect_litellm_metrics("http://127.0.0.1:9997")
        assert good["status"] == "up"
        clock.advance(1.0)
        with patch("app.urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("down")):
            stale = collect_litellm_metrics("http://127.0.0.1:9997")
        assert stale is not None and stale["status"] == "stale"
        clock.advance(20.0)  # beyond the 15 s hold
        with patch("app.urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("down")):
            assert collect_litellm_metrics("http://127.0.0.1:9997") is None


def test_collect_llm_auto_vllm_down_litellm_wins():
    """auto mode: vLLM down, LiteLLM up → LiteLLM returned."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.litellm_url = "http://127.0.0.1:9997"
    collector.llama_url = "http://127.0.0.1:9999"

    litellm_body = LITELLM_METRICS_SAMPLE.encode()

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        if url.endswith("/metrics") and "9997" in url:
            return _http_response(litellm_body)
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "litellm"
    assert data["model"] == "llama-3.1-8b"


def test_collect_llm_auto_vllm_beats_litellm():
    """auto mode: both vLLM and LiteLLM up → vLLM wins (direct engine)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.litellm_url = "http://127.0.0.1:9997"
    collector.llama_url = "http://127.0.0.1:9999"

    calls: list = []

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if url.endswith("/metrics"):
            if "9998" in url:
                return _http_response(VLLM_METRICS_SAMPLE.encode())
            if "9997" in url:
                return _http_response(LITELLM_METRICS_SAMPLE.encode())
        raise ConnectionRefusedError(f"no route for {url}")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "vllm"
    # LiteLLM was NOT scraped (vLLM wins before it)
    assert not any("9997" in c for c in calls)


def test_collect_llm_auto_litellm_down_falls_through_to_llama():
    """auto mode: vLLM + LiteLLM down, llama up → llama returned."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.litellm_url = "http://127.0.0.1:9997"
    collector.llama_url = "http://127.0.0.1:9999"

    llama_body = LLAMA_METRICS_SAMPLE.encode()

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        if "9999" in url:
            if url.endswith("/metrics"):
                return _http_response(llama_body)
            if url.endswith("/props"):
                return _http_response(
                    b'{"model_path": "/models/x.gguf", "total_slots": 1}')
            if url.endswith("/slots"):
                return _http_response(b"[]")
            if url.endswith("/health"):
                return _http_response(b'{"status":"ok"}')
            return _http_response(b"{}")
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "llama"


def test_collect_llm_pinned_litellm_skips_others():
    """--llm-backend litellm: only the proxy is scraped, even if others up."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="litellm")
    collector.vllm_url = "http://127.0.0.1:9998"
    collector.litellm_url = "http://127.0.0.1:9997"
    collector.llama_url = "http://127.0.0.1:9999"

    calls: list = []

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if url.endswith("/metrics") and "9997" in url:
            return _http_response(LITELLM_METRICS_SAMPLE.encode())
        raise ConnectionRefusedError("down")

    with patch("app.urllib.request.urlopen", side_effect=fake_urlopen):
        data = collector._collect_llm()

    assert data is not None
    assert data["backend"] == "litellm"
    assert all("9997" in c for c in calls)


def test_defaults_vllm_port_5001():
    """The default vLLM host port is 5001 (operator config)."""
    from app import DEFAULT_VLLM_URL

    assert DEFAULT_VLLM_URL == "http://127.0.0.1:5001"


# ---------------------------------------------------------------------------
# AMD GPU collector tests (amd-smi / rocm-smi / sysfs)
# ---------------------------------------------------------------------------

# Fixture reconstructed from the amd-smi CLI source + ROCm 7.14 docs
# (docs/research/amd-smi.md): monitor --json is an array keyed by "gpu",
# numerics wrapped as {"value": N, "unit": ...}, VRAM in MB, "N/A" for
# unavailable sensors (hotspot is N/A on many datacenter cards).
AMD_SMI_MONITOR_JSON = json.dumps([
    {
        "gpu": 0,
        "power_usage": {"value": 49, "unit": "W"},
        "hotspot_temperature": {"value": 27, "unit": "C"},
        "memory_temperature": {"value": 29, "unit": "C"},
        "gfx": {"value": 2, "unit": "%"},
        "gfx_clock": {"value": 43, "unit": "MHz"},
        "vram_used": {"value": 283, "unit": "MB"},
        "vram_total": {"value": 196592, "unit": "MB"},
    },
    {
        "gpu": 1,
        "power_usage": {"value": 42, "unit": "W"},
        "hotspot_temperature": "N/A",
        "memory_temperature": {"value": 30, "unit": "C"},
        "gfx": {"value": 0, "unit": "%"},
        "gfx_clock": {"value": 47, "unit": "MHz"},
        "vram_used": "N/A",
        "vram_total": {"value": 196592, "unit": "MB"},
    },
])

# Real rocm-smi 5.7.0 --csv header/row (llama-swap test fixture shape):
# VRAM columns are BYTES (unlike amd-smi's MB).
ROCM_SMI_CSV_SAMPLE = (
    "device,GPU ID,Temperature (Sensor edge) (C),"
    "Temperature (Sensor junction) (C),Temperature (Sensor memory) (C),"
    "Average Graphics Package Power (W),GPU use (%),GPU memory use (%),"
    "Memory Activity,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
    "card0,0x7551,38.0,38.0,39.0,1068.0,51.0,20.0,10.0,"
    "21458059264,27856896\n"
)

# Real ``amd-smi list --json`` shape: the tool's own ``bus_id`` per gpu.
AMD_SMI_LIST_JSON = json.dumps([
    {"gpu": 0, "bus_id": "0000:2d:00.0"},
    {"gpu": 1, "bus_id": "0000:61:00.0"},
])


def _fake_drm(drm_dir, cards):
    """Build a fake ``/sys/class/drm`` layout: ``{cardN: (vendor_hex, bdf)}``.
    Mirrors sysfs: each ``cardN/device`` is a symlink to a real directory
    named after the BDF, with a ``vendor`` file behind it."""
    for name, (vendor, bdf) in cards.items():
        card = drm_dir / name
        card.mkdir(parents=True)
        (card / bdf).mkdir()
        (card / bdf / "vendor").write_text(vendor + "\n")
        os.symlink(bdf, card / "device")


@pytest.fixture(autouse=True)
def reset_amd_state():
    """Reset AMD BDF/product-name caches around every test."""
    import app as app_mod

    original_absent = app_mod._amd_absent_since
    original_bdf_map = app_mod._amd_bdf_map
    original_bdf_neg = app_mod._amd_bdf_negative
    original_rocm_cards = app_mod._rocm_cards_cache
    original_product_names = app_mod._amd_product_names
    original_product_ts = app_mod._amd_product_names_ts
    app_mod._amd_absent_since = None
    app_mod._amd_bdf_map = None
    app_mod._amd_bdf_negative = {}
    app_mod._rocm_cards_cache = None
    app_mod._amd_product_names = {}
    app_mod._amd_product_names_ts = None
    yield
    app_mod._amd_absent_since = original_absent
    app_mod._amd_bdf_map = original_bdf_map
    app_mod._amd_bdf_negative = original_bdf_neg
    app_mod._rocm_cards_cache = original_rocm_cards
    app_mod._amd_product_names = original_product_names
    app_mod._amd_product_names_ts = original_product_ts


def _amd_smi_mock(stdout: str, returncode: int = 0):
    return MagicMock(returncode=returncode, stdout=stdout)


def test_amd_unwrap():
    from app import _amd_unwrap

    assert _amd_unwrap({"value": 49, "unit": "W"}) == 49.0
    assert _amd_unwrap(7) == 7.0
    assert _amd_unwrap("N/A") is None
    assert _amd_unwrap(None) is None
    assert _amd_unwrap({"value": "N/A"}) is None
    assert _amd_unwrap(True) is None  # bool is not a metric


def test_bdf_sysfs_form_handles_8_digit_domain():
    from app import _bdf_sysfs_form

    # 4-digit domain → unchanged (lowercased).
    assert _bdf_sysfs_form("0000:2d:00.0") == "0000:2d:00.0"
    # 8-digit domain → truncated to 4 digits (the sysfs form).
    assert _bdf_sysfs_form("00000000:2d:00.0") == "0000:2d:00.0"
    # Uppercase → lowercased.
    assert _bdf_sysfs_form("0000:2D:00.0") == "0000:2d:00.0"
    # Malformed / empty / None → None.
    assert _bdf_sysfs_form("not-a-bdf") is None
    assert _bdf_sysfs_form("") is None
    assert _bdf_sysfs_form(None) is None


def test_gpu_amd_smi_happy_path():
    from app import _gpu_amd_smi

    with (
        patch("shutil.which", return_value="/usr/bin/amd-smi"),
        patch("subprocess.run", return_value=_amd_smi_mock(AMD_SMI_MONITOR_JSON)),
        patch("app._sysfs_pci_bus_id", return_value=None),
    ):
        gpus = _gpu_amd_smi()

    assert len(gpus) == 2
    g0, g1 = gpus
    assert g0["index"] == 0
    assert g0["utilization_pct"] == 2.0
    assert g0["memory_used_gb"] == round(283 / 1024, 1)
    assert g0["memory_total_gb"] == round(196592 / 1024, 1)
    assert g0["memory_pct"] == round(283 / 196592 * 100, 1)
    assert g0["temperature_c"] == 27.0  # hotspot preferred
    assert g0["graphics_clock_mhz"] == 43.0
    assert g0["power_w"] == 49.0
    assert g0["name"] == "AMD GPU"
    assert g0["board_vendor"] == "unknown"
    assert g0["throttle"] == {"hw_thermal_slowdown": False,
                              "sw_thermal_slowdown": False}

    # GPU 1: hotspot N/A → memory_temperature fallback; vram_used N/A →
    # memory keys omitted (not 0), row still kept (util present)
    assert g1["temperature_c"] == 30.0
    assert "memory_used_gb" not in g1
    assert "memory_pct" not in g1
    assert g1["utilization_pct"] == 0.0


def test_gpu_amd_smi_binary_missing():
    from app import _gpu_amd_smi

    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        gpus = _gpu_amd_smi()

    assert gpus == []
    mock_run.assert_not_called()


def test_gpu_amd_smi_failure_modes():
    from app import _gpu_amd_smi

    with patch("shutil.which", return_value="/usr/bin/amd-smi"):
        # Non-zero exit (no AMD GPUs / driver not loaded)
        with patch("subprocess.run", return_value=_amd_smi_mock("", 1)):
            assert _gpu_amd_smi() == []
        # Unparseable output
        with patch("subprocess.run", return_value=_amd_smi_mock("not json")):
            assert _gpu_amd_smi() == []
        # Empty list
        with patch("subprocess.run", return_value=_amd_smi_mock("[]")):
            assert _gpu_amd_smi() == []
        # All-N/A row is dropped
        with patch("subprocess.run", return_value=_amd_smi_mock(
                json.dumps([{"gpu": 0, "gfx": "N/A",
                             "vram_used": "N/A"}]))):
            assert _gpu_amd_smi() == []


def test_gpu_rocm_smi_parses_csv_by_header_name():
    from app import _gpu_rocm_smi

    with (
        patch("shutil.which", return_value=None),
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
        patch("subprocess.run",
              return_value=_amd_smi_mock(ROCM_SMI_CSV_SAMPLE)),
        patch("app._sysfs_pci_bus_id", return_value=None),
    ):
        gpus = _gpu_rocm_smi()

    assert len(gpus) == 1
    g = gpus[0]
    assert g["utilization_pct"] == 51.0
    # BYTES → GiB (unlike amd-smi's MB)
    assert g["memory_used_gb"] == round(27856896 / 1024**3, 1)
    assert g["memory_total_gb"] == round(21458059264 / 1024**3, 1)  # 20.0
    assert g["memory_pct"] == round(27856896 / 21458059264 * 100, 1)
    assert g["temperature_c"] == 38.0  # first "Temperature (Sensor..." col
    assert g["power_w"] == 1068.0


def test_gpu_rocm_smi_binary_missing():
    from app import _gpu_rocm_smi

    with (
        patch("shutil.which", return_value=None),
        patch("os.path.isfile", return_value=False),
        patch("subprocess.run") as mock_run,
    ):
        assert _gpu_rocm_smi() == []
    mock_run.assert_not_called()


def test_gpu_amd_sysfs_last_resort(tmp_path):
    from app import _gpu_amd_sysfs

    dev = tmp_path / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "gpu_busy_percent").write_text("67\n")
    (dev / "mem_info_vram_used").write_text(str(8 * 1024**3) + "\n")
    (dev / "mem_info_vram_total").write_text(str(16 * 1024**3) + "\n")
    (dev / "temperature1").write_text("55250\n")  # milli-°C

    # A non-amdgpu card (no VRAM meminfo) must be excluded.
    other = tmp_path / "card1" / "device"
    other.mkdir(parents=True)
    (other / "gpu_busy_percent").write_text("3\n")

    with patch("app._sysfs_pci_bus_id", return_value=None):
        gpus = _gpu_amd_sysfs(drm_dir=str(tmp_path))

    assert len(gpus) == 1
    g = gpus[0]
    assert g["index"] == 0
    assert g["utilization_pct"] == 67
    assert g["memory_used_gb"] == 8.0
    assert g["memory_total_gb"] == 16.0
    assert g["memory_pct"] == 50.0
    assert g["temperature_c"] == round(55250 / 1000, 1)  # 55.2


def test_gpu_amd_sysfs_missing_dir():
    from app import _gpu_amd_sysfs

    assert _gpu_amd_sysfs(drm_dir="/nonexistent-drm-dir") == []


def test_gpu_amd_chain_falls_through_to_sysfs():
    """amd-smi missing → rocm-smi missing → sysfs result wins."""
    from app import _gpu_amd

    with (
        patch("shutil.which", return_value=None),
        patch("os.path.isfile", return_value=False),
        patch("subprocess.run") as mock_run,
        patch("app._gpu_amd_sysfs") as mock_sysfs,
    ):
        mock_sysfs.return_value = [{"index": 0, "name": "AMD GPU"}]
        gpus = _gpu_amd()

    assert gpus == [{"index": 0, "name": "AMD GPU"}]
    mock_run.assert_not_called()
    mock_sysfs.assert_called_once()


def test_collect_gpu_falls_back_to_amd():
    from app import collect_gpu_metrics

    with (
        patch("app._init_platform", return_value="desktop"),
        patch("app._gpu_pynvml", return_value=[]),
        patch("app._gpu_cli", return_value=[]),
        patch("app._gpu_amd") as mock_amd,
    ):
        mock_amd.return_value = [{"index": 0, "name": "AMD GPU"}]
        result = collect_gpu_metrics()

    assert result == [{"index": 0, "name": "AMD GPU", "vendor": "amd"}]
    mock_amd.assert_called_once()


def test_collect_gpu_mixed_vendors_merges_and_renumbers():
    """NVIDIA + AMD on one box: both are reported, indices 0..n-1."""
    from app import collect_gpu_metrics

    nvidia = [
        {"index": 0, "name": "RTX 4090", "utilization_pct": 10.0},
        {"index": 1, "name": "RTX 4090", "utilization_pct": 20.0},
    ]
    amd = [
        {"index": 0, "name": "RX 7900 XTX"},
        {"index": 1, "name": "RX 7900 XTX"},
    ]
    with (
        patch("app._init_platform", return_value="desktop"),
        patch("app._gpu_pynvml", return_value=nvidia),
        patch("app._gpu_cli") as mock_cli,
        patch("app._gpu_amd", return_value=amd),
    ):
        result = collect_gpu_metrics()

    mock_cli.assert_not_called()
    assert [g["index"] for g in result] == [0, 1, 2, 3]
    assert [g["vendor"] for g in result] == [
        "nvidia", "nvidia", "amd", "amd"]
    assert [g["name"] for g in result] == [
        "RTX 4090", "RTX 4090", "RX 7900 XTX", "RX 7900 XTX"]
    # Merged entries are copies — source dicts untouched
    assert "vendor" not in nvidia[0]
    assert nvidia[0]["index"] == 0
    assert amd[0]["index"] == 0


def test_merge_gpu_lists_offset_uses_max_index():
    """If a pynvml index is skipped the AMD offset is max+1, not count."""
    from app import _merge_gpu_lists

    nvidia = [{"index": 0}, {"index": 2}]  # device 1 failed to init
    amd = [{"index": 0}]
    merged = _merge_gpu_lists(nvidia, amd)
    assert [g["index"] for g in merged] == [0, 1, 2]


def test_merge_gpu_lists_empty_sides():
    from app import _merge_gpu_lists

    assert _merge_gpu_lists([], []) == []
    assert _merge_gpu_lists([{"index": 5}], []) == [
        {"index": 0, "vendor": "nvidia"}]
    assert _merge_gpu_lists([], [{"index": 3}]) == [
        {"index": 0, "vendor": "amd"}]


def test_amd_bdf_resolved_from_amd_smi_bus_id():
    """(a): ``amd-smi list --json`` bus_id is used; the vendor-local gpu
    index is never passed as a global cardN (no sysfs fallback runs)."""
    from app import (_AMD_SMI_BIN_CANDIDATES, _amd_gpu_bus_id)

    def fake_find_cli(candidates):
        if candidates == _AMD_SMI_BIN_CANDIDATES:
            return "/usr/bin/amd-smi"
        return None

    with (
        patch("app._find_cli_binary", side_effect=fake_find_cli),
        patch("subprocess.run", return_value=_amd_smi_mock(AMD_SMI_LIST_JSON)),
        patch("app._sysfs_pci_bus_id") as mock_sysfs,
    ):
        assert _amd_gpu_bus_id(0) == "0000:2d:00.0"
        assert _amd_gpu_bus_id(1) == "0000:61:00.0"
        # (a) satisfied → no sysfs fallback ran, so the vendor-local
        # index was never passed as a global cardN.
        mock_sysfs.assert_not_called()


def test_amd_bdf_fallback_rocm_cardn():
    """(b): amd-smi unavailable → rocm-smi's own cardN string (a global
    DRM number, e.g. "card2") is used, not a vendor-local index."""
    from app import (_AMD_SMI_BIN_CANDIDATES, _ROCM_SMI_BIN_CANDIDATES,
                     _amd_gpu_bus_id)

    rocm_csv = (
        "device,GPU ID,GPU use (%),VRAM Total Memory (B),"
        "VRAM Total Used Memory (B)\n"
        "card2,0x7551,10.0,21458059264,27856896\n"
        "card3,0x7551,20.0,21458059264,55713792\n"
    )

    def fake_find_cli(candidates):
        if candidates == _AMD_SMI_BIN_CANDIDATES:
            return None  # amd-smi unavailable
        if candidates == _ROCM_SMI_BIN_CANDIDATES:
            return "/usr/bin/rocm-smi"
        return None

    def fake_sysfs(card, drm_dir="/sys/class/drm"):
        return {"card2": "0000:2d:00.0",
                "card3": "0000:61:00.0"}.get(card)

    with (
        patch("app._find_cli_binary", side_effect=fake_find_cli),
        patch("subprocess.run", return_value=_amd_smi_mock(rocm_csv)),
        patch("app._sysfs_pci_bus_id", side_effect=fake_sysfs) as mock_sysfs,
        patch("app._sysfs_amd_card_bdfs", return_value=[]),
    ):
        assert _amd_gpu_bus_id(0) == "0000:2d:00.0"  # rocm's "card2"
        assert _amd_gpu_bus_id(1) == "0000:61:00.0"  # rocm's "card3"
    # rocm's own cardN strings — never the vendor-local ints 0/1:
    called_with = [c.args[0] for c in mock_sysfs.call_args_list]
    assert called_with == ["card2", "card3"]


def test_amd_bdf_fallback_sysfs_vendor_filter(tmp_path):
    """(c): both tools unavailable → ``/sys/class/drm/card*`` filtered
    by ``device/vendor == 0x1002``; an NVIDIA card0 is not picked for
    an AMD gpu."""
    from app import _amd_gpu_bus_id

    _fake_drm(tmp_path, {
        "card0": ("0x10de", "0000:44:00.0"),   # NVIDIA — must be skipped
        "card1": ("0x1002", "0000:2d:00.0"),   # AMD gpu 0
        "card2": ("0x1002", "0000:61:00.0"),   # AMD gpu 1
    })

    with (
        patch("app._find_cli_binary", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        assert _amd_gpu_bus_id(0, drm_dir=str(tmp_path)) == "0000:2d:00.0"
        assert _amd_gpu_bus_id(1, drm_dir=str(tmp_path)) == "0000:61:00.0"
        mock_run.assert_not_called()  # no tool was found → no subprocess


def test_amd_bdf_fake_drm_nvidia_card0(tmp_path):
    """Spec-mandated: fake drm layout where card0 is NVIDIA and AMD gpu
    indices are 0/1 — the AMD cards resolve to their own (AMD) BDFs,
    not card0's. The old bug passed the vendor-local index as a global
    cardN, so AMD gpu 0 resolved to the NVIDIA card."""
    from app import _amd_gpu_bus_id, _sysfs_pci_bus_id

    _fake_drm(tmp_path, {
        "card0": ("0x10de", "0000:44:00.0"),   # NVIDIA
        "card1": ("0x1002", "0000:2d:00.0"),   # AMD gpu 0
        "card2": ("0x1002", "0000:61:00.0"),   # AMD gpu 1
    })

    with patch("app._find_cli_binary", return_value=None):
        # What the old bug resolved: vendor-local index 0 → global card0
        # (the NVIDIA card's BDF).
        assert _sysfs_pci_bus_id(0, str(tmp_path)) == "0000:44:00.0"
        # The fix: AMD gpu 0/1 → their own AMD BDFs.
        assert _amd_gpu_bus_id(0, drm_dir=str(tmp_path)) == "0000:2d:00.0"
        assert _amd_gpu_bus_id(1, drm_dir=str(tmp_path)) == "0000:61:00.0"


def test_colliding_bdf_merge_guard(caplog):
    """Spec-mandated: two cards claiming the same (normalized) BDF —
    the guard flags (logs) and the payload contains no two entries with
    the same normalized BDF. The domain width differs across vendors
    (``00000000:`` vs ``0000:``) — normalization must catch it."""
    from app import _merge_gpu_lists

    nvidia = [{"index": 0, "name": "RTX 3090",
               "pci_bus_id": "00000000:61:00.0"}]
    amd = [{"index": 0, "name": "R9700",
            "pci_bus_id": "0000:61:00.0"}]

    with caplog.at_level(logging.WARNING):
        merged = _merge_gpu_lists(nvidia, amd)

    # Duplicate dropped (the first — NVIDIA — is kept), flagged with
    # both card names.
    assert len(merged) == 1
    assert merged[0]["name"] == "RTX 3090"
    assert merged[0]["vendor"] == "nvidia"
    assert any("BDF collision" in m and "RTX 3090" in m and "R9700" in m
               for m in caplog.messages)

    # Distinct BDFs (even with mixed domain width) are both kept and
    # renumbered dense.
    ok = _merge_gpu_lists(
        [{"index": 0, "name": "RTX 3090",
          "pci_bus_id": "00000000:44:00.0"}],
        [{"index": 0, "name": "R9700", "pci_bus_id": "0000:2d:00.0"}])
    assert [g["index"] for g in ok] == [0, 1]
    assert [g["vendor"] for g in ok] == ["nvidia", "amd"]


def test_sysfs_pci_vendor_id_reads_vendor_file(tmp_path):
    """The raw PCI vendor ID is read from the sysfs ``vendor`` file
    (pci.ids-independent). A fake device layout with a ``vendor`` file
    behind the BDF is resolved to the raw ID string."""
    from app import _sysfs_pci_vendor_id

    # Build a fake /sys/bus/pci/devices layout: {bdf: vendor_hex}.
    pci_dir = tmp_path / "pci"
    (pci_dir / "0000:2d:00.0").mkdir(parents=True)
    (pci_dir / "0000:2d:00.0" / "vendor").write_text("0x1002\n")
    (pci_dir / "0000:44:00.0").mkdir(parents=True)
    (pci_dir / "0000:44:00.0" / "vendor").write_text("0x10de\n")

    # The raw vendor ID is read from the vendor file.
    assert _sysfs_pci_vendor_id("0000:2d:00.0", str(pci_dir)) == "0x1002"
    assert _sysfs_pci_vendor_id("0000:44:00.0", str(pci_dir)) == "0x10de"
    # The 8-digit-domain form is normalized to the sysfs form.
    assert _sysfs_pci_vendor_id("00000000:2d:00.0", str(pci_dir)) == "0x1002"
    # A missing device node → None.
    assert _sysfs_pci_vendor_id("0000:ff:ff.0", str(pci_dir)) is None


def test_sysfs_pci_vendor_name_unknown_id_falls_back(tmp_path):
    """An unknown vendor ID falls back to the raw ID string (deterministic,
    pci.ids-independent). A fake device layout with an unknown vendor ID
    resolves to the raw ID string, not a name."""
    from app import _sysfs_pci_vendor_name

    pci_dir = tmp_path / "pci"
    (pci_dir / "0000:2d:00.0").mkdir(parents=True)
    (pci_dir / "0000:2d:00.0" / "vendor").write_text("0x9999\n")

    # Unknown ID → the raw ID string.
    assert _sysfs_pci_vendor_name("0000:2d:00.0", str(pci_dir)) == "0x9999"


def test_sysfs_pci_vendor_name_known_id_maps(tmp_path):
    """A known vendor ID is mapped to its name via the hardcoded map.
    A fake device layout with a known vendor ID resolves to the name."""
    from app import _sysfs_pci_vendor_name

    pci_dir = tmp_path / "pci"
    (pci_dir / "0000:2d:00.0").mkdir(parents=True)
    (pci_dir / "0000:2d:00.0" / "vendor").write_text("0x1002\n")
    (pci_dir / "0000:44:00.0").mkdir(parents=True)
    (pci_dir / "0000:44:00.0" / "vendor").write_text("0x10de\n")

    # Known IDs → names.
    assert _sysfs_pci_vendor_name("0000:2d:00.0", str(pci_dir)) == "AMD"
    assert _sysfs_pci_vendor_name("0000:44:00.0", str(pci_dir)) == "NVIDIA"


def test_gpu_name_prefers_tool_product_name():
    """The tool's own product/asic name (``amd-smi static --asic``) is
    preferred over lspci for the AMD path, so the label is not hostage
    to pci.ids freshness."""
    from app import _gpu_amd_smi

    monitor = json.dumps([
        {"gpu": 0, "gfx": {"value": 5, "unit": "%"},
         "vram_used": {"value": 100, "unit": "MB"},
         "vram_total": {"value": 32000, "unit": "MB"}},
    ])
    # The tool's own product/asic name (from amd-smi static --asic).
    static_asic = json.dumps([
        {"gpu": 0, "asic": "Radeon AI PRO R9700"},
    ])
    # lspci would return a different (plausible) name.
    lspci_name = (
        "Device:\tNavi 48 [Radeon Something Else]\n"
        "Class:\tVGA compatible controller\n"
        "Vendor:\tAdvanced Micro Devices, Inc. [AMD/ATI]\n"
    )

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "lspci":
            return _amd_smi_mock(lspci_name)
        if "static" in cmd:
            return _amd_smi_mock(static_asic)
        return _amd_smi_mock(monitor)

    with (
        patch("shutil.which", return_value="/usr/bin/amd-smi"),
        patch("subprocess.run", side_effect=fake_run),
        patch("app._amd_gpu_bus_id", return_value="0000:2d:00.0"),
    ):
        gpus = _gpu_amd_smi()

    assert len(gpus) == 1
    # The tool's own product name is preferred over the lspci name.
    assert gpus[0]["name"] == "Radeon AI PRO R9700"


def test_gpu_name_never_matches_bdf_shape():
    """Spec-mandated: run the real ``_gpu_amd_smi`` path on a fake host
    layout where lspci would return a BDF-shaped name (the poisoned
    first-``Device:``-line case) and assert no emitted GPU name matches a
    BDF/slot shape — the fix rejects it and falls back."""
    from app import _gpu_amd_smi, _is_implausible_gpu_name

    # lspci -vm with only a BDF-shaped Device: line (the bug: the
    # only/first Device: line is the slot, not the product name).
    lspci_bdf_only = (
        "Device:\t2d:00.0\n"
        "Class:\tVGA compatible controller\n"
        "Vendor:\tAdvanced Micro Devices, Inc. [AMD/ATI]\n"
    )
    monitor = json.dumps([
        {"gpu": 0, "gfx": {"value": 5, "unit": "%"},
         "vram_used": {"value": 100, "unit": "MB"},
         "vram_total": {"value": 32000, "unit": "MB"}},
    ])

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "lspci":
            return _amd_smi_mock(lspci_bdf_only)
        return _amd_smi_mock(monitor)  # amd-smi monitor/list/static

    with (
        patch("shutil.which", return_value="/usr/bin/amd-smi"),
        patch("subprocess.run", side_effect=fake_run),
        patch("app._amd_gpu_bus_id", return_value="0000:2d:00.0"),
    ):
        gpus = _gpu_amd_smi()

    assert len(gpus) == 1
    name = gpus[0]["name"]
    # The emitted name must never be a BDF/slot shape.
    assert not _is_implausible_gpu_name(name)
    # lspci name rejected and no tool name → the fallback.
    assert name == "AMD GPU"


def test_gpu_amd_negative_cache_skips_reprobe():
    """No AMD GPU → amd-smi is not re-run every poll; re-probe after TTL."""
    import app as app_mod

    calls = {"n": 0}

    def fake_amd_smi():
        calls["n"] += 1
        return []

    with (
        patch.object(app_mod, "_gpu_amd_smi", side_effect=fake_amd_smi),
        patch.object(app_mod, "_gpu_rocm_smi", return_value=[]) as mock_rocm,
        patch.object(app_mod, "_gpu_amd_sysfs", return_value=[]) as mock_sysfs,
    ):
        assert app_mod._gpu_amd() == []
        assert calls["n"] == 1
        rocm_calls = mock_rocm.call_count
        sysfs_calls = mock_sysfs.call_count
        # Second poll within the window: negative-cached, nothing runs
        assert app_mod._gpu_amd() == []
        assert calls["n"] == 1
        assert mock_rocm.call_count == rocm_calls
        assert mock_sysfs.call_count == sysfs_calls
        # TTL elapses → re-probe (driver may have been loaded)
        app_mod._amd_absent_since -= app_mod._AMD_NEGATIVE_REPROBE_S + 1
        assert app_mod._gpu_amd() == []
        assert calls["n"] == 2


def test_gpu_amd_negative_cache_cleared_on_success():
    import app as app_mod

    results = iter([[], [{"index": 0}], []])
    with (
        patch.object(app_mod, "_gpu_amd_smi", side_effect=lambda: next(results)),
        patch.object(app_mod, "_gpu_rocm_smi", return_value=[]),
        patch.object(app_mod, "_gpu_amd_sysfs", return_value=[]),
    ):
        assert app_mod._gpu_amd() == []
        assert app_mod._amd_absent_since is not None
        # GPU appears → negative cache cleared…
        app_mod._amd_absent_since -= 1000.0  # force re-probe due
        assert app_mod._gpu_amd() == [{"index": 0}]
        assert app_mod._amd_absent_since is None
        # …so the next poll probes again (no cache on the success path)
        assert app_mod._gpu_amd() == []


# ---------------------------------------------------------------------------
# config.yaml + CLI resolution
# ---------------------------------------------------------------------------

def test_load_config_valid(tmp_path):
    from app import load_config

    p = tmp_path / "config.yaml"
    p.write_text(
        "port: 8888\n"
        "poll_interval: 2.5\n"
        "llm_backend: litellm\n"
        "llama_enabled: false\n"
        "litellm_token: sk-test\n"
        "upstreams:\n"
        "  - name: athena\n"
        "    url: http://athena:5001\n"
        "unknown_key: 1\n"
    )
    cfg = load_config(str(p))
    assert cfg["port"] == 8888
    assert cfg["poll_interval"] == 2.5
    assert cfg["llm_backend"] == "litellm"
    assert cfg["llama_enabled"] is False
    assert cfg["litellm_token"] == "sk-test"
    assert cfg["upstreams"][0]["name"] == "athena"


def test_load_config_missing(tmp_path, monkeypatch):
    from app import load_config

    monkeypatch.chdir(tmp_path)  # no config.yaml here
    assert load_config(None) == {}
    # not explicit: a missing path is a silent no-config case
    assert load_config(str(tmp_path / "nope.yaml")) == {}


def test_load_config_malformed_exits(tmp_path):
    from app import load_config

    p = tmp_path / "bad.yaml"
    p.write_text("port: [unclosed\n  bad: : :\n")
    with pytest.raises(SystemExit):
        load_config(str(p))


def test_load_config_not_a_mapping_exits(tmp_path):
    from app import load_config

    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(SystemExit):
        load_config(str(p))


def test_config_explicit_missing_exits(tmp_path):
    from app import load_config

    with pytest.raises(SystemExit):
        load_config(str(tmp_path / "nope.yaml"), explicit=True)


def test_load_config_invalid_value_exits(tmp_path):
    from app import load_config

    for bad in ("port: notanint\n", "poll_interval: -1\n",
               "llm_backend: quantum\n", "llama_enabled: maybe\n"):
        p = tmp_path / "badval.yaml"
        p.write_text(bad)
        with pytest.raises(SystemExit):
            load_config(str(p))


def _ns(**kw):
    base = dict(host=None, port=None, poll_interval=None, llama_url=None,
                vllm_url=None, litellm_url=None, litellm_token=None,
                strata_url=None, llm_backend=None, default_view=None,
                no_llama=False, upstream=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_config_precedence_cli_wins():
    from app import resolve_settings

    # CLI beats config
    assert resolve_settings(_ns(port=9999), {"port": 8888})["port"] == 9999
    # config fills what CLI left unset
    assert resolve_settings(_ns(), {"port": 8888})["port"] == 8888
    # built-in default when neither provides a value
    assert resolve_settings(_ns(), {})["port"] == 5000
    assert resolve_settings(_ns(), {})["poll_interval"] == 1.0


def test_no_llama_flag_overrides_config():
    from app import resolve_settings

    # flag wins when present (even against config true)
    assert resolve_settings(_ns(no_llama=True),
                           {"llama_enabled": True})["llama_enabled"] is False
    # config false sticks when the flag is absent
    assert resolve_settings(_ns(),
                           {"llama_enabled": False})["llama_enabled"] is False
    # config true / default
    assert resolve_settings(_ns(),
                           {"llama_enabled": True})["llama_enabled"] is True
    assert resolve_settings(_ns(), {})["llama_enabled"] is True


def test_litellm_token_env_var_overrides_config(monkeypatch):
    from app import resolve_settings

    monkeypatch.setenv("RD_LITELLM_TOKEN", "env-token")
    # env wins over config
    assert resolve_settings(_ns(),
                           {"litellm_token": "cfg-token"})["litellm_token"] == "env-token"
    # env wins over the CLI flag too
    assert resolve_settings(_ns(litellm_token="cli-token"),
                           {"litellm_token": "cfg-token"})["litellm_token"] == "env-token"
    # and it reaches the legacy single-backend target
    s = resolve_settings(_ns(llm_backend="litellm"), {})
    assert s["llm_targets"][0].token == "env-token"


def test_litellm_token_env_var_unset_falls_back_to_config(monkeypatch):
    from app import resolve_settings

    monkeypatch.delenv("RD_LITELLM_TOKEN", raising=False)
    # config value used when the env var is absent
    assert resolve_settings(_ns(),
                           {"litellm_token": "cfg-token"})["litellm_token"] == "cfg-token"
    # CLI still beats config when the env var is absent
    assert resolve_settings(_ns(litellm_token="cli-token"),
                           {"litellm_token": "cfg-token"})["litellm_token"] == "cli-token"
    # and the legacy single-backend target falls back the same way
    s = resolve_settings(_ns(llm_backend="litellm"),
                        {"litellm_token": "cfg-token"})
    assert s["llm_targets"][0].token == "cfg-token"


def test_litellm_token_env_var_empty_falls_back_to_config(monkeypatch):
    from app import resolve_settings

    monkeypatch.setenv("RD_LITELLM_TOKEN", "")
    # an empty env var is treated as unset
    assert resolve_settings(_ns(),
                           {"litellm_token": "cfg-token"})["litellm_token"] == "cfg-token"
    s = resolve_settings(_ns(llm_backend="litellm"),
                        {"litellm_token": "cfg-token"})
    assert s["llm_targets"][0].token == "cfg-token"


def test_upstream_cli_merge_dedupe_cli_wins():
    from app import merge_upstreams

    merged = merge_upstreams(
        [{"name": "a", "url": "http://cfg-a"},
         {"name": "b", "url": "http://cfg-b"}],
        ["a=http://cli-a", "c=http://cli-c"])
    assert {u["name"]: u["url"] for u in merged} == {
        "a": "http://cli-a", "b": "http://cfg-b", "c": "http://cli-c"}
    # order: config order first, then CLI-only names in CLI order
    assert [u["name"] for u in merged] == ["a", "b", "c"]


def test_upstream_entry_malformed_skipped():
    from app import merge_upstreams

    merged = merge_upstreams(
        [{"name": "good", "url": "http://g"},
         {"name": "", "url": "http://x"},
         {"name": "nourl"},
         "notadict"],
        ["badspec", "=nourl", "ok=http://ok"])
    assert [u["name"] for u in merged] == ["good", "ok"]


# ---------------------------------------------------------------------------
# collect_upstream_metrics
# ---------------------------------------------------------------------------

def test_collect_upstream_happy_path():
    import app as app_mod
    from app import collect_upstream_metrics

    orig = app_mod._upstream_last_good
    app_mod._upstream_last_good = {}
    try:
        payload = {
            "timestamp": 1.0, "cpu": 12.5, "cpu_per_core": [1.0],
            "memory": {"percent": 50.0}, "gpus": [], "swap": None,
            "host_health": {}, "llm_enabled": True, "llm": None,
            "systems": [{"name": "nested"}], "app_level_key": "x",
        }
        with patch("app.urllib.request.urlopen",
                   side_effect=make_llama_server(
                       {"/api/metrics": json.dumps(payload).encode()})):
            # trailing slash on the configured URL is normalized away
            out = collect_upstream_metrics("http://127.0.0.1:9996/", "mock")
        assert out is not None
        assert out["status"] == "up"
        assert out["name"] == "mock"
        m = out["metrics"]
        assert m["cpu"] == 12.5
        assert m["memory"]["percent"] == 50.0
        assert "systems" not in m        # no recursion
        assert "app_level_key" not in m  # strict sanitizer
    finally:
        app_mod._upstream_last_good = orig


def test_collect_upstream_unreachable_returns_none():
    import app as app_mod
    from app import collect_upstream_metrics

    orig = app_mod._upstream_last_good
    app_mod._upstream_last_good = {}
    try:
        with patch("app.urllib.request.urlopen",
                   side_effect=ConnectionRefusedError("down")):
            assert collect_upstream_metrics(
                "http://127.0.0.1:9996", "mock") is None
    finally:
        app_mod._upstream_last_good = orig


def test_collect_upstream_stale_hold_then_expiry():
    import app as app_mod
    from app import collect_upstream_metrics

    clock = _FakeClock()
    payload = {"cpu": 5.0, "gpus": []}
    orig = app_mod._upstream_last_good
    app_mod._upstream_last_good = {}
    try:
        with patch("app.urllib.request.urlopen",
                   side_effect=make_llama_server(
                       {"/api/metrics": json.dumps(payload).encode()})), \
             patch("app.time.monotonic", clock):
            good = collect_upstream_metrics("http://127.0.0.1:9996", "mock")
            assert good is not None and good["status"] == "up"
            clock.advance(1.0)
            with patch("app.urllib.request.urlopen",
                       side_effect=ConnectionRefusedError("down")):
                stale = collect_upstream_metrics(
                    "http://127.0.0.1:9996", "mock")
            assert stale is not None and stale["status"] == "stale"
            assert stale["metrics"]["cpu"] == 5.0
            clock.advance(20.0)  # beyond the 15 s hold
            with patch("app.urllib.request.urlopen",
                       side_effect=ConnectionRefusedError("down")):
                assert collect_upstream_metrics(
                    "http://127.0.0.1:9996", "mock") is None
    finally:
        app_mod._upstream_last_good = orig


# ---------------------------------------------------------------------------
# MetricsCollector: upstream threads + snapshot systems array
# ---------------------------------------------------------------------------

def _wait_for_snapshot_with(collector, pred, timeout=5.0):
    """Poll get_latest until a snapshot satisfies *pred* (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = collector.get_latest(timeout=0.5)
        if data is not None:
            m = json.loads(data)
            if pred(m):
                return m
    return None


def test_snapshot_systems_array():
    from app import MetricsCollector

    payload = {"cpu": 42.0, "cpu_per_core": [1.0, 2.0],
              "memory": {"percent": 33.0},
              "gpus": [{"index": 0, "utilization_pct": 7}],
              "swap": None, "host_health": {}, "llm_enabled": True,
              "llm": None, "systems": [{"name": "should-not-nest"}],
              "app_level": 1}
    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start_upstreams(
        [{"name": "mock", "url": "http://127.0.0.1:9996"}])
    collector.start()
    try:
        with patch("app.urllib.request.urlopen",
                   side_effect=make_llama_server(
                       {"/api/metrics": json.dumps(payload).encode()})):
            metrics = _wait_for_snapshot_with(
                collector,
                lambda m: any(s["name"] == "mock" and s["status"] == "up"
                             for s in m["systems"]))
            assert metrics is not None
            by_name = {s["name"]: s for s in metrics["systems"]}
            # local entry first, status up, metrics match the top-level keys
            assert metrics["systems"][0]["name"] == "local"
            assert by_name["local"]["status"] == "up"
            assert by_name["local"]["metrics"]["cpu"] == metrics["cpu"]
            assert by_name["local"]["metrics"]["memory"] == metrics["memory"]
            # upstream entry sanitized
            assert by_name["mock"]["status"] == "up"
            assert by_name["mock"]["metrics"]["cpu"] == 42.0
            assert by_name["mock"]["metrics"]["gpus"][0]["utilization_pct"] == 7
            assert "systems" not in by_name["mock"]["metrics"]
            assert "app_level" not in by_name["mock"]["metrics"]
            # `systems` never nests
            assert "systems" not in by_name["local"]["metrics"]
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_collector_upstream_thread_populates_latest():
    from app import MetricsCollector

    payload = {"cpu": 7.5, "gpus": []}
    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start_upstreams(
        [{"name": "mock", "url": "http://127.0.0.1:9996"}])
    collector.start()
    try:
        with patch("app.urllib.request.urlopen",
                   side_effect=make_llama_server(
                       {"/api/metrics": json.dumps(payload).encode()})):
            deadline = time.time() + 5
            while time.time() < deadline and not (
                    collector._upstream_latest.get("mock") or {}).get(
                        "status") == "up":
                time.sleep(0.05)
            latest = collector._upstream_latest.get("mock")
            assert latest is not None
            assert latest["status"] == "up"
            assert latest["metrics"]["cpu"] == 7.5
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_snapshot_upstream_down_after_hold_expiry():
    import app as app_mod
    from app import MetricsCollector

    payload = {"cpu": 7.5, "gpus": []}
    up = {"ok": True}

    def fake_urlopen(req, timeout=None):
        if not up["ok"]:
            raise ConnectionRefusedError("down")
        return _http_response(json.dumps(payload).encode())

    clock = _FakeClock()
    orig = app_mod._upstream_last_good
    app_mod._upstream_last_good = {}
    collector = MetricsCollector(poll_interval=0.05)
    collector.llama_enabled = False
    try:
        with patch("app.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("app.time.monotonic", clock):
            collector.start_upstreams(
                [{"name": "mock", "url": "http://127.0.0.1:9996"}])
            collector.start()
            # first fetch lands (up)
            deadline = time.time() + 5
            while time.time() < deadline and (
                    collector._upstream_latest.get("mock") or {}).get(
                        "status") != "up":
                time.sleep(0.05)
            assert (collector._upstream_latest.get("mock") or {}).get(
                "status") == "up"
            # upstream goes down; within the hold window -> stale
            up["ok"] = False
            clock.advance(1.0)
            deadline = time.time() + 5
            while time.time() < deadline and (
                    collector._upstream_latest.get("mock") or {}).get(
                        "status") != "stale":
                time.sleep(0.05)
            assert (collector._upstream_latest.get("mock") or {}).get(
                "status") == "stale"
            # beyond the 15 s hold -> None -> snapshot entry "down"
            clock.advance(20.0)
            deadline = time.time() + 5
            while time.time() < deadline and (
                    collector._upstream_latest.get("mock") is not None):
                time.sleep(0.05)
            assert collector._upstream_latest["mock"] is None
            metrics = _wait_for_snapshot_with(
                collector,
                lambda m: any(s["name"] == "mock" and s["status"] == "down"
                             and s["metrics"] is None
                             for s in m["systems"]))
            assert metrics is not None
    finally:
        collector.stop()
        collector.join(timeout=2)
        app_mod._upstream_last_good = orig


def test_start_upstreams_empty_noop():
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llama_enabled = False
    collector.start_upstreams([])
    assert collector._upstream_names == []
    assert collector._upstream_threads == []
    collector.start()
    try:
        data = collector.get_latest(timeout=3.0)
        assert data is not None
        metrics = json.loads(data)
        # systems is always present: the local entry at minimum
        assert [s["name"] for s in metrics["systems"]] == ["local"]
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_load_config_autodetect_present(tmp_path, monkeypatch):
    from app import load_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "port: 5099\npoll_interval: 2.5\n"
    )
    # path is None -> auto-detect ./config.yaml when present
    cfg = load_config(None)
    assert cfg["port"] == 5099
    assert cfg["poll_interval"] == 2.5


def test_snapshot_upstream_initial_down():
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=1.0)
    collector.llama_enabled = False
    # A configured upstream before any fetch has landed (no
    # _upstream_last_good entry, thread not run / not yet fetched):
    # _upstream_latest is empty, so _collect must report "down".
    collector._upstream_names.append("mock")
    metrics = collector._collect()
    by_name = {s["name"]: s for s in metrics["systems"]}
    assert by_name["mock"]["status"] == "down"
    assert by_name["mock"]["metrics"] is None


# ---------------------------------------------------------------------------
# Strata backend (collect_strata_metrics + config plumbing)
# ---------------------------------------------------------------------------

@pytest.fixture()
def reset_strata_state():
    """Reset module-level Strata scrape history around a test."""
    import app as app_mod

    originals = (app_mod._strata_prev, app_mod._strata_last_good,
                 app_mod._strata_tie_warned)
    app_mod._strata_prev = {}
    app_mod._strata_last_good = {}
    app_mod._strata_tie_warned = False
    yield
    (app_mod._strata_prev, app_mod._strata_last_good,
     app_mod._strata_tie_warned) = originals


def _strata_doc(**over):
    """A minimal Strata /metrics document (serve/server.py metrics() shape)."""
    doc = {
        "engine": {"model": "qwen-test", "max_context": 262144,
                   "context": 262144, "kv": "int8", "images": False,
                   "version": "0.1.39"},
        "live": {"state": "idle", "queued": 0, "waiting": 0,
                 "parallel": 4, "running": 0,
                 "slots": [
                     {"slot": 0, "state": "idle", "held_tokens": 3692},
                     {"slot": 1, "state": "decoding", "prompt_tokens": 1000,
                      "generated": 50, "elapsed_s": 2.0, "tok_s": 40.0},
                     {"slot": 2, "state": "idle", "held_tokens": 0},
                     {"slot": 3, "state": "idle", "held_tokens": 0},
                 ]},
        "requests": [],
        "totals": {"since": time.time() - 100.0, "requests": 10,
                   "prompt_tokens": 10000.0, "reused": 5000.0,
                   "output_tokens": 500.0, "prompt_ms": 100.0,
                   "decode_ms": 200.0, "drafts_offered": 100.0,
                   "drafts_accepted": 75.0},
        "time": time.time(),
    }
    doc.update(over)
    return doc


def make_strata_server(body: bytes):
    """Side_effect for app.urllib.request.urlopen serving Strata /metrics.

    The Strata collector passes a plain URL string (no Request), unlike
    the header-carrying LiteLLM path.
    """
    def fake_urlopen(url, timeout=None):
        if str(url).endswith("/metrics"):
            return _http_response(body)
        raise ConnectionRefusedError(f"no route for {url}")
    return fake_urlopen


def test_collect_strata_happy_path(reset_strata_state):
    from app import collect_strata_metrics

    body = json.dumps(_strata_doc()).encode()
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(body)):
        result = collect_strata_metrics("http://127.0.0.1:9998")

    assert result is not None
    assert result["status"] == "up"
    assert result["model"] == "qwen-test"
    assert result["state"] == "idle"
    assert result["requests_processing"] == 0.0
    assert result["requests_waiting"] == 0.0
    assert result["free_slots"] == 4
    assert result["total_slots"] == 4
    # Slot normalisation into the llama-shaped UI contract
    slots = result["slots"]
    assert slots[0]["context_used"] == 3692  # idle: held KV
    assert slots[0]["is_processing"] is False
    assert slots[1]["context_used"] == 1050  # active: prompt + generated
    assert slots[1]["is_processing"] is True
    assert slots[1]["decoded"] == 50
    assert slots[1]["context_pct"] == round(100.0 * 1050 / 262144, 1)
    # Cumulative totals
    assert result["gen_tokens_total"] == 500.0
    assert result["prompt_tokens_total"] == 10000.0
    assert result["prompt_tokens_cached_total"] == 5000.0
    assert result["requests_completed_total"] == 10.0
    assert result["prefix_cache_hit_pct"] == 50.0  # reused / prompt
    assert result["spec_accept_pct"] == 75.0       # accepted / offered
    # First scrape: no Δ-window rates, no request-window latencies
    assert "gen_tokens_per_s" not in result
    assert "prompt_tokens_per_s" not in result
    assert "ttft_seconds" not in result
    # Usage: reused is NESTED in prompt (vLLM-style), not added
    usage = result["usage"]
    assert usage["total_tokens"] == 10500
    assert usage["gen_tokens"] == 500
    assert usage["prompt_cached_pct"] == 50.0
    assert usage["uptime_s"] >= 99
    assert result["config"]["version"] == "0.1.39"
    assert result["config"]["context"] == 262144
    assert result["config"]["parallel"] == 4


def test_collect_strata_rates_and_request_windows(reset_strata_state):
    """Second scrape yields Δ-counter rates + means over NEW requests."""
    from app import collect_strata_metrics

    clock = _FakeClock()
    first = json.dumps(_strata_doc()).encode()
    doc2 = _strata_doc()
    doc2["totals"]["output_tokens"] = 700.0   # +200 / 2 s -> Δ would say 100
    doc2["totals"]["prompt_tokens"] = 12000.0  # +2000 / 2 s -> Δ would say 1000
    doc2["requests"] = [
        {"time": 2600.0, "duration_s": 20.0, "prompt_ms": 4000.0,
         "prompt_tokens": 2000.0, "decode_tok_s": 50.0},
        {"time": 2500.0, "duration_s": 10.0, "prompt_ms": 2000.0,
         "prompt_tokens": 1000.0, "decode_tok_s": 50.0},
    ]
    second = json.dumps(doc2).encode()

    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(first)):
        r1 = collect_strata_metrics("http://127.0.0.1:9998")
    assert "gen_tokens_per_s" not in r1
    assert "ttft_seconds" not in r1  # first scrape: no window

    clock.advance(2.0)
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(second)):
        r2 = collect_strata_metrics("http://127.0.0.1:9998")
    # Completion tick (engine idle): the ring's TRUE speeds replace the
    # one-tick counter-burst (mean decode_tok_s; prompt tokens / prompt_ms)
    assert r2["gen_tokens_per_s"] == 50.0
    assert r2["prompt_tokens_per_s"] == 500.0   # 3000 tok / 6 s prefill
    # Means over the two NEW finished requests
    assert r2["ttft_seconds"] == 3.0            # mean(2.0, 4.0)
    assert r2["e2e_latency_seconds"] == 15.0    # mean(10, 20)
    assert r2["itl_ms"] == 20.0                 # mean(1000/50, 1000/50)

    # Third scrape, same ring buffer: watermark excludes old requests
    clock.advance(2.0)
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(second)):
        r3 = collect_strata_metrics("http://127.0.0.1:9998")
    assert "ttft_seconds" not in r3
    assert "e2e_latency_seconds" not in r3
    assert "itl_ms" not in r3


def test_collect_strata_live_rates_preferred_while_busy(reset_strata_state):
    """Strata totals.* only advance at request completion; while the
    engine is busy the live fields must drive the gauges instead."""
    from app import collect_strata_metrics

    clock = _FakeClock()
    first = json.dumps(_strata_doc()).encode()
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(first)):
        collect_strata_metrics("http://127.0.0.1:9998")  # baseline

    def scrape(doc):
        clock.advance(2.0)
        body = json.dumps(doc).encode()
        with patch("app.time.monotonic", clock), \
             patch("app.urllib.request.urlopen",
                   side_effect=make_strata_server(body)):
            return collect_strata_metrics("http://127.0.0.1:9998")

    # Solo decode: counters frozen (rate would be 0.0) -> live.tok_s
    solo = _strata_doc()
    solo["live"] = dict(solo["live"], state="generating", running=1,
                        tok_s=42.5, tok_s_mean=40.0, slots=[])
    r = scrape(solo)
    assert r["gen_tokens_per_s"] == 42.5

    # Batch decode: live.tok_s is the engine's WINDOWED aggregate (sum
    # of every running request's recent rate - what the Monitor tab shows). The
    # per-slot tok_s values are whole-request means (generated /
    # since-first-token) and must NOT be summed: a long-running or
    # slowing request's mean outruns its windowed rate, over-reporting
    # (e.g. 55+55=110 while the windowed aggregate is 75).
    batch = _strata_doc()
    batch["live"] = dict(batch["live"], state="generating", running=2,
                         tok_s=75.0, slots=[
                             {"slot": 0, "state": "decoding",
                              "prompt_tokens": 10, "generated": 5,
                              "elapsed_s": 1.0, "tok_s": 55.0},
                             {"slot": 1, "state": "decoding",
                              "prompt_tokens": 20, "generated": 8,
                              "elapsed_s": 1.0, "tok_s": 55.0},
                             {"slot": 2, "state": "idle",
                              "held_tokens": 0},
                         ])
    r = scrape(batch)
    assert r["gen_tokens_per_s"] == 75.0  # windowed sum, not 55+55

    # No live.tok_s (edge: state flipped to generating without one):
    # fall back to tok_s_mean, still NOT the slot-sum
    fallback = _strata_doc()
    fallback["live"] = dict(fallback["live"], state="generating",
                           running=1, tok_s=None, tok_s_mean=40.0,
                           slots=[
                               {"slot": 0, "state": "decoding",
                                "prompt_tokens": 10, "generated": 5,
                                "elapsed_s": 1.0, "tok_s": 55.0},
                           ])
    r = scrape(fallback)
    assert r["gen_tokens_per_s"] == 40.0

    # Prefill (reading): live.prefill_tok_s_mean drives the prompt gauge
    reading = _strata_doc()
    reading["live"] = dict(reading["live"], state="reading", running=1,
                           prefill_tok_s_mean=1234.5, slots=[])
    r = scrape(reading)
    assert r["prompt_tokens_per_s"] == 1234.5

    # Idle again: no live fields -> back to the (frozen) counter rate
    idle = _strata_doc()
    r = scrape(idle)
    assert r["gen_tokens_per_s"] == 0.0


def test_collect_strata_prompt_prefill_last5(reset_strata_state):
    """prompt_tokens_per_s = the last 5 completed requests' prefill
    speed as a vLLM-style windowed aggregate (Σ fresh tokens / Σ read
    time), staying current on every tick instead of spiking at
    completion and freezing."""
    from app import collect_strata_metrics

    def req(t, tok, ms):
        return {"time": t, "prompt_tokens": tok, "prompt_ms": ms}

    def scrape(doc):
        with patch("app.urllib.request.urlopen",
                   side_effect=make_strata_server(
                       json.dumps(doc).encode())):
            return collect_strata_metrics("http://127.0.0.1:9998")

    # 6 completions: the 6th (oldest, 10000 tok/s) must NOT enter the
    # mean - only the 5 newest (all 500 tok/s).
    doc = _strata_doc()
    doc["requests"] = [
        req(3000.0, 100.0, 200.0),    # 500 tok/s  (newest)
        req(2900.0, 200.0, 400.0),    # 500
        req(2800.0, 300.0, 600.0),    # 500
        req(2700.0, 400.0, 800.0),    # 500
        req(2600.0, 500.0, 1000.0),   # 500
        req(2500.0, 1000.0, 100.0),   # 10000 - excluded (6th)
    ]
    r = scrape(doc)
    assert r["prompt_tokens_per_s"] == 500.0  # not (5*500+10000)/6

    # vLLM-style AGGREGATE (Σ fresh / Σ read time), not the per-request
    # mean: 1100 tok / 10.1 s = 108.9, whereas mean(1000, 100) = 550
    doc2 = _strata_doc()
    doc2["requests"] = [
        req(3000.0, 100.0, 100.0),    # 1000 tok/s
        req(2900.0, 1000.0, 10000.0), # 100 tok/s
    ]
    r = scrape(doc2)
    assert r["prompt_tokens_per_s"] == 108.9

    # Entries without a usable prompt_ms (older engine) are skipped
    doc3 = _strata_doc()
    doc3["requests"] = [
        req(3000.0, 100.0, 100.0),    # 1000 tok/s
        {"time": 2900.0, "prompt_tokens": 100.0, "prompt_ms": None},
    ]
    r = scrape(doc3)
    assert r["prompt_tokens_per_s"] == 1000.0

    # Cached tokens are excluded (the engine's own prompt_per_second
    # definition): 20k tokens in 1 ms is a cache lookup, not a 20M
    # tok/s prefill - the FRESH (prompt_tokens - reused) work counts.
    # Both requests at 1000 tok/s -> aggregate 1100 / 1.1 s = 1000;
    # WITHOUT the subtraction it would be 20100 / 1.1 s = 18272.7.
    doc3b = _strata_doc()
    doc3b["requests"] = [
        req(3000.0, 100.0, 100.0),    # 1000 tok/s (no cache)
        {"time": 2900.0, "prompt_tokens": 20000.0, "reused": 19000.0,
         "prompt_ms": 1000.0},        # 1000 fresh tok / 1 s -> 1000 tok/s
    ]
    r = scrape(doc3b)
    assert r["prompt_tokens_per_s"] == 1000.0  # not 20100/1.1 = 18272.7

    # Fully cached (no fresh work) requests are skipped from the mean
    doc3c = _strata_doc()
    doc3c["requests"] = [
        req(3000.0, 100.0, 100.0),    # 1000 tok/s
        {"time": 2900.0, "prompt_tokens": 1000.0, "reused": 1000.0,
         "prompt_ms": 1.0},           # fully cached: skipped
    ]
    r = scrape(doc3c)
    assert r["prompt_tokens_per_s"] == 1000.0

    # Reading state with an EMPTY ring: the live prefill speed is the
    # fallback (no completed requests yet).
    doc4 = _strata_doc()
    doc4["live"] = dict(doc4["live"], state="reading",
                        prefill_tok_s_mean=999.0)
    r = scrape(doc4)
    assert r["prompt_tokens_per_s"] == 999.0

    # Reading state WITH ring data: the last-5 mean wins over the live
    # prefill (the gauge is a completed-requests window, vLLM-style).
    doc5 = _strata_doc()
    doc5["live"] = dict(doc5["live"], state="reading",
                        prefill_tok_s_mean=999.0)
    doc5["requests"] = [req(3000.0, 100.0, 100.0)]  # 1000 tok/s
    r = scrape(doc5)
    assert r["prompt_tokens_per_s"] == 1000.0


def test_collect_strata_rejects_non_strata_payloads(reset_strata_state):
    """JSON without engine+live shape, or non-JSON text -> None."""
    from app import collect_strata_metrics

    for body in (b'{"hello": "world"}', b'not json at all'):
        with patch("app.urllib.request.urlopen",
                   side_effect=make_strata_server(body)):
            assert collect_strata_metrics("http://127.0.0.1:9998") is None


def test_collect_strata_stale_hold_and_expiry(reset_strata_state):
    """Failed scrape holds the last snapshot as stale for 15 s."""
    from app import collect_strata_metrics

    clock = _FakeClock()
    body = json.dumps(_strata_doc()).encode()
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(body)):
        assert collect_strata_metrics("http://127.0.0.1:9998")["status"] == "up"

    clock.advance(2.0)
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen",
               side_effect=ConnectionRefusedError("down")):
        held = collect_strata_metrics("http://127.0.0.1:9998")
        assert held["status"] == "stale"
        clock.advance(20.0)
        assert collect_strata_metrics("http://127.0.0.1:9998") is None


def test_collect_llm_auto_prefers_strata_over_llama(reset_strata_state):
    """auto order: vLLM/LiteLLM down -> Strata wins before llama.cpp."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="auto")
    with patch("app.collect_vllm_metrics", return_value=None), \
         patch("app.collect_litellm_metrics", return_value=None), \
         patch("app.collect_strata_metrics",
               return_value={"status": "up"}), \
         patch("app.collect_llama_metrics") as fake_llama:
        data = collector._collect_llm()
    assert data == {"status": "up", "backend": "strata"}
    fake_llama.assert_not_called()


def test_collect_llm_pinned_strata_skips_others():
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llm_backend="strata",
                                 strata_url="http://s:8080")
    with patch("app.collect_vllm_metrics") as fake_vllm, \
         patch("app.collect_litellm_metrics") as fake_litellm, \
         patch("app.collect_llama_metrics") as fake_llama, \
         patch("app.collect_strata_metrics",
               return_value={"status": "up"}) as fake_strata:
        data = collector._collect_llm()
    assert data["backend"] == "strata"
    fake_strata.assert_called_once_with("http://s:8080",
                                        state_key="primary")
    fake_vllm.assert_not_called()
    fake_litellm.assert_not_called()
    fake_llama.assert_not_called()


def test_config_strata_and_default_view(tmp_path):
    from app import load_config, resolve_settings

    p = tmp_path / "config.yaml"
    p.write_text(
        "llm_backend: strata\n"
        "strata_url: http://192.168.5.219:8080\n"
        "default_view: fleet\n"
    )
    cfg = load_config(str(p))
    settings = resolve_settings(_ns(), cfg)
    assert settings["llm_backend"] == "strata"
    assert settings["strata_url"] == "http://192.168.5.219:8080"
    assert settings["default_view"] == "fleet"


def test_default_view_defaults_to_local():
    from app import resolve_settings

    settings = resolve_settings(_ns(), {})
    assert settings["default_view"] == "local"
    assert settings["strata_url"] == "http://127.0.0.1:8080"


def test_default_view_invalid_exits(tmp_path):
    from app import load_config

    p = tmp_path / "bad_view.yaml"
    p.write_text("default_view: sidebar\n")
    with pytest.raises(SystemExit):
        load_config(str(p))


# ---------------------------------------------------------------------------
# Multi-LLM (targets, keyed state, payload shape, fleet, config)
# ---------------------------------------------------------------------------

def _multi_strata_router(bodies: dict):
    """urlopen side_effect routing several Strata servers by host:port."""
    def fake_urlopen(url, timeout=None):
        u = str(url)
        if not u.endswith("/metrics"):
            raise ConnectionRefusedError(f"no route for {u}")
        for hostport, body in bodies.items():
            if hostport in u:
                if body is None:
                    raise ConnectionRefusedError(f"{hostport} down")
                return _http_response(body)
        raise ConnectionRefusedError(f"no route for {u}")
    return fake_urlopen


def test_multi_llm_state_keyed_per_target(reset_strata_state):
    """Two same-type targets keep independent rate windows and stale holds."""
    from app import collect_strata_metrics

    clock = _FakeClock()
    doc_a, doc_b = _strata_doc(), _strata_doc()
    doc_b["totals"]["output_tokens"] = 900.0
    doc_b["engine"]["model"] = "qwen-b"
    body_a = json.dumps(doc_a).encode()
    body_b = json.dumps(doc_b).encode()
    router = _multi_strata_router({"h1": body_a, "h2": body_b})

    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen", side_effect=router):
        a1 = collect_strata_metrics("http://h1:8080", state_key="strata-a")
        b1 = collect_strata_metrics("http://h2:8080", state_key="strata-b")
    assert "gen_tokens_per_s" not in a1  # first scrape per key, independent
    assert "gen_tokens_per_s" not in b1

    clock.advance(2.0)
    doc_a["totals"]["output_tokens"] = 700.0   # +200 / 2 s = 100 t/s
    doc_b["totals"]["output_tokens"] = 1100.0  # +200 / 2 s = 100 t/s
    doc_b_body = json.dumps(doc_b).encode()
    router2 = _multi_strata_router({"h1": json.dumps(doc_a).encode(),
                                    "h2": doc_b_body})
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen", side_effect=router2):
        a2 = collect_strata_metrics("http://h1:8080", state_key="strata-a")
        b2 = collect_strata_metrics("http://h2:8080", state_key="strata-b")
    assert a2["gen_tokens_per_s"] == 100.0
    assert b2["gen_tokens_per_s"] == 100.0
    assert a2["model"] == "qwen-test"
    assert b2["model"] == "qwen-b"

    # h1 down: stale-hold for strata-a only; strata-b unaffected
    clock.advance(2.0)
    router3 = _multi_strata_router({"h1": None, "h2": doc_b_body})
    with patch("app.time.monotonic", clock), \
         patch("app.urllib.request.urlopen", side_effect=router3):
        a3 = collect_strata_metrics("http://h1:8080", state_key="strata-a")
        b3 = collect_strata_metrics("http://h2:8080", state_key="strata-b")
    assert a3["status"] == "stale"
    assert b3["status"] == "up"


def _collector_snapshot(collector):
    """_collect() with host-metric probes stubbed."""
    with patch("app.collect_gpu_metrics", return_value=[]), \
         patch("app.collect_swap_metrics", return_value=None), \
         patch("app.collect_host_health_metrics",
               return_value={"psi_mem_avg10": None, "d_state_procs": 0}):
        return collector._collect()


def test_collector_multi_llm_entries_and_alias(reset_strata_state):
    """Two strata targets -> two ordered entries; `llm` = primary alias."""
    from app import LLMTarget, MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llm_targets = [
        LLMTarget("strata-a", "strata", "http://h1:8080"),
        LLMTarget("strata-b", "strata", "http://h2:8081"),
    ]
    doc_a, doc_b = _strata_doc(), _strata_doc()
    doc_b["totals"]["output_tokens"] = 900.0
    router = _multi_strata_router(
        {"h1:8080": json.dumps(doc_a).encode(),
         "h2:8081": json.dumps(doc_b).encode()})
    with patch("app.urllib.request.urlopen", side_effect=router):
        collector._llm_latest = [
            collector._collect_one(t) for t in collector._targets_snapshot()]
    snapshot = _collector_snapshot(collector)

    names = [e["name"] for e in snapshot["llms"]]
    assert names == ["strata-a", "strata-b"]
    assert all(e["backend"] == "strata" for e in snapshot["llms"])
    assert snapshot["llms"][1]["gen_tokens_total"] == 900.0
    # legacy alias: primary (first) entry, NOT an aggregate
    assert snapshot["llm"]["name"] == "strata-a"
    assert snapshot["llm"]["gen_tokens_total"] == 500.0


def test_collector_multi_llm_down_placeholders(reset_strata_state):
    """A dead target keeps a named down entry; alias None if it is primary."""
    from app import LLMTarget, MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llm_targets = [
        LLMTarget("strata-a", "strata", "http://h1:8080"),
        LLMTarget("strata-b", "strata", "http://h2:8081"),
    ]
    router = _multi_strata_router(
        {"h1:8080": None,  # down
         "h2:8081": json.dumps(_strata_doc()).encode()})
    with patch("app.urllib.request.urlopen", side_effect=router):
        collector._llm_latest = [
            collector._collect_one(t) for t in collector._targets_snapshot()]
    snapshot = _collector_snapshot(collector)

    a, b = snapshot["llms"]
    assert a == {"name": "strata-a", "backend": "strata", "status": "down"}
    assert b["status"] == "up"
    assert snapshot["llm"] is None  # primary down -> legacy contract


def test_collector_disabled_shape():
    """llama_enabled=False -> llm None AND llms [] (pinned contract)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    snapshot = _collector_snapshot(collector)
    assert snapshot["llm"] is None
    assert snapshot["llms"] == []


def test_legacy_config_single_primary_target():
    from app import resolve_settings

    settings = resolve_settings(_ns(llm_backend="vllm"), {})
    targets = settings["llm_targets"]
    assert len(targets) == 1
    assert targets[0].name == "primary"
    assert targets[0].backend == "vllm"
    assert targets[0].url == "http://127.0.0.1:5001"
    # auto keeps url None (probes per-type URLs on the collector)
    settings = resolve_settings(_ns(), {})
    assert settings["llm_targets"][0].backend == "auto"
    assert settings["llm_targets"][0].url is None


def test_llms_config_multi_and_primary_first(tmp_path):
    from app import load_config, resolve_settings

    p = tmp_path / "config.yaml"
    p.write_text(
        "llms:\n"
        "  - name: strata-main\n"
        "    backend: strata\n"
        "    url: http://192.168.5.219:8080\n"
        "  - name: coder\n"
        "    backend: strata\n"
        "    url: http://127.0.0.1:8081\n"
        "  - name: vllm-box\n"
        "    backend: vllm\n"
        "    url: http://127.0.0.1:8000\n"
        "    primary: true\n"
    )
    settings = resolve_settings(_ns(), load_config(str(p)))
    names = [t.name for t in settings["llm_targets"]]
    assert names == ["vllm-box", "strata-main", "coder"]


def test_llms_config_conflicts_with_legacy(tmp_path):
    from app import resolve_settings

    cfg = {"llms": [{"name": "a", "backend": "strata",
                     "url": "http://x:8080"}],
           "llm_backend": "strata"}
    with pytest.raises(SystemExit):
        resolve_settings(_ns(), cfg)
    # CLI-side legacy flag conflicts too (raw presence)
    with pytest.raises(SystemExit):
        resolve_settings(_ns(llm_backend="llama"), cfg)


def test_llms_config_validation_errors(tmp_path):
    from app import load_config

    cases = [
        "llms: []\n",                                        # empty
        "llms: notalist\n",                                  # not a list
        "llms:\n  - {name: a, backend: auto, url: 'http://x'}\n",
        "llms:\n  - {name: 'bad name!', backend: strata, url: 'http://x'}\n",
        "llms:\n  - {name: a, backend: strata}\n",           # missing url
        ("llms:\n  - {name: a, backend: strata, url: 'http://x'}\n"
         "  - {name: a, backend: strata, url: 'http://y'}\n"),  # dup name
        ("llms:\n  - {name: a, backend: strata, url: 'http://x',"
         " primary: true}\n"
         "  - {name: b, backend: strata, url: 'http://y',"
         " primary: true}\n"),                               # two primaries
    ]
    for i, text in enumerate(cases):
        p = tmp_path / f"bad{i}.yaml"
        p.write_text(text)
        with pytest.raises(SystemExit):
            load_config(str(p))


def test_upstream_sanitizer_forwards_llms():
    import app as app_mod
    from app import collect_upstream_metrics

    orig = app_mod._upstream_last_good
    app_mod._upstream_last_good = {}
    try:
        payload = {
            "timestamp": 1.0, "cpu": 5.0, "memory": {}, "gpus": [],
            "llm_enabled": True,
            "llm": {"name": "primary", "backend": "strata", "status": "up"},
            "llms": [
                {"name": "primary", "backend": "strata", "status": "up"},
                {"name": "second", "backend": "vllm", "status": "down"},
            ],
            "systems": [{"name": "nested"}],
        }
        with patch("app.urllib.request.urlopen",
                   side_effect=make_llama_server(
                       {"/api/metrics": json.dumps(payload).encode()})):
            out = collect_upstream_metrics("http://127.0.0.1:9996", "mock")
        m = out["metrics"]
        assert len(m["llms"]) == 2
        assert m["llms"][1]["name"] == "second"
        assert "systems" not in m
    finally:
        app_mod._upstream_last_good = orig


def test_llama_usage_state_path_per_target(tmp_path, monkeypatch):
    from app import _llama_usage_state_path

    monkeypatch.setenv("RD_LLAMA_USAGE_STATE",
                       str(tmp_path / "llama_usage_state.json"))
    assert _llama_usage_state_path() == str(
        tmp_path / "llama_usage_state.json")
    assert _llama_usage_state_path("coder") == str(
        tmp_path / ".llama_usage_state.coder.json")


def test_llama_process_dropped_for_multi_llama_targets():
    """The local process can only be attributed to the sole llama target."""
    from app import LLMTarget, MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llm_targets = [
        LLMTarget("llama-a", "llama", "http://127.0.0.1:8080"),
        LLMTarget("llama-b", "llama", "http://127.0.0.1:8081"),
    ]
    with patch("app.collect_llama_metrics",
               return_value={"status": "up", "process": {"rss_gb": 3.0}}):
        entry = collector._collect_one(collector.llm_targets[0])
    assert "process" not in entry


def test_llama_process_kept_for_sole_local_target():
    from app import LLMTarget, MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llm_targets = [LLMTarget("primary", "llama",
                                       "http://127.0.0.1:8080")]
    with patch("app.collect_llama_metrics",
               return_value={"status": "up", "process": {"rss_gb": 3.0}}):
        entry = collector._collect_one(collector.llm_targets[0])
    assert entry["process"] == {"rss_gb": 3.0}


def test_llama_process_dropped_for_remote_url():
    from app import LLMTarget, MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    collector.llm_targets = [LLMTarget("primary", "llama",
                                       "http://192.168.5.207:5001")]
    with patch("app.collect_llama_metrics",
               return_value={"status": "up", "process": {"rss_gb": 3.0}}):
        entry = collector._collect_one(collector.llm_targets[0])
    assert "process" not in entry


# ---------------------------------------------------------------------------
# T2 — fixed-rate cadence + subprocess negative caching off the snapshot path
# ---------------------------------------------------------------------------

def test_fixed_rate_schedule_holds_period_under_collect_cost():
    """Fixed-rate scheduling (next += interval): with a 0.5 s _collect()
    and a 1.0 s poll_interval the inter-snapshot period stays ≈1.0 s
    (the old sleep-after-collect produced ≈1.5 s)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=1.0, llama_enabled=False)
    stamps: list[float] = []

    def fake_collect() -> dict:
        stamps.append(time.time())  # cycle start ≈ slot start
        time.sleep(0.5)
        return {"timestamp": time.time()}

    collector._collect = fake_collect
    collector.start()
    try:
        time.sleep(4.5)
    finally:
        collector.stop()
        collector.join(timeout=3)
    assert len(stamps) >= 3
    periods = [b - a for a, b in zip(stamps, stamps[1:])]
    for p in periods:
        assert 0.6 <= p <= 1.4, \
            f"period {p:.3f}s — expected ≈1.0s, not ≈1.5s"


def test_cycle_duration_and_drift_logged(caplog):
    """Each cycle logs its _collect() duration and drift vs the fixed
    schedule (one DEBUG line per cycle; no INFO flood at the default
    level)."""
    import re

    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.2, llama_enabled=False)
    collector._collect = lambda: {"timestamp": time.time()}
    with caplog.at_level(logging.DEBUG, logger="app"):
        collector.start()
        try:
            time.sleep(0.7)
        finally:
            collector.stop()
            collector.join(timeout=2)
    cycle_logs = [r for r in caplog.records
                 if "cycle:" in r.getMessage()
                 and "drift=" in r.getMessage()]
    assert len(cycle_logs) >= 2
    pattern = re.compile(r"collect=\d+\.\d\d+s drift=[+-]\d+\.\d\d+s")
    assert any(pattern.search(r.getMessage()) for r in cycle_logs)


def test_sysfs_pci_vendor_name_never_raises(tmp_path):
    """The sysfs vendor-name resolution never raises — a missing device
    node, a malformed BDF, or an absent vendor file all resolve to a
    fallback (``unknown`` or the raw ID string) without stalling the
    snapshot path."""
    from app import _sysfs_pci_vendor_name

    # A BDF with no device node → "unknown" (no raise).
    assert _sysfs_pci_vendor_name("0000:ff:ff.0", str(tmp_path)) == "unknown"
    # A malformed BDF → "unknown" (no raise).
    assert _sysfs_pci_vendor_name("garbage", str(tmp_path)) == "unknown"
    # A device node with no vendor file → "unknown" (the open fails
    # cleanly, no raise).
    (tmp_path / "0000:2d:00.0").mkdir(parents=True)
    assert _sysfs_pci_vendor_name("0000:2d:00.0", str(tmp_path)) == "unknown"


def test_poll_interval_semantics_unchanged():
    """poll_interval still means the target period between snapshots:
    config plumbing (CLI > config > default) is intact and the run
    loop schedules at that rate."""
    from app import MetricsCollector, resolve_settings

    assert resolve_settings(_ns(), {})["poll_interval"] == 1.0
    assert resolve_settings(_ns(), {"poll_interval": 2.5})["poll_interval"] == 2.5
    assert resolve_settings(_ns(poll_interval=3.5),
                           {"poll_interval": 2.5})["poll_interval"] == 3.5

    collector = MetricsCollector(poll_interval=0.3, llama_enabled=False)
    assert collector.poll_interval == 0.3
    stamps: list[float] = []

    def fake_collect() -> dict:
        stamps.append(time.time())
        return {"timestamp": time.time()}

    collector._collect = fake_collect
    collector.start()
    try:
        time.sleep(1.2)
    finally:
        collector.stop()
        collector.join(timeout=2)
    assert len(stamps) >= 3
    for a, b in zip(stamps, stamps[1:]):
        assert 0.15 <= b - a <= 0.45, f"period {b - a:.3f}s != target 0.3s"


# ---------------------------------------------------------------------------
# T6 — LLM publish path busy-latch + Strata slot diagnosability
# ---------------------------------------------------------------------------

def test_llm_busy_latch_publishes_latest():
    """A busy tick the slow main loop missed (current sample is idle but a
    newer busy sample was latched) is still published — the busy state is
    not discarded."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=1.0)
    collector.llama_enabled = True
    # The current sample (what the main loop reads) is idle.
    collector._llm_latest = [
        {"name": "primary", "backend": "strata", "status": "up",
         "state": "idle", "requests_processing": 0.0}]
    # A busy sample was latched by the LLM thread (a busy tick the main
    # loop missed); it has not yet been published.
    collector._llm_busy_latch = [
        {"name": "primary", "backend": "strata", "status": "up",
         "state": "generating", "requests_processing": 1.0}]
    collector._llm_busy_latch_id = 1
    collector._llm_busy_latch_published_id = 0

    metrics = collector._collect()
    # The latched busy sample is published (not the idle current).
    assert metrics["llms"] == collector._llm_busy_latch
    assert metrics["llms"][0]["state"] == "generating"
    # The legacy `llm` key is the primary entry (busy, not down).
    assert metrics["llm"]["state"] == "generating"
    # The watermark advanced, so the latch is not re-published next cycle.
    assert collector._llm_busy_latch_published_id == 1

    # Next cycle: current is still idle, no new latch -> the idle current
    # is published (the latch is not re-published).
    metrics2 = collector._collect()
    assert metrics2["llms"] == collector._llm_latest
    assert metrics2["llms"][0]["state"] == "idle"


def test_strata_slot_diagnosability_fields(reset_strata_state):
    """The Strata LLM entry carries state_seen, slots_busy, last_busy_ts,
    slots_state_mismatch with correct types/values for a busy payload."""
    from app import collect_strata_metrics

    doc = _strata_doc()
    doc["live"]["state"] = "generating"
    doc["live"]["running"] = 1
    body = json.dumps(doc).encode()
    with patch("app.time.time", return_value=12345.0), \
         patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(body)):
        result = collect_strata_metrics("http://127.0.0.1:9998")

    assert result is not None
    # slots_busy: one busy slot (slot 1 is decoding).
    assert result["slots_busy"] == 1
    # slots_state_mismatch: slot activity (1 busy) agrees with running (1)
    # -> no mismatch.
    assert result["slots_state_mismatch"] is False
    # state_seen: the raw state string (generating).
    assert result["state_seen"] == "generating"
    # last_busy_ts: a float epoch ts (a slot is busy).
    assert isinstance(result["last_busy_ts"], float)
    assert result["last_busy_ts"] == 12345.0


def test_strata_is_processing_cheap_guard(reset_strata_state):
    """A slot whose state string is idle but has generated tokens (or
    elapsed time) is marked is_processing (the cheap guard catches short
    requests whose state lags the activity)."""
    from app import collect_strata_metrics

    # Positive: a slot with state "idle" but generated > 0 (a short
    # request whose state string lags the actual activity).
    doc = _strata_doc()
    doc["live"]["slots"] = [
        {"slot": 0, "state": "idle", "held_tokens": 0,
         "generated": 10, "elapsed_s": 0.5},
    ]
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(json.dumps(doc).encode())):
        result = collect_strata_metrics("http://127.0.0.1:9998",
                                       state_key="pos")
    assert result is not None
    # The cheap guard: state is idle, but generated > 0 -> processing.
    assert result["slots"][0]["is_processing"] is True
    assert result["slots_busy"] == 1

    # Negative: a truly idle slot (no generated, no elapsed) is not
    # processing.
    doc2 = _strata_doc()
    doc2["live"]["slots"] = [
        {"slot": 0, "state": "idle", "held_tokens": 5},
    ]
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(json.dumps(doc2).encode())):
        result2 = collect_strata_metrics("http://127.0.0.1:9998",
                                        state_key="neg")
    assert result2["slots"][0]["is_processing"] is False
    assert result2["slots_busy"] == 0


def test_strata_slots_state_mismatch(reset_strata_state):
    """slots_state_mismatch = any(active) != (running > 0): True when
    slot-level activity disagrees with the engine-level running count,
    False when they agree."""
    from app import collect_strata_metrics

    # Case 1 (mismatch): a busy slot but running = 0.
    doc1 = _strata_doc()
    doc1["live"]["running"] = 0
    doc1["live"]["slots"] = [
        {"slot": 0, "state": "decoding", "prompt_tokens": 100,
         "generated": 10, "elapsed_s": 1.0},
    ]
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(json.dumps(doc1).encode())):
        r1 = collect_strata_metrics("http://127.0.0.1:9998", state_key="m1")
    assert r1["slots_state_mismatch"] is True

    # Case 2 (agree, both active): a busy slot and running > 0.
    doc2 = _strata_doc()
    doc2["live"]["running"] = 1
    doc2["live"]["slots"] = [
        {"slot": 0, "state": "decoding", "prompt_tokens": 100,
         "generated": 10, "elapsed_s": 1.0},
    ]
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(json.dumps(doc2).encode())):
        r2 = collect_strata_metrics("http://127.0.0.1:9998", state_key="a1")
    assert r2["slots_state_mismatch"] is False

    # Case 3 (agree, both idle): no busy slot and running = 0.
    doc3 = _strata_doc()
    doc3["live"]["running"] = 0
    doc3["live"]["slots"] = [
        {"slot": 0, "state": "idle", "held_tokens": 5},
    ]
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(json.dumps(doc3).encode())):
        r3 = collect_strata_metrics("http://127.0.0.1:9998", state_key="i1")
    assert r3["slots_state_mismatch"] is False


def test_strata_empty_slot_list_not_silently_dropped(reset_strata_state):
    """An empty slot list is not silently discarded: out["slots"] is
    present (as []) so the empty case is explicit / diagnosable."""
    from app import collect_strata_metrics

    doc = _strata_doc()
    doc["live"]["slots"] = []
    body = json.dumps(doc).encode()
    with patch("app.urllib.request.urlopen",
               side_effect=make_strata_server(body)):
        result = collect_strata_metrics("http://127.0.0.1:9998")

    assert result is not None
    # The empty slot list is present (not silently dropped).
    assert "slots" in result
    assert result["slots"] == []
    # slots_busy is 0 (no busy slots).
    assert result["slots_busy"] == 0


# ---------------------------------------------------------------------------
# T8 — lifecycle + state-keying fixes
# ---------------------------------------------------------------------------

def test_stop_joins_upstream_threads():
    """stop() sets _upstream_stop and joins the upstream fetch threads
    (they are woken and joined on shutdown, not left running)."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    with patch("app.collect_upstream_metrics", return_value=None):
        collector.start_upstreams(
            [{"name": "peer-a", "url": "http://127.0.0.1:9997"}])
        assert len(collector._upstream_threads) == 1
        t = collector._upstream_threads[0]
        # Wait for the daemon thread to actually start.
        for _ in range(200):
            if t.is_alive():
                break
            time.sleep(0.01)
        assert t.is_alive()
        collector.stop(timeout=2.0)
    # The upstream thread is woken (_upstream_stop set) and joined.
    assert collector._upstream_stop.is_set()
    assert not t.is_alive()


def test_upstream_backoff_on_failure():
    """A failing upstream backs off (retry interval increases on failure,
    resets on success; does not hammer a dead upstream every cycle)."""
    import app as app_mod
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    wait_durations: list[float] = []
    call_count = 0

    def fake_collect(url, name):
        nonlocal call_count
        call_count += 1
        # Fail on calls 1,2; succeed on call 3 (reset); fail on 4,5,6.
        if call_count == 3:
            return {"name": "peer-a", "status": "up", "metrics": {}}
        return None

    def fake_wait(timeout):
        wait_durations.append(timeout)
        if len(wait_durations) >= 6:
            collector.running = False  # exit after 6 iterations
        return True

    orig_wait = collector._upstream_stop.wait
    collector._upstream_stop.wait = fake_wait
    with patch("app.collect_upstream_metrics", side_effect=fake_collect):
        try:
            collector._upstream_loop("peer-a", "http://127.0.0.1:9997")
        finally:
            collector._upstream_stop.wait = orig_wait
    # Consecutive failures back off (the retry interval increases).
    assert wait_durations[1] > wait_durations[0]
    # A success resets the backoff to the poll cadence.
    assert wait_durations[2] == collector.poll_interval
    # A subsequent failure backs off again from the reset value.
    assert wait_durations[3] > wait_durations[2]
    # The backoff is capped (never exceeds _UPSTREAM_BACKOFF_CAP_S).
    assert max(wait_durations) <= app_mod._UPSTREAM_BACKOFF_CAP_S


def test_upstream_removal_after_n_failures():
    """After N consecutive failures, the upstream is removed (the loop
    stops fetching it)."""
    import app as app_mod
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    collect_count = 0

    def fake_collect(url, name):
        nonlocal collect_count
        collect_count += 1
        return None  # always fail

    def fake_wait(timeout):
        return True  # return immediately (no real sleep)

    orig_wait = collector._upstream_stop.wait
    collector._upstream_stop.wait = fake_wait
    with patch("app.collect_upstream_metrics", side_effect=fake_collect):
        try:
            collector._upstream_loop("peer-a", "http://127.0.0.1:9997")
        finally:
            collector._upstream_stop.wait = orig_wait
    # The upstream is removed after N consecutive failures.
    assert "peer-a" in collector._upstream_removed
    # The loop stopped after exactly N failures (not more).
    assert collect_count == app_mod._UPSTREAM_REMOVAL_AFTER_FAILURES


def test_llama_proc_keyed_per_target():
    """_llama_proc is keyed per target (two different llama targets
    retain different process handles; no cross-target clobbering)."""
    import app as app_mod
    from app import collect_llama_metrics

    proc_a = _fake_llama_proc(rss_gb=10.0)
    proc_b = _fake_llama_proc(rss_gb=20.0)
    routes = {"/metrics": LLAMA_METRICS_SAMPLE.encode(),
             "/props": b"{}", "/slots": b"[]"}
    # Target A: process_iter yields only proc_a.
    with patch("app.psutil.process_iter", return_value=[proc_a]), \
         patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(routes)):
        result_a = collect_llama_metrics("http://127.0.0.1:8080",
                                        state_key="target-a")
    assert result_a["process"]["rss_gb"] == 10.0
    assert app_mod._llama_proc["target-a"] is proc_a
    # Target B: process_iter yields only proc_b (a different handle).
    with patch("app.psutil.process_iter", return_value=[proc_b]), \
         patch("app.urllib.request.urlopen",
               side_effect=make_llama_server(routes)):
        result_b = collect_llama_metrics("http://127.0.0.1:8081",
                                        state_key="target-b")
    assert result_b["process"]["rss_gb"] == 20.0
    assert app_mod._llama_proc["target-b"] is proc_b
    # No cross-target clobbering: target-a's handle is still proc_a.
    assert app_mod._llama_proc["target-a"] is proc_a
    assert app_mod._llama_proc["target-a"] is not app_mod._llama_proc["target-b"]


# ---------------------------------------------------------------------------
# T7 — self-observability (/health) + log rate limiting
# ---------------------------------------------------------------------------

def test_health_endpoint_reports_last_cycle_and_errors(monkeypatch):
    """GET /health returns JSON with last_cycle_duration_s (a number) and
    last_errors (a dict reflecting a seeded per-target error)."""
    import app as app_mod
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    collector.start()
    try:
        # Wait for a cycle to complete (so _last_cycle_duration is set).
        for _ in range(200):
            if collector._last_cycle_duration is not None:
                break
            time.sleep(0.05)
        assert collector._last_cycle_duration is not None, \
            "a cycle did not complete in time"
        # Seed a per-target error (an upstream that is down).
        collector._record_error("apollo", "down (fetch returned None)")
        # GET /health via the Flask test client.
        monkeypatch.setattr(app_mod, "collector", collector)
        resp = app_mod.app.test_client().get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        # last_cycle_duration_s is a number (>= 0).
        assert isinstance(body["last_cycle_duration_s"], (int, float))
        assert body["last_cycle_duration_s"] >= 0
        # last_errors is a dict reflecting the seeded per-target error.
        assert isinstance(body["last_errors"], dict)
        assert body["last_errors"]["apollo"] == "down (fetch returned None)"
        # status is "degraded" (a target has a recent error).
        assert body["status"] == "degraded"
    finally:
        collector.stop()
        collector.join(timeout=2)


def test_log_rate_limiting_suppresses_repeat_failures(caplog):
    """A persistently-failing target logs the exception the first time,
    then suppresses it within the cooldown (no every-cycle flood); a
    recovery logs once."""
    import logging

    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=1.0, llama_enabled=False)
    # Do not start the collector (we test _record_error/_clear_error directly).
    with caplog.at_level(logging.DEBUG, logger="app"):
        # First failure: log the exception (with traceback).
        collector._record_error("apollo", "boom", exc=Exception("boom"),
                               log_msg="Error collecting upstream apollo")
        # Subsequent failures (within the cooldown): suppressed (no flood).
        collector._record_error("apollo", "boom", exc=Exception("boom"),
                               log_msg="Error collecting upstream apollo")
        collector._record_error("apollo", "boom", exc=Exception("boom"),
                               log_msg="Error collecting upstream apollo")
        # Simulate a cooldown expiry (the last log was > 60 s ago).
        collector._last_error_log_ts["apollo"] = time.monotonic() - 61.0
        # After the cooldown: a single "still failing" warning is logged.
        collector._record_error("apollo", "boom", exc=Exception("boom"),
                               log_msg="Error collecting upstream apollo")
        # Recovery: log once.
        collector._clear_error("apollo", log_msg="Upstream apollo")
    # First failure: the exception is logged (with traceback / exc_info).
    first_failure = [r for r in caplog.records
                    if "Error collecting upstream apollo" in r.getMessage()
                    and r.exc_info is not None]
    assert len(first_failure) == 1, \
        f"expected 1 logged exception, got {len(first_failure)}"
    # Only 2 "Error collecting upstream apollo" logs total: the first
    # (exception) and the post-cooldown "still failing" (a warning, no exc).
    all_failure_logs = [r for r in caplog.records
                       if "Error collecting upstream apollo" in r.getMessage()]
    assert len(all_failure_logs) == 2, \
        f"expected 2 failure logs (first + post-cooldown), " \
        f"got {len(all_failure_logs)}"
    # The post-cooldown log is a "still failing" warning (no exc_info).
    still_failing = [r for r in caplog.records
                    if "still failing" in r.getMessage()]
    assert len(still_failing) == 1, \
        f"expected 1 'still failing' log, got {len(still_failing)}"
    assert still_failing[0].exc_info is None, \
        "the 'still failing' log should not carry an exception"
    # The recovery is logged once.
    recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
    assert len(recovered) == 1, f"expected 1 recovery log, got {len(recovered)}"


# ---------------------------------------------------------------------------
# T9: app factory + explicit bootstrap
# ---------------------------------------------------------------------------

def _t9_settings(**overrides):
    """A minimal settings dict (the shape resolve_settings returns) for the
    T9 bootstrap tests. LLM is disabled by default so the test collector
    does not probe any LLM server."""
    s = {
        "host": "127.0.0.1",
        "port": 5000,
        "poll_interval": 0.5,
        "llama_url": "http://127.0.0.1:5001",
        "vllm_url": "http://127.0.0.1:8000",
        "litellm_url": "http://127.0.0.1:4000",
        "litellm_token": None,
        "strata_url": "http://127.0.0.1:8080",
        "llm_backend": "auto",
        "default_view": "local",
        "llama_enabled": False,
        "llm_targets": [],
        "upstreams": [],
    }
    s.update(overrides)
    return s


def test_apply_settings_starts_collector(monkeypatch):
    """apply_settings applies every resolved setting to the module-level
    collector and starts it, so the collector runs on the resolved settings
    (not the built-in defaults it was created with)."""
    import time
    import app as app_mod
    from app import MetricsCollector

    fresh = MetricsCollector(poll_interval=99.0)  # built-in default (to detect override)
    monkeypatch.setattr(app_mod, "collector", fresh)
    try:
        app_mod.apply_settings(_t9_settings(poll_interval=0.5,
                                          llama_url="http://127.0.0.1:5001"))
        time.sleep(0.3)  # give the thread a moment to start
        assert fresh.is_alive(), "apply_settings should start the collector"
        assert fresh.poll_interval == 0.5, "poll_interval should be the resolved setting"
        assert fresh.llama_url == "http://127.0.0.1:5001", "llama_url should be the resolved setting"
    finally:
        fresh.stop()
        fresh.join(timeout=2)


def test_wsgi_entry_applies_config(monkeypatch):
    """The lazy before_request hook resolves the settings from the config
    (no CLI) and applies them on the FIRST request, so a WSGI/import-based
    serve runs on the config's settings, not built-in defaults. The
    collector is NOT started at import time (only on the first request)."""
    import app as app_mod
    from app import MetricsCollector

    # Simulate a config with a non-default poll_interval + llama disabled.
    fake_config = {"poll_interval": 3.3, "llama_enabled": False}
    monkeypatch.setattr(app_mod, "load_config",
                        lambda path=None, explicit=False: fake_config)
    monkeypatch.setattr(app_mod, "_settings_applied", False)

    fresh = MetricsCollector(poll_interval=99.0)  # built-in default (to detect override)
    monkeypatch.setattr(app_mod, "collector", fresh)
    try:
        # Import time does NOT start the collector; trigger the lazy bootstrap
        # via a request (before_request -> _ensure_bootstrap -> apply_settings).
        app_mod.app.test_client().get("/api/metrics")
        assert fresh.poll_interval == 3.3, "poll_interval should be the config's setting (not the 99.0 built-in default)"
        assert fresh.llama_enabled is False, "llama_enabled should be the config's setting"
    finally:
        fresh.stop()
        fresh.join(timeout=2)


def test_main_bootstrap_idempotent(monkeypatch):
    """Calling apply_settings twice does not double-start the collector: the
    second start() is a no-op via the RuntimeError guard (no crash, the
    collector keeps running, no second thread is spawned)."""
    import time
    import app as app_mod
    from app import MetricsCollector

    fresh = MetricsCollector(poll_interval=0.5)
    monkeypatch.setattr(app_mod, "collector", fresh)
    s = _t9_settings()
    try:
        app_mod.apply_settings(s)  # first call: starts the collector
        time.sleep(0.3)
        assert fresh.is_alive(), "first apply_settings should start the collector"
        app_mod.apply_settings(s)  # second call: no-op (no double-start, no crash)
        assert fresh.is_alive(), "collector should still be running after the second apply_settings"
    finally:
        fresh.stop()
        fresh.join(timeout=2)
