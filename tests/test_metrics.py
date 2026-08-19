"""Unit tests for the resource dashboard metrics collectors.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import urllib.error
from pathlib import Path
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
    from app import _gpu_cli, _lspci_vendor_map

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
# lspci vendor lookup tests
# ---------------------------------------------------------------------------

def test_lspci_vendor_map_parses_subsystem():
    from app import _lspci_vendor_map, _parse_lspci_subsystem

    # Machine-readable ``lspci -vm`` format uses a tab-separated SVendor line
    lspci_vm = (
        "Slot:\t01:00.0\n"
        "Class:\tVGA compatible controller\n"
        "Vendor:\tNVIDIA Corporation\n"
        "Device:\tGA102 [GeForce RTX 3080]\n"
        "SVendor:\tGigabyte Technology Co., Ltd\n"
        "SDevice:\tDevice 1234\n"
    )
    assert _parse_lspci_subsystem(lspci_vm) == "Gigabyte"

    # Verbose ``lspci -v`` format has a Subsystem line instead
    lspci_subsys = (
        "01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3080]\n"
        "        Subsystem: Gigabyte Technology Co., Ltd Device 1234\n"
    )
    assert _parse_lspci_subsystem(lspci_subsys) == "Gigabyte"

    # Test Dell format
    lspci_dell = (
        "02:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4070 Ti]\n"
        "        Subsystem: Dell Device 5678\n"
    )
    assert _parse_lspci_subsystem(lspci_dell) == "Dell"

    # Neither SVendor nor Subsystem present → empty string so caller fallbacks kick in
    lspci_none = (
        "Slot:\t01:00.0\n"
        "Class:\tVGA compatible controller\n"
        "Vendor:\tNVIDIA Corporation\n"
        "Device:\tGA102 [GeForce RTX 3080]\n"
    )
    assert _parse_lspci_subsystem(lspci_none) == ""


def test_lspci_vendor_map_not_found():
    from app import _lspci_vendor_map, _lspci_cache

    import app as app_mod
    original = app_mod._lspci_cache
    try:
        app_mod._lspci_cache = None  # force fresh call
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            mapping = _lspci_vendor_map(["0000:01:00.0"])
        assert mapping == {}
    finally:
        app_mod._lspci_cache = original


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
        # Seed the lspci cache so _gpu_pynvml picks up a vendor name
        app_mod._lspci_cache = {"0000:01:00.0": "Dell"}
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
    assert gpu["board_vendor"] == "Dell"


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
        collector.running = False
        collector.join(timeout=2)


def test_collector_queue_bounded():
    """When the queue is full, old data should be dropped."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    # Shrink queue for faster test
    collector._queue = queue.Queue(maxsize=3)
    collector.start()
    try:
        time.sleep(0.6)  # let it produce ~6 items (queue max is 3)
        # Drain all items
        items = []
        while not collector._queue.empty():
            items.append(collector._queue.get_nowait())
        assert len(items) <= 3
    finally:
        collector.running = False
        collector.join(timeout=2)


def test_collector_llama_disabled_performs_no_llama_calls():
    """llama_enabled=False (--no-llama): no llama HTTP calls, flag in snapshot."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    collector._queue = queue.Queue(maxsize=3)
    with patch("app.urllib.request.urlopen") as fake_urlopen:
        fake_urlopen.side_effect = ConnectionRefusedError("llama must not be scraped")
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["llama_enabled"] is False
            assert metrics.get("llama") is None
        finally:
            collector.running = False
            collector.join(timeout=2)
        assert fake_urlopen.call_count == 0


def test_collector_llama_enabled_by_default():
    """Default (no flag): llama_enabled True and llama scraped as before."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1)
    assert collector.llama_enabled is True
    collector._queue = queue.Queue(maxsize=3)
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
            assert metrics["llama_enabled"] is True
            assert metrics["llama"]["status"] == "up"
            assert metrics["llama"]["model"] == "x.gguf"
        finally:
            collector.running = False
            collector.join(timeout=2)


# ---------------------------------------------------------------------------
# collect_gpu_metrics integration (prefers pynvml, falls back to CLI)
# ---------------------------------------------------------------------------

def test_collect_gpu_falls_back_to_cli():
    from app import collect_gpu_metrics, _gpu_pynvml, _gpu_cli

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
    app_mod._llama_props_cache = None
    yield
    app_mod._llama_props_cache = original


@pytest.fixture(autouse=True)
def reset_llama_prev():
    """Reset module-level scrape history around every test."""
    import app as app_mod

    original_prev = app_mod._llama_prev
    original_last_good = app_mod._llama_last_good
    original_slot_prev = app_mod._llama_slot_prev
    original_decode_history = app_mod._llama_decode_history
    app_mod._llama_prev = None
    app_mod._llama_last_good = None
    app_mod._llama_slot_prev = None
    app_mod._llama_decode_history = []
    yield
    app_mod._llama_prev = original_prev
    app_mod._llama_last_good = original_last_good
    app_mod._llama_slot_prev = original_slot_prev
    app_mod._llama_decode_history = original_decode_history


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
    app_mod._llama_proc = None
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
    l = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "n_prompt_tokens_processed": 3,
        "next_token": [
            {"has_next_token": True, "has_new_line": False,
             "n_remain": 16374, "n_decoded": 10},
            {"n_decoded": 5},
            "garbage",
        ],
    }])
    assert l[0]["decoded"] == 15
    assert l[0]["prompt_processed"] == 3
    # context = prompt + decoded
    assert l[0]["context_used"] == 25


def test_parse_llama_slots_remain():
    """n_remain (tokens left to generate) is summed over the next_token
    items; -1 (idle) or absent -> None."""
    from app import _parse_llama_slots

    # list shape: sum positive n_remain over the items
    l = _parse_llama_slots([{
        "id": 0, "n_ctx": 100, "n_prompt_tokens": 10,
        "next_token": [
            {"n_decoded": 10, "n_remain": 16000},
            {"n_decoded": 5, "n_remain": 374},
        ],
    }])
    assert l[0]["remain"] == 16374
    assert l[0]["decoded"] == 15

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
        app_mod._llama_props_cache = None
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

        # 67 tokens decoded + 67 prompt tokens processed over 2 s
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

        # Same task: cached 0 -> 800, processed 0 -> 20 over 2 s
        clock.advance(2.0)
        set_task(101, 20, 800, 30)
        third = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["prompt_tokens_per_s"] == 0.0
    assert second["gen_tokens_per_s"] == 0.0
    assert third["prompt_tokens_per_s"] == 410.0
    assert third["gen_tokens_per_s"] == 15.0


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
    app_mod._llama_props_cache = (
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

    app_mod._llama_prev = {
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
    app_mod._llama_props_cache = None
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
    app_mod._llama_decode_history = [
        (1000.0 + i, value) for i in range(n)
    ]


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
    assert len(app_mod._llama_decode_history) == 10


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

    assert len(app_mod._llama_decode_history) == 10


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
    assert app_mod._llama_proc is proc
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
    assert app_mod._llama_proc is big
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
    assert app_mod._llama_proc is server


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
        app_mod._llama_proc = dead
        result = collect_llama_metrics("http://127.0.0.1:9999")

    assert result["process"]["rss_gb"] == 22.0
    assert app_mod._llama_proc is fresh


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
    collector._queue = queue.Queue(maxsize=3)
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
            collector.running = False
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
    collector._queue = queue.Queue(maxsize=3)
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
            collector.running = False
            collector.join(timeout=2)

    hh = metrics["host_health"]
    assert hh["d_state_procs"] == 1
    assert hh["psi_mem_avg10"] is None or isinstance(hh["psi_mem_avg10"], float)
