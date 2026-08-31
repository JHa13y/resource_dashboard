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


def test_collector_llm_disabled_performs_no_llm_calls():
    """llama_enabled=False (--no-llama): no LLM HTTP calls, flag in snapshot."""
    from app import MetricsCollector

    collector = MetricsCollector(poll_interval=0.1, llama_enabled=False)
    collector._queue = queue.Queue(maxsize=3)
    with patch("app.urllib.request.urlopen") as fake_urlopen:
        fake_urlopen.side_effect = ConnectionRefusedError("LLM must not be scraped")
        collector.start()
        try:
            data = collector.get_latest(timeout=3.0)
            assert data is not None
            metrics = json.loads(data)
            assert metrics["llm_enabled"] is False
            assert metrics.get("llm") is None
        finally:
            collector.running = False
            collector.join(timeout=2)
        assert fake_urlopen.call_count == 0


def test_collector_llm_enabled_by_default():
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
            assert metrics["llm_enabled"] is True
            assert metrics["llm"]["status"] == "up"
            assert metrics["llm"]["model"] == "x.gguf"
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
    original_req_rates = app_mod._llama_req_rates
    app_mod._llama_prev = None
    app_mod._llama_last_good = None
    app_mod._llama_slot_prev = None
    app_mod._llama_decode_history = []
    app_mod._llama_req_rates.clear()
    yield
    app_mod._llama_prev = original_prev
    app_mod._llama_last_good = original_last_good
    app_mod._llama_slot_prev = original_slot_prev
    app_mod._llama_decode_history = original_decode_history
    app_mod._llama_req_rates.extend(original_req_rates)


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

        # Same task: cached 0 -> 800, processed 0 -> 20 over 2 s.
        # processed + cached reaches the 800-token prompt: prefill done,
        # per-request rate 800 tok / 2 s = 400 t/s replaces the delta.
        clock.advance(2.0)
        set_task(101, 20, 800, 30)
        third = collect_llama_metrics("http://127.0.0.1:9999")

    assert second["prompt_tokens_per_s"] == 0.0
    assert second["gen_tokens_per_s"] == 0.0
    assert third["prompt_tokens_per_s"] == 400.0
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
    (prompt size / prefill duration): a large prefill must not spike for
    one scrape and then drop to 0, and the window must evict the oldest
    sample after 5 requests."""
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
            state["slots"] = [
                _llama_slot_fixture(task, total, min(per_step * i, total), 0)
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


def test_collect_llama_prompt_rate_cleared_on_counter_reset():
    """A server restart (cumulative counters decrease) clears the
    per-request history: the gauge falls back to the live rate."""
    import app as app_mod
    from app import collect_llama_metrics

    app_mod._llama_req_rates.append(1000.0)  # pretend a request completed
    app_mod._llama_prev = {"ts": 1000.0, "gen": 1000.0, "prompt": 1000.0}
    with patch("app.urllib.request.urlopen", side_effect=make_llama_server({
        "/metrics": ("llamacpp:tokens_predicted_total 100\n"
                     "llamacpp:prompt_tokens_total 50\n").encode(),
        "/props": b"{}",
        "/slots": b"[]",
    })):
        result = collect_llama_metrics("http://127.0.0.1:9999")

    # Restart detected (prev 1000 -> now 100): history cleared, no mean
    assert list(app_mod._llama_req_rates) == []
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

    original_prev = app_mod._vllm_prev
    original_last_good = app_mod._vllm_last_good
    original_config_cache = app_mod._vllm_config_cache
    original_req_rates = app_mod._vllm_req_rates
    app_mod._vllm_prev = None
    app_mod._vllm_last_good = None
    app_mod._vllm_config_cache = None
    app_mod._vllm_req_rates.clear()
    yield
    app_mod._vllm_prev = original_prev
    app_mod._vllm_last_good = original_last_good
    app_mod._vllm_config_cache = original_config_cache
    app_mod._vllm_req_rates.extend(original_req_rates)


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

    app_mod._vllm_prev = {"ts": 1000.0, "gen": 10000.0, "prompt": 10000.0}
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
    import app as app_mod
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
    """elapsed > 5 s: rate is omitted (gauges still present)."""
    import app as app_mod
    from app import collect_vllm_metrics

    body1 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 100.0\n'
    body2 = 'vllm:generation_tokens_total{engine="0",model_name="M"} 160.0\n'

    clock = _FakeClock()
    with patch("app.urllib.request.urlopen",
               side_effect=make_vllm_server(body1.encode())), \
         patch("app.time.monotonic", clock):
        collect_vllm_metrics("http://127.0.0.1:9999")

    clock.advance(6.0)  # > 5 s
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
    app_mod._vllm_config_cache = None  # bypass 30s cache
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
    collector._queue = queue.Queue(maxsize=3)

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
    collector._queue = queue.Queue(maxsize=3)
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

    assert result == [{"index": 0, "name": "fake"}]
    no_real_nvidia.assert_not_called()


def test_gpu_metrics_jetson_mode_logged_once(jetson_env, monkeypatch, caplog):
    import app

    monkeypatch.setattr(app, "_jetson_mode_logged", False)
    with caplog.at_level(logging.INFO):
        app.collect_gpu_metrics()
        app.collect_gpu_metrics()

    msgs = [r.getMessage() for r in caplog.records]
    assert sum("GPU mode: jetson (SoC: orin)" in m for m in msgs) == 1
