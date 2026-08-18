"""Unit tests for the resource dashboard metrics collectors.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
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
    app_mod._llama_prev = None
    app_mod._llama_last_good = None
    app_mod._llama_slot_prev = None
    yield
    app_mod._llama_prev = original_prev
    app_mod._llama_last_good = original_last_good
    app_mod._llama_slot_prev = original_slot_prev


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
