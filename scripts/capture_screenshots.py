#!/usr/bin/env python3
"""Capture dashboard screenshots with populated trend charts.

Waits for real elapsed time (the CLI screenshot path fires at first paint, and
--virtual-time-budget never settles against a never-ending SSE stream), then
asserts the redaction actually held in the rendered DOM before writing files.

Usage: python capture.py <proxy-url> <outdir> [settle-seconds]
"""
from __future__ import annotations

import os
import pathlib
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8901/?__view=local"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/shots")
SETTLE = int(sys.argv[3]) if len(sys.argv) > 3 else 35
SCALE = int(os.environ.get("SCALE", "2"))

# Anything that must never appear in a publishable capture. The wrapper feeds
# every --redact key in here, so a redaction rule that silently misses fails the
# run instead of shipping a leaky image. (Canvas-drawn labels — chart legends —
# are not in the DOM; they are covered because the payload itself is scrubbed.)
FORBIDDEN = [t for t in os.environ.get("FORBIDDEN", "").split(",") if t]
FORBIDDEN += [t for t in os.environ.get("FORBIDDEN_EXTRA", "").split(",") if t]

# Reuse a Playwright-downloaded Chromium if one is cached; otherwise let
# Playwright resolve its own default.
_CACHE = pathlib.Path.home() / "Library/Caches/ms-playwright"
_CANDS = sorted(_CACHE.glob("chromium-*/chrome-mac*/Google Chrome for Testing.app"
                            "/Contents/MacOS/Google Chrome for Testing"), reverse=True)
BIN = _CANDS[0] if _CANDS else pathlib.Path("default")

shots = [
    ("dashboard-local.png", 1600, 1000, False),   # README hero
    ("dashboard-detail.png", 1500, 1000, True),   # full page, documentation/
]

with sync_playwright() as pw:
    exe = str(BIN) if BIN.exists() else None
    browser = pw.chromium.launch(headless=True, executable_path=exe)
    try:
        for name, w, h, full in shots:
            page = browser.new_page(viewport={"width": w, "height": h},
                                    device_scale_factor=SCALE)
            page.goto(URL, wait_until="load", timeout=30000)
            page.wait_for_timeout(SETTLE * 1000)

            html = page.content()
            leaks = [t for t in FORBIDDEN if t in html]
            if leaks:
                raise SystemExit(f"REFUSING TO WRITE {name}: still contains {leaks}")
            live = page.evaluate(
                "() => document.querySelectorAll('canvas').length"
                " + '|' + (document.body.innerText.match(/tok\\/s/g) || []).length"
            )
            path = OUT / name
            page.screenshot(path=str(path), full_page=full)
            status = "clean" if not leaks else "LEAK"
            print(f"{name:26s} {path.stat().st_size//1024:5d} KB  redaction={status}"
                  f"  canvases|tok/s={live}")
            page.close()
    finally:
        browser.close()
