#!/usr/bin/env python3
"""Redacting reverse proxy for taking publishable screenshots.

Serves a local copy of a live dashboard with identifying strings replaced
(hostnames, fleet peer names, model names), so a screenshot of the page cannot
leak private infrastructure into a public README.

Usage:
    REDACT="athena=node-a,apollo=node-b" \
        python3 redact_proxy.py --upstream http://192.168.5.220:5000 --port 8899

# then capture a given view by adding the proxy-only __view parameter:
#   chrome --headless=new --screenshot=shot.png "http://127.0.0.1:8899/?__view=local"

Only GET is served. HTML / JSON / SSE bodies are rewritten; everything else
(Chart.js, etc.) is streamed through untouched.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:5000")
# "old=new,old2=new2" — longest key wins so "Qwen3.8" cannot eat "Qwen3.8-27B".
RULES: list[tuple[str, str]] = []
for pair in os.environ.get("REDACT", "").split(","):
    if "=" in pair:
        old, _, new = pair.partition("=")
        if old.strip():
            RULES.append((old.strip(), new.strip()))
RULES.sort(key=lambda p: -len(p[0]))

TEXT_TYPES = ("text/html", "application/json", "text/event-stream", "text/plain")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"



def scrub(text: str) -> str:
    for old, new in RULES:
        text = text.replace(old, new)
    return text


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:  # noqa: N802
        # __view is proxy-only (selects the initial view); never forwarded.
        path, _, query = self.path.partition("?")
        view = next((kv.partition("=")[2] for kv in query.split("&")
                     if kv.startswith("__view=")), "")
        keep = [kv for kv in query.split("&") if kv and not kv.startswith("__view=")]
        fwd = path + ("?" + "&".join(keep) if keep else "")
        req = urllib.request.Request(f"{UPSTREAM}{fwd}", headers={"User-Agent": UA})
        try:
            up = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            return
        except Exception as e:  # upstream unreachable
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy: {e}".encode())
            return

        ctype = up.headers.get("Content-Type", "")
        self.send_response(up.status)
        for h in ("Cache-Control", "X-Accel-Buffering"):
            if h in up.headers:
                self.send_header(h, up.headers[h])

        if not any(t in ctype for t in TEXT_TYPES):
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            while True:
                chunk = up.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
            up.close()
            return

        if "text/event-stream" in ctype:
            # SSE has no length; frame it with connection close so the client
            # never waits for a body end that is never announced.
            self.send_header("Content-Type", ctype)
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            # Stream, rewriting each SSE data line as it arrives.
            buf = b""
            while True:
                try:
                    piece = up.readline()
                except Exception:
                    break
                if not piece:
                    break
                buf += piece
                if buf.endswith(b"\n\n"):
                    try:
                        out = scrub(buf.decode("utf-8", "replace")).encode()
                    except Exception:
                        out = buf
                    self.wfile.write(out)
                    self.wfile.flush()
                    buf = b""
            up.close()
            return

        body = up.read().decode("utf-8", "replace")
        up.close()
        body = scrub(body)
        if "text/html" in ctype and view:
            # The page's initial view is server-rendered (index.html:
            # `selected: '{{ default_view }}'`). Rewriting that is deterministic;
            # a synthetic tab click lost a race with the page's own init.
            body = re.sub(r"selected:\s*'[A-Za-z0-9_-]+'",
                          "selected: '%s'" % view, body, count=1)
        data = body.encode()
        # Buffered text must declare Content-Length, or an HTTP/1.1 client waits
        # forever for a body end that is never announced (this is what hung
        # headless Chrome on the page load).
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args()
    globals()["UPSTREAM"] = a.upstream
    print(f"proxy http://127.0.0.1:{a.port} -> {a.upstream} | rules={len(RULES)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
