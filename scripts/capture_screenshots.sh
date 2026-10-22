#!/usr/bin/env bash
#
# capture_screenshots.sh — screenshot a live dashboard for the README/docs,
# optionally redacting identifying strings so a picture cannot leak private
# infrastructure into a public repository.
#
# Why a proxy: the dashboard prints its own hostname in the page title, its peer
# names in the Fleet strip, and model names in the LLM cards. A screenshot of a
# real box therefore publishes all of that. The proxy serves a rewritten copy
# (HTML, /api/metrics and the SSE frames), and the capture fails loudly if any
# value you asked to redact is still present in the rendered DOM.
#
# Usage:
#   scripts/capture_screenshots.sh --url http://192.168.1.10:5000 \
#       [--out documentation/assets] [--view local] [--settle 35] \
#       [--redact "realhost=node-a,peer=node-b,MyModelName=local-model-a"]
#
# Notes:
#   --redact  comma-separated old=new pairs; every OLD string is also treated as
#             forbidden in the output, so a missed rule fails the run.
#   --view    which system tab to show (local | fleet | <peer name>).
#   --settle  seconds of live data to wait for before capturing (trend charts
#             need real elapsed time; the LLM charts also need real traffic —
#             this script generates none on purpose).
#
# Requires python3 + Playwright. If the module is missing, a throwaway venv is
# created under $TMPDIR (never in your project); install its browser once with
#   $TMPDIR/pw-dashboard-venv/bin/playwright install chromium
#
set -eu

URL="" OUT="documentation/assets" VIEW="local" SETTLE=35 REDACT="" PORT=8391
while [ $# -gt 0 ]; do
    case "$1" in
        --url)    URL=$2; shift 2 ;;
        --out)    OUT=$2; shift 2 ;;
        --view)   VIEW=$2; shift 2 ;;
        --settle) SETTLE=$2; shift 2 ;;
        --redact) REDACT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$URL" ] || { echo "ERROR: --url is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)

# --- python with playwright -------------------------------------------------
PY=$(command -v python3)
if ! "$PY" -c "import playwright" 2>/dev/null; then
    PWVENV="${TMPDIR:-/tmp}/pw-dashboard-venv"
    [ -x "$PWVENV/bin/python" ] || { echo "==> creating $PWVENV"; python3 -m venv "$PWVENV"; }
    "$PWVENV/bin/python" -c "import playwright" 2>/dev/null || {
        echo "==> installing playwright (throwaway venv)"
        "$PWVENV/bin/pip" install -q --disable-pip-version-check playwright
    }
    PY="$PWVENV/bin/python"
fi

# --- what must never appear in the output -----------------------------------
FORBIDDEN=$(printf '%s' "$REDACT" | tr ',' '\n' | cut -d= -f1 | grep -v '^$' | paste -sd, -)

# --- proxy lifecycle --------------------------------------------------------
PROXY_PID=""
cleanup() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> proxy :$PORT -> $URL (rules: $(printf '%s' "$REDACT" | awk -F, '{print NF}'))"
REDACT="$REDACT" "$PY" "$HERE/redact_proxy.py" --upstream "$URL" --port "$PORT" \
    >"${TMPDIR:-/tmp}/redact_proxy.log" 2>&1 &
PROXY_PID=$!

ready=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)" = "200" ] \
        && { ready=1; break; }
    sleep 1
done
[ -n "$ready" ] || { echo "ERROR: proxy did not come up:"; cat "${TMPDIR:-/tmp}/redact_proxy.log"; exit 1; }

# --- capture ----------------------------------------------------------------
mkdir -p "$OUT"
FORBIDDEN="$FORBIDDEN" "$PY" "$HERE/capture_screenshots.py" \
    "http://127.0.0.1:$PORT/?__view=$VIEW" "$OUT" "$SETTLE"

echo "==> written to $OUT"
