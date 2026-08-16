#!/usr/bin/env bash
#
# install_service.sh — install resource-dashboard as a systemd service.
#
# Usage:
#   sudo bash install_service.sh   (or: sudo sh install_service.sh)
#
# The .service file lives next to this script in the project directory and is
# linked into /etc/systemd/system via `systemctl link`, so the canonical copy
# always stays with the code.
#
set -eu

INSTALL_DIR="/home/joshua/resource_dashboard"
SERVICE_NAME="resource-dashboard"
SERVICE_FILE="${INSTALL_DIR}/${SERVICE_NAME}.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: please run as root (sudo bash install_service.sh)" >&2
    exit 1
fi

if [ ! -f "${SERVICE_FILE}" ]; then
    echo "ERROR: ${SERVICE_FILE} not found" >&2
    exit 1
fi

if [ ! -x "${INSTALL_DIR}/.venv/bin/python" ]; then
    echo "ERROR: ${INSTALL_DIR}/.venv/bin/python not found. Create the venv first:" >&2
    echo "  cd ${INSTALL_DIR} && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "==> Linking service file"
# `systemctl link` creates a symlink in /etc/systemd/system pointing at the
# absolute path of the service file, keeping it in place in the project dir.
systemctl link "${SERVICE_FILE}"

echo "==> Reloading systemd daemon"
systemctl daemon-reload

echo "==> Enabling service (start after boot)"
systemctl enable "${SERVICE_NAME}.service"

echo "==> Starting service"
systemctl restart "${SERVICE_NAME}.service"

echo ""
echo "Done. Service status:"
systemctl status "${SERVICE_NAME}.service" --no-pager || true
