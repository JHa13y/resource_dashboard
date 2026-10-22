#!/usr/bin/env bash
#
# install_service.sh — install resource-dashboard as a systemd service.
#
# Usage:
#   sudo bash install_service.sh   (or: sudo sh install_service.sh)
#
# The .service file next to this script is a TEMPLATE: `User=dashboard` and the
# /opt/resource-dashboard paths are placeholders that are substituted here at
# install time, so the committed unit carries no machine-specific values. The
# rendered unit is written to /etc/systemd/system/<service>.service.
#
# Overridable environment variables:
#   INSTALL_DIR   project directory (default: the directory holding this script)
#   SERVICE_USER  account the service runs as (default: the invoking user)
#
#   sudo INSTALL_DIR=/srv/resource_dashboard bash install_service.sh
#   sudo SERVICE_USER=monitor bash install_service.sh
#
set -eu

INSTALL_DIR="${INSTALL_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SERVICE_NAME="resource-dashboard"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
SERVICE_TEMPLATE="${INSTALL_DIR}/${SERVICE_NAME}.service"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: please run as root (sudo bash install_service.sh)" >&2
    exit 1
fi

if [ ! -f "${SERVICE_TEMPLATE}" ]; then
    echo "ERROR: ${SERVICE_TEMPLATE} not found" >&2
    exit 1
fi

if [ ! -x "${INSTALL_DIR}/.venv/bin/python" ]; then
    echo "ERROR: ${INSTALL_DIR}/.venv/bin/python not found. Create the venv first:" >&2
    echo "  cd ${INSTALL_DIR} && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# A unit installed earlier with `systemctl link` is a symlink into the project
# directory; writing through it would overwrite the template in the repo.
if [ -L "${UNIT_FILE}" ]; then
    echo "==> Removing previously linked unit (${UNIT_FILE} -> $(readlink "${UNIT_FILE}"))"
    rm -f "${UNIT_FILE}"
fi

echo "==> Rendering ${UNIT_FILE} (user=${SERVICE_USER}, install=${INSTALL_DIR})"
# Escape the sed replacement metacharacters in the install path.
escaped_dir=$(printf '%s' "${INSTALL_DIR}" | sed -e 's/[&|]/\\&/g')
sed -e "s|/opt/resource-dashboard|${escaped_dir}|g" \
    -e "s|^User=dashboard\$|User=${SERVICE_USER}|" \
    "${SERVICE_TEMPLATE}" > "${UNIT_FILE}"

if grep -q 'User=dashboard$' "${UNIT_FILE}"; then
    echo "ERROR: could not substitute User= in ${UNIT_FILE}" >&2
    exit 1
fi

echo "==> Reloading systemd daemon"
systemctl daemon-reload

echo "==> Enabling service (start after boot)"
systemctl enable "${SERVICE_NAME}.service"

echo "==> Starting service"
systemctl restart "${SERVICE_NAME}.service"

echo ""
echo "Done. Service status:"
systemctl status "${SERVICE_NAME}.service" --no-pager || true
