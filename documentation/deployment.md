# Deployment

## Security posture — read first

The dashboard has **no authentication** and binds **`0.0.0.0`** by default. Anyone who
can route to the port can read your hostnames, GPU/model names, LLM traffic rates and
`/api/metrics` payloads.

- Keep it on a trusted/private network, or front it with a reverse proxy that
  authenticates.
- Bind local-only when that is enough: `python app.py --host 127.0.0.1`.
- There is no CSRF/session surface (read-only GETs), but there is also no access
  control or TLS — terminate TLS at a proxy if you expose it.
- `litellm_token` / `RD_LITELLM_TOKEN` is the only credential the app handles, and it is
  only sent to your LiteLLM proxy's `/metrics` endpoint.

## systemd

Two files ship with the repo:

- [`resource-dashboard.service`](../resource-dashboard.service) — the unit template.
- [`install_service.sh`](../install_service.sh) — renders and installs it.

The unit uses `/opt/resource-dashboard` as its install path and `User=dashboard` as its
service account. Both are **placeholders that `install_service.sh` substitutes at
install time**, so the committed file contains no machine-specific values.

```bash
cd /path/to/resource_dashboard
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo bash install_service.sh
```

Overrides:

```bash
# install from (and run out of) a non-default directory
sudo INSTALL_DIR=/srv/resource_dashboard bash install_service.sh

# run the service as a specific unprivileged user
sudo SERVICE_USER=monitor bash install_service.sh
```

`install_service.sh` renders `/etc/systemd/system/resource-dashboard.service` from the
shipped unit (replacing the install path and `User=`), reloads systemd, enables and
restarts the service, then prints its status. It refuses to run as non-root and checks
that `.venv/bin/python` exists in the install directory first.

Useful after that:

```bash
systemctl status resource-dashboard
journalctl -u resource-dashboard -f
systemctl stop resource-dashboard
```

The unit starts the app with `python -m app --config <install>/config.yaml`, i.e. the
explicit bootstrap that resolves settings (env > CLI > config) and starts the collector
on the resolved values.

## Production WSGI serving

The dev loop is fine for a LAN dashboard, but for anything load-bearing serve the
`wsgi` entry point with a real WSGI server (`waitress` is pinned in
`requirements.txt` for exactly this):

```bash
waitress-serve --listen=0.0.0.0:5000 app:wsgi
```

Importing `app` under a WSGI server applies the resolved config at import time, so the
server honours `config.yaml` instead of running on built-in defaults.

Fleet/aggregating dashboards hold long-lived SSE connections per browser, so a
single-threaded server will starve `/api/metrics`: **upstream instances must run a
threaded server** (this build or newer). Older single-threaded builds queue or return
`503` for `/api/metrics` while a browser holds `/events`, which the aggregating
dashboard shows as **stale** (values held 15 s) or **down**.

## Rollout order for a fleet

Update **aggregating (fleet) dashboards first**, then the leaf dashboards. Fleet
dashboards understand both payload shapes (they shim an old-style `llm`-only payload
into a single synthetic target), whereas an old fleet dashboard reading a new leaf simply
ignores its `llms` array and keeps showing the `llm` primary alias.

Version skew in general: upstreams should run the same or an older build. Older builds
without `systems` still work — they just appear as a single local entry.
