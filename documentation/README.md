# Documentation

Detailed reference material for the Resource & GPU Utilization Dashboard. The
[top-level README](../README.md) covers what the project is and the quick start;
everything else lives here.

| Document | Contents |
|---|---|
| [configuration.md](configuration.md) | `config.yaml` keys, precedence, environment variables, full CLI reference, multi-LLM (`llms:`) and `upstreams:` rules |
| [deployment.md](deployment.md) | systemd service, `install_service.sh`, production WSGI serving, security posture, fleet rollout order |
| [gpu-detection.md](gpu-detection.md) | NVIDIA / AMD / JetPack detection chains, `amd-smi` & `rocm-smi` output contracts and units, field sources, on-device smoke script |
| [llm-backends.md](llm-backends.md) | What each inference backend exposes (llama.cpp, vLLM, LiteLLM, Strata) and how each metric is derived |
| [api.md](api.md) | HTTP endpoints (`/`, `/events`, `/api/metrics`, `/health`), payload keys, payload-shape history |
| [architecture.md](architecture.md) | Process model, collection cadence, SSE fan-out hub, fleet relay model, known limitations |
| [troubleshooting.md](troubleshooting.md) | Symptom → cause → fix table, log behaviour, self-observability pointers |

Placeholders used throughout: `192.168.1.10` / `192.168.1.20` / `192.168.1.30` and
`node-a.example.com` stand in for *your* hosts; `127.0.0.1` means "same machine".
