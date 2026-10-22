# GPU detection and field sources

Three vendor chains, probed automatically at first GPU collection — no configuration.
Mixed-vendor boxes (NVIDIA + AMD) are supported: both card sets are shown, `index`
renumbered `0..n-1` across vendors (NVIDIA cards first, in their original order), each card
tagged with `vendor` (`nvidia` / `amd`) by the merge step. The **name** carries the chip
vendor and the board/package vendor (e.g. `NVIDIA GeForce RTX 3090 (Gigabyte)`,
`AMD Radeon AI PRO R9700 (ASRock)`), so the UI title is just the name. No two entries in
one payload may claim the same PCI BDF: a colliding BDF means a mis-resolved card, so it is
warned about and the duplicate dropped rather than silently rendered.

On NVIDIA-only boxes the AMD probe is negative-cached for 30 s so the `amd-smi` subprocess
isn't spawned every second, while a freshly loaded `amdgpu` driver is still picked up.

All tiers emit the same per-GPU dict shape the frontend expects, so unsupported fields
render as `–` rather than `0`. **`N/A` is always treated as absent, never zero.**

## NVIDIA

`pynvml` (`nvidia-ml-py`) → fallback to parsing `nvidia-smi --query-gpu=… --format=csv`.
If `nvidia-ml-py` fails to install (missing NVIDIA development headers) the CLI fallback
is automatic.

Card name hygiene: BDF/slot-shaped names are rejected, the tool's own product name wins
over `lspci`, board/package vendor is folded in when readable (e.g.
`NVIDIA RTX 3090 (Gigabyte)`), and the name lookup is TTL-cached.

## AMD — `amd-smi` → `rocm-smi` → sysfs

| Tier | Tool | Invocation | Notes |
|---|---|---|---|
| 1 (preferred) | `amd-smi` (ROCm 6.x+) | `amd-smi monitor --power-usage --temperature --gfx --vram-usage --json` | JSON is the stable contract. VRAM in **MB** (unit-aware: the `unit` field is respected). `hotspot_temperature` preferred, `memory_temperature` fallback. `gfx` = utilization %, `gfx_clock` = MHz, `power_usage` = W. |
| 2 (legacy) | `rocm-smi` (ROCm 5.x) | `rocm-smi --showuse --showmeminfo vram --showtemperatures --csv` | Columns matched by **header name, never position**. VRAM in **bytes** (1024× difference vs `amd-smi`). `GPU use (%)`, `VRAM Total/Used Memory (B)`, first `Temperature (Sensor…)` column, `…Graphics Package Power (W)`. |
| 3 (last resort) | sysfs | `/sys/class/drm/cardN/device/{gpu_busy_percent,mem_info_vram_used,mem_info_vram_total,temperature1}` | No clock/power. amdgpu discriminator = co-presence of all three attributes. Temperature in **milli-°C**. |

Card name / board vendor come from `lspci` (via the sysfs `device` → PCI BDF symlink).
The AMD BDF is resolved from the tool's own data (`amd-smi` `bus_id` → `rocm` card index →
vendor-filtered sysfs) with a colliding-BDF merge guard.

### `amd-smi` output contract

`amd-smi` is the official CLI replacement for `rocm-smi`; it ships with the
`amd-smi-lib` package as part of the standard ROCm install (since ~ROCm 6.0, binary under
`/opt/rocm/bin`). Every subcommand accepts `--json` / `--csv`.

- **JSON** (both `monitor` and `metric`): top level is an **array**, one object per GPU
  keyed by `"gpu"` (int). Numeric leaves are wrapped — `{"value": 140, "unit": "W"}` —
  and must be unwrapped; `"N/A"` marks a missing sensor.
- **CSV**: `gpu` first, then one flat snake_case column per leaf key. Column **order** is
  insertion order and can shift between versions — index by header name.
- Fields used: `gfx_activity` (%), `used_vram` / `total_vram` (MB), `hotspot` °C (fallback
  `edge`, which is `N/A` on many datacenter cards), `socket_power` / `power_usage` (W).
  `monitor` spells these `gfx`, `vram_used`/`vram_total`, `hotspot_temperature`,
  `power_usage`.
- Avoid in the scrape path: human-readable tables (they change by platform — XCP column,
  dynamic `GTT_USAGE`/`VRAM_USAGE` header) and `monitor --process` (emits a second CSV block).
- `metric --csv` on some virtualized variants accepts only one metric group per call;
  one group per call is the safe form on older builds.
- Metrics have a built-in cache (`AMDSMI_GPU_METRICS_CACHE_MS`, default 100 ms) — relevant
  if you poll faster than 10 Hz.
- With MI300X CPX/NPS4 partitioning several logical GPU rows share one physical card; only
  the primary XCP reports power/temp/utilization.

`rocm-smi` vs `amd-smi` differences a parser must handle:

| Aspect | `rocm-smi` | `amd-smi` |
|---|---|---|
| First CSV column | `device` = `card0`, `card1`… | `gpu` = `0`, `1`… |
| Utilization | `GPU use (%)` | `gfx_activity` (metric) / `gfx` (monitor) |
| VRAM | `VRAM Total Memory (B)` / `VRAM Total Used Memory (B)` — **bytes** | `total_vram` / `used_vram` — **MB** |
| Temperature | `Temperature (Sensor edge/junction/memory) (C)` | `edge` / `hotspot` / `mem` (metric); `hotspot_temperature` / `memory_temperature` (monitor) |
| Power | `…Socket Graphics Package Power (W)` | `socket_power` / `power_usage` (W) |
| JSON | flat-ish, no unit wrappers | array of objects with `{"value","unit"}` wrappers |
| Missing data | `N/A` | `N/A` (very common: `edge` on datacenter GPUs, `decoder`, fan on passive cards) |

Sources for the contract above: [AMD SMI CLI usage (ROCm docs, latest)](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html),
[AMD SMI 6.0.2 docs](https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.0.2/) (CLI ships with `amd-smi-lib`),
[AMD SMI 7.1.0 changelog](https://rocm.docs.amd.com/projects/amdsmi/en/docs-7.1.0/reference/changelog.html) (real `metric` key/value output),
[New Relic nri-flex amd-smi example](https://github.com/newrelic/nri-flex/blob/master/examples/amd-smi-gpu-monitoring.yml)
(third-party production parser corroborating the CSV header names),
[llama-swap monitor_unix_test.go](https://github.com/mostlygeek/llama-swap/blob/60226b63/internal/perf/monitor_unix_test.go)
(real `rocm-smi --csv` headers).

AMD throttle badges are rendered but always report *no throttle* on the AMD path — none of
the three tiers exposes a signal comparable to NVIDIA's throttle reason. Clock and power
are absent on the sysfs tier.

## JetPack / embedded (Orin, Thor)

JetPack boards are auto-detected and switch to a **sysfs-based** collector, because NVML is
unsupported on Orin (mostly "Not Supported" on Thor) and the Orin `nvidia-smi` stub reports
plausible-looking junk (0 % utilization, N/A memory).

**Detection** (checked once at first GPU collection, in order):

1. Device-tree `model` contains both `NVIDIA` and one of `jetson` / `orin` / `thor`
   (case-insensitive — the NVIDIA co-occurrence keeps custom aarch64 boards from matching).
2. Device-tree `compatible` contains `nvidia,tegra` (the stable SoC discriminator; covers
   vendor boards and model-string gaps).
3. `/etc/nv_tegra_release` first line matches `R\d+`.
4. The kernel release string contains `tegra` (visible inside containers where the
   device-tree is hidden).

**Field sources on JetPack:**

| Field | Source |
|---|---|
| Utilization % | `/sys/class/devfreq/<nvgpu>/device/load` (any SoC exposing an nvgpu sysfs); on Thor (no `load` file) from `nvidia-smi --query-gpu=utilization.gpu` |
| Memory (unified) | `/proc/meminfo` — `MemTotal` / `MemAvailable` of the single DRAM pool shared by CPU, OS and GPU |
| Temperature | `/sys/class/thermal/thermal_zone*/temp` of the zone whose `type` is `gpu-thermal` (matched by type, never by index; `−256000` "no sensor" readings are skipped) |
| Graphics clock | devfreq node `cur_freq` (Orin: e.g. `17000000.gpu`; Thor: `gpu-gpc-0`) |
| Name | SoC (`NVIDIA Thor` / `NVIDIA Orin`), then the device-tree model string, then `NVIDIA Jetson` |

**Intentionally absent on JetPack** (the UI renders `–`): power draw / power limit,
thermal-throttle badges, PCIe bus id, board vendor. There is no reliable root-free source
for rail power without a persistent `tegrastats` process; NVML power is Not Supported.

**Unified memory semantics.** JetPack has one memory pool, so the GPU card's gauge
(labeled **"Unified mem %"**, tooltip "Shared system memory (CPU+OS+GPU)") shows the same
`MemTotal`/`MemAvailable` as the system memory card and overlaps with it. Model weights and
the vLLM KV cache are CUDA allocations in that pool — starting or stopping a vLLM server
moves the gauge by roughly (weights + preallocated KV cache). The pool therefore looks
"partly full" with no GPU load; that is the OS working set, not the GPU.

**Behaviour notes:**

- The nvgpu devfreq node appears a few seconds after boot. Until it does, utilization and
  clock show `–` (never a false 0) and fill in automatically — no restart needed.
- On Thor the utilization read shells out to `nvidia-smi` with a 2 s timeout. The collector
  runs sequentially, so a wedged `nvidia-smi` can stall a whole snapshot (CPU/memory/LLM
  included) by up to 2 s per tick; SSE keeps serving the last good payload meanwhile. Same
  failure class as the desktop `nvidia-smi` path.
- Containers (e.g. CUDA images): the device-tree is usually hidden, so detection falls back
  to `nv_tegra_release` / the `-tegra` kernel string, and devfreq/thermal sysfs may be
  absent unless the host's `/sys` is mounted. The card degrades to whatever fields it can
  source (often just the unified-memory bar) rather than failing.

### On-device smoke script

Run on the JetPack host to see exactly what the dashboard will read:

```sh
#!/bin/sh
# jetpack_gpu_smoke.sh — print the inputs the dashboard parses
echo "--- /proc/device-tree/model:"
cat /proc/device-tree/model 2>/dev/null | tr '\0' ' '; echo
echo "--- /proc/device-tree/compatible:"
cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' '; echo
echo "--- /etc/nv_tegra_release:"
head -1 /etc/nv_tegra_release 2>/dev/null || echo "(absent)"
echo "--- devfreq nodes:"
ls /sys/class/devfreq/ 2>/dev/null || echo "(absent)"
for d in /sys/class/devfreq/*gpu*; do
    [ -d "$d" ] || continue
    echo "  $d: cur_freq=$(cat $d/cur_freq 2>/dev/null) load=$(cat $d/device/load 2>/dev/null || echo absent)"
done
echo "--- gpu-thermal zones:"
for z in /sys/class/thermal/thermal_zone*; do
    [ -f "$z/type" ] || continue
    [ "$(cat $z/type)" = "gpu-thermal" ] && echo "  $z: $(cat $z/temp) mC"
done
echo "--- nvidia-smi utilization:"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi unavailable)"
echo "--- meminfo (unified pool):"
grep -E 'MemTotal|MemAvailable' /proc/meminfo
```
