# KernelGym + rLLM (Megatron) — Co-training quickstart

This example runs **rLLM** training with **KernelGym** as the reward / evaluation backend on the **same machine** (shared accelerators). The sections below are ordered as: **training Docker image** → **KernelGym services** → **training entry script**.

---

## Environment setup (Docker image)

Your team ships a self-contained build layout under **`training_env/`**:

| Path | Purpose |
|------|---------|
| `training_env/Dockerfile` | Image recipe (Ascend / CANN / rLLM stack). |
| `training_env/pkgs/` | Vendored **source** of dependency libraries used by the build. |
| `training_env/command.txt` | The **full `docker build` command** (paths, tags, and flags) as intended by the maintainers. |
| Other files in `training_env/` | Supporting assets referenced by the Dockerfile or build context. |

**Build the image**

If you already have **`ascend-cann900-rllm:latest`** on the machine, skip to [What runs where](#what-runs-where).

1. Open `training_env/command.txt` and run the command it contains exactly (working directory and context paths are defined there).
2. The resulting image is tagged **`ascend-cann900-rllm:latest`**.

After the image exists, run your jobs or an interactive container the way your cluster expects (device mounts, `ASCEND_*` env, etc.). Everything from **Prerequisites** onward assumes that runtime is available on the training host.

---

## What runs where

```
rLLM (train_kernelgym_megatron_qwen30b.sh)
  └── Hydra config: reward_model.server_url → KernelGym HTTP API
        └── KernelGym (start_all_with_monitor.sh)
              ├── Redis
              ├── FastAPI (evaluate / health / …)
              ├── worker monitor
              └── NPU workers (one per entry in GPU_DEVICES)
```

---

## Prerequisites

### 1. Repositories

You need both checkouts on the training host (paths are examples):


| Role                  | Typical path          |
| --------------------- | --------------------- |
| rLLM                  | `rllm-071/`           |
| KernelGym (this fork) | `kernelGym-npu-main/` |


**Scripts referenced below**


| Step            | Script                                                            |
| --------------- | ----------------------------------------------------------------- |
| Generate `.env` | `kernelGym-npu-main/scripts/auto_configure.sh`                    |
| Start services  | `kernelGym-npu-main/start_all_with_monitor.sh`                    |
| Training        | `rllm-071/examples/kernelgym/train_kernelgym_megatron_qwen30b.sh` |


### 2. Redis

KernelGym expects **redis-server** on the host (**redis-cli** is recommended for the startup helper).

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y redis-server redis-tools
```

**Note:** `start_all_with_monitor.sh` can spawn a local `redis-server` if Redis is not listening yet and `REDIS_HOST` is `localhost` / `127.0.0.1`. You still need the `redis-server` binary installed.

### 3. Port alignment (KernelGym API vs training)

`train_kernelgym_megatron_qwen30b.sh` sets:

```text
reward_model.server_url="http://127.0.0.1:8002"
```

So **API_PORT** in KernelGym’s `.env` must be **8002**, unless you change both sides to the same URL.

**About port `8000` and vLLM:** `scripts/auto_configure.sh` picks the first *free* ports from a candidate list (defaults include `8000`–`8009`). If something (e.g. a vLLM OpenAI server) already listens on `8000`, that port is skipped. For a stable layout with this training script, **pin `API_PORT=8002`** in `.env` after auto-config (see Quick start below).

---

## Quick start

### 1. Configure KernelGym (first time or after port changes)

```bash
cd /path/to/kernelGym-npu-main
bash scripts/auto_configure.sh --force
```

Open `.env` and confirm at least:

```bash
API_HOST=127.0.0.1          # or your reachable bind / LAN IP
API_PORT=8002               # must match reward_model.server_url
REDIS_HOST=localhost
REDIS_PORT=8001             # example; any free port is fine
METRICS_PORT=8003           # example
GPU_DEVICES=[0,1,2,3,4,5,6,7]   # adjust to your node
```

If `auto_configure.sh` wrote different ports, either edit `.env` or re-run with `--force` after freeing ports / adjusting the script’s candidate pool (see KernelGym repo `README.md`).

### 2. Start KernelGym (API + monitor + workers + Redis if needed)

```bash
cd /path/to/kernelGym-npu-main
chmod +x start_all_with_monitor.sh
./start_all_with_monitor.sh
```

Logs default to `kernelGym-npu-main/logs/` (`api_server.log`, `worker_monitor.log`, `worker_npu_*.log`).

**Useful flags** (see KernelGym `README.md`):

```bash
./start_all_with_monitor.sh --force-config      # re-run auto_configure with existing .env
./start_all_with_monitor.sh --use-indexed-ports # use PORT0, PORT1, … as candidates
```

### 3. Sanity check before training

```bash
curl -sS "http://127.0.0.1:8002/health" | head
curl -sS "http://127.0.0.1:8002/workers/status" | head
```

You should get JSON from `/health` and worker entries from `/workers/status`. If not, tail the log files under `logs/`.

### 4. Launch training

KernelGym processes are started in the **background** by `start_all_with_monitor.sh`, so you can run training from the same shell (or any other) once the sanity check passes.

```bash
cd /path/to/rllm-071
bash examples/kernelgym/train_kernelgym_megatron_qwen30b.sh
```

That script (re)starts Ray and submits `python3 -m examples.kernelgym.train_kernelgym` with the Hydra overrides defined in the shell file.

---

## One-liner recap

```bash
# KernelGym (background)
cd /path/to/kernelGym-npu-main && bash scripts/auto_configure.sh --force
# edit .env: API_PORT=8002, fix GPU_DEVICES
./start_all_with_monitor.sh

# rLLM training
cd /path/to/rllm-071 && bash examples/kernelgym/train_kernelgym_megatron_qwen30b.sh
```

