# Installation Guide

This guide walks through the setup needed to run the Autonomous Local LLM SDLC
on consumer hardware. The reference target is **Windows 11 + WSL2 + Rocky
Linux 10**, but pointers for native Linux are included at the end.

If you just want to read what the project is and what it produces, see the
[README](README.md). This file is the *getting your machine ready* manual.

---

## 1. Hardware prerequisites

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **GPU** | NVIDIA CUDA-capable, **≥ 8 GB VRAM** | 16 GB VRAM | AMD/Intel GPUs work with Ollama in CPU mode only and will be very slow |
| **RAM** | **24 GB** | **32 GB+** | `deepseek-r1:14b` partially spills into RAM (~9 GB at runtime) |
| **Disk** | **30 GB free** | 60 GB free | Models 15 GB + Docker images + venv + Chromium |
| **CPU** | x86_64, 4 cores | 8+ cores | bge-m3 embedding throughput scales with CPU |

The performance numbers in the README (qwen 36.5 tok/s warm, deepseek-r1 8.3
tok/s warm) are measured on a **GTX 1080 / 8 GB VRAM + 32 GB RAM** Windows
host. Faster GPUs (RTX 30/40 series with more VRAM) eliminate the partial
offload and roughly triple deepseek-r1 throughput.

---

## 2. Software prerequisites — Windows + WSL2 path

### 2.1 Enable WSL2

On Windows 11 (or Windows 10 build 19041+), open **PowerShell as Administrator**
and run:

```powershell
wsl --install --no-distribution
wsl --set-default-version 2
```

If WSL was previously installed, ensure version 2 is the default:

```powershell
wsl --status
```

You should see `Default Version: 2`.

### 2.2 Install Rocky Linux 10

```powershell
wsl --install -d Rocky-10
```

If `Rocky-10` is not in your available list (`wsl --list --online`), grab it
from the [Microsoft Store](https://aka.ms/wslstore) or install the AlmaLinux 9
equivalent — the setup script also works on Rocky 9 and Alma 9.

After install, launch the distro once to create your user:

```powershell
wsl -d Rocky-10
```

Inside the Rocky prompt, give your user passwordless sudo (recommended — the
setup script runs many `sudo dnf` commands):

```bash
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER
sudo chmod 440 /etc/sudoers.d/$USER
```

### 2.3 Install Docker Desktop and enable WSL Integration

Install Docker Desktop from <https://www.docker.com/products/docker-desktop/>.

After install, open **Docker Desktop → Settings → Resources → WSL Integration**:

1. Verify **Enable integration with my default WSL distro** is ON.
2. Under **Enable integration with additional distros**, toggle **Rocky-10**
   to ON.
3. Click **Apply & Restart**.

Verify from WSL:

```bash
docker --version
docker info | head -10
```

`docker info` must succeed (no `Cannot connect to the Docker daemon`). If it
fails, re-check the WSL Integration toggle — Rocky 10 does not run native
`dockerd`; it borrows the daemon from Docker Desktop.

### 2.4 Install NVIDIA driver on Windows (for WSL2 GPU)

WSL2 GPU support requires the **Windows-side** NVIDIA driver only. Do *not*
install a Linux NVIDIA driver inside WSL — it will conflict.

Download the latest **Game Ready Driver** or **Studio Driver** for your GPU
from <https://www.nvidia.com/Download/index.aspx>. After installation and
reboot, verify from inside WSL:

```bash
nvidia-smi
```

You should see your GPU, VRAM total, and driver version. If `nvidia-smi` is
not found inside WSL but works in Windows, restart WSL with
`wsl --shutdown` from PowerShell, then re-enter the distro.

---

## 3. Software prerequisites — native Linux path

For users running Rocky 9/10, AlmaLinux 9, RHEL 9, or Fedora directly on
bare metal (no Windows / no WSL):

1. Install **Docker Engine** (not Desktop):
   <https://docs.docker.com/engine/install/rhel/>
2. Install the **NVIDIA proprietary driver** for your distribution:
   <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>
3. Install the **NVIDIA Container Toolkit** if you plan to run GPU workloads in
   Docker (Ollama runs on the host, so this is optional for this project):
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
4. Skip §2 entirely and proceed to §4.

Ubuntu/Debian users should follow the same outline but substitute `apt`
commands and skip the `dnf` portions of `setup_ai_env.sh` (see §7 below for
adaptation notes).

---

## 4. Bootstrap walkthrough

Clone the repo into your WSL home (or native Linux home — **not** under
`/mnt/c` or `/mnt/d`; cross-filesystem operations on Windows-mounted drives
are slow and have file-locking quirks):

```bash
cd ~
git clone https://github.com/smhakama/autonomous-local-llm.git
cd autonomous-local-llm
chmod +x setup_ai_env.sh
./setup_ai_env.sh
```

What happens at each step:

| Step | Action | Approx. time | Disk |
|---|---|---|---|
| **1/5** | `dnf upgrade --refresh` + install dev tools (python, git, gcc, **zstd**, jq, etc.) | 5–10 min | ~500 MB |
| **2/5** | Verify Docker, pull `qdrant/qdrant:latest`, start container on ports 6333/6334 with `~/qdrant_storage` mount | 1–2 min | ~150 MB |
| **3/5** | Install Ollama via official `install.sh`, pull `qwen2.5-coder:7b` (4.7 GB), `deepseek-r1:14b` (9.0 GB), `bge-m3` (1.2 GB) | 5–15 min depending on download speed | ~15 GB |
| **4/5** | Create venv at `~/ai_agents_env`, install `aider-chat`, `browser-use`, `langchain-ollama`, `playwright`, and Chromium | 3–5 min | ~1.5 GB |
| **5/5** | Smoke test endpoints (Ollama `/api/tags`, Qdrant `/`, `docker ps qdrant`) and print the cheat sheet | <30 s | — |

Total expected time on a 100 Mbps connection: **15–30 minutes**.

The script is **idempotent**: re-running it skips packages that are already
installed, recreates the Qdrant container, and re-uses the venv if present.

---

## 5. Verification checklist

After `setup_ai_env.sh` prints `[setup_ai_env]  環境構築完了`, run:

```bash
# 5.1 — Ollama is running and has all three models loaded
ollama list
# Expect: bge-m3, deepseek-r1:14b, qwen2.5-coder:7b

# 5.2 — Qdrant container is up
docker ps --filter name=qdrant
curl -s http://127.0.0.1:6333/ | head -c 200
# Expect: a JSON body with "title": "qdrant - vector search engine"

# 5.3 — venv has the agent packages
source ~/ai_agents_env/bin/activate
python -c "import aider, browser_use, langchain_ollama, playwright, qdrant_client; print('OK')"
deactivate

# 5.4 — Aider talks to Ollama end-to-end (1-minute smoke)
mkdir -p ~/_smoke && cd ~/_smoke && git init -q
echo 'def add(a, b): return a' > calc.py
git add calc.py && git -c user.email=test@local -c user.name=test commit -q -m seed
source ~/ai_agents_env/bin/activate
OLLAMA_API_BASE=http://127.0.0.1:11434 \
  aider --model ollama_chat/qwen2.5-coder:7b \
    --message "Fix add() to return a + b instead of just a." \
    --yes-always --no-stream --no-auto-commits \
    --no-show-model-warnings --no-check-update calc.py
cat calc.py
# Expect: def add(a, b): return a + b
deactivate
```

If 5.1–5.3 succeed but 5.4 hangs for more than 2 minutes on the *first* call,
that is normal — Ollama is loading `qwen2.5-coder:7b` into VRAM (~45 s cold).
Subsequent calls are fast.

---

## 6. Troubleshooting

The errors below all came up while building this project; the fixes are
verified.

### `ERROR: This version requires zstd for extraction.` (during Ollama install)

Rocky 10 minimal install does not include `zstd`. Run:

```bash
sudo dnf -y install zstd
curl -fsSL https://ollama.com/install.sh | sh
```

The latest `setup_ai_env.sh` already adds `zstd` to its dev tools list, so this
only bites on manual installs.

### `Cannot connect to the Docker daemon` / `docker info` fails

You are on WSL2 but Docker Desktop's WSL Integration is not toggled on for your
distro. Open Docker Desktop → Settings → Resources → WSL Integration → flip
your distro to ON → Apply & Restart. There is no `dockerd` service to start
inside Rocky 10 itself; it must come from Docker Desktop.

### `PermissionError: [Errno 13] Permission denied: 'aider'` (or `'pytest'`)

You are running a Python script with `~/ai_agents_env/bin/python script.py`
*without* activating the venv first, and the script does
`subprocess.run(["aider", ...])`. The venv's `bin/` is not on PATH for child
processes.

Fix inside the script — invoke venv binaries via absolute path:

```python
from pathlib import Path
import sys
AIDER_BIN = str(Path(sys.executable).parent / "aider")
subprocess.run([AIDER_BIN, ...])
```

Or just `source ~/ai_agents_env/bin/activate` before running the script.

### Aider says `OllamaException - 404 page not found`

You set `OLLAMA_API_BASE=http://127.0.0.1:11434/v1`. The `/v1` is for the
OpenAI-compatible endpoint, but `litellm`'s `ollama/` and `ollama_chat/`
providers call the native `/api/show` and `/api/chat` paths. Drop the `/v1`:

```bash
OLLAMA_API_BASE=http://127.0.0.1:11434 \
  aider --model ollama_chat/qwen2.5-coder:7b
```

Also prefer `ollama_chat/` over the legacy `ollama/` model prefix.

### `localhost` does not work but `127.0.0.1` does

Some WSL2 configurations resolve `localhost` to `::1` (IPv6 loopback), and
Ollama listens only on the IPv4 loopback. Always use `127.0.0.1` explicitly in
`OLLAMA_API_BASE` and similar settings.

### `deepseek-r1:14b` takes 85 seconds on every call

It is being evicted from VRAM between calls. Ollama's default keep-alive is
~5 minutes; the partial-offload reload of a 14B model takes about 85 s on
8 GB VRAM. For active sessions, bump it:

```bash
# Long-term: edit your ollama systemd drop-in or shell profile
export OLLAMA_KEEP_ALIVE=1h

# Per-request via the HTTP API:
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "deepseek-r1:14b",
  "messages": [...],
  "keep_alive": "1h"
}'
```

### `deepseek-r1:14b` answers feel cut off / contain only `<think>`

DeepSeek-R1 is a reasoning model that emits a `<think>…</think>` block before
its final answer. With Ollama's default `num_predict`, the budget can be spent
entirely inside the thinking block. Bump it:

```bash
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "deepseek-r1:14b",
  "messages": [...],
  "options": {"num_predict": 2048}
}'
```

### `playwright install-deps chromium` fails on Rocky 10

The Playwright maintainers do not ship Rocky 10 system-package dependencies.
The `setup_ai_env.sh` swallows the error and continues; you only feel it when
`browser-use` actually launches a browser and reports missing `libnss3` /
`libxkbcommon0` / similar. Install whatever Chromium asks for via `dnf`:

```bash
sudo dnf -y install nss libxkbcommon alsa-lib at-spi2-atk cups-libs \
                    libdrm mesa-libgbm pango cairo
```

### nvidia-smi works in Windows but not in WSL

After installing the Windows NVIDIA driver, shut WSL down completely so the
new driver shim picks up:

```powershell
wsl --shutdown
wsl -d Rocky-10
nvidia-smi
```

---

## 7. Adapting to Ubuntu / Debian / other distros

`setup_ai_env.sh` is mostly distro-agnostic except for **Step 1**, which uses
`dnf`. Concrete substitutions:

| Rocky/Fedora (`dnf`) | Ubuntu/Debian (`apt`) |
|---|---|
| `dnf -y upgrade --refresh` | `apt update && apt upgrade -y` |
| `dnf -y install python3 python3-pip python3-devel git curl wget gcc gcc-c++ make tar which jq zstd` | `apt install -y python3 python3-venv python3-pip python3-dev git curl wget build-essential jq zstd` |

The Docker, Ollama, Qdrant, venv, and Playwright sections work unchanged.

If you maintain a fork for another distro, please open a PR with a
`setup_ai_env_<distro>.sh` so others can find it.

---

## 8. Uninstall / reset

```bash
# Stop and remove the Qdrant container (your stored vectors persist in
# ~/qdrant_storage — delete that directory to wipe them)
docker rm -f qdrant
rm -rf ~/qdrant_storage

# Remove the venv
rm -rf ~/ai_agents_env

# Remove Ollama models (each model is ~5–10 GB)
ollama rm qwen2.5-coder:7b deepseek-r1:14b bge-m3

# Or uninstall Ollama entirely:
#   sudo rm -rf /usr/local/lib/ollama /usr/local/bin/ollama
#   sudo userdel ollama
```

---

## Need help?

Open an issue on the repository with:

1. Output of `cat /etc/os-release`
2. Output of `wsl --status` (if on Windows)
3. Output of `docker info | head -20`
4. Output of `ollama list`
5. The exact command you ran and the full error message
