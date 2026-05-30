# Autonomous Local LLM SDLC

A self-contained AI Software Development Lifecycle (SDLC) loop optimized for
consumer-grade hardware: **GPU VRAM 8GB / RAM 32GB / WSL2 + Rocky Linux 10**.

The system performs the full *research → plan → implement → audit → commit* cycle
locally, with **zero external API cost** and **no data leaving the host**.

See [docs/spec/Autonomous_Local_LLM_Specification.pdf](docs/spec/Autonomous_Local_LLM_Specification.pdf)
for the original architecture brief.

## Architecture

| Layer | Component | Role |
|---|---|---|
| **Brain** | Ollama (`qwen2.5-coder:7b` + `deepseek-r1:14b`) | 7b drives rapid edits via Aider; 14b plans + audits |
| **Action** | Aider + browser-use (Playwright) | Aider makes surgical edits; browser-use scrapes JS-heavy docs |
| **Memory** | Qdrant + `bge-m3` embedding | bge-m3 runs on CPU and stays out of GPU contention |

The two LLMs share the 8GB VRAM via Ollama's per-model loading, while the
embedding model and vector DB live entirely in system RAM so document indexing
and search **never evict the active LLM**.

## What's in this repo

| File | Purpose |
|---|---|
| `setup_ai_env.sh` | One-shot Rocky 10 WSL2 bootstrap (Ollama + models + Qdrant + venv + Playwright Chromium) |
| `embed_codebase.py` | PoC: walks a codebase, chunks files, embeds via bge-m3, stores in Qdrant, runs sample semantic searches |
| `hotfix_loop.py` | Minimal Phase A→C orchestrator: codebase → 14b plan → Aider 7b edit → pytest → re-plan on failure → commit on green |
| `docs/spec/` | Architecture specification |
| `examples/aider_smoke/` | Tiny demo target for `hotfix_loop.py` (intentionally broken `greet()` + failing pytest cases) |

## Quick start

```bash
# 1. Bootstrap the environment (idempotent)
./setup_ai_env.sh

# 2. Activate the Python venv
source ~/ai_agents_env/bin/activate

# 3. Try the embedding PoC against any codebase
python embed_codebase.py      # edit CODE_ROOT inside the script first

# 4. Try the Phase A→C orchestration on the bundled example
cd examples/aider_smoke && git init -q && git add -A && git commit -q -m "seed"
cd -
python hotfix_loop.py examples/aider_smoke \
  "Tests in tests/test_hello.py are failing. Make all tests pass."
```

Expected outcome of step 4: the orchestrator indexes the repo, asks `deepseek-r1:14b`
for a plan, hands the plan to Aider with `qwen2.5-coder:7b`, runs pytest, and
commits on green. End-to-end roughly 3 minutes on the reference hardware.

## Verified configuration

### Ollama-via-Aider

The OpenAI-compatible base URL and the model prefix have to match. The combination
that works through `litellm` is:

```bash
OLLAMA_API_BASE=http://127.0.0.1:11434 \
  aider --model ollama_chat/qwen2.5-coder:7b
```

Common mistakes:

- `OLLAMA_API_BASE=http://127.0.0.1:11434/v1` — the `/v1` makes litellm hit
  `/v1/api/show` which returns 404.
- `--model ollama/...` — uses the legacy `/api/generate` endpoint; prefer
  `ollama_chat/...` which uses `/api/chat`.
- `http://localhost:...` — on some setups resolves to `::1` and misses the
  IPv4 listener. Use `127.0.0.1` explicitly.

### browser-use on 8GB VRAM

Vision mode pushes screenshot tokens through the LLM and overflows 8GB VRAM
immediately. Pass `use_vision=False`:

```python
from browser_use import Agent
agent = Agent(task="...", llm=..., use_vision=False)
```

## Measured performance (GTX 1080 / 8GB VRAM / 32GB RAM)

| Model | Pure inference (warm) | Cold load | GPU/CPU split |
|---|---|---|---|
| `qwen2.5-coder:7b` | **36.5 tok/s** | ~45 s | 100% GPU (~7.5GB) |
| `deepseek-r1:14b` | **8.3 tok/s** | ~85 s | 35% CPU / 65% GPU (~10GB) |
| `bge-m3` (embed) | **~2.7 chunks/s** (CPU) | n/a | CPU only |

End-to-end Aider edit through `qwen2.5-coder:7b` adds roughly **8 seconds of
litellm/repo-map overhead per turn** on top of pure inference.

DeepSeek-R1 is a reasoning model that emits `<think>...</think>` blocks; set
`num_predict` ≥ 1024 if you want a complete plan after the thinking phase.

## Known gotchas

- **Rocky 10**: the Ollama installer requires `zstd`, which is not part of the
  minimal install. `setup_ai_env.sh` adds it to the dev tools list.
- **WSL2**: Docker daemon must come from Docker Desktop's WSL Integration; the
  native `docker.service` is intentionally disabled.
- **subprocess from venv-python**: invoking the script with `venv/bin/python`
  does *not* put `venv/bin/` on PATH for child processes. Call venv binaries
  through their absolute paths (`Path(sys.executable).parent / "aider"`).
- **`OLLAMA_KEEP_ALIVE`**: the default ~5 minute idle timeout causes
  `deepseek-r1:14b` to be evicted between calls, forcing an 85-second reload.
  Bump `OLLAMA_KEEP_ALIVE=1h` (or longer) for active sessions.

## License

MIT — see [LICENSE](LICENSE).
