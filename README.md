# Autonomous Local LLM SDLC

A self-contained AI Software Development Lifecycle (SDLC) loop optimized for
consumer-grade hardware: **GPU VRAM 8GB / 32GB host RAM (15GB WSL2 guest) /
WSL2 + Rocky Linux 10**.

The system performs the full *research → plan → implement → audit → commit* cycle
locally, with **zero external API cost** and **no data leaving the host**.

The project is **research-oriented**: it is designed so the underlying LLMs can
be hot-swapped as better open-weight models appear, and so every pipeline run
emits a JSONL record (Phase 3.7+) that can be replayed and re-analysed offline.
Production hardening is a deliberate second-stage goal — see the metrics
section.

See [docs/spec/Autonomous_Local_LLM_Specification.pdf](docs/spec/Autonomous_Local_LLM_Specification.pdf)
for the original architecture brief.

## Architecture

| Layer | Component | Role |
|---|---|---|
| **Brain** | Ollama: `qwen2.5-coder:7b` (Aider edits) · `deepseek-r1:14b` (plan + audit) · `gemma2:9b-instruct-q4_K_M` (CPU-only, Japanese-strong critic) | A small `MODEL_REGISTRY` (Phase 3.6) makes the primary / fallback choice swappable per run |
| **Action** | Aider + browser-use (Playwright) | Aider makes surgical edits; browser-use scrapes JS-heavy docs |
| **Memory** | Qdrant + `bge-m3` embedding | bge-m3 runs on CPU and stays out of GPU contention |
| **Search** | SearxNG (self-hosted, `:8888`) | Replaces DDG since commit `20e0cc6` — owner-controlled, no per-instance rate limits |

The GPU models share the 8GB VRAM via Ollama's per-model loading (set
`OLLAMA_MAX_LOADED_MODELS=2` to let `deepseek-r1:14b` and a second GPU model
coexist without unload thrash). `gemma2:9b-instruct-q4_K_M` is intentionally
CPU-only — it does not compete for VRAM, which makes asymmetric strategies
(e.g. 14b proposer + 9b critic) feasible. The embedding model and vector DB
live entirely in system RAM, so document indexing and search **never evict
the active LLM**.

## What's in this repo

| File / dir | Purpose |
|---|---|
| `setup_ai_env.sh` | One-shot Rocky 10 WSL2 bootstrap (Ollama + models + Qdrant + venv + Playwright Chromium) |
| `embed_codebase.py` | PoC: walks a codebase, chunks files, embeds via bge-m3, stores in Qdrant, runs sample semantic searches |
| `hotfix_loop.py` | Original Phase A→C orchestrator: codebase → 14b plan → Aider 7b edit → pytest → re-plan on failure → commit on green |
| `corpus2skill.py` | **Phase 3 main pipeline**: consumes pre-cleansed Qdrant chunks for a theme → 14b distill (with optional 7b fallback) → quality loop → emits a reusable Python module under `skills/<theme>.py` + a JSONL record |
| `bench/parallel_capacity_check.py` | Phase 3.7c/e: multi-model coexistence benchmark (gemma2:9b CPU + deepseek-r1:14b GPU), schema v3 with `num_thread` / `mem-stress` matrix knobs |
| `analyze_runs.py` | Phase 3.7b: pure-stdlib group-by aggregator over `metrics/distill_runs.jsonl` |
| `metrics/` | Append-only JSONL records — gitignored, intended for offline replay and cross-model comparison |
| `docs/spec/` | Architecture specification |
| `examples/aider_smoke/` | Tiny demo target for `hotfix_loop.py` (intentionally broken `greet()` + failing pytest cases) |

## Quick start

> First time on this machine? Read **[INSTALL.md](INSTALL.md)** for the full
> prerequisites (WSL2, Docker Desktop WSL Integration, NVIDIA driver, Rocky 10
> setup) and a troubleshooting section. The steps below assume those are
> already in place.

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

# 5. Distill a skill document for a topic (Phase 3 main pipeline)
python corpus2skill.py --theme asyncio --primary-model deepseek-r1:14b \
  --metrics-file metrics/distill_runs.jsonl

# 6. Aggregate metrics across runs (Phase 3.7b)
./analyze_runs.py --group-by theme,config.primary_model --table
```

Expected outcome of step 4: the orchestrator indexes the repo, asks `deepseek-r1:14b`
for a plan, hands the plan to Aider with `qwen2.5-coder:7b`, runs pytest, and
commits on green. End-to-end roughly 3 minutes on the reference hardware.

Expected outcome of step 5: a single Python module written to
`skills/<theme>.py` (importable as `skills.<theme>`) plus one JSONL line under
`metrics/distill_runs.jsonl` capturing the prompt, configuration, quality
verdict, elapsed time, and system snapshot. The skill module can be imported
directly by Aider, browser-use, or any future agent. Step 5 assumes the
`web_brain_clean` Qdrant collection already has chunks for the theme (use
`web_research.py` + `cleanse_chunk.py` to populate it first).

## Distillation pipeline (Phase 3)

`corpus2skill.py` is the main Phase 3 entry point. Given a theme (e.g.
`asyncio`, `kubernetes`), it pulls the matching chunks out of the
`web_brain_clean` Qdrant collection (populated upstream by the Phase 2.5
`web_research.py` + `cleanse_chunk.py` pipeline), distills them through
`deepseek-r1:14b` (with optional fallback to `qwen2.5-coder:7b`) into a
single reusable Python module written to `skills/<theme>.py`, and runs a
quality loop that validates the output before persisting.

| Phase | Feature | Detail |
|---|---|---|
| 3.1 | Multi-theme orchestrator | `--themes-file` batch with `flock` to prevent overlapping runs |
| 3.2 | Quality loop (L4 + L5) | mypy strict type-check + Gemini lint, with auto-retry on failure |
| 3.3 | RAG augmentation | Layer-1 retrieval from a `web_brain` Qdrant collection at distill time |
| 3.4 | Adaptive RAG | domain-selective injection schedule — avoids bloating prompts for generic themes |
| 3.5 | Inner retry + corrective hint | retries with a targeted hint when validation partially fails |
| 3.6 | Multi-model fallback | `MODEL_REGISTRY` + `--primary-model` for instant swap (e.g. `14b → 7b`) |
| 3.7 | Metrics JSONL | per-run record under `metrics/distill_runs.jsonl` for offline replay |

The Phase 3.6 fallback was validated by observing that a 7b retry of a failing
14b prompt reproduced the *same* hallucination — "same prompt → same wash-out".
That finding motivated the multi-model team direction explored under Phase 3.7c+
(asymmetric debate: 14b proposer + 9b critic running on different physical
resources).

## Research metrics (Phase 3.7)

Phase 3.7 introduces a JSONL-first measurement layer so swap decisions across
LLM generations can be grounded in numbers rather than recall:

- `metrics/distill_runs.jsonl` — one line per `corpus2skill.py` invocation.
  Includes the config snapshot, quality verdict, elapsed time, fallback usage,
  and a system snapshot (Phase 3.7d, `schema_version=2`).
- `analyze_runs.py` — pure-stdlib group-by aggregator.
  `./analyze_runs.py --group-by theme,config.primary_model --table` prints a
  quick cross-model comparison; the same command with `| jq` gives raw JSON.
- `bench/parallel_capacity_check.py` — multi-model coexistence benchmark
  (gemma2:9b CPU vs deepseek-r1:14b GPU) with per-run `eval_count` and a
  `wall_vs_total_max_ratio` field (`schema_version=3`, Phase 3.7e-2). The
  `--num-thread N`, `--mem-stress`, `--bind-cores-label`, and `--config-id`
  knobs make it easy to run a matrix and append to
  `metrics/parallel_capacity_checks.jsonl`.

**Measurement caveat (Phase 3.7e-1 finding)**: any throughput conclusion must
verify `eval_count == num_predict` on every run. A short prompt can early-EOS
at four tokens and produce a measurement window so small that wall is
dominated by `load_duration` and `prompt_eval_duration` — this fabricates a
near-1.0 interference ratio that does not reflect steady-state parallel cost.
Use `--prompt-mode long` (the current default) for any decision-grade
measurement; `--prompt-mode short` exists only to reproduce the Phase 3.7c
artifact.

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

### gemma2:9b as a CPU-only second model

`gemma2:9b-instruct-q4_K_M` is loaded with `num_gpu=0` so it lives entirely in
system RAM (~5.8GB). This lets it coexist with `deepseek-r1:14b`'s 6.6GB GPU
partial offload without VRAM contention:

```python
ollama_generate(model="gemma2:9b-instruct-q4_K_M",
                options={"num_gpu": 0, "num_thread": 6, "temperature": 0.1})
```

`num_thread` matters more than it looks. On a 12-core / 24-thread Ryzen,
solo gemma throughput at the default thread count (≈12) is roughly the same
as at `num_thread=6` (5.4 vs 5.7 tok/s), but `num_thread=24` *halves* it
(2.9 tok/s) — over-threading triggers SMT contention. Use Phase 3.7e-2's
matrix runner to pick a sensible value on your CPU.

### Multi-model coexistence (Ollama drop-in)

For two models loaded at once, add a systemd drop-in so Ollama keeps both
warm:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`. Without
this, Ollama unloads the previous model on each load, which costs ~80s per
swap for the 14b.

## Measured performance (GTX 1080 / 8GB VRAM / Ryzen 9 3900 / WSL2 15GB)

### Single-model warm inference

| Model | Warm | Cold load | Split |
|---|---|---|---|
| `qwen2.5-coder:7b` | 36.5 tok/s | ~45 s | 100% GPU (~7.5GB VRAM) |
| `deepseek-r1:14b` | 7.1 tok/s | ~85 s | ~35% CPU / ~65% GPU (6.6GB VRAM + ~2.4GB RAM) |
| `gemma2:9b-instruct-q4_K_M` | 5.1 tok/s | ~30 s | 100% CPU (~5.8GB RAM) |
| `bge-m3` (embed) | ~2.7 chunks/s | n/a | CPU only |

The 7.1 / 5.1 tok/s figures are 3-run means with a ~720-character English
prompt and `num_predict=300` (Phase 3.7e-1). The earlier "8.3 tok/s for
deepseek" number from Phase 3.7c was inflated by `load_duration` bleeding
into a short measurement window — see the *Measurement caveat* under
[Research metrics](#research-metrics-phase-37).

### Concurrent gemma + deepseek (Phase 3.7e-1, n=3, long prompt)

| Model | Solo | Concurrent | Interference ratio |
|---|---|---|---|
| `gemma2:9b-instruct-q4_K_M` | 5.07 tok/s | 3.18 tok/s | 0.628 |
| `deepseek-r1:14b` | 7.11 tok/s | 3.66 tok/s | 0.515 |
| **Combined throughput** | n/a | **6.84 tok/s** | (~0.99× of solo deepseek) |
| `wall_vs_total_max_ratio` | n/a | 1.0001 – 1.0003 | (Ollama internal queue ≈ 0) |

When both models run together, each loses roughly half its solo throughput,
and the combined token rate is essentially the same as running deepseek
alone. The `wall_vs_total_max_ratio ≈ 1.0` rules out Ollama-side queue
overhead, so the bottleneck is either CPU-thread contention or RAM bandwidth
(a back-of-envelope DDR4-3200 ceiling for the combined ~8.2 GB/token scan
is ~6.27 tok/s — close to the measured 6.84). Phase 3.7e-2 (`schema v3`)
adds a `num_thread` / `mem-stress` matrix to separate the two hypotheses
formally.

### Other notes

End-to-end Aider edits via `qwen2.5-coder:7b` add roughly **8 s of
litellm/repo-map overhead per turn** on top of pure inference.

DeepSeek-R1 is a reasoning model that emits `<think>...</think>` blocks; set
`num_predict` ≥ 1024 if you want a complete plan after the thinking phase.

## Known gotchas

- **Rocky 10**: the Ollama installer requires `zstd`, which is not part of the
  minimal install. `setup_ai_env.sh` adds it to the dev tools list.
- **WSL2**: Docker daemon must come from Docker Desktop's WSL Integration; the
  native `docker.service` is intentionally disabled.
- **WSL2 guest RAM**: a 32GB host typically exposes ~15GB to the guest unless
  `~/.wslconfig` raises it. With gemma2:9b (~5.8GB) + deepseek-r1:14b's
  CPU layer (~2.4GB) loaded, you have ~6GB left for the OS, Aider, and
  scratch — bench in isolation if you start hitting swap.
- **subprocess from venv-python**: invoking the script with `venv/bin/python`
  does *not* put `venv/bin/` on PATH for child processes. Call venv binaries
  through their absolute paths (`Path(sys.executable).parent / "aider"`).
- **`OLLAMA_KEEP_ALIVE`**: the default ~5 minute idle timeout causes
  `deepseek-r1:14b` to be evicted between calls, forcing an 85-second reload.
  Bump `OLLAMA_KEEP_ALIVE=1h` (or longer) for active sessions.
- **`num_thread` over-subscription**: on a 12c/24t CPU, calling Ollama with
  `options.num_thread=24` halves throughput vs the default. The Phase 3.7e-2
  matrix is the right way to find the sweet spot for a given CPU.
- **`bench/parallel_capacity_check.py --mem-stress`** requires `stress-ng`
  (`sudo dnf install stress-ng` on Rocky 10). `perf` is *not* required;
  stress-ng acts as a proxy for RAM-bandwidth saturation.

## License

MIT — see [LICENSE](LICENSE).
