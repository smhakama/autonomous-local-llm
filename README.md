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

> **Naming note.** A separate public project also called **Corpus2Skill**
> ([dukesun99/Corpus2Skill](https://github.com/dukesun99/Corpus2Skill),
> Sun et al. 2026) shares the name but solves a different problem: it
> distils a corpus into an
> [Anthropic Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
> navigation tree (`.claude/skills/SKILL.md` + `INDEX.md`) so an LLM agent
> can traverse the hierarchy at serve time without a vector store. This
> repo's `corpus2skill.py` instead produces an importable Python module
> (`skills/<theme>.py`) from theme-tagged Qdrant chunks. The two outputs
> are complementary — Markdown navigation tree for Claude Code, Python
> module for Aider / browser-use.

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

## Router / multi-model team (Phase 3.8b)

Phase 3.8b introduces a new `router/` subpackage that orchestrates two models
running in parallel on the same chunks. The first concrete strategy —
`AsymmetricDebateStrategy` — fires a *proposer* (`deepseek-r1:14b` on GPU) and
an independent *critic* (`gemma2:9b-instruct-q4_K_M` with `num_gpu=0`)
concurrently, leveraging the Phase 3.8a NT6 verdict
(`options.num_thread=6` for both, sum_conc 9.39 tok/s, wall ≈ max of both totals).

The PoC merge step is intentionally dumb: `chosen_text = proposer.text`; the
critic output is parsed into bullet findings and recorded in
`metrics/router_runs.jsonl` (schema v1) for offline analysis. The goal is to
measure *critic signal* in isolation before adding any feedback loop (Phase 3.8c).

Opt in per run via `--router-strategy asymmetric_debate`:

```bash
python corpus2skill.py \
  --theme "Kubernetes pod security standards" \
  --router-strategy asymmetric_debate
```

Default behaviour is `--router-strategy none`: the legacy single-`call_14b`
code path is preserved verbatim. The router fires only on the first attempt
(`attempt == 1`); subsequent retries (corrective hint, 7B fallback,
RAG-adaptive) stay on the single primary model — the critic exists to give an
*initial independent view*, not retry-hint refinement.

### Smoke run (Phase 3.8b, kubernetes, commit a74ef4d)

Real-world end-to-end smoke against fresh `web_brain_clean` (2 cleansed
chunks from `kubernetes.io` pod security standards docs):

| Metric | Value |
|---|---|
| `parallel_wall_sec` | 197.96 |
| proposer `total_duration` | 197.34 s |
| critic `total_duration`  | 194.78 s |
| `wall_vs_total_max` ratio | 1.003 (Phase 3.8a pattern reproduced live) |
| proposer `eval_count` | 1009 tokens |
| critic `eval_count`   | 158 tokens |
| `critic_findings_count` | 8 |

Sample of the critic's findings (gemma2:9b reading only the source chunks,
not the proposer's output):

- *"Hallucinating methods or attributes for Kubernetes objects based on incomplete documentation."*
- *"Confusing the allowed values for sysctls with a whitelist, potentially missing edge cases."*
- *"Overlooking the distinction between `audit`, `warn`, and `enforce` modes, resulting in inappropriate behavior."*

These are kubernetes-specific pitfalls about the actual semantics of the
Pod Security Standards API — useful signal that the existing Gemini L5 critic
(which audits the *generated module*, Phase 3.2) does not produce because it
never sees the source docs.

### Schema bump (`distill_runs.jsonl` v2 → v3)

`metrics/distill_runs.jsonl` now carries `router_strategy`, `router_wall_sec`,
and `router_critic_findings_count` (all null when the router did not fire),
plus the router CLI knobs in the `config` snapshot. The change is additive
and nullable, so v2 parsers continue to work on v3 records.

The router itself is wired behind a narrow `ModelRunner` Protocol so the
backend can be swapped (Ollama → vLLM → a remote API) by writing one new
class. The Phase 3.8a parallel-capacity verdict pins the runner defaults
(`num_thread=6`, critic `num_gpu=0`); revisit those when the underlying
hardware changes.

### Qdrant note (2026-06-01 incident)

Phase 3.8b's smoke setup found that Docker Desktop on WSL2 silently overlays
bind mounts to `/home/<user>/...` with `tmpfs`, so Qdrant restarts wipe all
collections without warning. The repository now ships a `docker-compose.yml`
that uses a Docker *named volume* (`qdrant_data`) instead of a bind mount;
bring Qdrant up with `docker compose up -d`.

## Critic→proposer merge loop (Phase 3.8c)

Phase 3.8b recorded critic findings in `router_runs.jsonl` but never fed them
back to the proposer. Phase 3.8c closes that loop: critic findings can now be
injected into the proposer prompt on retries via a `--router-feedback` flag.

Three modes, all opt-in (only fire when `--router-strategy asymmetric_debate`):

| `--router-feedback` | router runs | hint injection | proposer prompts per attempt |
|---|---|---|---|
| `none` | attempt 1 only | never | base prompt + L1/L2/L3 feedback only (Phase 3.8b parity) |
| `on-retry` | attempt 1 only | attempts 2+ reuse memoized findings | base + feedback + same hint each retry |
| `every-attempt` (default) | every attempt | attempt N+1 sees attempt N's critic | base + feedback + fresh hint per retry |

Findings are formatted as a `PRIOR INDEPENDENT REVIEWER ... FLAGGED THESE
PITFALLS PRE-EMPTIVELY` block appended after L1/L2/L3 corrective feedback —
clearly distinct from hard-error retries so the model treats the bullets as
advisory, not as compile errors. Each block is clipped to 10 bullets to keep
prompts bounded.

Example invocation:

```bash
python corpus2skill.py \
  --theme "Kubernetes pod security standards" \
  --router-strategy asymmetric_debate \
  --router-feedback every-attempt
```

### Smoke run (Phase 3.8c, kubernetes, commit 07b79a3)

Same `web_brain_clean` corpus (2 cleansed chunks from `kubernetes.io` pod
security standards docs) ran through all 3 modes back-to-back. Models had
already been pulled but the daemon was cold for the first run.

| Mode | attempts | wall (s) | router_wall_sec | critic findings | injected_count |
|---|---|---|---|---|---|
| `none` | 2 (L2 import fail → call_14b retry) | 552.6 | 338.9 (cold load) | 9 | 0 |
| `on-retry` | 1 (L1+L2+L3 PASS) | 194.3 | 194.0 (warm) | 8 | 0 |
| `every-attempt` | 1 (L1+L2+L3 PASS) | 192.3 | 192.3 (warm) | 8 | 0 |

All three runs succeeded. **Two findings worth recording:**

1. **`deepseek-r1:14b` on a 2-chunk corpus passed L1+L2+L3 on attempt 1 in
   2/3 runs, so the merge loop dispatch fired but `injected_count = 0` for
   all modes.** Hint injection is exercised by the monkeypatched integration
   tests (`tests/test_corpus2skill_integration.py`), not the live smoke. A
   harder corpus or a weaker proposer would be needed to trigger real-world
   injection — pencilled in for the deferred experiment harness.
2. **Critic findings were near-identical across runs**: the top 2 bullets
   were verbatim the same in all 3 modes (e.g.
   *"Misinterpreting Undefined/nil as a valid value..."*,
   *"Incorrectly assuming all spec fields are always present..."*), and 6 of
   8 bullets overlapped pairwise. With the chunks fixed and the critic prompt
   stable, `every-attempt` produces ≈ the same findings each iteration as
   `on-retry` — so the extra token cost (one full critic pass per attempt)
   buys very little new signal in this corpus shape. A future Phase 3.8d
   could either raise critic temperature or move to *critic-on-proposer-
   output* (Phase 3.8c proposal C) for genuinely fresh per-attempt feedback.

### Schema bump (`distill_runs.jsonl` v3 → v4)

`router_feedback_mode` and `router_findings_injected_count` are now top-level
fields on each record (both `null` when no injection ran), plus `router_feedback`
is appended to the `config` snapshot. Additive and nullable, so v3 parsers
keep working on v4 records — same compatibility contract as the v2→v3 bump.

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

### Phase 3.7e-2 matrix verdict (5 configs, n=3 each, long prompt)

| cfg | num_thread | bind | g_intf | d_intf | wall (s) |
|---|---|---|---|---|---|
| E0 (baseline) | server default | — | 0.604 | 0.517 | 89.5 |
| **E1** | 6 | — | **0.729** | **0.700** | **75.8** |
| E2 | 24 | — | 0.071 | 0.048 | 577.8 |
| E3 | 6 | taskset `0-11` | 0.687 | 0.642 | 83.9 |
| E4 | 6 | — (`stress-ng --vm 2 --vm-bytes 1G`) | 0.638 | 0.607 | 90.0 |

The matrix isolates two effects:

- **Primary: SMT over-subscription (hypothesis C).** Capping `options.num_thread`
  at 6 (E1) lifts both interference ratios by 21–35 % and cuts wall by 15 %
  versus the server default. Pushing it to 24 (E2) collapses throughput;
  even *solo* gemma drops from 5.7 → 3.2 tok/s and `wall_vs_total_max_ratio`
  falls to 0.33, meaning the kernel scheduler is no longer overlapping the
  two requests.
- **Secondary: RAM bandwidth (hypothesis D).** A 2 GB working-set
  `stress-ng --vm` background load (E4) costs about 13 % of the
  interference-ratio gain. Real, but small compared to E2.

The systemd `taskset -c 0-11` drop-in (E3) actually *hurt* throughput here.
WSL2 hides the underlying CCD topology from the guest, so binding vCPUs 0–11
does not translate into pinning to one CCD or to physical-core siblings, and
the Hyper-V SMT-aware scheduler ends up working against the affinity mask.
On bare-metal Linux on a Ryzen with exposed CCDs the verdict could flip;
under WSL2 it does not.

**Practical takeaway for this environment** (Ryzen 9 3900 12C24T, WSL2,
15 GB guest RAM): set `options.num_thread=6` for the CPU-only second model
and leave Ollama's affinity to the kernel default. Do not add a
`taskset` drop-in. Re-run the matrix on hardware where the CCD topology is
visible to the OS before extrapolating.

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
