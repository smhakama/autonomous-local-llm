# experiments/ — Phase 3.8a nightly matrix runner

This directory holds yaml-driven experiment matrices and a runner that
executes them as a sequence of subprocesses. Designed to be triggered by
`cron/llm-nightly.timer` (systemd user unit) so that overnight runs
accumulate data without manual intervention.

The runner does **not** define new measurement code — it only invokes
existing bench scripts (currently `bench/parallel_capacity_check.py`)
with different argument permutations. Whatever the bench appends to
`metrics/*.jsonl` is the actual data; this directory just records which
invocation produced which line.

## Layout

```
experiments/
├── matrices/                       # yaml definitions (versioned)
│   └── phase_38a_num_thread_sweep.yaml
├── runner.py                       # CLI: --matrix <path> [--dry-run]
├── runs/                           # per-execution outputs (.gitignored)
│   └── <matrix_id>/<local-timestamp>/
│       ├── summary.jsonl           # one line per sweep entry
│       └── <config_id>.log         # raw stdout+stderr from that entry
└── README.md
```

## Matrix yaml schema (v1)

```yaml
schema: 1                           # required, must be 1
matrix_id: phase_38a_num_thread_sweep
description: free text (recorded in summary, not strict)
bench: bench/parallel_capacity_check.py    # path relative to repo root
common_args:                        # passed to every sweep entry
  - --runs
  - "3"
  - --prompt-mode
  - long
sweep:                              # required, one entry per config
  - config_id: NT4                  # required, unique within the matrix
    extra_args: [--num-thread, "4"] # optional, appended after common_args
  - config_id: NT6
    extra_args: [--num-thread, "6"]
```

The runner appends `--config-id <config_id>` automatically unless
`extra_args` already contains it. The same `config_id` then shows up in
the bench's `metrics/*.jsonl` row, which is how you filter sweeps later:

```bash
jq -c 'select(.config.config_id|startswith("NT"))' \
  metrics/parallel_capacity_checks.jsonl
```

## Run manually

```bash
# Dry-run (print the commands, no execution)
~/ai_agents_env/bin/python experiments/runner.py \
  --matrix experiments/matrices/phase_38a_num_thread_sweep.yaml --dry-run

# Real run (requires ollama.service active)
~/ai_agents_env/bin/python experiments/runner.py \
  --matrix experiments/matrices/phase_38a_num_thread_sweep.yaml
```

`--skip-ollama-check` bypasses the precheck when you know what you are
doing (e.g. running the bench against a remote Ollama).

## Scheduled run (systemd user timer)

See `cron/llm-nightly.service.template` for install steps. tl;dr:

```bash
mkdir -p ~/.config/systemd/user
sed "s|__REPO__|$HOME/projects/autonomous-local-llm|g" \
  cron/llm-nightly.service.template > ~/.config/systemd/user/llm-nightly.service
sed "s|__REPO__|$HOME/projects/autonomous-local-llm|g" \
  cron/llm-nightly.timer.template > ~/.config/systemd/user/llm-nightly.timer
systemctl --user daemon-reload
systemctl --user enable --now llm-nightly.timer
```

WSL2 note: `systemctl --user` requires the user manager to be running
(systemd-logind sets it up on login). If `loginctl enable-linger
$USER` was already applied for other services (e.g. fnm shim), the
timer will fire even when the user is not logged in.

## Failure handling

- The runner exits non-zero if any sweep entry returns non-zero, but it
  still runs every entry. Look at `summary.jsonl` for `rc` per stage.
- `ollama.service` not active: the wrapper script and the runner both
  short-circuit before launching the bench.
- bench-internal errors (Ollama 500, OOM): land in the per-entry
  `<config_id>.log` and are summarized in `summary.jsonl`.

## Adding a new matrix

1. Drop a new yaml under `experiments/matrices/`.
2. `runner.py --matrix <path> --dry-run` to sanity-check.
3. Either point `cron/llm_nightly_experiment.sh` at it via
   `MATRIX_YAML=...` or invoke manually.

A few ideas for follow-up matrices, none implemented yet:
- `phase_38a_temperature_sweep.yaml` — vary `--temperature` (needs a
  bench flag; not yet wired).
- `phase_38a_prompt_mode_sweep.yaml` — alternate `--prompt-mode long`
  and `--prompt-mode short` to track early-EOS rate over time.
- `phase_38b_router_smoke.yaml` — once a router PoC lands, replay a
  small set of representative prompts through it and record per-route
  latency / verdict.

## Related

- `bench/parallel_capacity_check.py` — the only bench currently used.
- `metrics/parallel_capacity_checks.jsonl` — where bench output lands.
- `analyze_runs.py` — group-by aggregator over the same JSONL.
- `cron/llm_nightly_experiment.sh` — the entrypoint that systemd calls.
