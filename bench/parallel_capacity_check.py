#!/usr/bin/env python3
"""Phase 3.7c / 3.7e-1 / 3.7e-2: Parallel capacity benchmark.

Measures whether gemma2:9b (CPU only, num_gpu=0) and deepseek-r1:14b (default
GPU partial offload) can coexist under 8GB VRAM / 15GB RAM, and the throughput
penalty when both are queried in parallel. One JSONL record appended to
metrics/parallel_capacity_checks.jsonl.

Phase 3.7e-1 (schema v2):
  - Records per-run RunStats (eval_count, prompt_eval_count, *_duration_ns)
    so interference can be separated from early-EOS (Phase 3.7c short prompt
    finished with eval_count=4 — wall was dominated by load + prompt_eval).
  - --prompt-mode {short,long}; long uses a ~800-char English prompt with
    num_predict=300 to keep the measurement window > early-EOS.
  - wall_vs_total_max_ratio = wall / max(gemma.total_duration, deepseek.total_duration).
    ≈ 1.0 means wall is bounded by the slower model's total_duration (no
    queue overhead). > 1.0 means thread/queue overhead beyond Ollama-internal.

Phase 3.7e-2 (schema v3): C vs D hypothesis split.
  - --num-thread N: pass options.num_thread to both models (None=Ollama default).
  - --mem-stress: run stress-ng --vm 2 --vm-bytes 1G --vm-keep alongside the
    concurrent measurement to add RAM-bandwidth pressure (proxy for hypothesis D).
  - --bind-cores-label: free-form label recorded in config (actual taskset of
    `ollama serve` is done outside via systemd override; the script only labels).
  - --config-id: free-form matrix tag (e.g. E0, E1, E3, E4) attached to record.

Sequence:
  1. baseline VRAM/RAM
  2. load gemma (warmup) → gemma_only VRAM/RAM (delta = CUDA context overhead)
  3. unload gemma
  4. load deepseek (warmup) → deepseek_only VRAM/RAM
  5. re-load gemma alongside deepseek → both_loaded VRAM/RAM
  6. solo throughput: gemma x N, deepseek x N
  7. concurrent throughput: gemma+deepseek in parallel threads x N
  8. write JSONL
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
GEMMA = "gemma2:9b-instruct-q4_K_M"
DEEPSEEK = "deepseek-r1:14b"

SHORT_PROMPT = "What is the capital of Japan? Answer in one word."
SHORT_NUM_PREDICT = 50
LONG_PROMPT = (
    "Explain in detail what a large language model is, how it is trained, "
    "and how it differs from earlier statistical language models. Cover the "
    "transformer architecture, the role of self-attention, the distinction "
    "between pre-training and fine-tuning, the role of reinforcement learning "
    "from human feedback, common evaluation benchmarks such as MMLU and "
    "HumanEval, and the engineering challenges of scaling parameter counts "
    "from millions to hundreds of billions. Discuss the trade-offs between "
    "model size, inference cost, latency, and quality, and explain why "
    "smaller specialised models are sometimes preferred over larger general "
    "ones. Finish with a short paragraph on open-weight models and their "
    "impact on local inference."
)
LONG_NUM_PREDICT = 300

WARMUP_NUM_PREDICT = 10
DEFAULT_RUNS = 3
SCHEMA_VERSION = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_FILE = PROJECT_ROOT / "metrics" / "parallel_capacity_checks.jsonl"

STRESS_NG_VM_INSTANCES = 2
STRESS_NG_VM_BYTES = "1G"


def nvidia_smi_vram_used_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def loadavg() -> list[float]:
    return list(os.getloadavg())


def ram_used_mb() -> int:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split(":")
        if len(parts) == 2:
            info[parts[0].strip()] = int(parts[1].split()[0])  # kB
    return (info["MemTotal"] - info["MemAvailable"]) // 1024


def ollama_generate(
    model: str,
    prompt: str,
    num_predict: int,
    num_gpu: int | None,
    num_thread: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": num_predict, "temperature": 0.1}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_thread is not None:
        options["num_thread"] = num_thread
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def ollama_loaded() -> list[dict]:
    r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def ollama_unload(model: str) -> None:
    """Send empty generate with keep_alive=0 to force unload."""
    requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
        timeout=30,
    )


@dataclass(frozen=True)
class RunStats:
    """Phase 3.7e-1 schema v2: full Ollama /api/generate timing record.

    eval_tps is derived from eval_count / eval_duration_s, matching Phase 3.7c
    semantics. Use total_duration_ns when comparing against concurrent wall.
    """

    eval_count: int
    prompt_eval_count: int
    eval_duration_ns: int
    prompt_eval_duration_ns: int
    total_duration_ns: int
    load_duration_ns: int
    eval_tps: float

    @staticmethod
    def from_response(resp: dict[str, Any] | None) -> "RunStats":
        if not resp:
            return RunStats(0, 0, 0, 0, 0, 0, 0.0)
        ec = int(resp.get("eval_count", 0) or 0)
        ed = int(resp.get("eval_duration", 0) or 0)
        return RunStats(
            eval_count=ec,
            prompt_eval_count=int(resp.get("prompt_eval_count", 0) or 0),
            eval_duration_ns=ed,
            prompt_eval_duration_ns=int(resp.get("prompt_eval_duration", 0) or 0),
            total_duration_ns=int(resp.get("total_duration", 0) or 0),
            load_duration_ns=int(resp.get("load_duration", 0) or 0),
            eval_tps=(ec / (ed / 1e9)) if ec and ed else 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def warmup(model: str, num_gpu: int | None, num_thread: int | None = None) -> None:
    print(f"  warmup {model} (num_gpu={num_gpu}, num_thread={num_thread})...", flush=True)
    ollama_generate(model, "Hello", WARMUP_NUM_PREDICT, num_gpu, num_thread)


def measure_solo(
    model: str,
    num_gpu: int | None,
    runs: int,
    prompt: str,
    num_predict: int,
    num_thread: int | None = None,
) -> list[RunStats]:
    results: list[RunStats] = []
    for i in range(runs):
        resp = ollama_generate(model, prompt, num_predict, num_gpu, num_thread)
        stats = RunStats.from_response(resp)
        results.append(stats)
        print(
            f"  {model} solo run {i + 1}/{runs}: "
            f"ec={stats.eval_count} "
            f"tps={stats.eval_tps:.2f} "
            f"pec={stats.prompt_eval_count} "
            f"total={stats.total_duration_ns / 1e9:.2f}s",
            flush=True,
        )
    return results


def measure_concurrent(
    runs: int, prompt: str, num_predict: int, num_thread: int | None = None
) -> tuple[list[RunStats], list[RunStats], list[float]]:
    gemma_results: list[RunStats] = []
    deepseek_results: list[RunStats] = []
    wall_results: list[float] = []
    for i in range(runs):
        results: dict[str, dict | None] = {"gemma": None, "deepseek": None}

        def call_gemma() -> None:
            results["gemma"] = ollama_generate(
                GEMMA, prompt, num_predict, num_gpu=0, num_thread=num_thread
            )

        def call_deepseek() -> None:
            results["deepseek"] = ollama_generate(
                DEEPSEEK, prompt, num_predict, num_gpu=None, num_thread=num_thread
            )

        t_start = time.monotonic()
        t1 = threading.Thread(target=call_gemma)
        t2 = threading.Thread(target=call_deepseek)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        wall = time.monotonic() - t_start

        gstats = RunStats.from_response(results["gemma"])
        dstats = RunStats.from_response(results["deepseek"])
        gemma_results.append(gstats)
        deepseek_results.append(dstats)
        wall_results.append(wall)
        print(
            f"  concurrent run {i + 1}/{runs}: "
            f"gemma tps={gstats.eval_tps:.2f} ec={gstats.eval_count}, "
            f"deepseek tps={dstats.eval_tps:.2f} ec={dstats.eval_count}, "
            f"wall {wall:.2f}s",
            flush=True,
        )
    return gemma_results, deepseek_results, wall_results


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def safe_ratio(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def stress_ng_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["stress-ng", "--version"], text=True, stderr=subprocess.STDOUT
        )
        return out.strip().splitlines()[0]
    except Exception:
        return None


def start_mem_stress() -> subprocess.Popen | None:
    """Phase 3.7e-2 D-hypothesis proxy: launch stress-ng vm load.

    Returns Popen so caller can terminate in finally block. Returns None and
    prints a warning if stress-ng is unavailable.
    """
    if not stress_ng_version():
        print("  [warn] stress-ng not found; --mem-stress is a no-op", flush=True)
        return None
    cmd = [
        "stress-ng",
        "--vm",
        str(STRESS_NG_VM_INSTANCES),
        "--vm-bytes",
        STRESS_NG_VM_BYTES,
        "--vm-keep",
    ]
    print(f"  starting mem stress: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_mem_stress(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    print("  terminating mem stress", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] stress-ng cleanup failed: {exc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    ap.add_argument("--dry-run", action="store_true", help="skip JSONL write")
    ap.add_argument(
        "--prompt-mode",
        choices=["short", "long"],
        default="long",
        help="short = Phase 3.7c reproducer (early-EOS, ~4 tokens); "
        "long (default) = ~800-char English prompt with num_predict=300",
    )
    ap.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Phase 3.7e-2: options.num_thread for both models "
        "(None = Ollama default). Used to test hypothesis C (SMT contention).",
    )
    ap.add_argument(
        "--mem-stress",
        action="store_true",
        help="Phase 3.7e-2: run stress-ng --vm 2 --vm-bytes 1G --vm-keep "
        "alongside concurrent measurement (hypothesis D: RAM bandwidth).",
    )
    ap.add_argument(
        "--bind-cores-label",
        type=str,
        default=None,
        help="Phase 3.7e-2: free-form label recorded in config. Actual taskset "
        "of `ollama serve` is done outside via systemd override.",
    )
    ap.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Phase 3.7e-2: free-form matrix tag (e.g. E0, E1, E3, E4).",
    )
    args = ap.parse_args()

    if args.prompt_mode == "long":
        prompt, num_predict = LONG_PROMPT, LONG_NUM_PREDICT
    else:
        prompt, num_predict = SHORT_PROMPT, SHORT_NUM_PREDICT

    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()
    print(f"=== Phase 3.7c parallel capacity check ({started_at.isoformat()}) ===")

    print("\n[1/8] baseline VRAM/RAM")
    baseline_vram = nvidia_smi_vram_used_mb()
    baseline_ram = ram_used_mb()
    print(f"  vram={baseline_vram}MB ram={baseline_ram}MB load={loadavg()}")

    print(f"\n[2/8] {GEMMA} load (num_gpu=0) + warmup")
    warmup(GEMMA, num_gpu=0, num_thread=args.num_thread)
    gemma_only_vram = nvidia_smi_vram_used_mb()
    gemma_only_ram = ram_used_mb()
    print(
        f"  vram={gemma_only_vram}MB ram={gemma_only_ram}MB "
        f"(Δvram={gemma_only_vram - baseline_vram}MB = CUDA context overhead estimate)"
    )

    print(f"\n[3/8] unload {GEMMA}")
    ollama_unload(GEMMA)
    time.sleep(3)
    loaded_after_unload = [m["name"] for m in ollama_loaded()]
    print(f"  loaded after unload: {loaded_after_unload}")

    print(f"\n[4/8] {DEEPSEEK} load + warmup")
    warmup(DEEPSEEK, num_gpu=None, num_thread=args.num_thread)
    deepseek_only_vram = nvidia_smi_vram_used_mb()
    deepseek_only_ram = ram_used_mb()
    print(
        f"  vram={deepseek_only_vram}MB ram={deepseek_only_ram}MB "
        f"(Δvram={deepseek_only_vram - baseline_vram}MB = deepseek offload)"
    )

    print(f"\n[5/8] {GEMMA} re-load (both should coexist)")
    warmup(GEMMA, num_gpu=0, num_thread=args.num_thread)
    loaded = [m["name"] for m in ollama_loaded()]
    both_vram = nvidia_smi_vram_used_mb()
    both_ram = ram_used_mb()
    print(f"  loaded: {loaded}")
    print(
        f"  vram={both_vram}MB ram={both_ram}MB "
        f"(Δfrom_deepseek_only={both_vram - deepseek_only_vram}MB)"
    )

    print(
        f"\n[6/8] solo throughput x{args.runs} "
        f"(mode={args.prompt_mode}, num_predict={num_predict})"
    )
    print("  -- gemma solo --")
    gemma_solo = measure_solo(
        GEMMA,
        num_gpu=0,
        runs=args.runs,
        prompt=prompt,
        num_predict=num_predict,
        num_thread=args.num_thread,
    )
    print("  -- deepseek solo --")
    deepseek_solo = measure_solo(
        DEEPSEEK,
        num_gpu=None,
        runs=args.runs,
        prompt=prompt,
        num_predict=num_predict,
        num_thread=args.num_thread,
    )

    print(f"\n[7/8] concurrent throughput x{args.runs}")
    stress_proc: subprocess.Popen | None = None
    try:
        if args.mem_stress:
            stress_proc = start_mem_stress()
        gemma_concurrent, deepseek_concurrent, wall_concurrent = measure_concurrent(
            args.runs,
            prompt=prompt,
            num_predict=num_predict,
            num_thread=args.num_thread,
        )
    finally:
        stop_mem_stress(stress_proc)

    duration = time.monotonic() - t0
    finished_at = datetime.now(timezone.utc)
    print(f"\n[8/8] writing JSONL ({duration:.1f}s total elapsed)")

    gemma_solo_tps = [s.eval_tps for s in gemma_solo]
    deepseek_solo_tps = [s.eval_tps for s in deepseek_solo]
    gemma_concurrent_tps = [s.eval_tps for s in gemma_concurrent]
    deepseek_concurrent_tps = [s.eval_tps for s in deepseek_concurrent]
    gemma_solo_avg = statistics.fmean(gemma_solo_tps) if gemma_solo_tps else 0.0
    deepseek_solo_avg = (
        statistics.fmean(deepseek_solo_tps) if deepseek_solo_tps else 0.0
    )
    gemma_concurrent_avg = (
        statistics.fmean(gemma_concurrent_tps) if gemma_concurrent_tps else 0.0
    )
    deepseek_concurrent_avg = (
        statistics.fmean(deepseek_concurrent_tps) if deepseek_concurrent_tps else 0.0
    )

    # Phase 3.7e-1: wall vs max(total_duration) ratio per concurrent run.
    # ≈ 1.0  → wall is bounded by the slower model's total_duration (no extra queue overhead)
    # > 1.0  → thread/queue overhead beyond Ollama-internal max
    concurrent_total_max_ms: list[float] = []
    wall_vs_total_max: list[float] = []
    for g, d, wall_s in zip(gemma_concurrent, deepseek_concurrent, wall_concurrent):
        max_total_ns = max(g.total_duration_ns, d.total_duration_ns)
        concurrent_total_max_ms.append(round(max_total_ns / 1e6, 3))
        wall_vs_total_max.append(
            round(wall_s / (max_total_ns / 1e9), 4) if max_total_ns > 0 else 0.0
        )

    record = {
        "theme": "parallel_capacity_check",
        "models": {"gemma": GEMMA, "deepseek": DEEPSEEK},
        "loaded_after_both": loaded,
        "vram_mb": {
            "baseline": baseline_vram,
            "gemma_only": gemma_only_vram,
            "deepseek_only": deepseek_only_vram,
            "both_loaded": both_vram,
            "cuda_context_overhead_estimate": gemma_only_vram - baseline_vram,
            "deepseek_offload_estimate": deepseek_only_vram - baseline_vram,
            "both_minus_deepseek": both_vram - deepseek_only_vram,
        },
        "ram_mb": {
            "baseline": baseline_ram,
            "gemma_only": gemma_only_ram,
            "deepseek_only": deepseek_only_ram,
            "both_loaded": both_ram,
        },
        "throughput_tps": {
            "gemma_solo": gemma_solo_tps,
            "deepseek_solo": deepseek_solo_tps,
            "gemma_concurrent": gemma_concurrent_tps,
            "deepseek_concurrent": deepseek_concurrent_tps,
        },
        "solo_stats": {
            "gemma": [s.to_dict() for s in gemma_solo],
            "deepseek": [s.to_dict() for s in deepseek_solo],
        },
        "concurrent_stats": {
            "gemma": [s.to_dict() for s in gemma_concurrent],
            "deepseek": [s.to_dict() for s in deepseek_concurrent],
        },
        "concurrent_total_duration_max_ms": concurrent_total_max_ms,
        "wall_vs_total_max_ratio": wall_vs_total_max,
        "throughput_avg_tps": {
            "gemma_solo_avg": round(gemma_solo_avg, 3),
            "deepseek_solo_avg": round(deepseek_solo_avg, 3),
            "gemma_concurrent_avg": round(gemma_concurrent_avg, 3),
            "deepseek_concurrent_avg": round(deepseek_concurrent_avg, 3),
            "gemma_interference_ratio": (
                round(r, 4)
                if (r := safe_ratio(gemma_concurrent_avg, gemma_solo_avg)) is not None
                else None
            ),
            "deepseek_interference_ratio": (
                round(r, 4)
                if (r := safe_ratio(deepseek_concurrent_avg, deepseek_solo_avg))
                is not None
                else None
            ),
        },
        "concurrent_wall_sec": [round(w, 3) for w in wall_concurrent],
        "cpu_loadavg_final": loadavg(),
        "config": {
            "prompt_mode": args.prompt_mode,
            "prompt": prompt,
            "prompt_length_chars": len(prompt),
            "num_predict": num_predict,
            "warmup_num_predict": WARMUP_NUM_PREDICT,
            "runs_per_phase": args.runs,
            "ollama_base_url": OLLAMA_BASE_URL,
            "ollama_max_loaded_models_env": os.environ.get("OLLAMA_MAX_LOADED_MODELS")
            or "service(2)",
            "ollama_keep_alive_env": os.environ.get("OLLAMA_KEEP_ALIVE") or "service(5m)",
            "gemma_num_gpu": 0,
            "deepseek_num_gpu": None,
            "temperature": 0.1,
            "num_thread": args.num_thread,
            "mem_stress": bool(args.mem_stress),
            "bind_cores_label": args.bind_cores_label,
            "config_id": args.config_id,
        },
        "system": {
            "stress_ng_version": stress_ng_version() if args.mem_stress else None,
        },
        "meta": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": round(duration, 3),
            "git_commit": git_commit(),
            "host": socket.gethostname(),
            "schema_version": SCHEMA_VERSION,
        },
    }

    print("\n--- record ---")
    print(json.dumps(record, indent=2, ensure_ascii=False))

    if not args.dry_run:
        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nappended to {args.metrics_file}")
    else:
        print("\n[dry-run] JSONL not written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
