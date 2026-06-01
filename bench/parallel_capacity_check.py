#!/usr/bin/env python3
"""Phase 3.7c: Parallel capacity benchmark.

Measures whether gemma2:9b (CPU only, num_gpu=0) and deepseek-r1:14b (default
GPU partial offload) can coexist under 8GB VRAM / 15GB RAM, and the throughput
penalty when both are queried in parallel. One JSONL record appended to
metrics/parallel_capacity_checks.jsonl.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
GEMMA = "gemma2:9b-instruct-q4_K_M"
DEEPSEEK = "deepseek-r1:14b"
PROMPT = "What is the capital of Japan? Answer in one word."
NUM_PREDICT = 50
WARMUP_NUM_PREDICT = 10
DEFAULT_RUNS = 3
SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_FILE = PROJECT_ROOT / "metrics" / "parallel_capacity_checks.jsonl"


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
    model: str, prompt: str, num_predict: int, num_gpu: int | None
) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": num_predict, "temperature": 0.1}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
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


def tps(resp: dict | None) -> float:
    if not resp:
        return 0.0
    ec = resp.get("eval_count", 0)
    ed = resp.get("eval_duration", 0)  # ns
    return (ec / (ed / 1e9)) if ec and ed else 0.0


def warmup(model: str, num_gpu: int | None) -> None:
    print(f"  warmup {model} (num_gpu={num_gpu})...", flush=True)
    ollama_generate(model, "Hello", WARMUP_NUM_PREDICT, num_gpu)


def measure_solo(model: str, num_gpu: int | None, runs: int) -> list[float]:
    results: list[float] = []
    for i in range(runs):
        resp = ollama_generate(model, PROMPT, NUM_PREDICT, num_gpu)
        results.append(tps(resp))
        print(f"  {model} solo run {i + 1}/{runs}: {results[-1]:.2f} t/s", flush=True)
    return results


def measure_concurrent(runs: int) -> tuple[list[float], list[float], list[float]]:
    gemma_results: list[float] = []
    deepseek_results: list[float] = []
    wall_results: list[float] = []
    for i in range(runs):
        results: dict[str, dict | None] = {"gemma": None, "deepseek": None}

        def call_gemma() -> None:
            results["gemma"] = ollama_generate(GEMMA, PROMPT, NUM_PREDICT, num_gpu=0)

        def call_deepseek() -> None:
            results["deepseek"] = ollama_generate(
                DEEPSEEK, PROMPT, NUM_PREDICT, num_gpu=None
            )

        t_start = time.monotonic()
        t1 = threading.Thread(target=call_gemma)
        t2 = threading.Thread(target=call_deepseek)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        wall = time.monotonic() - t_start

        gtps = tps(results["gemma"])
        dtps = tps(results["deepseek"])
        gemma_results.append(gtps)
        deepseek_results.append(dtps)
        wall_results.append(wall)
        print(
            f"  concurrent run {i + 1}/{runs}: gemma {gtps:.2f}, "
            f"deepseek {dtps:.2f} t/s, wall {wall:.2f}s",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    ap.add_argument("--dry-run", action="store_true", help="skip JSONL write")
    args = ap.parse_args()

    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()
    print(f"=== Phase 3.7c parallel capacity check ({started_at.isoformat()}) ===")

    print("\n[1/8] baseline VRAM/RAM")
    baseline_vram = nvidia_smi_vram_used_mb()
    baseline_ram = ram_used_mb()
    print(f"  vram={baseline_vram}MB ram={baseline_ram}MB load={loadavg()}")

    print(f"\n[2/8] {GEMMA} load (num_gpu=0) + warmup")
    warmup(GEMMA, num_gpu=0)
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
    warmup(DEEPSEEK, num_gpu=None)
    deepseek_only_vram = nvidia_smi_vram_used_mb()
    deepseek_only_ram = ram_used_mb()
    print(
        f"  vram={deepseek_only_vram}MB ram={deepseek_only_ram}MB "
        f"(Δvram={deepseek_only_vram - baseline_vram}MB = deepseek offload)"
    )

    print(f"\n[5/8] {GEMMA} re-load (both should coexist)")
    warmup(GEMMA, num_gpu=0)
    loaded = [m["name"] for m in ollama_loaded()]
    both_vram = nvidia_smi_vram_used_mb()
    both_ram = ram_used_mb()
    print(f"  loaded: {loaded}")
    print(
        f"  vram={both_vram}MB ram={both_ram}MB "
        f"(Δfrom_deepseek_only={both_vram - deepseek_only_vram}MB)"
    )

    print(f"\n[6/8] solo throughput x{args.runs}")
    print("  -- gemma solo --")
    gemma_solo = measure_solo(GEMMA, num_gpu=0, runs=args.runs)
    print("  -- deepseek solo --")
    deepseek_solo = measure_solo(DEEPSEEK, num_gpu=None, runs=args.runs)

    print(f"\n[7/8] concurrent throughput x{args.runs}")
    gemma_concurrent, deepseek_concurrent, wall_concurrent = measure_concurrent(args.runs)

    duration = time.monotonic() - t0
    finished_at = datetime.now(timezone.utc)
    print(f"\n[8/8] writing JSONL ({duration:.1f}s total elapsed)")

    gemma_solo_avg = statistics.fmean(gemma_solo) if gemma_solo else 0.0
    deepseek_solo_avg = statistics.fmean(deepseek_solo) if deepseek_solo else 0.0
    gemma_concurrent_avg = (
        statistics.fmean(gemma_concurrent) if gemma_concurrent else 0.0
    )
    deepseek_concurrent_avg = (
        statistics.fmean(deepseek_concurrent) if deepseek_concurrent else 0.0
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
            "gemma_solo": gemma_solo,
            "deepseek_solo": deepseek_solo,
            "gemma_concurrent": gemma_concurrent,
            "deepseek_concurrent": deepseek_concurrent,
        },
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
            "prompt": PROMPT,
            "num_predict": NUM_PREDICT,
            "warmup_num_predict": WARMUP_NUM_PREDICT,
            "runs_per_phase": args.runs,
            "ollama_base_url": OLLAMA_BASE_URL,
            "ollama_max_loaded_models_env": os.environ.get("OLLAMA_MAX_LOADED_MODELS")
            or "service(2)",
            "ollama_keep_alive_env": os.environ.get("OLLAMA_KEEP_ALIVE") or "service(5m)",
            "gemma_num_gpu": 0,
            "deepseek_num_gpu": None,
            "temperature": 0.1,
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
