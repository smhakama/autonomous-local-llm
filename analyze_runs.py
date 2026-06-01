#!/usr/bin/env python3
"""Aggregate metrics/distill_runs.jsonl into group-by summaries.

Phase 3.7b: JSONL aggregator. Pure stdlib (json/argparse/statistics/datetime/pathlib).
Default output is JSON (jq-friendly); --table switches to fixed-width text.

Examples:
    ./analyze_runs.py
    ./analyze_runs.py --group-by theme
    ./analyze_runs.py --group-by theme,config.primary_model --table
    ./analyze_runs.py --since 2026-06-01 --theme kubernetes
    ./analyze_runs.py | jq '.groups[] | {key, count, avg_elapsed_sec}'
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_METRICS_FILE = Path("metrics/distill_runs.jsonl")

# Group-by fields the user is allowed to reference. dotted paths are resolved
# against the record dict. Limit to fields that produce stable, hashable keys.
ALLOWED_GROUP_FIELDS = {
    "theme",
    "valid",
    "fallback_model_used",
    "config.primary_model",
    "config.fallback_model",
    "config.enable_fallback",
    "config.quality_loop",
    "config.rag_augmented",
    "config.rag_adaptive",
    "config.rag_top_k",
    "config.rag_adaptive_schedule",
    "config.collection",
    "meta.host",
    "meta.git_commit",
    "meta.schema_version",
    # Phase 3.7d: system snapshot bool fields (numeric fields like vram_used_mb
    # are continuous and not useful as group keys; list fields like
    # concurrent_models are not hashable). Filter/inspect those via jq instead.
    "system.baseline.nvidia_smi_available",
    "system.end.nvidia_smi_available",
}


def warn(msg: str) -> None:
    print(f"analyze_runs: warning: {msg}", file=sys.stderr)


def parse_iso(value: str) -> datetime:
    """Parse ISO-8601 date or datetime; naive input is treated as UTC."""
    # datetime.fromisoformat accepts "YYYY-MM-DD" since 3.11, but we also accept
    # full timestamps. Normalize trailing "Z" for older inputs.
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def resolve_dotted(record: dict, path: str) -> Any:
    """Resolve a dotted path (e.g. 'config.primary_model') against a record."""
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def read_records(path: Path) -> Iterable[dict]:
    """Stream JSONL records. Broken lines are skipped with a stderr warning."""
    if not path.exists():
        warn(f"metrics file not found: {path}")
        return
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                warn(f"{path}:{line_no} skipped (invalid JSON: {e.msg})")


def record_finished_at(record: dict) -> datetime | None:
    raw = resolve_dotted(record, "meta.finished_at")
    if not isinstance(raw, str):
        return None
    try:
        return parse_iso(raw)
    except ValueError:
        return None


def passes_filters(
    record: dict,
    *,
    since: datetime | None,
    until: datetime | None,
    theme_substr: str | None,
    primary_model: str | None,
) -> bool:
    if since is not None or until is not None:
        ts = record_finished_at(record)
        if ts is None:
            return False
        if since is not None and ts < since:
            return False
        if until is not None and ts > until:
            return False
    if theme_substr is not None:
        theme = record.get("theme")
        if not isinstance(theme, str) or theme_substr.lower() not in theme.lower():
            return False
    if primary_model is not None:
        model = resolve_dotted(record, "config.primary_model")
        if model != primary_model:
            return False
    return True


def group_key(record: dict, fields: list[str]) -> tuple:
    """Build a tuple of (field, value) pairs for grouping; tuple is hashable."""
    return tuple((f, resolve_dotted(record, f)) for f in fields)


def aggregate(records: list[dict]) -> dict:
    elapsed = [
        r["elapsed_sec"]
        for r in records
        if isinstance(r.get("elapsed_sec"), (int, float))
    ]
    success = sum(1 for r in records if r.get("valid") is True)
    fallback = sum(1 for r in records if r.get("fallback_model_used") is not None)
    count = len(records)
    return {
        "count": count,
        "avg_elapsed_sec": round(statistics.fmean(elapsed), 3) if elapsed else None,
        "median_elapsed_sec": (
            round(statistics.median(elapsed), 3) if elapsed else None
        ),
        "success_rate": round(success / count, 4) if count else None,
        "fallback_fired_rate": round(fallback / count, 4) if count else None,
    }


def build_result(
    records: list[dict],
    *,
    group_fields: list[str],
    filters: dict,
) -> dict:
    if not group_fields:
        groups = [{"key": {}, **aggregate(records)}] if records else []
    else:
        buckets: dict[tuple, list[dict]] = {}
        for r in records:
            buckets.setdefault(group_key(r, group_fields), []).append(r)
        groups = [
            {"key": dict(key), **aggregate(bucket)}
            for key, bucket in buckets.items()
        ]
        groups.sort(key=lambda g: tuple(str(g["key"].get(f)) for f in group_fields))
    return {
        "filters": filters,
        "group_by": group_fields,
        "total_records": len(records),
        "groups": groups,
    }


def format_table(result: dict) -> str:
    group_fields = result["group_by"]
    rows = result["groups"]
    if not rows:
        return f"No records (total_records={result['total_records']}).\n"

    metric_cols = [
        ("count", "count", lambda v: str(v)),
        ("avg_elapsed_sec", "avg_s", lambda v: "—" if v is None else f"{v:.1f}"),
        ("median_elapsed_sec", "med_s", lambda v: "—" if v is None else f"{v:.1f}"),
        ("success_rate", "succ", lambda v: "—" if v is None else f"{v * 100:.1f}%"),
        (
            "fallback_fired_rate",
            "fb",
            lambda v: "—" if v is None else f"{v * 100:.1f}%",
        ),
    ]
    headers = [*group_fields, *(label for _, label, _ in metric_cols)]
    table_rows = []
    for row in rows:
        cells = [str(row["key"].get(f, "")) for f in group_fields]
        cells += [render(row[key]) for key, _, render in metric_cols]
        table_rows.append(cells)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]
    sep = " | "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for r in table_rows:
        lines.append(sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append(f"\ntotal_records={result['total_records']}")
    return "\n".join(lines) + "\n"


def parse_group_by(value: str) -> list[str]:
    fields = [f.strip() for f in value.split(",") if f.strip()]
    unknown = [f for f in fields if f not in ALLOWED_GROUP_FIELDS]
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_GROUP_FIELDS))
        raise argparse.ArgumentTypeError(
            f"unknown group-by field(s): {unknown}. allowed: {allowed}"
        )
    return fields


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate distill_runs.jsonl into group-by summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help=f"path to JSONL (default: {DEFAULT_METRICS_FILE})",
    )
    p.add_argument(
        "--group-by",
        type=parse_group_by,
        default=[],
        metavar="FIELDS",
        help="comma-separated dotted fields (e.g. theme,config.primary_model)",
    )
    p.add_argument("--since", type=parse_iso, default=None, metavar="ISO_DATE")
    p.add_argument("--until", type=parse_iso, default=None, metavar="ISO_DATE")
    p.add_argument("--theme", type=str, default=None, metavar="SUBSTR")
    p.add_argument("--primary-model", type=str, default=None, metavar="EXACT")
    p.add_argument("--table", action="store_true", help="text table (default: JSON)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [
        r
        for r in read_records(args.metrics_file)
        if passes_filters(
            r,
            since=args.since,
            until=args.until,
            theme_substr=args.theme,
            primary_model=args.primary_model,
        )
    ]
    filters = {
        "since": args.since.isoformat() if args.since else None,
        "until": args.until.isoformat() if args.until else None,
        "theme": args.theme,
        "primary_model": args.primary_model,
    }
    result = build_result(records, group_fields=args.group_by, filters=filters)
    if args.table:
        sys.stdout.write(format_table(result))
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
