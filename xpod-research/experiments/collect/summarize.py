#!/usr/bin/env python3
# summarize.py: 对 metrics.csv 做按 mode 与节点维度的统计汇总，输出 summary CSV 供对比与画图使用。Modified: 2026-04-28
import argparse
import csv
import logging
import os
import sys
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    idx = (n - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(n - 1, lo + 1)
    frac = idx - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(mean(vals))


def fmt_num(x: Optional[float], ndigits: int = 6) -> str:
    if x is None:
        return ""
    return f"{x:.{ndigits}f}"


def fmt_rate(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


def read_metrics_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def group_by_mode(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        mode = (row.get("mode") or "").strip()
        if mode not in out:
            out[mode] = []
        out[mode].append(row)
    return out


def is_truthy(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def summarize_overall(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    jct_s: List[float] = []
    queue_wait_s: List[float] = []
    transfer_s: List[float] = []
    cold_start_1 = 0
    sla_violated_1 = 0
    for row in rows:
        jct_ms = parse_float(row.get("jct_ms"))
        if jct_ms is not None:
            jct_s.append(jct_ms / 1000.0)

        qw_ms = parse_float(row.get("queue_wait_ms"))
        if qw_ms is not None:
            queue_wait_s.append(qw_ms / 1000.0)

        tr_ms = parse_float(row.get("estimated_transfer_ms"))
        if tr_ms is not None:
            transfer_s.append(tr_ms / 1000.0)

        if str(row.get("cold_start") or "").strip() == "1":
            cold_start_1 += 1
        if is_truthy(row.get("sla_violated") or ""):
            sla_violated_1 += 1

    jct_s_sorted = sorted(jct_s)
    queue_wait_s_sorted = sorted(queue_wait_s)
    transfer_s_sorted = sorted(transfer_s)
    count = len(rows)
    return {
        "count": count,
        "avg_jct_s": safe_mean(jct_s),
        "p50_jct_s": percentile(jct_s_sorted, 50),
        "p95_jct_s": percentile(jct_s_sorted, 95),
        "p99_jct_s": percentile(jct_s_sorted, 99),
        "avg_queue_wait_s": safe_mean(queue_wait_s),
        "p95_queue_wait_s": percentile(queue_wait_s_sorted, 95),
        "cold_start_rate": (cold_start_1 / count) if count else None,
        "sla_violation_rate": (sla_violated_1 / count) if count else None,
        "avg_transfer_s": safe_mean(transfer_s),
        "p95_transfer_s": percentile(transfer_s_sorted, 95),
    }


def summarize_nodes(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        mode = (row.get("mode") or "").strip()
        node = (row.get("scheduled_node") or "").strip()
        key = (mode, node)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (mode, node), rs in grouped.items():
        jct_s: List[float] = []
        for row in rs:
            jct_ms = parse_float(row.get("jct_ms"))
            if jct_ms is not None:
                jct_s.append(jct_ms / 1000.0)
        jct_s_sorted = sorted(jct_s)
        out[(mode, node)] = {
            "mode": mode,
            "node": node,
            "count": len(rs),
            "avg_jct_s": safe_mean(jct_s),
            "p95_jct_s": percentile(jct_s_sorted, 95),
        }
    return out


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", vars(args))

    metrics_path = os.path.join("results", "raw", str(args.run_id), "metrics.csv")
    rows = read_metrics_rows(metrics_path)
    if not rows:
        sys.stderr.write(f"metrics not found or empty: {metrics_path}\n")
        return 2

    by_mode = group_by_mode(rows)

    overall_rows: List[Dict[str, Any]] = []
    for mode, rs in sorted(by_mode.items(), key=lambda kv: kv[0]):
        s = summarize_overall(rs)
        overall_rows.append(
            {
                "mode": mode,
                "count": s["count"],
                "avg_jct_s": fmt_num(s["avg_jct_s"], 6),
                "p50_jct_s": fmt_num(s["p50_jct_s"], 6),
                "p95_jct_s": fmt_num(s["p95_jct_s"], 6),
                "p99_jct_s": fmt_num(s["p99_jct_s"], 6),
                "avg_queue_wait_s": fmt_num(s["avg_queue_wait_s"], 6),
                "p95_queue_wait_s": fmt_num(s["p95_queue_wait_s"], 6),
                "cold_start_rate": fmt_rate(s["cold_start_rate"]),
                "sla_violation_rate": fmt_rate(s["sla_violation_rate"]),
                "avg_transfer_s": fmt_num(s["avg_transfer_s"], 6),
                "p95_transfer_s": fmt_num(s["p95_transfer_s"], 6),
            }
        )

    nodes_stats = summarize_nodes(rows)
    node_rows: List[Dict[str, Any]] = []
    for key in sorted(nodes_stats.keys()):
        s = nodes_stats[key]
        node_rows.append(
            {
                "mode": s["mode"],
                "node": s["node"],
                "count": s["count"],
                "avg_jct_s": fmt_num(s["avg_jct_s"], 6),
                "p95_jct_s": fmt_num(s["p95_jct_s"], 6),
            }
        )

    out_dir = os.path.join("results", "summary", str(args.run_id))
    overall_path = os.path.join(out_dir, "summary_overall.csv")
    nodes_path = os.path.join(out_dir, "summary_nodes.csv")

    write_csv(
        overall_path,
        [
            "mode",
            "count",
            "avg_jct_s",
            "p50_jct_s",
            "p95_jct_s",
            "p99_jct_s",
            "avg_queue_wait_s",
            "p95_queue_wait_s",
            "cold_start_rate",
            "sla_violation_rate",
            "avg_transfer_s",
            "p95_transfer_s",
        ],
        overall_rows,
    )
    write_csv(nodes_path, ["mode", "node", "count", "avg_jct_s", "p95_jct_s"], node_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
