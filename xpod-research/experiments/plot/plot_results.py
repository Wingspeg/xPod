#!/usr/bin/env python3
# plot_results.py: 从原始 metrics.csv 直接生成对比图（CDF/柱状/箱线/节点分布）并保存到 results/figures/{run_id}/。Modified: 2026-04-28
import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

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


def is_truthy(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def read_metrics(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def group_modes(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        mode = (row.get("mode") or "").strip()
        if mode not in out:
            out[mode] = []
        out[mode].append(row)
    return out


def compute_cdf(values: List[float]) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def annotate_bar_values(ax, bars, fmt: str = "{:.2f}%") -> None:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


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
    rows = read_metrics(metrics_path)
    if not rows:
        sys.stderr.write(f"metrics not found or empty: {metrics_path}\n")
        return 2

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.stderr.write(f"matplotlib import failed: {e}\n")
        return 3

    plt.style.use("seaborn-v0_8-paper")

    out_dir = os.path.join("results", "figures", str(args.run_id))
    ensure_dir(out_dir)

    by_mode = group_modes(rows)
    modes = [m for m in sorted(by_mode.keys()) if m != ""]
    if "" in by_mode and "" not in modes:
        modes.append("")

    mode_to_jct_s: Dict[str, List[float]] = {}
    mode_to_transfer_s: Dict[str, List[float]] = {}
    mode_to_cold_rate: Dict[str, float] = {}
    mode_to_sla_rate: Dict[str, float] = {}

    for mode, rs in by_mode.items():
        jcts: List[float] = []
        transfers: List[float] = []
        cold_1 = 0
        sla_1 = 0
        for row in rs:
            jct_ms = parse_float(row.get("jct_ms"))
            if jct_ms is not None:
                jcts.append(jct_ms / 1000.0)
            tr_ms = parse_float(row.get("estimated_transfer_ms"))
            if tr_ms is not None:
                transfers.append(tr_ms / 1000.0)
            if str(row.get("cold_start") or "").strip() == "1":
                cold_1 += 1
            if is_truthy(row.get("sla_violated") or ""):
                sla_1 += 1
        mode_to_jct_s[mode] = jcts
        mode_to_transfer_s[mode] = transfers
        mode_to_cold_rate[mode] = (cold_1 / len(rs)) * 100.0 if rs else 0.0
        mode_to_sla_rate[mode] = (sla_1 / len(rs)) * 100.0 if rs else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for mode in modes:
        xs, ys = compute_cdf(mode_to_jct_s.get(mode, []))
        if not xs:
            continue
        ax.plot(xs, ys, label=mode or "(unknown)")
    ax.set_xlabel("JCT (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "jct_cdf.png"), dpi=300)
    plt.close(fig)

    labels = [m or "(unknown)" for m in modes]
    cold_rates = [mode_to_cold_rate.get(m, 0.0) for m in modes]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, cold_rates, color="C0")
    ax.set_xlabel("mode")
    ax.set_ylabel("cold_start_rate (%)")
    annotate_bar_values(ax, bars, fmt="{:.2f}%")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cold_start_rate.png"), dpi=300)
    plt.close(fig)

    sla_rates = [mode_to_sla_rate.get(m, 0.0) for m in modes]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, sla_rates, color="C1")
    ax.set_xlabel("mode")
    ax.set_ylabel("sla_violation_rate (%)")
    annotate_bar_values(ax, bars, fmt="{:.2f}%")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sla_violation_rate.png"), dpi=300)
    plt.close(fig)

    box_data = [mode_to_transfer_s.get(m, []) for m in modes]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.boxplot(box_data, labels=labels, showfliers=False)
    ax.set_xlabel("mode")
    ax.set_ylabel("transfer_time (s)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transfer_time_boxplot.png"), dpi=300)
    plt.close(fig)

    node_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    nodes_set = set()
    for row in rows:
        mode = (row.get("mode") or "").strip()
        node = (row.get("scheduled_node") or "").strip()
        nodes_set.add(node)
        node_counts[(mode, node)] += 1
    nodes = sorted(nodes_set)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.5 * len(nodes)), 4.0))
    x = list(range(len(nodes)))
    mcount = max(1, len(modes))
    total_width = 0.8
    bar_w = total_width / mcount
    for i, mode in enumerate(modes):
        offsets = [xi - total_width / 2.0 + (i + 0.5) * bar_w for xi in x]
        ys = [node_counts.get((mode, node), 0) for node in nodes]
        ax.bar(offsets, ys, width=bar_w, label=mode or "(unknown)")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=30, ha="right")
    ax.set_xlabel("node")
    ax.set_ylabel("schedule count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "node_schedule_dist.png"), dpi=300)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
