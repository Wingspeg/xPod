import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Sequence

from . import io, yamlutil
from .scheduler import CacheState, ClusterConfig, NodeLoadTracker, ServiceTracker, schedule_one
from .specs import (
    build_fluid_dataset_manifest,
    build_fluid_runtime_manifest,
    build_volcano_job_manifest,
    build_xpod_manifest,
)

try:
    from scheduler.xpod_scheduler import schedule_one_baseline, schedule_one_xpod
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scheduler.xpod_scheduler import schedule_one_baseline, schedule_one_xpod

logger = logging.getLogger(__name__)


def write_docs(docs: Sequence[Dict[str, Any]], fmt: str, out: Any) -> None:
    if fmt == "json":
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
        return
    first = True
    for d in docs:
        if not first:
            out.write("\n---\n")
        first = False
        out.write(yamlutil.to_yaml(d))
        out.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--format", choices=["yaml", "json"], default="yaml")
    ap.add_argument("--output", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--mode", choices=["xpod", "advanced", "baseline"], default="xpod")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)

    ap.add_argument("--data-nodes", default="data-node-1,data-node-2")
    ap.add_argument("--gpu-nodes", default="gpu-node-1")
    ap.add_argument("--cpu-nodes", default="cpu-node-1,cpu-node-2")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--base-latency-ms", type=float, default=2.0)
    ap.add_argument("--base-bandwidth-mbps", type=float, default=2000.0)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--sla-penalty", type=float, default=10.0)

    ap.add_argument("--xpod-alpha", type=float, default=1.0)
    ap.add_argument("--xpod-beta", type=float, default=1.5)
    ap.add_argument("--xpod-gamma", type=float, default=0.8)
    ap.add_argument("--disable-hash-tiebreak", action="store_true", help="关闭 hash tie-break,用于 ablation 实验")
    ap.add_argument("--disable-las-priority", action="store_true", help="关闭 Tiresias 2D-LAS,用于 ablation 实验")
    ap.add_argument("--disable-contention", action="store_true", help="关闭 contention-aware t_compute,用于 ablation 实验")
    ap.add_argument("--node-capacity", default="gpu-node-1:1,data-node-1:1,data-node-2:1,algo-node-1:1,cpu-node-1:4,cpu-node-2:4",
                    help="节点容量配置，格式: node1:cap1,node2:cap2,...")
    return ap


def _parse_node_capacity(s: str) -> Dict[str, float]:
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def main(argv: List[str]) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", vars(args))

    cluster = ClusterConfig(
        data_nodes=tuple(io.split_csv_list(args.data_nodes)),
        gpu_nodes=tuple(io.split_csv_list(args.gpu_nodes)),
        cpu_nodes=tuple(io.split_csv_list(args.cpu_nodes)),
        seed=args.seed,
        base_latency_ms=args.base_latency_ms,
        base_bandwidth_mbps=args.base_bandwidth_mbps,
    )

    cache = CacheState()
    service_tracker = ServiceTracker()
    capacity = _parse_node_capacity(args.node_capacity)
    load_tracker = NodeLoadTracker(capacity)
    docs: List[Dict[str, Any]] = []
    n = 0
    for job_index, row in enumerate(io.parse_csv(args.input)):
        if n >= args.limit:
            break
        ts = float(row.get("submit_time", 0))
        load_tracker.release_until(ts)
        if args.mode == "baseline":
            decision = schedule_one_baseline(row, cluster, cache, service_tracker, load_tracker)
            scheduler_name = "default-scheduler"
        elif args.mode == "advanced":
            decision = schedule_one_xpod(
                row,
                cluster,
                cache,
                service_tracker,
                load_tracker,
                job_index=job_index,
                alpha=args.xpod_alpha,
                beta=args.xpod_beta,
                gamma=args.xpod_gamma,
                enable_hash_tiebreak=(not args.disable_hash_tiebreak),
                enable_las_priority=(not args.disable_las_priority),
                enable_contention=(not args.disable_contention),
            )
            node_prefix = (decision.compute_node.split("-")[0] if decision else "").lower()
            if node_prefix in ("gpu", "cpu"):
                scheduler_name = "volcano"
            elif node_prefix == "algo":
                scheduler_name = "koord-scheduler"
            elif node_prefix == "data":
                scheduler_name = "default-scheduler"
            else:
                scheduler_name = "volcano"
        else:
            decision = schedule_one(row, cluster, cache, args.alpha, args.beta, args.gamma, args.sla_penalty)
            scheduler_name = "volcano"
        if decision is None:
            continue
        xpod_doc = build_xpod_manifest(row, decision, args.namespace)
        if args.mode == "baseline":
            xpod_doc.get("spec", {}).get("compute", {})["scheduler"] = "default-scheduler"
            xpod_doc.get("spec", {}).get("data", {})["scheduler"] = ""
            xpod_doc.get("spec", {}).get("algorithm", {})["scheduler"] = "default-scheduler"
        elif args.mode == "advanced":
            xpod_doc.get("spec", {}).get("compute", {})["scheduler"] = scheduler_name
        docs.append(xpod_doc)
        if args.mode != "baseline":
            docs.append(build_fluid_dataset_manifest(row, decision, args.namespace))
            docs.append(build_fluid_runtime_manifest(row, decision, args.namespace))
            docs.append(build_volcano_job_manifest(row, decision, args.namespace, scheduler_name=scheduler_name))
        else:
            docs.append(build_volcano_job_manifest(row, decision, args.namespace, scheduler_name="default-scheduler"))
        n += 1

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            write_docs(docs, args.format, f)
    else:
        write_docs(docs, args.format, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
