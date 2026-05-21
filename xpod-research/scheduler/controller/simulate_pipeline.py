#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import Any, Dict, List

from xpodgen import dag, io, specs, yamlutil
from xpodgen.scheduler import CacheState, ClusterConfig, schedule_one

logger = logging.getLogger(__name__)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def print_dag(g: dag.XPodDAG, out: Any) -> None:
    out.write("DAG\n")
    out.write(f"- xpod_id: {g.xpod_id}\n")
    out.write(
        "- algorithm: "
        + f"framework={fmt(g.algorithm.framework)} "
        + f"type={fmt(g.algorithm.algo_type)} "
        + f"qos={fmt(g.algorithm.qos_class)} "
        + f"priority={fmt(g.algorithm.priority)} "
        + f"sla_ms={fmt(g.algorithm.sla_ms)}\n"
    )
    out.write(
        "- data: "
        + f"dataset_id={fmt(g.data.dataset_id)} "
        + f"data_node={fmt(g.data.data_node)} "
        + f"dataset_size_bytes={fmt(g.data.dataset_size_bytes)}\n"
    )
    out.write(
        "- compute: "
        + f"type={fmt(g.compute.compute_type)} "
        + f"duration_s={fmt(g.compute.duration_s)}\n"
    )
    out.write("- edges:\n")
    out.write("  - data -> compute (transfer + network)\n")
    out.write("  - algorithm -> compute (control)\n")


def print_decision(d: Any, out: Any) -> None:
    out.write("Decision\n")
    out.write(f"- compute_node: {d.compute_node}\n")
    out.write(f"- data_node: {d.data_node}\n")
    out.write(f"- cold_start: {d.cold_start}\n")
    out.write(f"- bytes_to_load: {d.bytes_to_load}\n")
    out.write(f"- network_latency_ms: {d.network_latency_ms}\n")
    out.write(f"- network_bandwidth_mbps: {d.network_bandwidth_mbps}\n")
    out.write(f"- estimated_transfer_ms: {d.estimated_transfer_ms}\n")
    out.write(f"- score: {d.score}\n")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--xpod-id", default="")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--emit-manifests", action="store_true")

    ap.add_argument("--data-nodes", default="data-node-1,data-node-2")
    ap.add_argument("--gpu-nodes", default="gpu-node-1,gpu-node-2,gpu-node-3")
    ap.add_argument("--cpu-nodes", default="cpu-node-1,cpu-node-2")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--base-latency-ms", type=float, default=2.0)
    ap.add_argument("--base-bandwidth-mbps", type=float, default=2000.0)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--sla-penalty", type=float, default=10.0)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)
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
    matched = 0
    for row in io.parse_csv(args.input):
        if args.xpod_id and (row.get("xpod_id") or "") != args.xpod_id:
            continue
        g = dag.build_dag(row)
        d = schedule_one(row, cluster, cache, args.alpha, args.beta, args.gamma, args.sla_penalty)
        if d is None:
            continue
        print_dag(g, sys.stdout)
        print_decision(d, sys.stdout)
        if args.emit_manifests:
            docs: List[Dict[str, Any]] = [
                specs.build_xpod_manifest(row, d, args.namespace),
                specs.build_fluid_dataset_manifest(row, d, args.namespace),
                specs.build_fluid_runtime_manifest(row, d, args.namespace),
                specs.build_volcano_job_manifest(row, d, args.namespace),
            ]
            sys.stdout.write("---\n")
            first = True
            for doc in docs:
                if not first:
                    sys.stdout.write("\n---\n")
                first = False
                sys.stdout.write(yamlutil.to_yaml(doc))
                sys.stdout.write("\n")
        matched += 1
        if matched >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
