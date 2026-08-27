#!/usr/bin/env python3
# replay_workload.py: 回放 workload CSV，生成/提交 manifests，并落盘调度决策日志（含 run_id/mode 等实验可观测字段）。Modified: 2026-04-28
import argparse
import csv
import json
import heapq
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


def add_repo_import_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


add_repo_import_paths()

from scheduler.controller.xpodgen import io, specs, yamlutil  # noqa: E402
from scheduler.controller.xpodgen.scheduler import ClusterConfig, NodeLoadTracker, ResourceCacheState, ServiceTracker  # noqa: E402
from scheduler.xpod_scheduler import (
    NodeHistory,
    schedule_one_decoupled_cd,
    schedule_one_firstfit_hash,
    schedule_one_k8s_default,
    schedule_one_pure_random,
    schedule_one_tetris,
    schedule_one_xpod,
)  # noqa: E402


def parse_number(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


@dataclass(frozen=True)
class ReplayItem:
    submit_time: float
    row: Dict[str, str]


def read_rows(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, str]]:
    t0 = time.monotonic()
    total = 0
    logger.info("read_rows start path=%s limit=%s", path, limit)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            header_line = f.readline()
            if not header_line:
                return
            try:
                fieldnames = next(csv.reader([header_line]))
            except Exception:
                logger.warning("CSV header parse failed path=%s raw=%r", path, header_line)
                return

            for line_no, raw_line in enumerate(f, start=2):
                raw = raw_line.rstrip("\n")
                try:
                    vals = next(csv.reader([raw]))
                except Exception:
                    logger.warning("CSV row parse failed path=%s line=%d raw=%r", path, line_no, raw)
                    continue
                row = {k: (vals[i] if i < len(vals) else "") for i, k in enumerate(fieldnames)}
                total += 1
                logger.debug("CSV row parsed path=%s line=%d row=%s", path, line_no, row)
                yield row
                if limit is not None and total >= limit:
                    break
    except Exception:
        logger.exception("read_rows failed path=%s", path)
        raise
    finally:
        elapsed_s = time.monotonic() - t0
        logger.info("read_rows done path=%s rows=%d elapsed_s=%.3f", path, total, elapsed_s)


def row_submit_time(row: Dict[str, str], ts_field: str) -> Optional[float]:
    return parse_number(row.get(ts_field), None)


def write_chunk(rows: List[ReplayItem], tmpdir: str, chunk_index: int) -> str:
    rows.sort(key=lambda x: x.submit_time)
    out_path = os.path.join(tmpdir, f"chunk_{chunk_index:06d}.csv")
    fieldnames = list(rows[0].row.keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in rows:
            w.writerow(it.row)
    return out_path


def external_sort_by_ts(
    input_path: str,
    ts_field: str,
    chunk_rows: int,
    tmpdir: str,
    limit: Optional[int] = None,
) -> List[str]:
    t0 = time.monotonic()
    total = 0
    kept = 0
    IN_MEMORY_LIMIT_THRESHOLD = 100000

    if limit is not None and limit > 0 and limit < IN_MEMORY_LIMIT_THRESHOLD:
        logger.info("limit=%d specified, using in-memory top-N selection", limit)
        # Apply start_ts/end_ts window before top-N so limit+window compose correctly
        import sys as _sys
        _start_ts = None
        _end_ts = None
        for _i, _a in enumerate(_sys.argv):
            if _a == "--start-ts" and _i + 1 < len(_sys.argv):
                try:
                    _start_ts = float(_sys.argv[_i + 1])
                except ValueError:
                    pass
            elif _a == "--end-ts" and _i + 1 < len(_sys.argv):
                try:
                    _end_ts = float(_sys.argv[_i + 1])
                except ValueError:
                    pass
        def item_generator():
            nonlocal total, kept
            for row in read_rows(input_path):
                total += 1
                ts = row_submit_time(row, ts_field)
                if ts is None:
                    continue
                if _start_ts is not None and ts < _start_ts:
                    continue
                if _end_ts is not None and ts > _end_ts:
                    continue
                kept += 1
                yield ReplayItem(ts, row)
        top_items = heapq.nsmallest(limit, item_generator(), key=lambda x: x.submit_time)
        chunk_files = [write_chunk(top_items, tmpdir, 0)]
        elapsed_s = time.monotonic() - t0
        logger.info(
            "external_sort_by_ts in-memory top-N done input=%s read_rows=%d kept_rows=%d top_n=%d elapsed_s=%.3f",
            input_path,
            total,
            kept,
            len(top_items),
            elapsed_s,
        )
        return chunk_files

    logger.info("no limit or limit too large, using external sort")
    chunk_files: List[str] = []
    buf: List[ReplayItem] = []
    chunk_index = 0
    for row in read_rows(input_path):
        total += 1
        ts = row_submit_time(row, ts_field)
        if ts is None:
            continue
        kept += 1
        buf.append(ReplayItem(ts, row))
        if len(buf) >= chunk_rows:
            chunk_files.append(write_chunk(buf, tmpdir, chunk_index))
            buf = []
            chunk_index += 1
    if buf:
        chunk_files.append(write_chunk(buf, tmpdir, chunk_index))
    elapsed_s = time.monotonic() - t0
    logger.info(
        "external_sort_by_ts done input=%s read_rows=%d kept_rows=%d chunks=%d elapsed_s=%.3f",
        input_path,
        total,
        kept,
        len(chunk_files),
        elapsed_s,
    )
    return chunk_files


def merge_sorted_chunks(chunk_files: List[str], ts_field: str) -> Iterator[Dict[str, str]]:
    readers: List[Tuple[csv.DictReader, Any]] = []
    for p in chunk_files:
        f = open(p, "r", encoding="utf-8", errors="ignore", newline="")
        r = csv.DictReader(f)
        readers.append((r, f))

    heap: List[Tuple[float, int, Dict[str, str]]] = []
    for i, (r, _) in enumerate(readers):
        try:
            row = next(r)
        except StopIteration:
            continue
        ts = row_submit_time(row, ts_field)
        if ts is None:
            continue
        heap.append((ts, i, row))
    heapq.heapify(heap)

    try:
        while heap:
            ts, i, row = heapq.heappop(heap)
            yield row
            r, _ = readers[i]
            try:
                nxt = next(r)
            except StopIteration:
                continue
            nts = row_submit_time(nxt, ts_field)
            if nts is None:
                continue
            heapq.heappush(heap, (nts, i, nxt))
    finally:
        for _, f in readers:
            try:
                f.close()
            except Exception:
                pass


def _parse_node_capacity(s: str) -> Dict[str, float]:
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def build_cluster(args: argparse.Namespace) -> ClusterConfig:
    rack_map: Dict[str, str] = {}
    if args.rack_map:
        for kv in args.rack_map.split(","):
            kv = kv.strip()
            if ":" in kv:
                k, v = kv.split(":", 1)
                rack_map[k.strip()] = v.strip()
    return ClusterConfig(
        data_nodes=tuple(io.split_csv_list(args.data_nodes)),
        gpu_nodes=tuple(io.split_csv_list(args.gpu_nodes)),
        cpu_nodes=tuple(io.split_csv_list(args.cpu_nodes)),
        seed=args.seed,
        base_latency_ms=args.base_latency_ms,
        base_bandwidth_mbps=args.base_bandwidth_mbps,
        algo_nodes=tuple(io.split_csv_list(args.algo_nodes)),
        rack_map=rack_map if rack_map else None,
    )


def make_manifest_yaml(row: Dict[str, str], decision: Any, namespace: str) -> str:
    docs = [
        specs.build_xpod_manifest(row, decision, namespace),
        specs.build_fluid_dataset_manifest(row, decision, namespace),
        specs.build_fluid_runtime_manifest(row, decision, namespace),
        specs.build_volcano_job_manifest(row, decision, namespace),
    ]
    parts = []
    first = True
    for d in docs:
        if not first:
            parts.append("---")
        first = False
        parts.append(yamlutil.to_yaml(d))
    return "\n".join(parts) + "\n"


def kubectl_apply(yaml_text: str, kube_context: str, namespace: str, dry_run: bool) -> subprocess.CompletedProcess:
    cmd = ["kubectl"]
    if kube_context:
        cmd += ["--context", kube_context]
    if namespace:
        cmd += ["-n", namespace]
    cmd += ["apply", "-f", "-"]
    if dry_run:
        cmd += ["--dry-run=client"]
    logger.info("kubectl_apply start dry_run=%s namespace=%s context=%s bytes=%d", bool(dry_run), namespace, kube_context, len(yaml_text))
    logger.debug("kubectl_apply manifest(head_500)=%r", yaml_text[:500])
    try:
        res = subprocess.run(cmd, input=yaml_text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        logger.exception("kubectl_apply exception")
        raise

    out = res.stdout.decode("utf-8", errors="ignore")
    err = res.stderr.decode("utf-8", errors="ignore")
    out_s = (out[:500] + "...") if len(out) > 500 else out
    err_s = (err[:500] + "...") if len(err) > 500 else err
    logger.info("kubectl_apply done rc=%d stdout(head_500)=%r stderr(head_500)=%r", res.returncode, out_s, err_s)
    if res.returncode != 0:
        logger.error("kubectl_apply failed rc=%d stderr=%s", res.returncode, err)
    return res


def kubectl_delete_volcano_job(name: str, kube_context: str, namespace: str) -> None:
    if not name:
        return
    base = ["kubectl"]
    if kube_context:
        base += ["--context", kube_context]
    if namespace:
        base += ["-n", namespace]
    for res in ("vcjob", "jobs.batch.volcano.sh"):
        cmd = base + ["delete", res, name, "--ignore-not-found=true", "--wait=true", "--timeout=60s"]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass


def should_keep_row(row: Dict[str, str], ts: float, args: argparse.Namespace) -> bool:
    if args.filter_source and (row.get("source") or "") != args.filter_source:
        return False
    if args.xpod_id and (row.get("xpod_id") or "") != args.xpod_id:
        return False
    if args.start_ts is not None and ts < args.start_ts:
        return False
    if args.end_ts is not None and ts > args.end_ts:
        return False
    return True


def join_manifests(request_yamls: List[str]) -> str:
    if not request_yamls:
        return ""
    cleaned = []
    for y in request_yamls:
        cleaned.append(y.rstrip("\n"))
    return "\n---\n".join(cleaned) + "\n"


def open_log(path: str):
    if not path:
        return None, None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existed = os.path.exists(path)
    f = open(path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(
        f,
        fieldnames=[
            "run_id",
            "mode",
            "hash_tiebreak",
            "las_priority",
            "contention_aware",
            "idx",
            "xpod_id",
            "submit_time",
            "wall_time",
            "compute_node",
            "data_node",
            "algo_node",
            "cold_start",
            "bytes_to_load",
            "estimated_transfer_ms",
            "jct_s",
            "contention_factor",
            "score",
            "sla_ms",
            "apply",
            "dry_run",
            "kubectl_rc",
            "algo_transfer_ms",
            "image_cached",
            "algo_bytes_to_load",
        ],
    )
    if not existed:
        w.writeheader()
        f.flush()
    return f, w


def replay(
    rows: Iterable[Dict[str, str]],
    ts_field: str,
    cluster: ClusterConfig,
    capacity: Dict[str, float],
    args: argparse.Namespace,
) -> int:
    logger.info(
        "replay start mode=%s run_id=%s limit=%s speedup=%s data_nodes=%s gpu_nodes=%s cpu_nodes=%s",
        args.mode,
        args.run_id,
        args.limit,
        args.speedup,
        ",".join(cluster.data_nodes),
        ",".join(cluster.gpu_nodes),
        ",".join(cluster.cpu_nodes),
    )
    cache = ResourceCacheState()
    service_tracker = ServiceTracker()
    load_tracker = NodeLoadTracker(capacity)
    t0_wall = time.monotonic()
    t0_ts: Optional[float] = None
    last_ts: Optional[float] = None
    sent = 0
    attempted = 0
    apply_ok = 0
    apply_fail = 0
    pending_yamls: List[str] = []
    pending_meta: List[Tuple[int, str, float, Any, str]] = []

    log_f, log_w = open_log(args.replay_log_file)

    for row in rows:
        logger.debug("replay input row=%s", row)
        ts = row_submit_time(row, ts_field)
        if ts is None:
            continue
        if not should_keep_row(row, ts, args):
            continue
        if args.strict_order and last_ts is not None and ts < last_ts:
            logger.error("timestamp not non-decreasing prev=%s cur=%s", last_ts, ts)
            sys.stderr.write(f"timestamp not non-decreasing: prev={last_ts}, cur={ts}\n")
            return 2
        last_ts = ts
        if t0_ts is None:
            t0_ts = ts

        if args.speedup > 0:
            virtual_elapsed = (ts - t0_ts) / args.speedup
            target_wall = t0_wall + max(0.0, virtual_elapsed)
            now = time.monotonic()
            if target_wall > now:
                time.sleep(target_wall - now)

        job_id = (row.get("job_id") or row.get("xpod_id") or row.get("task_id") or "").strip()
        logger.info("workload submit idx=%d job_id=%s submit_time=%s", attempted + 1, job_id, ts)

        load_tracker.release_until(ts)

        if args.mode == "pure_random":
            logger.debug("schedule_one_pure_random input job_id=%s row=%s", job_id, row)
            decision = schedule_one_pure_random(row, cluster, cache, service_tracker, load_tracker)
            logger.debug("schedule_one_pure_random output job_id=%s decision=%s", job_id, decision)
        elif args.mode == "firstfit_hash":
            logger.debug("schedule_one_firstfit_hash input job_id=%s row=%s", job_id, row)
            decision = schedule_one_firstfit_hash(row, cluster, cache, service_tracker, load_tracker)
            logger.debug("schedule_one_firstfit_hash output job_id=%s decision=%s", job_id, decision)
        elif args.mode == "k8s_default":
            logger.debug("schedule_one_k8s_default input job_id=%s row=%s", job_id, row)
            decision = schedule_one_k8s_default(row, cluster, cache, service_tracker, load_tracker)
            logger.debug("schedule_one_k8s_default output job_id=%s decision=%s", job_id, decision)
        elif args.mode == "decoupled_cd":
            logger.debug("schedule_one_decoupled_cd input job_id=%s row=%s", job_id, row)
            decision = schedule_one_decoupled_cd(row, cluster, cache, service_tracker, load_tracker)
            logger.debug("schedule_one_decoupled_cd output job_id=%s decision=%s", job_id, decision)
        elif args.mode == "tetris":
            logger.debug("schedule_one_tetris input job_id=%s row=%s", job_id, row)
            decision = schedule_one_tetris(row, cluster, cache, service_tracker, load_tracker)
            logger.debug("schedule_one_tetris output job_id=%s decision=%s", job_id, decision)
        else:
            logger.debug("schedule_one_xpod input job_id=%s row=%s attempted=%d", job_id, row, attempted)
            decision = schedule_one_xpod(
                row,
                cluster,
                cache,
                service_tracker,
                load_tracker,
                job_index=attempted,
                alpha=args.xpod_alpha,
                beta=args.xpod_beta,
                gamma=args.xpod_gamma,
                xpod_dim=args.xpod_dim,
                enable_hash_tiebreak=(not args.disable_hash_tiebreak),
                enable_las_priority=(not args.disable_las_priority),
                enable_contention=(not args.disable_contention),
                enable_compute_hash=args.enable_compute_hash,
                enable_data_hash=args.enable_data_hash,
                enable_algo_hash=args.enable_algo_hash,
                cache_bonus=args.cache_bonus,
                rack_bonus=args.rack_bonus,
                popularity_weight=args.popularity_weight,
                dataset_popularity=args.dataset_popularity,
                image_popularity=args.image_popularity,
                look_ahead_penalty=args.look_ahead_penalty,
                look_ahead_window=args.look_ahead_window,
                node_history=args.node_history,
            )
            logger.debug("schedule_one_xpod output job_id=%s decision=%s", job_id, decision)
        if decision is None:
            continue

        logger.info(
            "decision job_id=%s compute_node=%s data_node=%s algo_node=%s cold_start=%s bytes_to_load=%s est_transfer_ms=%s score=%s",
            job_id,
            getattr(decision, "compute_node", ""),
            getattr(decision, "data_node", ""),
            getattr(decision, "algo_node", ""),
            getattr(decision, "cold_start", ""),
            getattr(decision, "bytes_to_load", ""),
            getattr(decision, "estimated_transfer_ms", ""),
            getattr(decision, "score", ""),
        )

        y = make_manifest_yaml(row, decision, args.namespace)
        logger.debug("manifest_yaml job_id=%s head_500=%r", job_id, y[:500])

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            name = row.get("xpod_id") or row.get("task_id") or f"req-{sent}"
            out_path = os.path.join(args.output_dir, f"{name}.yaml")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(y)

        attempted += 1
        pending_yamls.append(y)
        pending_meta.append((attempted, row.get("xpod_id") or "", ts, decision, str(row.get("sla_ms") or "")))

        flush_now = (not args.apply) or (args.apply_batch_size <= 1) or (len(pending_yamls) >= args.apply_batch_size)
        if flush_now:
            batch_yaml = join_manifests(pending_yamls)
            logger.info(
                "apply_batch start apply=%s dry_run=%s batch_size=%d pending=%d",
                bool(args.apply),
                bool(args.dry_run),
                args.apply_batch_size,
                len(pending_yamls),
            )
            kubectl_rc = ""
            if args.apply:
                if not args.dry_run:
                    for _, xpod_id, _, _, _ in pending_meta:
                        kubectl_delete_volcano_job(xpod_id, args.kube_context, args.namespace)
                res = kubectl_apply(batch_yaml, args.kube_context, args.namespace, args.dry_run)
                kubectl_rc = str(res.returncode)
                if res.returncode != 0:
                    apply_fail += 1
                    sys.stderr.write(res.stderr.decode("utf-8", errors="ignore"))
                    if not args.continue_on_error:
                        return res.returncode
                else:
                    apply_ok += 1
                if args.verbose:
                    sys.stdout.write(res.stdout.decode("utf-8", errors="ignore"))
            else:
                apply_ok += 1
                if args.verbose:
                    sys.stdout.write(batch_yaml)

            if log_w is not None:
                wall_now = time.time()
                for idx, xpod_id, submit_time, d, sla_ms in pending_meta:
                    log_w.writerow(
                        {
                            "run_id": args.run_id,
                            "mode": args.mode,
                            "hash_tiebreak": (not args.disable_hash_tiebreak),
                            "las_priority": (not args.disable_las_priority),
                            "contention_aware": (not args.disable_contention),
                            "idx": idx,
                            "xpod_id": xpod_id,
                            "submit_time": submit_time,
                            "wall_time": wall_now,
                            "compute_node": getattr(d, "compute_node", ""),
                            "data_node": getattr(d, "data_node", ""),
                            "algo_node": getattr(d, "algo_node", ""),
                            "cold_start": getattr(d, "cold_start", ""),
                            "bytes_to_load": getattr(d, "bytes_to_load", ""),
                            "estimated_transfer_ms": getattr(d, "estimated_transfer_ms", ""),
                            "jct_s": getattr(d, "jct_s", ""),
                            "contention_factor": getattr(d, "contention_factor", ""),
                            "score": getattr(d, "score", ""),
                            "sla_ms": sla_ms,
                            "apply": bool(args.apply),
                            "dry_run": bool(args.dry_run),
                            "kubectl_rc": kubectl_rc,
                            "algo_transfer_ms": getattr(d, "algo_transfer_ms", ""),
                            "image_cached": getattr(d, "image_cached", ""),
                            "algo_bytes_to_load": getattr(d, "algo_bytes_to_load", ""),
                        }
                    )
                log_f.flush()

            pending_yamls = []
            pending_meta = []

        sent += 1
        remaining = ""
        if args.limit and args.limit > 0:
            remaining = str(max(0, int(args.limit) - sent))
        logger.info("workload done sent=%d remaining=%s sim_submit_time=%s", sent, remaining, ts)
        if args.progress_every and sent % args.progress_every == 0:
            elapsed = time.monotonic() - t0_wall
            logger.info("progress replayed=%d elapsed_s=%.3f last_submit_time=%s", sent, elapsed, ts)
        if args.limit and sent >= args.limit:
            break
    if args.apply and pending_yamls:
        batch_yaml = join_manifests(pending_yamls)
        res = kubectl_apply(batch_yaml, args.kube_context, args.namespace, args.dry_run)
        if res.returncode != 0:
            apply_fail += 1
            sys.stderr.write(res.stderr.decode("utf-8", errors="ignore"))
            if not args.continue_on_error:
                return res.returncode
        else:
            apply_ok += 1
        if args.verbose:
            sys.stdout.write(res.stdout.decode("utf-8", errors="ignore"))

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t0_wall
    logger.info("replay done elapsed_s=%.3f submitted=%d apply_ok=%d apply_fail=%d", elapsed_s, sent, apply_ok, apply_fail)
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--timestamp-field", default="submit_time")
    ap.add_argument("--presorted", action="store_true")
    ap.add_argument("--strict-order", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=200000)
    ap.add_argument("--speedup", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter-source", default="")
    ap.add_argument("--xpod-id", default="")
    ap.add_argument("--start-ts", type=float, default=None)
    ap.add_argument("--end-ts", type=float, default=None)
    ap.add_argument("--namespace", default="default")

    ap.add_argument(
        "--mode",
        choices=["xpod", "pure_random", "firstfit_hash", "k8s_default", "decoupled_cd", "tetris"],
        default="xpod",
        help="Scheduler routing: xpod=3-dim argmin; pure_random=uniform (no cache); "
             "firstfit_hash=least-loaded c + consistent-hash d/a; k8s_default=K8s first-available; "
             "decoupled_cd=Tiresias 2D-LAS + Quiver consistent-hash; "
             "tetris=3-dim max-min remaining capacity (Grandl et al. SIGCOMM'14).",
    )
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply-batch-size", type=int, default=1)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--kube-context", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--replay-log-file", default="")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)

    ap.add_argument("--data-nodes", default="data-node-1,data-node-2")
    ap.add_argument("--gpu-nodes", default="gpu-node-1,gpu-node-2,gpu-node-3")
    ap.add_argument("--cpu-nodes", default="cpu-node-1,cpu-node-2")
    ap.add_argument("--algo-nodes", default="")
    ap.add_argument("--node-capacity", default="gpu-node-1:1,data-node-1:1,data-node-2:1,algo-node-1:1,cpu-node-1:4,cpu-node-2:4",
                    help="节点容量配置，格式: node1:cap1,node2:cap2,...")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--base-latency-ms", type=float, default=30.0)
    ap.add_argument("--base-bandwidth-mbps", type=float, default=200.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--sla-penalty", type=float, default=10.0)
    ap.add_argument("--xpod-alpha", type=float, default=1.0)
    ap.add_argument("--xpod-beta", type=float, default=1.5)
    ap.add_argument("--xpod-gamma", type=float, default=0.8)
    ap.add_argument("--xpod-dim", type=int, default=3, choices=[1, 2, 3], help="1=compute-only, 2=compute+data, 3=compute+data+algo")
    ap.add_argument("--disable-hash-tiebreak", action="store_true", help="关闭 hash tie-break (master switch),用于 ablation 实验")
    ap.add_argument("--disable-las-priority", action="store_true", help="关闭 Tiresias 2D-LAS,用于 ablation 实验")
    ap.add_argument("--disable-contention", action="store_true", help="关闭 contention-aware t_compute,用于 ablation 实验")
    ap.add_argument("--no-compute-hash", action="store_true", help="(legacy) 关闭 compute 维度 hash, 现在 default 已经是 enumerate, 此 flag 是 no-op")
    ap.add_argument("--no-data-hash", action="store_true", help="(legacy) 同上, no-op")
    ap.add_argument("--no-algo-hash", action="store_true", help="(legacy) 同上, no-op")
    ap.add_argument("--enable-compute-hash", action="store_true", help="ablation: 启用 c 维度 consistent-hash, d/a 仍 enumerate")
    ap.add_argument("--enable-data-hash", action="store_true", help="ablation: 启用 d 维度 consistent-hash, c/a 仍 enumerate")
    ap.add_argument("--enable-algo-hash", action="store_true", help="ablation: 启用 a 维度 consistent-hash, c/d 仍 enumerate")
    ap.add_argument("--rack-map", default="", help="逗号分隔的 rack_map, 格式 node1:rack1,node2:rack2,...; 空=不启用 rack-aware topology")
    ap.add_argument("--cache-bonus", type=float, default=1.0, help="cache hit 基础奖励 (score -= cache_bonus * cached_dims)")
    ap.add_argument("--rack-bonus", type=float, default=0.5, help="same-rack 奖励 (score -= rack_bonus * same_rack_dims)")
    ap.add_argument("--popularity-weight", type=float, default=1.0, help="A: dataset/image popularity 加权 (cache_bonus += popularity_weight * log1p(pop))")
    ap.add_argument("--popularity-json", default="", help="path to pre-scan popularity JSON (dataset_popularity + image_popularity)")
    ap.add_argument("--look-ahead-penalty", type=float, default=1.0, help="B: look-ahead contention penalty coefficient")
    ap.add_argument("--look-ahead-window", type=int, default=1000, help="B: sliding window size (number of recent tasks per c)")
    args = ap.parse_args(argv)

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", vars(args))

    if not args.replay_log_file:
        args.replay_log_file = os.path.join("results", "raw", str(args.run_id), "replay_log.csv")

    if args.apply_batch_size < 1:
        args.apply_batch_size = 1
    if args.speedup < 0:
        args.speedup = 0

    cluster = build_cluster(args)
    capacity = _parse_node_capacity(args.node_capacity)

    # Load popularity dict (A: dataset/image popularity-aware cache bonus)
    dataset_popularity = None
    image_popularity = None
    if args.popularity_json and os.path.exists(args.popularity_json):
        with open(args.popularity_json, "r", encoding="utf-8") as f:
            pop = json.load(f)
        dataset_popularity = pop.get("dataset_popularity", {})
        image_popularity = pop.get("image_popularity", {})
        logger.info("loaded popularity: %d datasets, %d images", len(dataset_popularity), len(image_popularity))

    # Create node history (B: sliding window look-ahead contention)
    node_history = NodeHistory(window=args.look_ahead_window)
    # Attach to args for replay() to use
    args.dataset_popularity = dataset_popularity
    args.image_popularity = image_popularity
    args.node_history = node_history

    limit_arg = args.limit if args.limit and args.limit > 0 else None

    if args.presorted:
        rows = read_rows(args.input, limit_arg)
        return replay(rows, args.timestamp_field, cluster, capacity, args)

    with tempfile.TemporaryDirectory(prefix="xpod-replay-", dir=None) as tmpdir:
        chunk_files = external_sort_by_ts(args.input, args.timestamp_field, args.chunk_rows, tmpdir, limit_arg)
        rows = merge_sorted_chunks(chunk_files, args.timestamp_field)
        return replay(rows, args.timestamp_field, cluster, capacity, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
