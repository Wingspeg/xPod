import hashlib
import logging
import math
import random
from typing import Dict, Optional

from .controller.xpodgen import dag
from .controller.xpodgen.scheduler import CacheState, ClusterConfig, NodeLoadTracker, ResourceCacheState, ScheduleDecision, ServiceTracker, estimate_transfer_ms

logger = logging.getLogger(__name__)


def _as_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _as_int(x: object, default: int = 0) -> int:
    v = _as_float(x, float(default))
    try:
        return int(v)
    except Exception:
        return default


def _dominant_resource_amount(row: Dict[str, str]) -> float:
    compute_type = (row.get("compute_type") or "").upper()
    if compute_type == "GPU":
        g = _as_int(row.get("gpu"), 0)
        return float(max(1, g))
    cpu = _as_float(row.get("cpu"), 0.0)
    if cpu <= 0:
        return 1.0
    return float(max(1.0, math.ceil(cpu - 1e-12)))


def _p_2dlas(attained_service: float) -> float:
    """Tiresias 2D-LAS 优先级：已得服务越少，优先级越高。

    新任务 (attained=0) → 优先级 = 1.0（最高）
    任务累计运行越久 → attained 越大 → 优先级越低
    """
    return 1.0 / max(1.0, float(attained_service))


def _locality_penalty_s(bytes_to_load: int, data_node: str, compute_node: str, bandwidth_mbps: float) -> float:
    if not data_node or not compute_node:
        return 0.0
    if data_node == compute_node:
        return 0.0
    if bytes_to_load <= 0 or bandwidth_mbps <= 0:
        return 0.0
    mu_bytes_per_s = (bandwidth_mbps * 1_000_000.0) / 8.0
    base = float(bytes_to_load) / mu_bytes_per_s
    if data_node.split("-")[0] == compute_node.split("-")[0]:
        return 0.3 * base
    return 1.0 * base


def _completion_time_s(duration_s: float, dominant_amount: float, transfer_ms: float) -> float:
    a = max(0.0, float(duration_s))
    r = max(1e-9, float(dominant_amount))
    t_compute = a * (r ** -0.5)
    return t_compute + float(transfer_ms) / 1000.0


def schedule_one_xpod(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
    job_index: int,
    alpha: float = 1.0,
    beta: float = 1.5,
    gamma: float = 0.8,
    enable_hash_tiebreak: bool = True,
    enable_las_priority: bool = True,
    enable_contention: bool = True,
) -> Optional[ScheduleDecision]:
    compute_type = (row.get("compute_type") or "").upper()
    compute_candidates = cluster.candidates_for_compute_type(compute_type)
    if not compute_candidates:
        return None

    data_candidates = cluster.all_data_nodes()
    algo_candidates = cluster.all_algo_nodes()

    if not data_candidates:
        data_candidates = ("",)
    if not algo_candidates:
        algo_candidates = ("",)

    dataset_id = row.get("dataset_id") or ""
    dataset_size_bytes = _as_int(row.get("dataset_size_bytes"), 0)
    algorithm_image = row.get("algorithm_image") or ""
    duration_s = _as_float(row.get("duration"), 0.0)
    sla_ms = _as_float(row.get("sla_ms"), 0.0)
    xpod_id = row.get("xpod_id") or ""
    submit_time_s = _as_float(row.get("submit_time"), 0.0)

    def _hashed_node_priority(node: str, key: str) -> int:
        h = hashlib.md5(f"{key}|{node}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    xpod_id_for_hash = row.get("xpod_id") or ""

    if enable_hash_tiebreak:
        data_candidates_ranked = tuple(sorted(data_candidates, key=lambda n: _hashed_node_priority(n, dataset_id)))
        compute_candidates_ranked = tuple(sorted(compute_candidates, key=lambda n: _hashed_node_priority(n, xpod_id_for_hash)))
        algo_candidates_ranked = tuple(sorted(algo_candidates, key=lambda n: _hashed_node_priority(n, algorithm_image)))
    else:
        data_candidates_ranked = tuple(data_candidates)
        compute_candidates_ranked = tuple(compute_candidates)
        algo_candidates_ranked = tuple(algo_candidates)

    dominant_amount = _dominant_resource_amount(row)
    attained = service_tracker.get_attained(xpod_id)
    if enable_las_priority:
        p = _p_2dlas(attained)
    else:
        p = 0.0

    best = None

    for c_node in compute_candidates_ranked:
        for d_node in data_candidates_ranked:
            for a_node in algo_candidates_ranked:
                dataset_cached = bool(dataset_id) and bool(d_node) and cache.is_dataset_cached_on(d_node, dataset_id)
                bytes_to_load = 0 if dataset_cached else max(0, dataset_size_bytes)
                data_lat_ms, data_bw_mbps = cluster.network_profile(d_node, c_node)
                data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

                image_cached = bool(algorithm_image) and bool(a_node) and cache.is_image_cached_on(a_node, algorithm_image)
                algo_bytes_to_load = 0 if image_cached else 100 * 1024 * 1024
                algo_lat_ms, algo_bw_mbps = cluster.network_profile(a_node, c_node)
                algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)

                t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
                if enable_contention:
                    contention = load_tracker.contention_factor(c_node, dominant_amount)
                else:
                    contention = 1.0
                t_compute = t_compute_base * contention
                t_total = t_compute + data_transfer_ms / 1000.0 + algo_transfer_ms / 1000.0

                data_locality = _locality_penalty_s(bytes_to_load, d_node, c_node, data_bw_mbps)
                score = (alpha * t_total) + (beta * data_locality) - (gamma * p)

                if best is None or score < best[0]:
                    best = (
                        score, c_node, d_node, a_node,
                        {
                            "bytes_to_load": int(bytes_to_load),
                            "data_lat_ms": float(data_lat_ms),
                            "data_bw_mbps": float(data_bw_mbps),
                            "data_transfer_ms": float(data_transfer_ms),
                            "algo_transfer_ms": float(algo_transfer_ms),
                            "image_cached": image_cached,
                            "algo_bytes_to_load": int(algo_bytes_to_load),
                            "dataset_cached": dataset_cached,
                        }
                    )

    if best is None:
        return None

    score, c_node, d_node, a_node, m = best

    if dataset_id and d_node:
        cache.mark_dataset_cached(d_node, dataset_id)
    if algorithm_image and a_node:
        cache.mark_image_cached(a_node, algorithm_image)

    cold_start = 0 if m["dataset_cached"] else 1

    service_tracker.add_service(xpod_id, duration_s, dominant_amount)

    if enable_contention:
        final_contention = load_tracker.contention_factor(c_node, dominant_amount)
    else:
        final_contention = 1.0
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    final_t_compute = t_compute_base * final_contention
    jct_s = final_t_compute + m["data_transfer_ms"] / 1000.0 + m["algo_transfer_ms"] / 1000.0
    release_time = submit_time_s + max(0.0, duration_s)
    load_tracker.add_task(c_node, release_time, dominant_amount)

    return ScheduleDecision(
        compute_node=c_node,
        data_node=d_node,
        algo_node=a_node,
        bytes_to_load=m["bytes_to_load"],
        network_latency_ms=m["data_lat_ms"],
        network_bandwidth_mbps=m["data_bw_mbps"],
        estimated_transfer_ms=m["data_transfer_ms"],
        cold_start=cold_start,
        score=float(score),
        jct_s=jct_s,
        contention_factor=final_contention,
        algo_transfer_ms=m["algo_transfer_ms"],
        image_cached=1 if m["image_cached"] else 0,
        algo_bytes_to_load=m["algo_bytes_to_load"],
    )


def schedule_one_baseline(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
) -> Optional[ScheduleDecision]:
    g = dag.build_dag(row)
    compute_type = (g.compute.compute_type or "").upper()
    candidates = cluster.candidates_for_compute_type(compute_type)
    if not candidates:
        return None

    data_candidates = cluster.all_data_nodes()
    algo_candidates = cluster.all_algo_nodes()

    if not data_candidates:
        data_candidates = ("",)
    if not algo_candidates:
        algo_candidates = ("",)

    xpod_id = g.xpod_id or ""
    seed_material = f"{cluster.seed}|{xpod_id}".encode("utf-8", errors="ignore")
    seed_int = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
    rng = random.Random(seed_int)
    compute_node = rng.choice(list(candidates))
    data_node = rng.choice(list(data_candidates))
    algo_node = rng.choice(list(algo_candidates))

    dataset_id = g.data.dataset_id or ""
    dataset_size_bytes = int(g.data.dataset_size_bytes)
    dataset_cached = bool(dataset_id) and bool(data_node) and cache.is_dataset_cached_on(data_node, dataset_id)
    bytes_to_load = 0 if dataset_cached else max(0, dataset_size_bytes)
    cold_start = 0 if dataset_cached else 1

    if dataset_id and data_node:
        cache.mark_dataset_cached(data_node, dataset_id)

    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

    algorithm_image = row.get("algorithm_image") or ""
    image_cached = bool(algorithm_image) and bool(algo_node) and cache.is_image_cached_on(algo_node, algorithm_image)
    algo_bytes_to_load = 0 if image_cached else 100 * 1024 * 1024
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)

    if algorithm_image and algo_node:
        cache.mark_image_cached(algo_node, algorithm_image)

    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    duration_s = _as_float(row.get("duration"), 0.0)
    dominant_amount = _dominant_resource_amount(row)

    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute = t_compute_base * contention
    jct_s = t_compute + data_transfer_ms / 1000.0 + algo_transfer_ms / 1000.0

    release_time = submit_time_s + max(0.0, duration_s)
    load_tracker.add_task(compute_node, release_time, dominant_amount)

    return ScheduleDecision(
        compute_node=compute_node,
        data_node=data_node,
        algo_node=algo_node,
        bytes_to_load=int(max(0, bytes_to_load)),
        network_latency_ms=float(data_lat_ms),
        network_bandwidth_mbps=float(data_bw_mbps),
        estimated_transfer_ms=float(data_transfer_ms),
        cold_start=cold_start,
        score=0.0,
        jct_s=jct_s,
        contention_factor=contention,
        algo_transfer_ms=float(algo_transfer_ms),
        image_cached=1 if image_cached else 0,
        algo_bytes_to_load=int(algo_bytes_to_load),
    )
