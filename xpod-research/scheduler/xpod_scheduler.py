import hashlib
import logging
import math
import random
from typing import Any, Dict, Optional, Tuple

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
    enable_hash_tiebreak: bool = False,
    enable_las_priority: bool = True,
    enable_contention: bool = True,
    enable_compute_hash: bool = False,
    enable_data_hash: bool = False,
    enable_algo_hash: bool = False,
    cache_bonus: float = 1.0,
    rack_bonus: float = 0.5,
    popularity_weight: float = 1.0,
    dataset_popularity: Optional[Dict[str, int]] = None,
    image_popularity: Optional[Dict[str, int]] = None,
    look_ahead_penalty: float = 50.0,
    look_ahead_window: int = 10000,
    node_history: Optional[Any] = None,
    xpod_dim: int = 3,
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
        algo_candidates = ("")

    # v7 ablation: 1-dim (c-only, d/a fixed), 2-dim (c+d, a fixed), 3-dim (c+d+a, default)
    if xpod_dim <= 1:
        if data_candidates:
            data_candidates = (data_candidates[0],)
        if algo_candidates:
            algo_candidates = (algo_candidates[0],)
    elif xpod_dim == 2:
        if algo_candidates:
            algo_candidates = (algo_candidates[0],)

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

    # enable_hash_tiebreak is a legacy master switch: when True, all 3 dims use
    # consistent-hash (default xpod behavior). For 3-dim ablation, set
    # enable_hash_tiebreak=False and toggle the per-dim flags below:
    #   no_compute_hash: enable_compute_hash=False, others=True
    #   no_data_hash:    enable_data_hash=False, others=True
    #   no_algo_hash:    enable_algo_hash=False, others=True
    #   full_joint:      all 3 False (4x4x4 = 64 combinations per row)
    if enable_hash_tiebreak:
        use_data_hash = True
        use_compute_hash = True
        use_algo_hash = True
    else:
        use_data_hash = enable_data_hash
        use_compute_hash = enable_compute_hash
        use_algo_hash = enable_algo_hash

    if use_data_hash:
        data_candidates_ranked = tuple(sorted(data_candidates, key=lambda n: _hashed_node_priority(n, dataset_id)))
    else:
        data_candidates_ranked = tuple(data_candidates)
    if use_compute_hash:
        compute_candidates_ranked = tuple(sorted(compute_candidates, key=lambda n: _hashed_node_priority(n, xpod_id_for_hash)))
    else:
        compute_candidates_ranked = tuple(compute_candidates)
    if use_algo_hash:
        algo_candidates_ranked = tuple(sorted(algo_candidates, key=lambda n: _hashed_node_priority(n, algorithm_image)))
    else:
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
                dataset_cached = False  # v7: cache disabled
                bytes_to_load = 0 if dataset_cached else max(0, dataset_size_bytes)
                data_lat_ms, data_bw_mbps = cluster.network_profile(d_node, c_node)
                data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

                image_cached = False  # v7: cache disabled
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

                # Joint optimization with intelligent features:
                # A. Popularity-aware cache bonus: hot datasets/images get higher cache_bonus
                #    (preserves cache for popular items, lets cold items spread out)
                # B. Look-ahead contention penalty: penalize c with high recent assignment rate
                #    (preventive load balancing, not just reactive)
                # C. Hash tie-break: avoid first-c-node bias when scores tie
                # X: Size-aware cache bonus: ds_bonus = cache_bonus * (1 + log10(bytes/1GB))
                # 1GB dataset: bonus = 1.0 * 1.0 = 1.0
                # 100GB dataset: bonus = 1.0 * 3.0 = 3.0
                # 1TB dataset: bonus = 1.0 * 4.0 = 4.0
                # Combined with popularity multiplier: hot large datasets get biggest bonus
                ds_size_factor = 1.0 + math.log10(max(1.0, float(dataset_size_bytes)) / 1e9)
                if dataset_popularity and dataset_id:
                    ds_pop = dataset_popularity.get(dataset_id, 0)
                    ds_bonus = cache_bonus * ds_size_factor + popularity_weight * math.log1p(ds_pop) * ds_size_factor
                else:
                    ds_bonus = cache_bonus * ds_size_factor

                algo_size_factor = 1.0 + math.log10(100 * 1024 * 1024 / 1e9)  # algo image is 100MB
                if image_popularity and algorithm_image:
                    img_pop = image_popularity.get(algorithm_image, 0)
                    img_bonus = cache_bonus * algo_size_factor + popularity_weight * math.log1p(img_pop) * algo_size_factor
                else:
                    img_bonus = cache_bonus * algo_size_factor
                ds_cached_score = (1.0 if dataset_cached else 0.0) * ds_bonus
                img_cached_score = (1.0 if image_cached else 0.0) * img_bonus
                rack_score = (1.0 if cluster.same_rack(d_node, c_node) else 0.0) + (1.0 if cluster.same_rack(a_node, c_node) else 0.0)
                # Look-ahead penalty (B): c with high recent load gets penalty
                la_penalty = 0.0
                if node_history is not None:
                    la_penalty = look_ahead_penalty * node_history.recent_load(c_node) / max(1, look_ahead_window)
                score = (alpha * t_total) + (beta * data_locality) - (gamma * p) - (ds_cached_score + img_cached_score) - (rack_bonus * rack_score) + la_penalty

                # Hash tie-break (C): when scores tie, deterministic per-xpod-id pick
                hash_tb = int.from_bytes(
                    hashlib.md5(f"{xpod_id}|{c_node}|{d_node}|{a_node}".encode("utf-8")).digest()[:4], "big"
                )
                candidate = (score, hash_tb, c_node, d_node, a_node, {
                    "bytes_to_load": int(bytes_to_load),
                    "data_lat_ms": float(data_lat_ms),
                    "data_bw_mbps": float(data_bw_mbps),
                    "data_transfer_ms": float(data_transfer_ms),
                    "algo_transfer_ms": float(algo_transfer_ms),
                    "image_cached": image_cached,
                    "algo_bytes_to_load": int(algo_bytes_to_load),
                    "dataset_cached": dataset_cached,
                })
                if best is None or (score, hash_tb) < (best[0], best[1]):
                    best = candidate

    if best is None:
        return None

    score, _hash_tb, c_node, d_node, a_node, m = best

    # Record assignment in node history (B: look-ahead)
    if node_history is not None:
        node_history.record(c_node)

    if dataset_id and d_node:
        pass  # v7: cache removed for fair routing-only test
    if algorithm_image and a_node:
        pass  # v7: cache removed for fair routing-only test

    cold_start = 0 if m["dataset_cached"] else 1

    service_tracker.add_service(xpod_id, duration_s, dominant_amount)

    if enable_contention:
        final_contention = load_tracker.contention_factor(c_node, dominant_amount)
    else:
        final_contention = 1.0
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    final_t_compute = t_compute_base * final_contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer.
    service_time = final_t_compute + (m["data_transfer_ms"] + m["algo_transfer_ms"]) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        c_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

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


def schedule_one_pure_random(
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
    # Per user spec: pure_random never sees cache, so always full size + cold start
    dataset_cached = False
    bytes_to_load = max(0, dataset_size_bytes)
    cold_start = 1 if dataset_id else 0


    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

    algorithm_image = row.get("algorithm_image") or ""
    # Per user spec: pure_random never sees cache for image either
    image_cached = False
    algo_bytes_to_load = 100 * 1024 * 1024 if algorithm_image else 0
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)


    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    duration_s = _as_float(row.get("duration"), 0.0)
    dominant_amount = _dominant_resource_amount(row)

    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute = t_compute_base * contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer (transfer happens serially at compute_node,
    # waiting for data to arrive from d/a before compute can start).
    service_time = t_compute + (data_transfer_ms + algo_transfer_ms) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        compute_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

    if data_node and dataset_id:
        pass  # v7: cache removed for fair routing-only test
    if algo_node and algorithm_image:
        pass  # v7: cache removed for fair routing-only test

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


def schedule_one_firstfit_hash(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
    **kwargs: Any,
) -> Optional[ScheduleDecision]:
    """Pure Random baseline (paper Section V baseline #4).\n\n    Per user spec (2026-08-11):\n    - (c, d, a) sampled independently from uniform distribution\n    - NO cache consultation, NO cache mark, NO cost function\n    - bytes_to_load = full dataset_size always; cold_start = 1 always\n\n    Difference vs xPod: no intelligence, baseline for "no scheduling" comparison.\n    Difference vs K8s Default: K8s picks first-by-name (deterministic);\n      Pure Random samples uniformly (depends on seed|xpod_id).\n    """
    g = dag.build_dag(row)
    compute_type = (g.compute.compute_type or "").upper()
    compute_candidates = list(cluster.candidates_for_compute_type(compute_type))
    if not compute_candidates:
        return None

    data_candidates = list(cluster.all_data_nodes())
    algo_candidates = list(cluster.all_algo_nodes())
    if not data_candidates:
        data_candidates = [""]
    if not algo_candidates:
        algo_candidates = [""]

    dataset_id = g.data.dataset_id or ""
    algorithm_image = row.get("algorithm_image") or ""
    duration_s = _as_float(row.get("duration"), 0.0)
    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    dominant_amount = _dominant_resource_amount(row)
    dataset_size_bytes = int(g.data.dataset_size_bytes)

    # c: least-loaded (min post_load_ratio), tiebreak by node name
    def _post_load_ratio(c_node: str) -> float:
        cap = load_tracker.capacity_of(c_node)
        if cap <= 0:
            return float("inf")
        post = load_tracker.current_load(c_node) + max(0.0, dominant_amount)
        return post / cap

    compute_node = min(
        compute_candidates,
        key=lambda n: (_post_load_ratio(n), n),
    )

    # d: pure consistent-hash, hash(dataset_id) % N_data. NO cache check, NO cache mark.
    if dataset_id and data_candidates and data_candidates[0]:
        d_idx = int.from_bytes(
            hashlib.md5(f"firstfit|{dataset_id}".encode("utf-8")).digest()[:4], "big"
        ) % len(data_candidates)
        data_node = data_candidates[d_idx]
    else:
        data_node = data_candidates[0] if data_candidates else ""

    # a: pure consistent-hash, hash(algorithm_image) % N_algo. NO cache check, NO cache mark.
    if algorithm_image and algo_candidates and algo_candidates[0]:
        a_idx = int.from_bytes(
            hashlib.md5(f"firstfit|{algorithm_image}".encode("utf-8")).digest()[:4], "big"
        ) % len(algo_candidates)
        algo_node = algo_candidates[a_idx]
    else:
        algo_node = algo_candidates[0] if algo_candidates else ""

    # FirstFit+Hash does not consult cache: assume NOT cached
    bytes_to_load = max(0, dataset_size_bytes)
    cold_start = 1 if dataset_id else 0
    dataset_cached = False
    image_cached = False
    algo_bytes_to_load = 100 * 1024 * 1024 if algorithm_image else 0

    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)

    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    t_compute = t_compute_base * contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer (transfer happens serially at compute_node,
    # waiting for data to arrive from d/a before compute can start).
    service_time = t_compute + (data_transfer_ms + algo_transfer_ms) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        compute_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

    if data_node and dataset_id:
        pass  # v7: cache removed for fair routing-only test
    if algo_node and algorithm_image:
        pass  # v7: cache removed for fair routing-only test

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




def schedule_one_k8s_default(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
    **kwargs: Any,
) -> Optional[ScheduleDecision]:
    """K8s default scheduler baseline (paper Section V baseline #1).

    Mimics vanilla Kubernetes scheduling policy: pick the first compute node
    (sorted by name, deterministic) that has capacity, then pick the first
    data and algo node (also sorted by name). No cost function, no priority,
    no joint optimization, no cache reuse at decision time.

    Difference vs xpod:
    - xpod does 3-dim argmin over cost function S; K8s default picks the
      first node by name in each dimension independently.
    - K8s default does not consider data/algo locality, attained_service
      priority, or contention at decision time.

    Difference vs Pure Random (baseline):
    - Pure Random samples (c, d, a) from a uniform distribution; K8s default
      always picks the lexicographically first node, making it deterministic
      and reproducible.

    Cache state is still updated for fair comparison with other modes; this
    only affects what subsequent tasks see, not the K8s default routing itself.
    """
    g = dag.build_dag(row)
    compute_type = (g.compute.compute_type or "").upper()
    compute_candidates = sorted(cluster.candidates_for_compute_type(compute_type))
    if not compute_candidates:
        return None

    data_candidates = sorted(cluster.all_data_nodes())
    algo_candidates = sorted(cluster.all_algo_nodes())

    if not data_candidates:
        data_candidates = ("",)
    if not algo_candidates:
        algo_candidates = ("")

    compute_node = compute_candidates[0]
    data_node = data_candidates[0]
    algo_node = algo_candidates[0]

    dataset_id = g.data.dataset_id or ""
    dataset_size_bytes = int(g.data.dataset_size_bytes)
    algorithm_image = row.get("algorithm_image") or ""
    duration_s = _as_float(row.get("duration"), 0.0)
    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    dominant_amount = _dominant_resource_amount(row)

    dataset_cached = False  # baseline does not consult cache
    bytes_to_load = max(0, dataset_size_bytes)
    cold_start = 1


    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

    image_cached = False  # baseline does not consult cache
    algo_bytes_to_load = 100 * 1024 * 1024
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)


    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    t_compute = t_compute_base * contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer (transfer happens serially at compute_node,
    # waiting for data to arrive from d/a before compute can start).
    service_time = t_compute + (data_transfer_ms + algo_transfer_ms) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        compute_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

    if data_node and dataset_id:
        pass  # v7: cache removed for fair routing-only test
    if algo_node and algorithm_image:
        pass  # v7: cache removed for fair routing-only test

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


def schedule_one_decoupled_cd(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
    **kwargs: Any,
) -> Optional[ScheduleDecision]:
    """Tiresias+Quiver decoupled baseline (paper Section V baseline #2).

    Mimics the natural alternative: use Tiresias 2D-LAS for compute selection
    (priority-driven, sorted by attained_service) and Quiver consistent-hash
    for data/algo selection. Each dimension is decided independently, NOT
    jointly argmin-ed.

    Difference vs xpod:
    - xpod 3-dim argmin over (c, d, a) using cost S = alpha*t_total + beta*locality - gamma*priority
    - Decoupled C+D: c picked by priority-first-fit, d picked by hash(dataset_id),
      a picked by hash(algorithm_image). The 3 dimensions do NOT co-optimize.

    Difference vs Pure Random (baseline):
    - Pure Random samples (c, d, a) uniformly; Decoupled C+D uses Tiresias 2D-LAS
      priority for c and Quiver consistent-hash for d/a, making it the strongest
      decoupled alternative.

    Reviewer Q1 answer: this baseline tests whether combining Tiresias + Quiver
    can match xpod without explicit joint optimization. The expected outcome is
    near-equal JCT (transfer volume is 0.01% of total JCT) but worse load balance
    (2D-LAS does not enforce balanced placement).
    """
    g = dag.build_dag(row)
    compute_type = (g.compute.compute_type or "").upper()
    compute_candidates = list(cluster.candidates_for_compute_type(compute_type))
    if not compute_candidates:
        return None

    data_candidates = list(cluster.all_data_nodes())
    algo_candidates = list(cluster.all_algo_nodes())

    if not data_candidates:
        data_candidates = [""]
    if not algo_candidates:
        algo_candidates = [""]

    dataset_id = g.data.dataset_id or ""
    dataset_size_bytes = int(g.data.dataset_size_bytes)
    algorithm_image = row.get("algorithm_image") or ""
    duration_s = _as_float(row.get("duration"), 0.0)
    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    dominant_amount = _dominant_resource_amount(row)
    xpod_id = g.xpod_id or ""

    # Tiresias 2D-LAS: pick compute with highest priority (lowest attained_service).
    # attained_service is per-xPod, not per-node; we approximate priority as
    # the (current_load + new_load)/capacity ratio, so 2D-LAS effectively picks
    # the most-available node.
    def _tiresias_score(c_node: str) -> Tuple[float, str]:
        cap = load_tracker.capacity_of(c_node)
        if cap <= 0:
            return (float("inf"), c_node)
        post = load_tracker.current_load(c_node) + max(0.0, dominant_amount)
        return (post / cap, c_node)

    compute_node = min(compute_candidates, key=_tiresias_score)

    # Quiver consistent-hash for data: hash(dataset_id) % N_data
    if dataset_id and data_candidates and data_candidates[0]:
        d_idx = int.from_bytes(
            hashlib.md5(f"data-quiver|{dataset_id}".encode("utf-8")).digest()[:4], "big"
        ) % len(data_candidates)
        data_node = data_candidates[d_idx]
    else:
        data_node = data_candidates[0] if data_candidates else ""

    # Quiver consistent-hash for algo: hash(algorithm_image) % N_algo
    if algorithm_image and algo_candidates and algo_candidates[0]:
        a_idx = int.from_bytes(
            hashlib.md5(f"algo-quiver|{algorithm_image}".encode("utf-8")).digest()[:4], "big"
        ) % len(algo_candidates)
        algo_node = algo_candidates[a_idx]
    else:
        algo_node = algo_candidates[0] if algo_candidates else ""

    dataset_cached = False  # baseline does not consult cache
    bytes_to_load = max(0, dataset_size_bytes)
    cold_start = 1


    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

    image_cached = False  # baseline does not consult cache
    algo_bytes_to_load = 100 * 1024 * 1024
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)


    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    t_compute = t_compute_base * contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer (transfer happens serially at compute_node,
    # waiting for data to arrive from d/a before compute can start).
    service_time = t_compute + (data_transfer_ms + algo_transfer_ms) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        compute_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

    if data_node and dataset_id:
        pass  # v7: cache removed for fair routing-only test
    if algo_node and algorithm_image:
        pass  # v7: cache removed for fair routing-only test

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


def schedule_one_tetris(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    service_tracker: ServiceTracker,
    load_tracker: NodeLoadTracker,
    **kwargs: Any,
) -> Optional[ScheduleDecision]:
    """Tetris multi-resource packing baseline (paper Section V baseline #5).

    Implements the SIGCOMM'14 Grandl et al. multi-resource packing strategy:
    enumerate all (c, d, a) tuples and pick the one that maximizes the minimum
    remaining capacity ratio across all 3 dimensions. This is a 3-dim argmin
    but on a different objective than xPod (Tetris: balance utilization;
    xPod: minimize completion time + locality - priority).

    Difference vs xpod:
    - xpod minimizes S = alpha*(t_compute + t_data_transfer + t_algo_transfer)
      + beta*data_locality - gamma*priority
    - Tetris maximizes min(remaining_capacity_ratio) over c, d, a
    - Tetris ignores transfer time, locality, and attained_service priority

    Difference vs FirstFit+Hash:
    - FirstFit+Hash makes 3 INDEPENDENT decisions (c least-loaded, d consistent-hash, a consistent-hash)
    - Tetris makes 1 JOINT decision enumerating all (c, d, a) combos and picking max-min balance
    - Tetris does NOT consult or update cache (pure remaining capacity across all 3 dims)

    Difference vs Decoupled C+D:
    - Decoupled C+D makes 3 independent decisions (c priority-fit, d hash, a hash)
    - Tetris makes 1 joint decision over (c, d, a) for max-min balance

    Score definition (per user spec 2026-08-11): for each (c, d, a), compute
    score = min(c_remaining/c_cap, d_remaining/d_cap, a_remaining/a_cap)
    where remaining_cap = max(0, 1 - current_count / capacity).
    d_remaining uses cache.cached_dataset_count(d_node);
    a_remaining uses cache.cached_image_count(a_node).
    Tetris does NOT check is_dataset_cached_on / is_image_cached_on (no cache hit bonus).
    Tiebreak: deterministic by node name.
    """
    g = dag.build_dag(row)
    compute_type = (g.compute.compute_type or "").upper()
    compute_candidates = list(cluster.candidates_for_compute_type(compute_type))
    if not compute_candidates:
        return None

    data_candidates = list(cluster.all_data_nodes())
    algo_candidates = list(cluster.all_algo_nodes())

    if not data_candidates:
        data_candidates = [""]
    if not algo_candidates:
        algo_candidates = [""]

    dataset_id = g.data.dataset_id or ""
    dataset_size_bytes = int(g.data.dataset_size_bytes)
    algorithm_image = row.get("algorithm_image") or ""
    duration_s = _as_float(row.get("duration"), 0.0)
    submit_time_s = _as_float(row.get("submit_time"), 0.0)
    dominant_amount = _dominant_resource_amount(row)

    # Enumerate all (c, d, a) tuples and score by max-min remaining capacity.
    best = None
    for c_node in compute_candidates:
        c_cap = load_tracker.capacity_of(c_node)
        if c_cap <= 0:
            c_remaining = 0.0
        else:
            post = load_tracker.current_load(c_node) + max(0.0, dominant_amount)
            c_remaining = max(0.0, 1.0 - post / c_cap)
        # Per user spec (2026-08-11): d/a score = remaining CAPACITY, NOT cache hit
        data_cap = (getattr(cluster, 'data_capacity', {}) or {})
        algo_cap = (getattr(cluster, 'algo_capacity', {}) or {})
        for d_node in data_candidates:
            d_cap = float(data_cap.get(d_node, 50)) if d_node else 0.0
            if d_cap <= 0:
                d_remaining = 0.0
            else:
                d_remaining = max(0.0, 1.0 - cache.cached_dataset_count(d_node) / d_cap)
            for a_node in algo_candidates:
                a_cap = float(algo_cap.get(a_node, 50)) if a_node else 0.0
                if a_cap <= 0:
                    a_remaining = 0.0
                else:
                    a_remaining = max(0.0, 1.0 - cache.cached_image_count(a_node) / a_cap)
                tetris_score = min(c_remaining, d_remaining, a_remaining)
                key = (tetris_score, c_node, d_node, a_node)
                if best is None or key > best:
                    best = key

    if best is None:
        return None

    _, compute_node, data_node, algo_node = best

    dataset_cached = False  # baseline does not consult cache
    bytes_to_load = max(0, dataset_size_bytes)
    cold_start = 1


    data_lat_ms, data_bw_mbps = cluster.network_profile(data_node, compute_node)
    data_transfer_ms = estimate_transfer_ms(bytes_to_load, data_bw_mbps, data_lat_ms)

    image_cached = False  # baseline does not consult cache
    algo_bytes_to_load = 100 * 1024 * 1024
    algo_lat_ms, algo_bw_mbps = cluster.network_profile(algo_node, compute_node)
    algo_transfer_ms = estimate_transfer_ms(algo_bytes_to_load, algo_bw_mbps, algo_lat_ms)


    contention = load_tracker.contention_factor(compute_node, dominant_amount)
    t_compute_base = duration_s * (max(1e-9, dominant_amount) ** -0.5)
    t_compute = t_compute_base * contention
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # Per user spec 2026-08-11: real JCT = finish - submit (queueing model).
    # service_time = t_compute + data/algo transfer (transfer happens serially at compute_node,
    # waiting for data to arrive from d/a before compute can start).
    service_time = t_compute + (data_transfer_ms + algo_transfer_ms) / 1000.0
    start_time, finish_time = load_tracker.try_acquire(
        compute_node, submit_time_s, service_time, dominant_amount
    )
    jct_s = (finish_time - submit_time_s) if finish_time > submit_time_s else service_time  # v7: real jct = wait + service

    if data_node and dataset_id:
        pass  # v7: cache removed for fair routing-only test
    if algo_node and algorithm_image:
        pass  # v7: cache removed for fair routing-only test

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



class NodeHistory:
    """Sliding window of recent task assignments per compute node.

    Used by xPod (A+B) to predict near-future contention: nodes that have
    received many recent tasks get a penalty in the score function, pushing
    the scheduler to spread load. Pure online: no pre-scan needed.
    """

    def __init__(self, window: int = 1000) -> None:
        from collections import deque
        self.window = window
        self._queues: Dict[str, Any] = {}

    def record(self, node: str) -> None:
        from collections import deque
        if node not in self._queues:
            self._queues[node] = deque()
        self._queues[node].append(1)
        while len(self._queues[node]) > self.window:
            self._queues[node].popleft()

    def recent_load(self, node: str) -> int:
        q = self._queues.get(node)
        return len(q) if q is not None else 0
