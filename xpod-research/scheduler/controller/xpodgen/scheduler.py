import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from . import dag

logger = logging.getLogger(__name__)


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


def as_int(x: Any, default: int = 0) -> int:
    v = parse_number(x, None)
    if v is None:
        return default
    return int(v)


def as_float(x: Any, default: float = 0.0) -> float:
    v = parse_number(x, None)
    if v is None:
        return default
    return float(v)


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


@dataclass(frozen=True)
class ClusterConfig:
    data_nodes: Tuple[str, ...]
    gpu_nodes: Tuple[str, ...]
    cpu_nodes: Tuple[str, ...]
    seed: int
    base_latency_ms: float
    base_bandwidth_mbps: float
    algo_nodes: Tuple[str, ...] = ()
    rack_map: Optional[Dict[str, str]] = None  # node_name -> rack_id; None = use prefix-based fallback

    def candidates_for_compute_type(self, compute_type: str) -> Tuple[str, ...]:
        if (compute_type or "").upper() == "GPU":
            return self.gpu_nodes
        return self.cpu_nodes

    def network_profile(self, data_node: str, compute_node: str) -> Tuple[float, float]:
        if not data_node or not compute_node:
            return self.base_latency_ms, self.base_bandwidth_mbps
        if data_node == compute_node:
            return clamp(self.base_latency_ms * 0.25, 0.1, 100.0), clamp(self.base_bandwidth_mbps * 5.0, 100.0, 100000.0)
        # rack_map takes precedence over prefix-based fallback
        if (self.rack_map is not None
                and data_node in self.rack_map
                and compute_node in self.rack_map
                and self.rack_map[data_node] == self.rack_map[compute_node]):
            return clamp(self.base_latency_ms * 0.5, 0.1, 100.0), clamp(self.base_bandwidth_mbps * 2.0, 100.0, 100000.0)
        if data_node.split("-")[0] == compute_node.split("-")[0]:
            return clamp(self.base_latency_ms * 0.5, 0.1, 100.0), clamp(self.base_bandwidth_mbps * 2.0, 100.0, 100000.0)
        return self.base_latency_ms, self.base_bandwidth_mbps

    def same_rack(self, a_node: str, b_node: str) -> bool:
        """True if a and b are in the same rack (per rack_map), False otherwise."""
        if self.rack_map is None:
            return False
        return (a_node in self.rack_map
                and b_node in self.rack_map
                and self.rack_map[a_node] == self.rack_map[b_node])

    def all_data_nodes(self) -> Tuple[str, ...]:
        return self.data_nodes

    def all_algo_nodes(self) -> Tuple[str, ...]:
        return self.algo_nodes


class ResourceCacheState:
    """跟踪每个 data_node 缓存了哪些 dataset，每个 algo_node 缓存了哪些 image。"""

    def __init__(self) -> None:
        self._dataset_on_data_node: Dict[str, Set[str]] = {}
        self._image_on_algo_node: Dict[str, Set[str]] = {}

    def is_dataset_cached_on(self, data_node: str, dataset_id: str) -> bool:
        return dataset_id in self._dataset_on_data_node.get(data_node, set())

    def mark_dataset_cached(self, data_node: str, dataset_id: str) -> None:
        self._dataset_on_data_node.setdefault(data_node, set()).add(dataset_id)

    def is_image_cached_on(self, algo_node: str, image: str) -> bool:
        return image in self._image_on_algo_node.get(algo_node, set())

    def mark_image_cached(self, algo_node: str, image: str) -> None:
        self._image_on_algo_node.setdefault(algo_node, set()).add(image)

    def cached_dataset_count(self, data_node: str) -> int:
        """Number of distinct datasets currently cached on this data node.
        Used by Tetris (multi-resource packing) to compute remaining capacity.
        """
        return len(self._dataset_on_data_node.get(data_node, set()))

    def cached_image_count(self, algo_node: str) -> int:
        """Number of distinct algorithm images currently cached on this algo node."""
        return len(self._image_on_algo_node.get(algo_node, set()))

class ServiceTracker:
    """跟踪每个 xPod 的累计 attained service。

    Tiresias 2D-LAS 语义：attained_service = Σ (duration_s × dominant_resource_amount)
    优先级 = 1 / max(1, attained_service)，已得服务少的优先级高。
    """

    def __init__(self) -> None:
        self._attained: Dict[str, float] = {}

    def get_attained(self, xpod_id: str) -> float:
        return self._attained.get(xpod_id, 0.0)

    def add_service(self, xpod_id: str, duration_s: float, dominant_amount: float) -> None:
        if not xpod_id:
            return
        delta = max(0.0, float(duration_s)) * max(1.0, float(dominant_amount))
        self._attained[xpod_id] = self._attained.get(xpod_id, 0.0) + delta


class NodeLoadTracker:
    """跟踪每个 compute_node 当前的活跃任务负载。

    任务在调度时立刻占用节点（add_load），在 submit_time + duration_s 后释放（通过 release_until 推进时间）。
    contention factor = max(1.0, (current_load + new_task_load) / node_capacity)
    """

    def __init__(self, capacity: Dict[str, float]) -> None:
        self._capacity: Dict[str, float] = dict(capacity)
        self._active: Dict[str, list] = {n: [] for n in capacity}
        self._next_free: Dict[str, float] = {}  # next available time per node (for O(1) queue)

    def release_until(self, current_time_s: float) -> None:
        for node, tasks in self._active.items():
            self._active[node] = [(t, l) for (t, l) in tasks if t > current_time_s]

    def current_load(self, node: str) -> float:
        return sum(l for (_, l) in self._active.get(node, []))

    def capacity_of(self, node: str) -> float:
        return self._capacity.get(node, 1.0)

    def contention_factor(self, node: str, new_load: float) -> float:
        """Reverted to v2-style O(N) contention (sum active loads)."""
        cap = self.capacity_of(node)
        total = self.current_load(node) + max(0.0, new_load)
        return max(1.0, total / max(1e-9, cap))

    def add_task(self, node: str, release_time_s: float, load_units: float) -> None:
        if node not in self._active:
            self._active[node] = []
        self._active[node].append((float(release_time_s), float(max(0.0, load_units))))

    def try_acquire(self, node: str, submit_time_s: float, service_time_s: float,
                    load_units: float) -> Tuple[float, float]:
        """Real O(1) queue (v7: 4 mode routing trade-off).

        start = max(submit_time, next_free[node])  # wait if node busy
        finish = start + service_time
        next_free[node] = finish  # node busy until then

        JCT = finish - submit_time = wait_time + service_time.
        """
        start = max(float(submit_time_s), self._next_free.get(node, 0.0))
        finish = start + max(0.0, float(service_time_s))
        self._next_free[node] = finish
        return start, finish


@dataclass
class ScheduleDecision:
    compute_node: str
    data_node: str
    algo_node: str = ""
    bytes_to_load: int = 0
    network_latency_ms: float = 0.0
    network_bandwidth_mbps: float = 0.0
    estimated_transfer_ms: float = 0.0
    cold_start: int = 0
    score: float = 0.0
    jct_s: float = 0.0
    contention_factor: float = 1.0
    algo_transfer_ms: float = 0.0
    image_cached: int = 0
    algo_bytes_to_load: int = 0


class CacheState:
    def __init__(self) -> None:
        self._cached: Set[Tuple[str, str]] = set()

    def is_cached(self, compute_node: str, dataset_id: str) -> bool:
        return (compute_node, dataset_id) in self._cached

    def mark_cached(self, compute_node: str, dataset_id: str) -> None:
        self._cached.add((compute_node, dataset_id))


def estimate_transfer_ms(bytes_to_load: int, bandwidth_mbps: float, latency_ms: float) -> float:
    if bytes_to_load <= 0 or bandwidth_mbps <= 0:
        return float(latency_ms)
    return float(latency_ms) + (bytes_to_load * 8.0) / (bandwidth_mbps * 1_000_000.0) * 1000.0


def score_request(
    duration_s: float,
    transfer_ms: float,
    latency_ms: float,
    sla_ms: float,
    alpha: float,
    beta: float,
    gamma: float,
    penalty_weight: float,
) -> float:
    base = alpha * max(duration_s, 0.0) + beta * (transfer_ms / 1000.0) + gamma * (latency_ms / 1000.0)
    if sla_ms <= 0:
        return base
    slack_ms = sla_ms - (duration_s * 1000.0 + transfer_ms)
    penalty = max(0.0, -slack_ms) / 1000.0 * penalty_weight
    return base + penalty


def schedule_one(
    row: Dict[str, str],
    cluster: ClusterConfig,
    cache: ResourceCacheState,
    alpha: float,
    beta: float,
    gamma: float,
    penalty_weight: float,
) -> Optional[ScheduleDecision]:
    g = dag.build_dag(row)
    compute_candidates = cluster.candidates_for_compute_type(g.compute.compute_type)
    if not compute_candidates:
        return None

    data_candidates = cluster.all_data_nodes()
    algo_candidates = cluster.all_algo_nodes()

    if not data_candidates:
        data_candidates = ("",)
    if not algo_candidates:
        algo_candidates = ("",)

    dataset_id = g.data.dataset_id
    dataset_size_bytes = int(g.data.dataset_size_bytes)
    duration_s = float(g.compute.duration_s)
    sla_ms = float(g.compute.sla_ms)

    best: Optional[ScheduleDecision] = None

    for c_node in compute_candidates:
        for d_node in data_candidates:
            for a_node in algo_candidates:
                dataset_cached = bool(dataset_id) and bool(d_node) and cache.is_dataset_cached_on(d_node, dataset_id)
                bytes_to_load = 0 if dataset_cached else max(0, dataset_size_bytes)
                lat_ms, bw_mbps = cluster.network_profile(d_node, c_node)
                transfer_ms = estimate_transfer_ms(bytes_to_load, bw_mbps, lat_ms)

                s = score_request(duration_s, transfer_ms, lat_ms, sla_ms, alpha, beta, gamma, penalty_weight)

                if best is None or s < best.score:
                    best = ScheduleDecision(
                        compute_node=c_node,
                        data_node=d_node,
                        algo_node=a_node,
                        bytes_to_load=int(bytes_to_load),
                        network_latency_ms=lat_ms,
                        network_bandwidth_mbps=bw_mbps,
                        estimated_transfer_ms=transfer_ms,
                        cold_start=0 if dataset_cached else 1,
                        score=s,
                    )
    if best is None:
        return None
    if dataset_id and best.data_node:
        cache.mark_dataset_cached(best.data_node, dataset_id)
    return best


def fmt_cpu(cpu: Any) -> str:
    v = parse_number(cpu, None)
    if v is None:
        return "1"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return str(v)


def fmt_mem_gi(mem: Any) -> str:
    v = parse_number(mem, None)
    if v is None:
        return "1Gi"
    gi = max(float(v), 0.001)
    if abs(gi - round(gi)) < 1e-9:
        return f"{int(round(gi))}Gi"
    return f"{gi}Gi"
