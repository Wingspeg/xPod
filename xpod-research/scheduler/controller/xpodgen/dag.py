import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


@dataclass(frozen=True)
class AlgorithmNode:
    framework: str
    algo_type: str
    image: str
    qos_class: str
    priority: int
    sla_ms: float


@dataclass(frozen=True)
class DataNode:
    dataset_id: str
    data_node: str
    dataset_size_bytes: int


@dataclass(frozen=True)
class ComputeNode:
    compute_type: str
    duration_s: float


@dataclass(frozen=True)
class XPodDAG:
    xpod_id: str
    algorithm: AlgorithmNode
    data: DataNode
    compute: ComputeNode

    def evaluate(
        self,
        compute_node: str,
        cached: bool,
        network_latency_ms: float,
        network_bandwidth_mbps: float,
        alpha: float,
        beta: float,
        gamma: float,
        penalty_weight: float,
    ) -> Tuple[int, float, float]:
        bytes_to_load = 0 if cached else int(self.data.dataset_size_bytes)
        transfer_ms = estimate_transfer_ms(bytes_to_load, network_bandwidth_mbps, network_latency_ms)
        score = score_request(
            self.compute.duration_s,
            transfer_ms,
            network_latency_ms,
            self.algorithm.sla_ms,
            alpha,
            beta,
            gamma,
            penalty_weight,
        )
        return bytes_to_load, transfer_ms, score


def build_dag(row: Dict[str, str]) -> XPodDAG:
    xpod_id = row.get("xpod_id") or f"xpod-{row.get('task_id') or 'unknown'}"
    algorithm = AlgorithmNode(
        framework=row.get("algorithm_framework") or "",
        algo_type=row.get("algorithm_type") or "",
        image=row.get("algorithm_image") or "",
        qos_class=row.get("qos_class") or "",
        priority=as_int(row.get("priority"), 0),
        sla_ms=as_float(row.get("sla_ms"), 0.0),
    )
    data = DataNode(
        dataset_id=row.get("dataset_id") or "",
        data_node=row.get("data_node") or "",
        dataset_size_bytes=as_int(row.get("dataset_size_bytes"), 0),
    )
    compute = ComputeNode(
        compute_type=(row.get("compute_type") or "").upper(),
        duration_s=as_float(row.get("duration"), 0.0),
    )
    return XPodDAG(xpod_id=xpod_id, algorithm=algorithm, data=data, compute=compute)
