import logging
from typing import Any, Dict

from .scheduler import ScheduleDecision, as_float, as_int, fmt_cpu, fmt_mem_gi

logger = logging.getLogger(__name__)


def build_xpod_manifest(row: Dict[str, str], decision: ScheduleDecision, namespace: str) -> Dict[str, Any]:
    xpod_id = row.get("xpod_id") or f"xpod-{row.get('task_id') or 'unknown'}"
    return {
        "apiVersion": "scheduling.xcloud.io/v1",
        "kind": "XPod",
        "metadata": {
            "name": xpod_id,
            "namespace": namespace,
            "annotations": {
                "xpod-research/selected-compute-node": decision.compute_node,
                "xpod-research/selected-data-node": decision.data_node,
                "xpod-research/estimated-transfer-ms": str(round(decision.estimated_transfer_ms, 6)),
                "xpod-research/cold-start": str(int(decision.cold_start)),
            },
        },
        "spec": {
            "xpod_id": xpod_id,
            "source": row.get("source") or "",
            "compute": {
                "type": (row.get("compute_type") or "").upper(),
                "cpu": fmt_cpu(row.get("cpu")),
                "memory": fmt_mem_gi(row.get("mem")),
                "gpu": as_int(row.get("gpu"), 0),
                "gpuType": row.get("gpu_type") or "",
                "gangSize": as_int(row.get("gang_size"), 1),
                "topology": row.get("topology_req") or "none",
                "targetNode": decision.compute_node,
                "scheduler": "volcano",
            },
            "data": {
                "dataset": row.get("dataset_id") or "",
                "location": decision.data_node,
                "sizeBytes": as_int(row.get("dataset_size_bytes"), 0),
                "bytesToLoad": int(decision.bytes_to_load),
                "coldStart": bool(decision.cold_start),
                "scheduler": "fluid",
            },
            "algorithm": {
                "image": row.get("algorithm_image") or "",
                "framework": row.get("algorithm_framework") or "",
                "type": row.get("algorithm_type") or "",
                "batchSize": as_int(row.get("batch_size"), 0),
                "learningRate": as_float(row.get("learning_rate"), 0.0),
                "qosClass": row.get("qos_class") or "",
                "priority": as_int(row.get("priority"), 0),
                "slaMs": as_int(row.get("sla_ms"), 0),
                "scheduler": "koord-scheduler",
            },
        },
    }


def build_volcano_job_manifest(
    row: Dict[str, str],
    decision: ScheduleDecision,
    namespace: str,
    scheduler_name: str = "volcano",
) -> Dict[str, Any]:
    name = row.get("xpod_id") or f"xpod-{row.get('task_id') or 'unknown'}"
    compute_type = (row.get("compute_type") or "").upper()
    requested_replicas = max(1, as_int(row.get("gang_size"), 1))
    replicas = requested_replicas
    if compute_type == "GPU":
        replicas_cap = as_int(row.get("vgpu_replicas_cap") or row.get("vgpu_cap") or row.get("vgpu_max") or "", 100)
        replicas = min(replicas, max(1, replicas_cap))

    requests: Dict[str, Any] = {"cpu": "100m", "memory": "64Mi"}
    limits: Dict[str, Any] = {"cpu": "100m", "memory": "64Mi"}
    if compute_type == "GPU":
        requests["volcano.sh/vgpu-number"] = "1"
        limits["volcano.sh/vgpu-number"] = "1"
        requests["volcano.sh/vgpu-memory"] = "20"
        limits["volcano.sh/vgpu-memory"] = "20"

    container = {
        "name": "workload",
        "image": "busybox:latest",
        "imagePullPolicy": "IfNotPresent",
        "command": ["sh", "-lc", "sleep 1"],
        "resources": {"requests": requests, "limits": limits},
        "env": [
            {"name": "XPOD_ID", "value": name},
            {"name": "DATASET_ID", "value": row.get("dataset_id") or ""},
            {"name": "DATA_NODE", "value": decision.data_node},
            {"name": "COLD_START", "value": str(int(decision.cold_start))},
        ],
    }

    template = {
        "metadata": {"labels": {"xpod": name}},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"kubernetes.io/hostname": decision.compute_node},
            "containers": [container],
        },
    }
    template["metadata"]["annotations"] = {
        "xpod-research/decision-compute-node": decision.compute_node,
        "xpod-research/decision-data-node": decision.data_node,
        "xpod-research/decision-algo-node": decision.algo_node,
    }
    if compute_type == "GPU":
        template["metadata"]["annotations"].update(
            {
                "xpod-research/original-cpu": str(row.get("cpu") or ""),
                "xpod-research/original-mem": str(row.get("mem") or ""),
                "xpod-research/original-gpu": str(row.get("gpu") or ""),
                "xpod-research/original-gang-size": str(row.get("gang_size") or ""),
                "volcano.sh/vgpu-mode": "hami-core",
            }
        )

    return {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "minAvailable": replicas,
            "schedulerName": scheduler_name,
            "queue": "default",
            "plugins": {"svc": [], "env": [], "ssh": []},
            "tasks": [
                {
                    "name": "worker",
                    "replicas": replicas,
                    "template": template,
                }
            ],
        },
    }


def build_fluid_dataset_manifest(row: Dict[str, str], decision: ScheduleDecision, namespace: str) -> Dict[str, Any]:
    dataset_id = row.get("dataset_id") or "dataset"
    name = f"ds-{dataset_id}"
    return {
        "apiVersion": "data.fluid.io/v1alpha1",
        "kind": "Dataset",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                "xpod-research/dataset-id": dataset_id,
                "xpod-research/dataset-size-bytes": str(as_int(row.get("dataset_size_bytes"), 0)),
                "xpod-research/data-node": decision.data_node,
            },
        },
        "spec": {
            "mounts": [
                {
                    "mountPoint": "/dataset",
                    "name": dataset_id,
                    "options": {
                        "source": "synthetic",
                        "location": decision.data_node,
                    },
                }
            ]
        },
    }


def build_fluid_runtime_manifest(row: Dict[str, str], decision: ScheduleDecision, namespace: str) -> Dict[str, Any]:
    dataset_id = row.get("dataset_id") or "dataset"
    name = f"rt-{dataset_id}"
    return {
        "apiVersion": "data.fluid.io/v1alpha1",
        "kind": "AlluxioRuntime",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": 1,
            "tieredstore": {"levels": [{"mediumtype": "MEM", "path": "/dev/shm", "quota": "1Gi"}]},
            "properties": {
                "xpod-research/target-compute-node": decision.compute_node,
                "xpod-research/cold-start": str(int(decision.cold_start)),
            },
        },
    }
