#!/usr/bin/env python3
import argparse
import csv
import glob
import hashlib
import json
import logging
import math
import os
import pandas as pd
import random
import sys

logger = logging.getLogger(__name__)


def norm_key(k):
    return k.strip().lower().replace(" ", "_")


def find_key(d, candidates):
    dk = {norm_key(k): k for k in d.keys()}
    for c in candidates:
        nc = norm_key(c)
        if nc in dk:
            return dk[nc]
    return None


def parse_number(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return default


def detect_file(raw_dir, prefix):
    candidates = sorted(glob.glob(os.path.join(raw_dir, f"{prefix}*.csv")))
    return candidates[0] if candidates else None


def read_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def read_csv_noheader(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r, start=1):
            yield i, row


def write_csv(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


def load_job_table(path):
    m = {}
    if not path:
        return m
    for row in read_csv(path):
        k_job = find_key(row, ["job_id", "jobid", "job"])
        if not k_job:
            continue
        job_id = str(row[k_job]).strip()
        submit_k = find_key(row, ["submit_time", "start_time", "create_time", "ts", "timestamp"])
        group_k = find_key(row, ["group", "group_id", "group_name"])
        user_k = find_key(row, ["user", "user_id", "username"])
        prio_k = find_key(row, ["priority", "prio"])
        m[job_id] = {
            "submit_time": parse_number(row.get(submit_k), None) if submit_k else None,
            "group": row.get(group_k) if group_k else None,
            "user": row.get(user_k) if user_k else None,
            "priority": row.get(prio_k) if prio_k else None,
        }
    return m


def load_task_table(path):
    m = {}
    if not path:
        return m
    for line_no, row in read_csv_noheader(path):
        if not row:
            continue
        task_id = str(row[0]).strip() if len(row) > 0 else None
        if not task_id:
            continue
        framework = str(row[1]).strip() if len(row) > 1 else ""
        gpu = parse_number(row[2], 0) if len(row) > 2 else 0
        status = str(row[3]).strip() if len(row) > 3 else ""
        start_time = parse_number(row[4], None) if len(row) > 4 else None
        end_time = parse_number(row[5], None) if len(row) > 5 else None
        cpu = parse_number(row[6], None) if len(row) > 6 else None
        mem = parse_number(row[7], None) if len(row) > 7 else None
        gpu_type = str(row[9]).strip() if len(row) > 9 else ""
        m[task_id] = {
            "framework": framework,
            "gpu": gpu or 0,
            "gpu_type": gpu_type,
            "cpu": cpu,
            "mem": mem,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
        }
    return m


def stable_rng(seed, *parts):
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return random.Random(int.from_bytes(h.digest()[:8], "big"))


def exp_sample(rng, mean):
    u = 1.0 - rng.random()
    return -mean * math.log(u)


def weighted_choice(rng, weighted_items):
    total = 0.0
    for _, w in weighted_items:
        total += float(w)
    if total <= 0:
        return weighted_items[0][0]
    x = rng.random() * total
    acc = 0.0
    for item, w in weighted_items:
        acc += float(w)
        if x <= acc:
            return item
    return weighted_items[-1][0]


def default_dataset_catalog():
    """50+ datasets covering CV/NLP/Speech/Tabular/Graph/TimeSeries domains.
    Sized to give 30-50% dataset cache hit rate at 4 data_node (2026-08-24)."""
    gi = 1024**3
    mi = 1024**2
    return [
        # Vision - large
        {"dataset_id": "imagenet", "size_bytes": 150 * gi, "object_count": 1500000, "hot_fraction": 0.20},
        {"dataset_id": "OpenImages", "size_bytes": 500 * gi, "object_count": 9000000, "hot_fraction": 0.08},
        {"dataset_id": "coco", "size_bytes": 25 * gi, "object_count": 300000, "hot_fraction": 0.25},
        {"dataset_id": "Places365", "size_bytes": 180 * gi, "object_count": 1800000, "hot_fraction": 0.10},
        {"dataset_id": "cityscapes", "size_bytes": 12 * gi, "object_count": 25000, "hot_fraction": 0.05},
        {"dataset_id": "ADE20K", "size_bytes": 20 * gi, "object_count": 25000, "hot_fraction": 0.04},
        {"dataset_id": "LVIS", "size_bytes": 80 * gi, "object_count": 2000000, "hot_fraction": 0.06},
        # Vision - small/medium
        {"dataset_id": "CIFAR-10", "size_bytes": 170 * mi, "object_count": 60000, "hot_fraction": 0.05},
        {"dataset_id": "CIFAR-100", "size_bytes": 160 * mi, "object_count": 60000, "hot_fraction": 0.04},
        {"dataset_id": "MNIST", "size_bytes": 12 * mi, "object_count": 70000, "hot_fraction": 0.05},
        {"dataset_id": "FashionMNIST", "size_bytes": 30 * mi, "object_count": 70000, "hot_fraction": 0.03},
        {"dataset_id": "ImageNet-100", "size_bytes": 7 * gi, "object_count": 130000, "hot_fraction": 0.06},
        {"dataset_id": "StanfordCars", "size_bytes": 2 * gi, "object_count": 16000, "hot_fraction": 0.04},
        {"dataset_id": "Flowers102", "size_bytes": 350 * mi, "object_count": 8000, "hot_fraction": 0.03},
        {"dataset_id": "Food101", "size_bytes": 5 * gi, "object_count": 100000, "hot_fraction": 0.04},
        {"dataset_id": "celeba", "size_bytes": 30 * gi, "object_count": 200000, "hot_fraction": 0.05},
        # NLP / Text
        {"dataset_id": "wikitext", "size_bytes": 5 * gi, "object_count": 50000, "hot_fraction": 0.35},
        {"dataset_id": "C4", "size_bytes": 750 * gi, "object_count": 300000000, "hot_fraction": 0.10},
        {"dataset_id": "BookCorpus", "size_bytes": 5 * gi, "object_count": 11000000, "hot_fraction": 0.08},
        {"dataset_id": "OpenWebText", "size_bytes": 38 * gi, "object_count": 8000000, "hot_fraction": 0.06},
        {"dataset_id": "PubMed", "size_bytes": 20 * gi, "object_count": 25000000, "hot_fraction": 0.05},
        {"dataset_id": "StackExchange", "size_bytes": 60 * gi, "object_count": 15000000, "hot_fraction": 0.04},
        {"dataset_id": "ArXiv", "size_bytes": 12 * gi, "object_count": 1700000, "hot_fraction": 0.03},
        {"dataset_id": "Gutenberg", "size_bytes": 8 * gi, "object_count": 70000, "hot_fraction": 0.03},
        {"dataset_id": "CC100", "size_bytes": 250 * gi, "object_count": 100000000, "hot_fraction": 0.05},
        # Speech / Audio
        {"dataset_id": "librispeech", "size_bytes": 60 * gi, "object_count": 200000, "hot_fraction": 0.15},
        {"dataset_id": "AudioSet", "size_bytes": 100 * gi, "object_count": 2000000, "hot_fraction": 0.04},
        {"dataset_id": "VoxCeleb", "size_bytes": 40 * gi, "object_count": 1500000, "hot_fraction": 0.04},
        {"dataset_id": "CommonVoice", "size_bytes": 25 * gi, "object_count": 1500000, "hot_fraction": 0.05},
        {"dataset_id": "LJSpeech", "size_bytes": 3 * gi, "object_count": 13000, "hot_fraction": 0.03},
        {"dataset_id": "FMA", "size_bytes": 80 * gi, "object_count": 100000, "hot_fraction": 0.03},
        # Tabular / Recommendation / Logs
        {"dataset_id": "clicklog", "size_bytes": 500 * gi, "object_count": 50000000, "hot_fraction": 0.05},
        {"dataset_id": "embedding", "size_bytes": 80 * gi, "object_count": 1000, "hot_fraction": 0.60},
        {"dataset_id": "Criteo", "size_bytes": 70 * gi, "object_count": 45000000, "hot_fraction": 0.04},
        {"dataset_id": "MovieLens-25M", "size_bytes": 250 * mi, "object_count": 25000000, "hot_fraction": 0.05},
        {"dataset_id": "AmazonReviews", "size_bytes": 8 * gi, "object_count": 23000000, "hot_fraction": 0.04},
        {"dataset_id": "AlibabaTrace", "size_bytes": 100 * gi, "object_count": 1000000000, "hot_fraction": 0.04},
        {"dataset_id": "NYC-Taxi", "size_bytes": 18 * gi, "object_count": 200000000, "hot_fraction": 0.03},
        {"dataset_id": "wikipedia-2024", "size_bytes": 90 * gi, "object_count": 65000000, "hot_fraction": 0.04},
        # Graph
        {"dataset_id": "ogbn-papers100M", "size_bytes": 50 * gi, "object_count": 111000000, "hot_fraction": 0.04},
        {"dataset_id": "ogbn-products", "size_bytes": 2 * gi, "object_count": 2400000, "hot_fraction": 0.05},
        {"dataset_id": "Reddit", "size_bytes": 30 * gi, "object_count": 50000000, "hot_fraction": 0.03},
        {"dataset_id": "MAG-Scholar", "size_bytes": 12 * gi, "object_count": 40000000, "hot_fraction": 0.03},
        # Video / TimeSeries
        {"dataset_id": "Kinetics", "size_bytes": 400 * gi, "object_count": 650000, "hot_fraction": 0.05},
        {"dataset_id": "YouTube-8M", "size_bytes": 1.5 * 1024 * gi, "object_count": 8000000, "hot_fraction": 0.04},
        {"dataset_id": "UCF101", "size_bytes": 7 * gi, "object_count": 13000, "hot_fraction": 0.03},
        {"dataset_id": "Something-Something", "size_bytes": 30 * gi, "object_count": 220000, "hot_fraction": 0.03},
        {"dataset_id": "Electricity", "size_bytes": 2 * gi, "object_count": 1000000, "hot_fraction": 0.04},
        {"dataset_id": "ETTh1", "size_bytes": 12 * mi, "object_count": 17000, "hot_fraction": 0.04},
        {"dataset_id": "WTH", "size_bytes": 800 * mi, "object_count": 500000, "hot_fraction": 0.04},
        # 3D / Point Cloud
        {"dataset_id": "KITTI", "size_bytes": 15 * gi, "object_count": 80000, "hot_fraction": 0.03},
        {"dataset_id": "ShapeNet", "size_bytes": 30 * gi, "object_count": 5000000, "hot_fraction": 0.03},
        {"dataset_id": "ScanNet", "size_bytes": 1.5 * 1024 * gi, "object_count": 2500000, "hot_fraction": 0.02},
    ]


# PAI framework name -> academic framework name, split by compute_type (2026-08-24)
PAI_TO_ACADEMIC_FRAMEWORK_GPU = {
    "tensorflow":        "TensorFlow",
    "PyTorchWorker":     "PyTorch",
    "worker":            "TensorFlow",
    "ps":                "TensorFlow",
    "xComputeWorker":    "PyTorch",
    "TensorboardTask":   "TensorFlow",
    "evaluator":         "PyTorch",
    "DecoderWorker":     "PyTorch",
    "ReduceTask":        "TVM",
    "JupyterTask":       "PyTorch-Lightning",
    "TVMTuneMain":       "TVM",
    "OpenmpiTracker":    "Horovod",
    "OpenmpiWorker":     "Horovod",
    "OssToVolumeWorker": "PyTorch",
    "MWorker":           "PyTorch",
    "chief":             "TensorFlow",
    "BladeMain":         "PyTorch",
    "aligraph":          "AliGraph",
    "TransformGraph":    "TensorFlow",
    "M1":                "PyTorch",
}
PAI_TO_ACADEMIC_FRAMEWORK_CPU = {
    "tensorflow":        "TensorFlow-serving",
    "PyTorchWorker":     "PyTorch-Lightning",
    "worker":            "ONNXRuntime",
    "ps":                "TensorFlow-serving",
    "xComputeWorker":    "PyTorch-Lightning",
    "TensorboardTask":   "TensorFlow-serving",
    "evaluator":         "scikit-learn",
    "DecoderWorker":     "PyTorch-Lightning",
    "ReduceTask":        "MapReduce",
    "JupyterTask":       "Python-notebook",
    "TVMTuneMain":       "TVM",
    "OpenmpiTracker":    "Horovod",
    "OpenmpiWorker":     "Horovod",
    "OssToVolumeWorker": "Python-script",
    "MWorker":           "scikit-learn",
    "chief":             "TensorFlow-serving",
    "BladeMain":         "PyTorch-Lightning",
    "aligraph":          "DeepDetect",
    "TransformGraph":    "TensorFlow-serving",
    "M1":                "PyTorch-Lightning",
}

# GPU-only / CPU-only / shared framework pools (5 + 5 + 7 = 17 framework, matches paper)
GPU_FRAMEWORKS = ["TensorFlow", "PyTorch", "JAX", "MXNet", "PaddlePaddle"]
CPU_FRAMEWORKS = ["TensorFlow-serving", "ONNXRuntime", "scikit-learn", "Spark-ML", "Ray"]
SHARED_FRAMEWORKS = ["TVM", "Horovod", "AliGraph", "TensorFlow-Lite", "PyTorch-Lightning",
                     "DeepDetect", "MapReduce"]


def pick_algorithm_type(rng, framework, compute_type):
    fw = (framework or "").lower()
    if compute_type == "GPU":
        if "tensorflow" in fw:
            return weighted_choice(rng, [("CNN", 0.5), ("Transformer", 0.25), ("GNN", 0.15), ("RL", 0.1)])
        if "pytorch" in fw:
            return weighted_choice(rng, [("Transformer", 0.45), ("CNN", 0.3), ("GNN", 0.15), ("RL", 0.1)])
        return weighted_choice(rng, [("CNN", 0.35), ("Transformer", 0.35), ("GNN", 0.2), ("RL", 0.1)])
    return weighted_choice(rng, [("ETL", 0.30), ("Preprocess", 0.22), ("Inference", 0.18), ("Feature", 0.13), ("Search", 0.05), ("Speech", 0.12)])


def pick_dataset_id(rng, algo_type):
    """50+ datasets mapped by algo_type domain (2026-08-24).
    1.74M tasks distributed across 50+ datasets for 30-50% cache hit at 4 data_node."""
    m = {
        # CNN: vision large + small
        "CNN": [
            ("imagenet", 0.20), ("OpenImages", 0.10), ("coco", 0.08), ("Places365", 0.08),
            ("CIFAR-10", 0.08), ("CIFAR-100", 0.05), ("MNIST", 0.05), ("celeba", 0.06),
            ("StanfordCars", 0.05), ("Flowers102", 0.04), ("Food101", 0.05), ("FashionMNIST", 0.04),
            ("ImageNet-100", 0.06), ("ADE20K", 0.03), ("LVIS", 0.03),
        ],
        # Transformer: NLP/text + logs
        "Transformer": [
            ("clicklog", 0.040), ("wikitext", 0.06), ("C4", 0.10), ("BookCorpus", 0.08),
            ("OpenWebText", 0.08), ("PubMed", 0.06), ("embedding", 0.02), ("AudioSet", 0.05),
            ("ArXiv", 0.05), ("Gutenberg", 0.04), ("wikipedia-2024", 0.05), ("CC100", 0.05),
            ("AlibabaTrace", 0.04),
        ],
        # GNN: graph + tabular
        "GNN": [
            ("clicklog", 0.04), ("embedding", 0.020), ("ogbn-papers100M", 0.10),
            ("ogbn-products", 0.08), ("Reddit", 0.08), ("MAG-Scholar", 0.08),
            ("OpenImages", 0.06), ("Criteo", 0.08), ("AmazonReviews", 0.07),
            ("MovieLens-25M", 0.06), ("AlibabaTrace", 0.07), ("wikipedia-2024", 0.07),
        ],
        # RL: mixed, mostly embedding + video
        "RL": [
            ("embedding", 0.050), ("clicklog", 0.030), ("Kinetics", 0.12), ("YouTube-8M", 0.10),
            ("UCF101", 0.08), ("Something-Something", 0.07), ("CIFAR-10", 0.06),
            ("OpenImages", 0.06), ("MNIST", 0.05), ("AudioSet", 0.05), ("KITTI", 0.06),
            ("ShapeNet", 0.05),
        ],
        # ETL: logs + tabular
        "ETL": [
            ("clicklog", 0.04), ("wikitext", 0.10), ("Criteo", 0.10), ("AmazonReviews", 0.08),
            ("AlibabaTrace", 0.10), ("NYC-Taxi", 0.08), ("AudioSet", 0.05), ("embedding", 0.08),
            ("PubMed", 0.05), ("StackExchange", 0.05), ("wikipedia-2024", 0.06),
        ],
        # Preprocess: vision + tabular
        "Preprocess": [
            ("coco", 0.12), ("imagenet", 0.10), ("clicklog", 0.040), ("CIFAR-10", 0.08),
            ("Places365", 0.08), ("MNIST", 0.06), ("celeba", 0.06), ("OpenImages", 0.07),
            ("Food101", 0.05), ("StanfordCars", 0.05), ("ImageNet-100", 0.05),
            ("ADE20K", 0.04), ("cityscapes", 0.05), ("Flowers102", 0.04), ("FashionMNIST", 0.05),
        ],
        # Inference: small + multi-modal
        "Inference": [
            ("embedding", 0.025), ("CIFAR-10", 0.10), ("MNIST", 0.08), ("Kinetics", 0.08),
            ("AudioSet", 0.06), ("librispeech", 0.05), ("clicklog", 0.08), ("imagenet", 0.06),
            ("Food101", 0.05), ("Flowers102", 0.04), ("ETTh1", 0.05), ("WTH", 0.04),
            ("StanfordCars", 0.04), ("CommonVoice", 0.04), ("VoxCeleb", 0.04), ("KITTI", 0.04),
        ],
        # Feature: logs + tabular
        "Feature": [
            ("clicklog", 0.04), ("embedding", 0.020), ("MNIST", 0.06), ("Criteo", 0.10),
            ("AmazonReviews", 0.08), ("MovieLens-25M", 0.07), ("CIFAR-10", 0.06),
            ("AlibabaTrace", 0.10), ("wikipedia-2024", 0.05), ("WTH", 0.05),
            ("ETTh1", 0.05), ("Electricity", 0.05), ("StackExchange", 0.05),
        ],
        # Search: misc
        "Search": [
            ("clicklog", 0.040), ("embedding", 0.10), ("CIFAR-10", 0.08), ("AlibabaTrace", 0.10),
            ("wikipedia-2024", 0.08), ("AmazonReviews", 0.08), ("Criteo", 0.08),
            ("NYC-Taxi", 0.06), ("AudioSet", 0.05), ("MNIST", 0.05), ("WTH", 0.05),
            ("ETTh1", 0.04), ("Electricity", 0.03),
        ],
        # Speech: audio
        "Speech": [
            ("librispeech", 0.30), ("AudioSet", 0.20), ("VoxCeleb", 0.15),
            ("CommonVoice", 0.15), ("LJSpeech", 0.10), ("FMA", 0.10),
        ],
    }
    return weighted_choice(rng, m.get(algo_type, [("clicklog", 1.0)]))


def pick_qos(rng, algo_type):
    if algo_type in {"Inference", "Search"}:
        return weighted_choice(rng, [("gold", 0.45), ("silver", 0.4), ("bronze", 0.15)])
    return weighted_choice(rng, [("gold", 0.2), ("silver", 0.55), ("bronze", 0.25)])


def qos_to_priority(qos):
    return {"gold": 1000, "silver": 100, "bronze": 10}.get(qos, 100)


def qos_to_sla_ms(qos):
    return {"gold": 2000, "silver": 10000, "bronze": 60000}.get(qos, 10000)


def compute_node_for(machine_id, compute_type, gpu_nodes, cpu_nodes, seed):
    if compute_type == "GPU":
        nodes = gpu_nodes
    else:
        nodes = cpu_nodes
    if not nodes:
        return ""
    rng = stable_rng(seed, "compute_node", machine_id, compute_type)
    return nodes[int(rng.random() * len(nodes)) % len(nodes)]


def data_node_for(dataset_id, data_nodes, seed):
    if not data_nodes:
        return ""
    rng = stable_rng(seed, "data_node", dataset_id)
    return data_nodes[int(rng.random() * len(data_nodes)) % len(data_nodes)]


def network_profile(data_node, compute_node, seed):
    if not data_node or not compute_node:
        return 2.0, 1000.0
    same = (data_node.split("-")[0] == compute_node.split("-")[0])
    if same:
        return 0.5, 10000.0
    rng = stable_rng(seed, "net", data_node, compute_node)
    return 2.0 + rng.random() * 3.0, 1000.0 + rng.random() * 4000.0


def estimate_transfer_ms(bytes_to_load, bandwidth_mbps, latency_ms):
    b = parse_number(bytes_to_load, 0) or 0
    bw = parse_number(bandwidth_mbps, 0) or 0
    if b <= 0 or bw <= 0:
        return latency_ms
    return latency_ms + (b * 8.0) / (bw * 1_000_000.0) * 1000.0


def classify_gpu(task_res):
    if not task_res:
        return False
    gpu = task_res.get("gpu")
    if gpu is None:
        return False
    try:
        return float(gpu) > 0
    except Exception:
        return False


def extract_instances(
    raw_dir,
    out_cpu,
    out_gpu,
    out_joint,
    out_data_requests,
    out_datasets,
    job_m,
    task_m,
    seed,
    data_nodes,
    gpu_nodes,
    cpu_nodes,
    max_duration_s,
    min_cpu_rows,
    cpu_synth_rate,
):
    p = detect_file(raw_dir, "pai_instance_table")
    if not p:
        return
    datasets = default_dataset_catalog()
    dataset_by_id = {d["dataset_id"]: d for d in datasets}
    f_ds, w_ds = write_csv(out_datasets, ["dataset_id", "size_bytes", "object_count", "hot_fraction", "data_node"])
    for d in datasets:
        w_ds.writerow(
            {
                "dataset_id": d["dataset_id"],
                "size_bytes": d["size_bytes"],
                "object_count": d["object_count"],
                "hot_fraction": d["hot_fraction"],
                "data_node": data_node_for(d["dataset_id"], data_nodes, seed),
            }
        )
    f_ds.close()

    out_fields = [
        "source",
        "submit_time",
        "xpod_id",
        "task_id",
        "instance_id",
        "compute_type",
        "cpu",
        "mem",
        "gpu",
        "gpu_type",
        "gang_size",
        "topology_req",
        "algorithm_framework",
        "algorithm_type",
        "algorithm_image",
        "batch_size",
        "learning_rate",
        "qos_class",
        "priority",
        "sla_ms",
        "dataset_id",
        "dataset_size_bytes",
        "status",
        "start_time",
        "end_time",
        "duration",
    ]
    f_cpu, w_cpu = write_csv(out_cpu, out_fields)
    f_gpu, w_gpu = write_csv(out_gpu, out_fields)
    f_joint, w_joint = write_csv(out_joint, out_fields)

    cpu_written = 0
    cpu_synth_written = 0

    for line_no, row in read_csv_noheader(p):
        if not row:
            continue
        task_id = str(row[0]).strip() if len(row) > 0 else ""
        inst_id = str(row[2]).strip() if len(row) > 2 else ""
        status = str(row[5]).strip() if len(row) > 5 else ""
        start_ts = parse_number(row[6], None) if len(row) > 6 else None
        end_ts = parse_number(row[7], None) if len(row) > 7 else None
        if start_ts is None or end_ts is None:
            continue
        duration = None
        try:
            duration = float(end_ts) - float(start_ts)
        except Exception:
            duration = None
        if duration is None or duration < 0 or (max_duration_s is not None and duration > max_duration_s):
            continue
        machine_id = row[8] if len(row) > 8 else ""
        task_res = task_m.get(task_id)
        if not task_res:
            continue

        gpu = task_res.get("gpu") or 0
        compute_type = "GPU" if classify_gpu(task_res) else "CPU"

        rng = stable_rng(seed, "task", task_id)
        # Map PAI framework name to academic name, split by compute_type (2026-08-24)
        raw_fw = task_res.get("framework") or ""
        fw_pool = (GPU_FRAMEWORKS + SHARED_FRAMEWORKS) if compute_type == "GPU" else (CPU_FRAMEWORKS + SHARED_FRAMEWORKS)
        fw_map = PAI_TO_ACADEMIC_FRAMEWORK_GPU if compute_type == "GPU" else PAI_TO_ACADEMIC_FRAMEWORK_CPU
        academic_fw = fw_map.get(raw_fw)
        if academic_fw in fw_pool:
            framework = academic_fw
        else:
            framework = rng.choice(fw_pool)
        algo_type = pick_algorithm_type(rng, framework, compute_type)
        qos = pick_qos(rng, algo_type)
        prio = qos_to_priority(qos)
        sla_ms = qos_to_sla_ms(qos)

        if compute_type == "GPU":
            try:
                gpu_f = float(gpu)
            except Exception:
                gpu_f = 0.0
            gang_size = int(math.ceil(max(gpu_f, 1.0) / 8.0))
            topology_req = "NVLink" if gpu_f >= 8 else "PCIe"
        else:
            gang_size = 1
            topology_req = "none"

        cpu = task_res.get("cpu")
        mem = task_res.get("mem")
        if cpu is None:
            if compute_type == "GPU":
                cpu = int(4 + rng.random() * 60)
            else:
                cpu = int(1 + rng.random() * 15)
        if mem is None:
            if compute_type == "GPU":
                mem = round(8 + rng.random() * 248, 6)
            else:
                mem = round(1 + rng.random() * 63, 6)

        bs = int(weighted_choice(rng, [(16, 0.2), (32, 0.35), (64, 0.3), (128, 0.15)]))
        lr = round(10 ** (-4 + rng.random() * 2.0), 8)
        image_ver = int(rng.random() * 7) + 1  # 1..7 versions
        image = f"{(framework or 'generic').lower()}-{algo_type.lower()}-v{image_ver}:latest"

        dataset_id = pick_dataset_id(rng, algo_type)
        ds = dataset_by_id.get(dataset_id) or {"size_bytes": 0}
        dataset_size_bytes = int(ds.get("size_bytes") or 0)

        arrival = task_res.get("start_time") if task_res.get("start_time") is not None else start_ts
        delay_mean = {"gold": 10.0, "silver": 60.0, "bronze": 300.0}.get(qos, 60.0)
        submit_time = max(0.0, float(arrival) - exp_sample(rng, delay_mean)) if arrival is not None else float(start_ts)
        xpod_id = f"xpod-{task_id}"

        rec = {
            "source": "trace",
            "submit_time": submit_time,
            "xpod_id": xpod_id,
            "task_id": task_id,
            "instance_id": inst_id,
            "compute_type": compute_type,
            "cpu": cpu,
            "mem": mem,
            "gpu": gpu,
            "gpu_type": task_res.get("gpu_type") or "",
            "gang_size": gang_size,
            "topology_req": topology_req,
            "algorithm_framework": framework,
            "algorithm_type": algo_type,
            "algorithm_image": image,
            "batch_size": bs,
            "learning_rate": lr,
            "qos_class": qos,
            "priority": prio,
            "sla_ms": sla_ms,
            "dataset_id": dataset_id,
            "dataset_size_bytes": dataset_size_bytes,
            "status": status,
            "start_time": start_ts,
            "end_time": end_ts,
            "duration": duration,
        }

        w_joint.writerow(rec)
        if compute_type == "GPU":
            w_gpu.writerow(rec)
        else:
            w_cpu.writerow(rec)
            cpu_written += 1

        need_more_cpu = min_cpu_rows is not None and cpu_written < min_cpu_rows
        if need_more_cpu and compute_type == "GPU":
            rng2 = stable_rng(seed, "cpu_synth_gate", task_id, inst_id)
            if rng2.random() < (cpu_synth_rate or 0.0):
                synth_id = f"syn-cpu-{task_id}-{cpu_synth_written}"
                rngs = stable_rng(seed, "syn_cpu", synth_id)

                # Re-map framework to CPU academic name (synthetic is always CPU)
                fw_pool_s = CPU_FRAMEWORKS + SHARED_FRAMEWORKS
                fw_map_s = PAI_TO_ACADEMIC_FRAMEWORK_CPU
                academic_fw_s = fw_map_s.get(framework)
                if academic_fw_s in fw_pool_s:
                    framework_s = academic_fw_s
                else:
                    framework_s = rngs.choice(fw_pool_s)
                algo_type_s = pick_algorithm_type(rngs, framework_s, "CPU")
                image_ver_s = int(rngs.random() * 7) + 1
                qos_s = pick_qos(rngs, algo_type_s)
                prio_s = qos_to_priority(qos_s)
                sla_s = qos_to_sla_ms(qos_s)

                submit_s = max(0.0, float(start_ts) - exp_sample(rngs, {"gold": 5.0, "silver": 20.0, "bronze": 60.0}.get(qos_s, 20.0)))
                wait_s = exp_sample(rngs, {"gold": 2.0, "silver": 10.0, "bronze": 30.0}.get(qos_s, 10.0))

                start_s = submit_s + wait_s
                dur_mean = {"ETL": 120.0, "Preprocess": 45.0, "Inference": 8.0, "Feature": 25.0, "Search": 5.0, "Speech": 30.0}.get(algo_type_s, 30.0)
                duration_s = max(0.01, exp_sample(rngs, dur_mean))
                end_s = start_s + duration_s

                cpu_s = int(1 + rngs.random() * 15)
                mem_s = round(1 + rngs.random() * 63, 6)

                ds_id_s = pick_dataset_id(rngs, algo_type_s)
                ds_s = dataset_by_id.get(ds_id_s) or {"size_bytes": 0}
                ds_bytes_s = int(ds_s.get("size_bytes") or 0)

                rec_s = dict(rec)
                rec_s.update(
                    {
                        "source": "synthetic",
                        "submit_time": round(submit_s, 6),
                        "xpod_id": f"xpod-{synth_id}",
                        "task_id": synth_id,
                        "instance_id": f"syn-inst-{task_id}-{cpu_synth_written}",
                        "compute_type": "CPU",
                        "cpu": cpu_s,
                        "mem": mem_s,
                        "gpu": 0,
                        "gpu_type": "",
                        "gang_size": 1,
                        "topology_req": "none",
                        "algorithm_framework": framework_s,
                        "algorithm_type": algo_type_s,
                        "algorithm_image": f"{(framework_s or 'generic').lower()}-{algo_type_s.lower()}-v{image_ver_s}:latest",
                        "batch_size": int(weighted_choice(rngs, [(16, 0.2), (32, 0.35), (64, 0.3), (128, 0.15)])),
                        "learning_rate": round(10 ** (-4 + rngs.random() * 2.0), 8),
                        "qos_class": qos_s,
                        "priority": prio_s,
                        "sla_ms": sla_s,
                        "dataset_id": ds_id_s,
                        "dataset_size_bytes": ds_bytes_s,
                        "status": "Synthetic",
                        "start_time": round(start_s, 6),
                        "end_time": round(end_s, 6),
                        "duration": round(duration_s, 6),
                    }
                )

                w_joint.writerow(rec_s)
                w_cpu.writerow(rec_s)
                cpu_written += 1
                cpu_synth_written += 1
    f_cpu.close()
    f_gpu.close()
    f_joint.close()

    logger.info("starting xpod-level aggregation for scene4-joint")
    df = pd.read_csv(out_joint)
    row_count_before = len(df)
    logger.info("instance-level rows: %d", row_count_before)

    df_sorted = df.sort_values(by=["instance_id"], ascending=[True])

    agg_dict = {
        "source": "first",
        "submit_time": "first",
        "task_id": "first",
        "compute_type": "first",
        "cpu": "first",
        "mem": "first",
        "gpu": "first",
        "gpu_type": "first",
        "topology_req": "first",
        "algorithm_framework": "first",
        "algorithm_type": "first",
        "algorithm_image": "first",
        "batch_size": "first",
        "learning_rate": "first",
        "qos_class": "first",
        "priority": "first",
        "sla_ms": "first",
        "dataset_id": "first",
        "dataset_size_bytes": "first",
        "status": "first",
        "gang_size": "count",
        "duration": "max",
        "end_time": "max",
        "start_time": "min",
        "instance_id": lambda x: "",
    }

    df_agg = df_sorted.groupby("xpod_id").agg(agg_dict).reset_index()

    col_order = [
        "source",
        "submit_time",
        "xpod_id",
        "task_id",
        "instance_id",
        "compute_type",
        "cpu",
        "mem",
        "gpu",
        "gpu_type",
        "gang_size",
        "topology_req",
        "algorithm_framework",
        "algorithm_type",
        "algorithm_image",
        "batch_size",
        "learning_rate",
        "qos_class",
        "priority",
        "sla_ms",
        "dataset_id",
        "dataset_size_bytes",
        "status",
        "start_time",
        "end_time",
        "duration",
    ]
    df_agg = df_agg[col_order]

    df_agg.to_csv(out_joint, index=False)
    row_count_after = len(df_agg)
    compression_ratio = row_count_before / row_count_after if row_count_after > 0 else float("inf")
    logger.info(
        "xpod-level aggregation complete: before=%d, after=%d, compression_ratio=%.2fx",
        row_count_before,
        row_count_after,
        compression_ratio,
    )


def extract_machine_metrics(raw_dir, out_metrics, out_sensors):
    p_metric = detect_file(raw_dir, "pai_machine_metric")
    p_sensor = detect_file(raw_dir, "pai_sensor_table")
    if p_metric:
        f, w = write_csv(
            out_metrics,
            [
                "machine_id",
                "start_time",
                "end_time",
                "m5",
                "m6",
                "m7",
                "m8",
                "m9",
                "m10",
                "m11",
                "m12",
            ],
        )
        for line_no, row in read_csv_noheader(p_metric):
            if not row:
                continue
            rec = {
                "machine_id": row[0] if len(row) > 0 else "",
                "start_time": parse_number(row[2], None) if len(row) > 2 else "",
                "end_time": parse_number(row[3], None) if len(row) > 3 else "",
                "m5": row[4] if len(row) > 4 else "",
                "m6": row[5] if len(row) > 5 else "",
                "m7": row[6] if len(row) > 6 else "",
                "m8": row[7] if len(row) > 7 else "",
                "m9": row[8] if len(row) > 8 else "",
                "m10": row[9] if len(row) > 9 else "",
                "m11": row[10] if len(row) > 10 else "",
                "m12": row[11] if len(row) > 11 else "",
            }
            w.writerow(rec)
        f.close()
    if p_sensor:
        f, w = write_csv(
            out_sensors,
            [
                "machine_id",
                "device",
                "s2",
                "s3",
                "s4",
                "s5",
                "s7",
                "s8",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
            ],
        )
        for line_no, row in read_csv_noheader(p_sensor):
            if not row:
                continue
            rec = {
                "machine_id": row[0] if len(row) > 0 else "",
                "device": row[5] if len(row) > 5 else "",
                "s2": row[1] if len(row) > 1 else "",
                "s3": row[2] if len(row) > 2 else "",
                "s4": row[3] if len(row) > 3 else "",
                "s5": row[4] if len(row) > 4 else "",
                "s7": row[6] if len(row) > 6 else "",
                "s8": row[7] if len(row) > 7 else "",
                "s9": row[8] if len(row) > 8 else "",
                "s10": row[9] if len(row) > 9 else "",
                "s11": row[10] if len(row) > 10 else "",
                "s12": row[11] if len(row) > 11 else "",
                "s13": row[12] if len(row) > 12 else "",
                "s14": row[13] if len(row) > 13 else "",
                "s15": row[14] if len(row) > 14 else "",
                "s16": row[15] if len(row) > 15 else "",
            }
            w.writerow(rec)
        f.close()


def build_joint_requests(raw_dir, out_joint, job_m, task_m):
    p = detect_file(raw_dir, "pai_instance_table")
    if not p:
        return
    f, w = write_csv(
        out_joint,
        [
            "timestamp",
            "task_id",
            "instance_id",
            "service_type",
            "status",
            "start_time",
            "end_time",
            "cpu",
            "mem",
            "gpu",
            "duration",
            "machine_id",
        ],
    )
    for line_no, row in read_csv_noheader(p):
        if not row:
            continue
        task_id = str(row[0]).strip() if len(row) > 0 else ""
        inst_id = str(row[2]).strip() if len(row) > 2 else ""
        status = str(row[5]).strip() if len(row) > 5 else ""
        start_ts = parse_number(row[6], None) if len(row) > 6 else None
        end_ts = parse_number(row[7], None) if len(row) > 7 else None
        duration = None
        if start_ts is not None and end_ts is not None:
            try:
                duration = float(end_ts) - float(start_ts)
            except Exception:
                duration = None
        machine_id = row[8] if len(row) > 8 else ""
        task_res = task_m.get(task_id)
        cpu = task_res.get("cpu") if task_res else None
        mem = task_res.get("mem") if task_res else None
        gpu = task_res.get("gpu") if task_res else 0
        ts = start_ts
        svc = "compute_gpu" if classify_gpu(task_res) else "compute_cpu"
        rec = {
            "timestamp": ts if ts is not None else "",
            "task_id": task_id,
            "instance_id": inst_id,
            "service_type": svc,
            "status": status,
            "start_time": start_ts if start_ts is not None else "",
            "end_time": end_ts if end_ts is not None else "",
            "cpu": cpu if cpu is not None else "",
            "mem": mem if mem is not None else "",
            "gpu": gpu if gpu is not None else "",
            "duration": duration if duration is not None else "",
            "machine_id": machine_id if machine_id is not None else "",
        }
        w.writerow(rec)
    f.close()


def write_field_mapping(out_path, raw_dir):
    schema = {
        "dataset": "Alibaba Cluster Trace GPU v2020",
        "raw_dir": raw_dir,
        "raw_tables": {
            "pai_task_table.csv": {
                "columns_by_position": {
                    "1": "task_id",
                    "2": "framework_or_type",
                    "3": "gpu_requested",
                    "4": "status",
                    "5": "start_time",
                    "6": "end_time",
                    "7": "cpu_requested",
                    "8": "mem_requested",
                    "9": "unknown_9",
                    "10": "gpu_type_or_queue",
                }
            },
            "pai_instance_table.csv": {
                "columns_by_position": {
                    "1": "task_id",
                    "2": "role",
                    "3": "instance_id",
                    "4": "unknown_4",
                    "5": "unknown_5",
                    "6": "status",
                    "7": "start_time",
                    "8": "end_time",
                    "9": "machine_id",
                }
            },
            "pai_machine_metric.csv": {
                "columns_by_position": {
                    "1": "machine_id",
                    "2": "unknown_2",
                    "3": "start_time",
                    "4": "end_time",
                    "5": "metric_5",
                    "6": "metric_6",
                    "7": "metric_7",
                    "8": "metric_8",
                    "9": "metric_9",
                    "10": "metric_10",
                    "11": "metric_11",
                    "12": "metric_12",
                }
            },
            "pai_sensor_table.csv": {
                "columns_by_position": {
                    "1": "machine_id",
                    "2": "role",
                    "3": "unknown_3",
                    "4": "unknown_4",
                    "5": "unknown_5",
                    "6": "device",
                    "7": "sensor_7",
                    "8": "sensor_8",
                    "9": "sensor_9",
                    "10": "sensor_10",
                    "11": "sensor_11",
                    "12": "sensor_12",
                    "13": "sensor_13",
                    "14": "sensor_14",
                    "15": "sensor_15",
                    "16": "sensor_16",
                }
            },
        },
        "processed_files": {
            "scene1-algo/cpu_requests.csv": {
                "derived_from": ["pai_instance_table.csv", "pai_task_table.csv"],
                "joins": [{"left": "pai_instance_table.col1(task_id)", "right": "pai_task_table.col1(task_id)"}],
                "field_mapping": {
                    "task_id": "pai_instance_table.col1",
                    "instance_id": "pai_instance_table.col3",
                    "status": "pai_instance_table.col6",
                    "start_time": "pai_instance_table.col7",
                    "end_time": "pai_instance_table.col8",
                    "machine_id": "pai_instance_table.col9",
                    "cpu": "pai_task_table.col7",
                    "mem": "pai_task_table.col8",
                    "gpu": "pai_task_table.col3",
                    "timestamp": "pai_instance_table.col7",
                    "duration": "pai_instance_table.col8 - pai_instance_table.col7",
                },
            },
            "scene3-compute/gpu_requests.csv": {
                "derived_from": ["pai_instance_table.csv", "pai_task_table.csv"],
                "joins": [{"left": "pai_instance_table.col1(task_id)", "right": "pai_task_table.col1(task_id)"}],
                "field_mapping": {
                    "task_id": "pai_instance_table.col1",
                    "instance_id": "pai_instance_table.col3",
                    "status": "pai_instance_table.col6",
                    "start_time": "pai_instance_table.col7",
                    "end_time": "pai_instance_table.col8",
                    "machine_id": "pai_instance_table.col9",
                    "cpu": "pai_task_table.col7",
                    "mem": "pai_task_table.col8",
                    "gpu": "pai_task_table.col3",
                    "timestamp": "pai_instance_table.col7",
                    "duration": "pai_instance_table.col8 - pai_instance_table.col7",
                },
                "filter": "pai_task_table.col3(gpu_requested) > 0",
            },
            "scene2-data/machine_metrics.csv": {
                "derived_from": ["pai_machine_metric.csv"],
                "field_mapping": {
                    "machine_id": "pai_machine_metric.col1",
                    "start_time": "pai_machine_metric.col3",
                    "end_time": "pai_machine_metric.col4",
                    "m5": "pai_machine_metric.col5",
                    "m6": "pai_machine_metric.col6",
                    "m7": "pai_machine_metric.col7",
                    "m8": "pai_machine_metric.col8",
                    "m9": "pai_machine_metric.col9",
                    "m10": "pai_machine_metric.col10",
                    "m11": "pai_machine_metric.col11",
                    "m12": "pai_machine_metric.col12",
                },
            },
            "scene2-data/sensor_events.csv": {
                "derived_from": ["pai_sensor_table.csv"],
                "field_mapping": {
                    "machine_id": "pai_sensor_table.col1",
                    "device": "pai_sensor_table.col6",
                    "s2": "pai_sensor_table.col2",
                    "s3": "pai_sensor_table.col3",
                    "s4": "pai_sensor_table.col4",
                    "s5": "pai_sensor_table.col5",
                    "s7": "pai_sensor_table.col7",
                    "s8": "pai_sensor_table.col8",
                    "s9": "pai_sensor_table.col9",
                    "s10": "pai_sensor_table.col10",
                    "s11": "pai_sensor_table.col11",
                    "s12": "pai_sensor_table.col12",
                    "s13": "pai_sensor_table.col13",
                    "s14": "pai_sensor_table.col14",
                    "s15": "pai_sensor_table.col15",
                    "s16": "pai_sensor_table.col16",
                },
                "note": "sensor 表未观察到显式时间戳列，输出不包含 timestamp",
            },
            "scene4-joint/joint_requests.csv": {
                "derived_from": ["pai_instance_table.csv", "pai_task_table.csv"],
                "joins": [{"left": "pai_instance_table.col1(task_id)", "right": "pai_task_table.col1(task_id)"}],
                "field_mapping": {
                    "task_id": "pai_instance_table.col1",
                    "instance_id": "pai_instance_table.col3",
                    "status": "pai_instance_table.col6",
                    "start_time": "pai_instance_table.col7",
                    "end_time": "pai_instance_table.col8",
                    "machine_id": "pai_instance_table.col9",
                    "cpu": "pai_task_table.col7",
                    "mem": "pai_task_table.col8",
                    "gpu": "pai_task_table.col3",
                    "timestamp": "pai_instance_table.col7",
                    "duration": "pai_instance_table.col8 - pai_instance_table.col7",
                    "service_type": "compute_gpu if pai_task_table.col3 > 0 else compute_cpu",
                },
            },
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="datasets/alibaba-cluster-trace-gpu-v2020/raw")
    ap.add_argument("--out-scene1", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene1-algo/cpu_requests.csv")
    ap.add_argument("--out-scene3", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene3-compute/gpu_requests.csv")
    ap.add_argument("--out-scene2-metrics", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene2-data/machine_metrics.csv")
    ap.add_argument("--out-scene2-sensors", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene2-data/sensor_events.csv")
    ap.add_argument("--out-scene2-data-requests", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene2-data/data_requests.csv")
    ap.add_argument("--out-scene2-datasets", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene2-data/datasets.csv")
    ap.add_argument("--out-scene4", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv")
    ap.add_argument("--out-schema", default="datasets/alibaba-cluster-trace-gpu-v2020/processed/_meta/field_mapping.json")
    ap.add_argument("--scenes", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--data-nodes", default="data-node-1,data-node-2")
    ap.add_argument("--gpu-nodes", default="gpu-node-1,gpu-node-2,gpu-node-3")
    ap.add_argument("--cpu-nodes", default="cpu-node-1,cpu-node-2")
    ap.add_argument("--max-duration-s", type=float, default=7 * 24 * 3600)
    ap.add_argument("--min-cpu-rows", type=int, default=1000000)
    ap.add_argument("--cpu-synth-rate", type=float, default=0.25)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    from scheduler.logging_config import setup_logging

    setup_logging(args.log_level, args.log_file)
    logger.info("main start args=%s", vars(args))
    raw_dir = args.raw_dir
    scenes = set([s.strip() for s in args.scenes.split(",") if s.strip() in {"1", "2", "3", "4"}])
    job_path = detect_file(raw_dir, "pai_job_table")
    task_path = detect_file(raw_dir, "pai_task_table")
    job_m = load_job_table(job_path)
    task_m = load_task_table(task_path)
    write_field_mapping(args.out_schema, raw_dir)
    data_nodes = [x.strip() for x in (args.data_nodes or "").split(",") if x.strip()]
    gpu_nodes = [x.strip() for x in (args.gpu_nodes or "").split(",") if x.strip()]
    cpu_nodes = [x.strip() for x in (args.cpu_nodes or "").split(",") if x.strip()]
    if scenes.intersection({"1", "2", "3", "4"}):
        extract_instances(
            raw_dir,
            args.out_scene1,
            args.out_scene3,
            args.out_scene4,
            args.out_scene2_data_requests,
            args.out_scene2_datasets,
            job_m,
            task_m,
            args.seed,
            data_nodes,
            gpu_nodes,
            cpu_nodes,
            args.max_duration_s,
            args.min_cpu_rows,
            args.cpu_synth_rate,
        )
    if "2" in scenes:
        extract_machine_metrics(raw_dir, args.out_scene2_metrics, args.out_scene2_sensors)


if __name__ == "__main__":
    sys.exit(main())
