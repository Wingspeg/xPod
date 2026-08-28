# xpod-research

**xPod: A Three-Element Joint Scheduling System for Kubernetes** (IEEE TNSM submission)

xPod is the first Kubernetes scheduler to formalize **joint scheduling of compute, data, and algorithm** as a unified optimization problem. This repository contains the scheduling algorithm implementation, the trace-replay simulator, the full v7 experiment code, and the paper analysis scripts.

The underlying Kubernetes infrastructure (KubeVirt VMs, Volcano, Multus, etc.) is described in the parent [../README.md](../README.md).

## Research Highlights (v7)

On the Alibaba GPU cluster trace (v2020, 1.74M tasks, 68.6 days) under the v7 cache-stripped + real-queue setting, replaying the full 1.74M-task trace against five baselines:

| Metric | xPod | best baseline (k8s_default / tetris) | worst baseline (firstfit_hash) |
|--------|------|--------------------------------------|--------------------------------|
| Mean JCT | 439.1 M sim s | 1,764.9 M | 2,646.2 M |
| p50 JCT | 316.8 M sim s | 1,277.5 M | 2,258.6 M |
| p99 JCT | 1,403.0 M sim s | 5,630.9 M | 7,537.0 M |
| cold-start % | 100% (cache-stripped) | 100% | 100% |
| data xfer (TB) | 203,940 | 203,940 | 203,940 |
| top-1 load share | 14.4% (8 nodes) | 57.4% (2 nodes) | 57.4% (2 nodes) |

**Headline speedups**:
- **4.02× mean** / **4.03× p50** JCT over k8s_default / tetris (best baseline)
- **6.03× mean** over firstfit_hash (worst baseline)
- **8.10× p50** at 32-node cluster scale (vs 4.03× at 16-node — the advantage grows with scale)
- **4.0×** stable across all 7 JCT deciles (p10–p99, 0.7% spread)

**Headline ablation (V.F.5b)**:
- 1-dim xpod (compute only) = k8s_default 1.00× bit-for-bit
- 2-dim xpod (compute + data) = k8s_default 1.00× bit-for-bit
- 3-dim xpod (compute + data + algorithm) = 4.02× advantage

→ **The 3-dim argmin is the sole structural source of the load-distribution advantage.** 1-dim and 2-dim both collapse to first-fit. The 5-weight ablation (α / β / γ / r / λ) and the 3-flag ablation (hash_tiebreak / las_priority / contention_aware) all tie at 1.00×.

**Live K8s validation (Section V.C)**: a 995-task live Volcano + Fluid deployment on a 16-node 4-rack cluster, 6 modes with 100% admission; xpod / firstfit ratio 0.25× matches the 4.0× simulation advantage.

## Repository Structure

```
xpod-research/
├── README.md                              # This file
├── scheduler/                             # ⭐ xPod scheduler core (v7, ~900 lines)
│   ├── xpod_scheduler.py                  # 6 schedule_one_* modes (xpod + 5 baselines);
│   │                                      # 1/2/3-dim argmin; 5-weight / 3-flag ablation
│   ├── logging_config.py
│   ├── controller/
│   │   ├── generate_manifests.py
│   │   └── xpodgen/
│   │       ├── scheduler.py               # ClusterConfig / NodeLoadTracker
│   │       │                              # / ResourceCacheState / ServiceTracker
│   │       ├── cli.py
│   │       ├── dag.py
│   │       ├── io.py
│   │       ├── specs.py                   # K8s resource specs (with _k8s_safe_name)
│   │       └── yamlutil.py
│   ├── crd/
│   └── router/
├── experiments/                           # ⭐ Experiment code (v7)
│   ├── replay/
│   │   └── replay_workload.py             # Main entry: --mode {xpod, pure_random, firstfit_hash,
│   │                                      # k8s_default, decoupled_cd, tetris}
│   │                                      # --xpod-dim {1,2,3} --speedup 3600
│   │                                      # --start-ts / --end-ts (PAI dense window)
│   ├── datagen/
│   │   └── parse_alibaba_gpu_v2020.py     # 1.74M-task PAI trace preprocessing (1138 lines)
│   ├── analysis/                          # Paper figures + numeric analysis
│   ├── baseline/                          # 5 baseline implementations
│   ├── collect/
│   └── plot/
├── docs/
│   ├── paper/                             # Paper-related documents
│   │   └── 调度研究思路整理.md
│   └── worklog/                           # Daily work logs (2026-03-01 ~ 2026-08-27)
└── results/                               # Experiment artifacts (raw data not tracked)
    ├── raw/                               # Scheduling decision CSVs (.gitignore; 30+ csv, 1.74M rows)
    └── figures/                           # Paper figures (tracked)
```

## Core Concepts

### Three-Element Scheduling

xPod models every scheduling decision as a triple node selection: `(compute_node, data_node, algo_node)`. The scheduler performs joint argmin over the cartesian product:

```
S = α · t_total + β · Φ_loc − γ · p_2dlas
    − w_cache · (B_ds + B_img) + r · R + λ · ℓ(c)

where:
  t_total  = t_compute · φ(c, m_k) + data_transfer_ms / 1000 + algo_transfer_ms / 1000
  Φ_loc    = network cost (same prefix / same rack preferred)
  p_2dlas  = Tiresias 2D-LAS priority
  B_ds / B_img = cache bonus (set to 0 in v7 cache-stripped; 0/1/2× in ablation)
  R        = rack-affinity bonus
  ℓ(c)     = look-ahead contention penalty
```

### Three Key Mechanisms

| Component | Role | v7 default |
|-----------|------|------------|
| 3-dim argmin | argmin S over the (c, d, a) cartesian product | enabled |
| 2D-LAS Priority | Queue fairness; candidate-independent offset | γ = 0.8 |
| Contention-aware t_compute | t_compute scaled by the task node's used load | enabled |

### v7 Cache-Stripped Setting

All 6 modes set `dataset_cached = False, image_cached = False`, providing a fair routing-only comparison and a clean ablation of the 3-dim argmin contribution. The cache-augmented setting is the v6 production version; v7 is the ablation.

## Reproducing the Experiments (v7)

### 1. Install dependencies

```bash
cd xpod-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Download and preprocess the Alibaba trace

```bash
# Original trace source
# https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

python3 -m experiments.datagen.parse_alibaba_gpu_v2020 \
    --input <downloaded-trace-dir> \
    --output datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/
```

Output: `xpod_requests.csv` (~250 MB, 1,741,164 rows).

### 3. Main experiment (v7 cache-stripped, 6 modes × 3 caps)

```bash
# 16-worker 4-rack topology
NODES='--gpu-nodes gpu-vnode-1,gpu-vnode-2,gpu-vnode-3,gpu-vnode-4 \
       --cpu-nodes cpu-node-1,cpu-node-2,cpu-node-3,cpu-node-4 \
       --data-nodes data-node-1,data-node-2,data-node-3,data-node-4 \
       --algo-nodes algo-node-1,algo-node-2,algo-node-3,algo-node-4'
CAP='--node-capacity gpu-vnode-1:8,gpu-vnode-2:8,gpu-vnode-3:8,gpu-vnode-4:8,cpu-node-1:200,cpu-node-2:200,cpu-node-3:200,cpu-node-4:200,data-node-1:50,data-node-2:50,data-node-3:50,data-node-4:50,algo-node-1:50,algo-node-2:50,algo-node-3:50,algo-node-4:50'

# Run each of the 6 modes once
for mode in xpod pure_random firstfit_hash k8s_default decoupled_cd tetris; do
    python3 -m experiments.replay.replay_workload \
        --input datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv \
        --mode $mode --speedup 3600 --dry-run \
        $NODES $CAP \
        --replay-log-file results/raw/v7_${mode}/replay_log.csv
done
```

### 4. 1/2/3-dim argmin ablation (paper headline)

```bash
# 1-dim / 2-dim / 3-dim each, compared against k8s / tetris
for dim in 1 2 3; do
    python3 -m experiments.replay.replay_workload \
        --input .../xpod_requests.csv \
        --mode xpod --xpod-dim $dim --speedup 3600 --dry-run \
        $NODES $CAP \
        --replay-log-file results/raw/v7_dim${dim}_xpod/replay_log.csv
done
```

### 5. 5-weight + 3-flag ablation

Controlled via `--xpod-alpha`, `--xpod-beta`, `--xpod-gamma`, `--rack-bonus`, `--look-ahead-penalty`.
See `docs/worklog/2026-08-27.md` for the full sweep details.

### 6. Cap sweep (V.B.1) + cluster scale (V.B.2)

- Cap sweep: `--node-capacity gpu-vnode-1:8,...:200` across three capacity levels
- 32n scale: extend the 16n setup with 8 new `gpu-vnode-5..8` nodes

### 7. Real K8s deployment (V.C, 995-task live)

```bash
# 1h dense window: PAI start_ts=5798889, end_ts=5799040 (995 tasks)
# 5 modes × speedup 10, sequential, ~25 min each, ~2 h total
python3 -m experiments.replay.replay_workload \
    --input .../xpod_requests.csv \
    --mode xpod --apply-batch-size 50 --speedup 10 \
    --start-ts 5798889 --end-ts 5799040 \
    --namespace xpod-test \
    $NODES $CAP
```

### 8. Generate paper figures

```bash
# 3 v7 perf figures (matplotlib) → figs/fig_*.pdf + .svg
python3 /tmp/gen_v7_figs.py
```

The .svg files are editable in vector editors such as Sketch / Inkscape; after editing, export as .pdf and replace the corresponding `figs/fig_*.pdf` for the paper compile.

### 9. Compile the paper

```bash
cd ../IEEE_TNSM_xPod__A_Collaborative_Scheduling_Method_of_Compute__Data__and_Algorithm_Resources_for_Computing_Power_Networks
latexmk -f -interaction=nonstopmode -pdf xpod_paper_en.tex
# 16-page PDF, 0 errors
```

## Experimental Data Not Tracked

The complete v7 experiment decision CSVs (~10 GB+, 30+ csv) and KubeVirt VM disk images (~120 GB) are **not tracked by git**; they are excluded via `.gitignore`:

```
results/raw/      # Scheduling decision CSVs (v7_*/replay_log.csv — reproducible from the commands above)
datasets/         # Raw PAI trace
*.raw / *.qcow2   # VM disk images
```

Only `results/figures/` (paper figures, PNG/PDF, < 10 MB) is tracked.

## Experiment Data Archive

For direct access to the experiment data of this paper (avoiding the 8+ h local rerun), an archive package will be released with a DOI on Zenodo / Alibaba Cloud OSS. The complete v7 vtable is available in `v7_paper_full_vtable.md` (all numbers for the 12 paper tables + 5 key findings + caveats).

## Related Papers

- v7 paper: "xPod: A Three-Element Joint Scheduling Method for Compute, Data, and Algorithm Resources in Computing Power Networks" (IEEE TNSM submission, 2026-08)
- v6 paper (superseded): the 32-39× numbers in v6's cache-augmented setting were distorted by four cache-aware baselines misusing `cache.query()`; replaced by v7

## License

[MIT](../LICENSE) © 2026 Wingspeg.
