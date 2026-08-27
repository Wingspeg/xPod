# xpod-research

**xPod: 三元素联合调度的 Kubernetes 调度系统**(IEEE TNSM 投稿主体)

xPod 是首个将 **compute / data / algorithm** 三元素联合调度形式化为统一优化问题的 Kubernetes 调度器。本仓库包含调度算法实现、trace 重放模拟器、完整 v7 实验代码与论文分析脚本。

底层 Kubernetes 基础设施(KubeVirt VM、Volcano、Multus 等)详见上级目录 [../README.md](../README.md)。

## 研究亮点 (v7)

在 Alibaba GPU 集群 trace(v2020, 1.74M 任务, 68.6 天)上,v7 cache-stripped + real-queue 设置下,xPod 相比 5 个 baseline 的 1.74M-task 完整 trace 重放:

| 指标 | xPod | best baseline (k8s_default / tetris) | worst baseline (firstfit_hash) |
|------|------|---------|---------|
| Mean JCT | 439.1 M sim s | 1,764.9 M | 2,646.2 M |
| p50 JCT | 316.8 M sim s | 1,277.5 M | 2,258.6 M |
| p99 JCT | 1,403.0 M sim s | 5,630.9 M | 7,537.0 M |
| cold-start % | 100% (cache-stripped) | 100% | 100% |
| data xfer (TB) | 203,940 | 203,940 | 203,940 |
| top-1 load share | 14.4% (8 nodes) | 57.4% (2 nodes) | 57.4% (2 nodes) |

**Headline speedups**:
- **4.02× mean** / **4.03× p50** JCT over k8s\_default / tetris (best baseline)
- **6.03× mean** over firstfit\_hash (worst baseline)
- **8.10× p50** at 32-node cluster scale (vs 4.03× at 16-node, 优势随 scale 增大)
- **4.0×** stable across all 7 JCT deciles (p10-p99, 0.7% spread)

**Headline ablation (V.F.5b)**:
- 1-dim xpod (compute only) = k8s\_default 1.00× bit-for-bit
- 2-dim xpod (compute + data) = k8s\_default 1.00× bit-for-bit
- 3-dim xpod (compute + data + algorithm) = 4.02× advantage

→ **3-dim argmin 是 load-distribution 优势的唯一 structural source**. 1-dim/2-dim 都 collapse to first-fit. 5 weight ablation (α/β/γ/r/λ) + 3 flag ablation (hash\_tiebreak / las\_priority / contention\_aware) 全部 1.00× tiebreak.

**Live K8s validation (Section V.C)**: 995-task live Volcano + Fluid deployment on 16-node 4-rack, 6 mode 100% admission, xpod / firstfit 0.25× ratio matches 4.0× sim advantage.

## 仓库结构

```
xpod-research/
├── README.md                              # 本文件
├── scheduler/                             # ⭐ xPod 调度器核心 (v7, 900 lines)
│   ├── xpod_scheduler.py                  # 6 mode schedule_one_* (xpod + 5 baseline)
│   │                                      # 1/2/3-dim argmin, 5-weight/3-flag ablation
│   ├── logging_config.py
│   ├── controller/
│   │   ├── generate_manifests.py
│   │   └── xpodgen/
│   │       ├── scheduler.py               # ClusterConfig / NodeLoadTracker
│   │       │                              # / ResourceCacheState / ServiceTracker
│   │       ├── cli.py
│   │       ├── dag.py
│   │       ├── io.py
│   │       ├── specs.py                   # K8s 资源规格 (with _k8s_safe_name)
│   │       └── yamlutil.py
│   ├── crd/
│   └── router/
├── experiments/                           # ⭐ 实验代码 (v7)
│   ├── replay/
│   │   └── replay_workload.py             # 主入口: --mode {xpod, pure_random, firstfit_hash,
│   │                                      # k8s_default, decoupled_cd, tetris}
│   │                                      # --xpod-dim {1,2,3} --speedup 3600
│   │                                      # --start-ts / --end-ts (PAI dense window)
│   ├── datagen/
│   │   └── parse_alibaba_gpu_v2020.py     # 1.74M 任务 PAI trace 预处理 (1138 lines)
│   ├── analysis/                          # 论文图 + 数字分析
│   ├── baseline/                          # 5 个 baseline 实现
│   ├── collect/
│   └── plot/
├── docs/
│   ├── paper/                             # 论文相关文档
│   │   └── 调度研究思路整理.md
│   └── worklog/                           # 每日工作日志 (2026-03-01 ~ 2026-08-27)
└── results/                               # 实验产物(数据不入库)
    ├── raw/                               # 调度决策 CSV(.gitignore, 30+ csv, 1.74M rows)
    └── figures/                           # 论文图(可入库)
```

## 核心概念

### 三元素调度
xPod 把每个调度决策建模为三元节点选择:`(compute_node, data_node, algo_node)`。调度器在三元笛卡尔积上做联合 argmin 优化:

```
S = α · t_total + β · Φ_loc − γ · p_2dlas
  − w_cache · (B_ds + B_img) + r · R + λ · ℓ(c)

其中:
  t_total  = t_compute · φ(c, m_k) + data_transfer_ms / 1000 + algo_transfer_ms / 1000
  Φ_loc    = network 代价(同 prefix / 同 rack 优先)
  p_2dlas  = Tiresias 2D-LAS 优先级
  B_ds/B_img = cache-bonus (v7 cache-stripped 设为 0, ablation 中 0/1/2 倍)
  R        = rack-affinity bonus
  ℓ(c)     = look-ahead contention penalty
```

### 三个关键机制

| 组件 | 作用 | v7 默认值 |
|------|------|------|
| 3-dim argmin | 在 (c, d, a) 笛卡尔积上 argmin S | enable |
| 2D-LAS Priority | 排队公平性,候选独立 offset | γ=0.8 |
| Contention-aware t_compute | 任务节点已用 load 放大 t_compute | enable |

### v7 cache-stripped setting
所有 6 mode 都 `dataset_cached = False, image_cached = False`,纯 routing 公平比较,clean ablation of 3-dim argmin 贡献。Cache-augmented setting 是 v6 的 production,这里 v7 是 ablation。

## 复现实验 (v7)

### 1. 安装依赖
```bash
cd xpod-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载并预处理 Alibaba trace
```bash
# 原始 trace 来源
# https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

python3 -m experiments.datagen.parse_alibaba_gpu_v2020 \
    --input <下载的 trace 目录> \
    --output datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/
```
输出 `xpod_requests.csv` (~250MB, 1,741,164 行)。

### 3. 主实验 (v7 cache-stripped, 6 mode × 3 cap)
```bash
# 16 worker 4-rack 拓扑
NODES='--gpu-nodes gpu-vnode-1,gpu-vnode-2,gpu-vnode-3,gpu-vnode-4 \
       --cpu-nodes cpu-node-1,cpu-node-2,cpu-node-3,cpu-node-4 \
       --data-nodes data-node-1,data-node-2,data-node-3,data-node-4 \
       --algo-nodes algo-node-1,algo-node-2,algo-node-3,algo-node-4'
CAP='--node-capacity gpu-vnode-1:8,gpu-vnode-2:8,gpu-vnode-3:8,gpu-vnode-4:8,cpu-node-1:200,cpu-node-2:200,cpu-node-3:200,cpu-node-4:200,data-node-1:50,data-node-2:50,data-node-3:50,data-node-4:50,algo-node-1:50,algo-node-2:50,algo-node-3:50,algo-node-4:50'

# 6 mode 各跑一次
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
# 1-dim / 2-dim / 3-dim 各自跑, 跟 k8s/tetris 对比
for dim in 1 2 3; do
    python3 -m experiments.replay.replay_workload \
        --input .../xpod_requests.csv \
        --mode xpod --xpod-dim $dim --speedup 3600 --dry-run \
        $NODES $CAP \
        --replay-log-file results/raw/v7_dim${dim}_xpod/replay_log.csv
done
```

### 5. 5 weight + 3 flag ablation
通过 `--xpod-alpha`, `--xpod-beta`, `--xpod-gamma`, `--rack-bonus`, `--look-ahead-penalty` 参数控制。
详细 sweep 见 `docs/worklog/2026-08-27.md`。

### 6. Cap sweep (V.B.1) + Cluster scale (V.B.2)
- Cap sweep: `--node-capacity gpu-vnode-1:8,...:200` 三档
- 32n scale: 在 16n 基础上加 `gpu-vnode-5..8` 8 个新节点

### 7. Real K8s deployment (V.C, 995 task live)
```bash
# 1h dense window: PAI start_ts=5798889, end_ts=5799040 (995 task)
# 5 mode × speedup 10 顺序跑,每次 ~25 min,total ~2h
python3 -m experiments.replay.replay_workload \
    --input .../xpod_requests.csv \
    --mode xpod --apply-batch-size 50 --speedup 10 \
    --start-ts 5798889 --end-ts 5799040 \
    --namespace xpod-test \
    $NODES $CAP
```

### 8. 生成论文图
```bash
# 3 张 v7 perf fig (matplotlib) → figs/fig_*.pdf + .svg
python3 /tmp/gen_v7_figs.py
```
在 Sketch / Inkscape 等矢量编辑器中可编辑 .svg,改完 export .pdf 替换 figs/fig_*.pdf 走 paper compile。

### 9. 编译 paper
```bash
cd ../IEEE_TNSM_xPod__A_Collaborative_Scheduling_Method_of_Compute__Data__and_Algorithm_Resources_for_Computing_Power_Networks
latexmk -f -interaction=nonstopmode -pdf xpod_paper_en.tex
# 16 页 PDF, 0 errors
```

## 实验数据不入库
完整 v7 实验决策 CSV (~10GB+, 30+ csv) 与 KubeVirt VM 磁盘镜像 (~120GB) **不入 git**,在 `.gitignore` 中排除:
```
results/raw/      # 调度决策 CSV (v7_*/replay_log.csv, 可通过上面命令复现)
datasets/         # 原始 PAI trace
*.raw / *.qcow2   # VM 磁盘镜像
```

仅 `results/figures/` (论文配图, PNG/PDF, < 10MB) 可入库。

## 实验数据档案
若需直接获取本论文的实验数据(避免本地重跑 8+ 小时), 后续将通过 Zenodo / 阿里云 OSS 发布 archive 包并提供 DOI。 v7 完整 vtable 见 `v7_paper_full_vtable.md` (12 paper table 数字齐 + 5 key findings + caveats)。

## 相关论文
- v7 paper: "xPod: A Three-Element Joint Scheduling Method for Compute, Data, and Algorithm Resources in Computing Power Networks" (IEEE TNSM 投稿, 2026-08)
- v6 paper (superseded): v6 cache-augmented 32-39x 数字因 4 个 cache-aware baseline 误用 cache.query() 而失真,被 v7 替换

## 许可
[MIT](../LICENSE) © 2026 Wingspeg。
