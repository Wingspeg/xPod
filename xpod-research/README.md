# xpod-research

**xPod: 三元素联合调度的 Kubernetes 调度系统**(博士论文研究主体)

xPod 是首个将 **compute / data / algorithm** 三元素联合调度形式化为统一优化问题的 Kubernetes 调度器。本仓库包含调度算法实现、dry-run 重放模拟器、完整 trace 实验代码与论文分析脚本。

底层 Kubernetes 基础设施(KubeVirt VM、Volcano、Multus 等)详见上级目录 [../README.md](../README.md)。

## 研究亮点

在 Alibaba GPU 集群 trace(v2020,1.74M 任务,68.6 天)上,xPod 相比 Random Baseline(业界独立组件实践的代理):

| 指标 | xPod | Baseline | 改进 |
|------|------|----------|------|
| 数据传输量 | 760 GB | 1520 GB | **2.00× ↓** |
| 镜像传输量 | 14.75 GB | 43.75 GB | **2.97× ↓** |
| 冷启动收敛索引 | idx=5 | idx=64 | **12.8× faster** |
| GPU 节点间均衡度 | std=0.138 | std=0.866 | **6.27× better** |
| 累计 JCT | 18.8M h | 19.5M h | **3.57% ↓** |
| p99 JCT | 740,416 s | 772,929 s | **4.21% ↓** |

Sensitivity 实验进一步证明:
- **vGPU 数 2→8 时,JCT 改进从 1.36% 单调上升至 7.52%**(scalability)
- **data/algo 节点数 N 增加时,xPod 保持 O(1) 存储足迹,Baseline 呈 O(N) 线性增长**(zero-replication)

## 仓库结构

```
xpod-research/
├── README.md                              # 本文件
├── scheduler/                             # ⭐ xPod 调度器核心
│   ├── xpod_scheduler.py                  # schedule_one_xpod / schedule_one_baseline
│   ├── logging_config.py
│   ├── controller/
│   │   ├── generate_manifests.py          # K8s Manifest 生成
│   │   ├── simulate_pipeline.py
│   │   └── xpodgen/
│   │       ├── scheduler.py               # ClusterConfig / ResourceCacheState
│   │       │                              # / ServiceTracker / NodeLoadTracker
│   │       │                              # / ScheduleDecision
│   │       ├── cli.py                     # 调度器 CLI 入口
│   │       ├── dag.py                     # xPod 任务 DAG
│   │       ├── io.py                      # I/O 辅助
│   │       ├── specs.py                   # K8s 资源规格生成
│   │       └── yamlutil.py                # YAML 渲染
│   ├── crd/                               # xPod CRD 定义
│   └── router/                            # 路由层
├── experiments/                           # ⭐ 实验代码
│   ├── replay/
│   │   └── replay_workload.py             # 主入口:回放 trace 生成调度决策
│   ├── datagen/
│   │   └── parse_alibaba_gpu_v2020.py     # Alibaba trace 预处理
│   ├── analysis/
│   │   ├── make_paper_figures.py          # 6 张论文图生成
│   │   └── README.md
│   ├── baseline/                          # Random Baseline 实现
│   ├── collect/
│   │   ├── collect_metrics.py             # 指标收集
│   │   └── summarize.py
│   └── plot/
│       └── plot_results.py
├── docs/
│   ├── paper/                             # 论文相关文档
│   │   └── 调度研究思路整理.md
│   └── worklog/                           # 每日工作日志
└── results/                               # 实验产物(数据不入库,见下方)
    ├── raw/                               # 调度决策 CSV(.gitignore)
    └── figures/                           # 论文图(入库)
```

## 核心概念

### 三元素调度

xPod 把每个调度决策建模为三元节点选择:`(compute_node, data_node, algo_node)`。调度器在三元笛卡尔积上做联合优化,而非由独立组件分别处理。

### 打分公式

```
score = α · t_total + β · data_locality − γ · p_2dlas

其中:
  t_total = t_compute(duration, dominant_amount, contention) 
            + data_transfer_ms / 1000 
            + algo_transfer_ms / 1000
  data_locality = network 代价(同前缀优先)
  p_2dlas = Tiresias 2D-LAS 优先级,1 / max(1, attained_service)
```

### 三个关键机制

| 组件 | 作用 | 影响 |
|------|------|------|
| Hash tie-break | score 平局时按 md5(key\|node) 排序 | N 节点间均衡分布 |
| 2D-LAS Priority | Tiresias 风格的服务公平性 | 排队语义占位(论文中讨论参数空间) |
| Contention-aware t_compute | 任务节点上同时占用的负载放大 t_compute | 模拟真实集群资源争用 |

## 复现实验

### 1. 安装依赖

```bash
cd xpod-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载并预处理 Alibaba trace

```bash
# 原始 trace 来源(本仓库不附带,数据集 ~几 GB)
# https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

python3 -m experiments.datagen.parse_alibaba_gpu_v2020 \
    --input <下载的 trace 目录> \
    --output datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/
```

### 3. 主实验(xPod vs Baseline)

```bash
# xPod
python3 -m experiments.replay.replay_workload \
  --input datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv \
  --mode xpod --speedup 3600 --dry-run \
  --data-nodes "data-node-1,data-node-2" \
  --gpu-nodes "gpu-vnode-1,gpu-vnode-2,gpu-vnode-3,gpu-vnode-4" \
  --cpu-nodes "cpu-node-1,cpu-node-2" \
  --algo-nodes "algo-node-1,algo-node-2,algo-node-3" \
  --node-capacity "gpu-vnode-1:50,gpu-vnode-2:50,gpu-vnode-3:50,gpu-vnode-4:50,cpu-node-1:200,cpu-node-2:200,data-node-1:50,data-node-2:50,algo-node-1:50,algo-node-2:50,algo-node-3:50" \
  --replay-log-file results/raw/xpod_full.csv

# Baseline:同命令,--mode baseline,改 --replay-log-file
```

完整 trace ~30 分钟。

### 4. Sensitivity 与 Ablation 实验

提供批量脚本 `run_overnight.sh`(项目根目录),约 13 小时跑完完整 sensitivity(A/B/C/D 四维度 9 配置)+ ablation(3 组件 + baseline,共 5 配置)。

### 5. 生成论文图

```bash
python3 -m experiments.analysis.make_paper_figures
# 输出到 results/figures/(PNG + PDF)
```

## 实验数据不入库

完整实验决策 CSV(~10GB+)与 KubeVirt VM 磁盘镜像(~120GB)**不入 git**,在 `.gitignore` 中排除:

```
results/raw/      # 实验决策 CSV(可通过上面命令复现)
datasets/         # 原始 trace(从 Aliyun 公开数据集下载)
*.raw / *.qcow2   # VM 磁盘镜像
```

仅 `results/figures/`(论文配图,PNG/PDF,< 10MB)入库。

## 实验数据档案

若需直接获取本论文的实验数据(避免本地重跑 10 小时),后续将通过 Zenodo / 阿里云 OSS 发布 archive 包并提供 DOI。

## 相关论文

[待补充论文标题、作者、会议、DOI]

## 许可

待定(建议 Apache 2.0)。
