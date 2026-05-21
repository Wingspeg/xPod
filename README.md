# k8s-workspace

xPod 研究项目的 Kubernetes 基础设施层。本仓库提供研究所需的集群部署、KubeVirt 虚拟机、Volcano 调度器、GPU 设备插件等运维配置,作为 [xpod-research](./xpod-research/) 研究代码的底层环境支撑。

研究背景与算法实现请参考 [xpod-research/README.md](./xpod-research/README.md)。

## 仓库结构

```
k8s-workspace/
├── README.md                      # 本文件
├── xpod-research/                 # ⭐ xPod 研究主体(独立 Python 项目,见其 README)
├── manifests/                     # Kubernetes 资源清单
│   ├── base/                      # 基础资源
│   │   ├── kubevirt/              # KubeVirt 虚拟机定义
│   │   │   ├── multus.yaml             # Multus CNI 网络
│   │   │   ├── bridge-conf.yaml        # Linux Bridge 配置
│   │   │   ├── gpu-node-1.yaml         # GPU 节点 VM(PCI passthrough A10)
│   │   │   ├── cpu-node-{1,2}.yaml     # CPU 节点 VM
│   │   │   ├── data-node-{1,2}.yaml    # 数据节点 VM
│   │   │   └── algo-node-{1,2,3}.yaml  # 算法节点 VM(支持横向扩展)
│   │   ├── deployments/           # 应用部署
│   │   ├── services/              # Service 定义
│   │   ├── configmaps/            # ConfigMap
│   │   ├── secrets/               # Secret(本仓库不含实际密钥)
│   │   ├── namespaces/            # Namespace
│   │   ├── pv-pvc/                # 存储卷
│   │   └── ingress/               # Ingress 规则
│   ├── overlays/                  # Kustomize overlays(dev/prod 差异)
│   └── volcano/                   # Volcano 调度器安装清单
├── clusters/                      # 集群配置
│   ├── dev/                       # 开发集群配置
│   ├── prod/                      # 生产集群配置(模板)
│   └── backup/                    # 集群备份配置
├── scripts/                       # 运维脚本
│   └── vm/                        # 虚拟机管理脚本
│       ├── create-vms.sh                # 批量创建 VM
│       ├── setup-gpu-passthrough.sh     # GPU PCI 直通配置
│       └── debs/                        # 离线 deb 包(kubeadm/kubelet/cri-tools)
├── k8s-device-plugin/             # NVIDIA Device Plugin(vendored 编译源)
├── docs/                          # 运维文档(部署、故障排查)
├── certs/                         # 集群证书(本仓库不含实际证书)
├── volcano-crds.yaml              # Volcano CRD
├── volcano-development.yaml       # Volcano 部署清单
└── test-volcano-job.yaml          # Volcano 冒烟测试 Job
```

## 软硬件环境

### 宿主机

| 项目 | 配置 |
|------|------|
| 主板 | JGINYUE X99-8D4/2.5G |
| CPU | 2× Intel Xeon E5-2696 v3(72 vCPU) |
| 内存 | 251 GiB DDR4 |
| 存储 | 1.8 TB SSD |
| GPU | 1× NVIDIA A10 (24GB),PCI Passthrough 至 gpu-node-1 |
| OS | Ubuntu 24.04.3 LTS,Kernel 6.8.0-110-generic |

### 软件栈

- **Kubernetes**:v1.31.14(kubeadm 部署)
- **容器运行时**:containerd
- **CNI**:Multus + Linux Bridge(`br0`)
- **虚拟化**:KubeVirt(8 个 VM,见下表)
- **批调度**:Volcano v1.9.0
- **GPU 切分**:HAMi 风格 vGPU(调度抽象层模拟,详见 xpod-research/README.md)

### 集群拓扑

| 角色 | VM 名 | IP | 资源 |
|------|-------|-----|------|
| Master(物理) | leosuek8s | — | 宿主机 |
| GPU 节点 | gpu-node-1 | 192.168.1.121 | 8c / 32G + A10 GPU |
| CPU 节点 | cpu-node-1, cpu-node-2 | .131, .132 | 4c / 8G |
| 数据节点 | data-node-1, data-node-2 | .141, .142 | 4c / 8G |
| 算法节点 | algo-node-1, algo-node-2, algo-node-3 | .151, .152, .153 | 4c / 8G |

## 部署步骤

```bash
# 1. 宿主机准备:KVM + KubeVirt + Multus + Volcano(参考 docs/ 详细文档)

# 2. 启动所有 VM
kubectl apply -f manifests/base/kubevirt/multus.yaml
kubectl apply -f manifests/base/kubevirt/bridge-conf.yaml
kubectl apply -f manifests/base/kubevirt/

# 3. 检查 VM 就绪
kubectl get vmi -A

# 4. 安装 Volcano
kubectl apply -f volcano-crds.yaml
kubectl apply -f volcano-development.yaml

# 5. 冒烟测试
kubectl apply -f test-volcano-job.yaml
```

## 与 xpod-research 的关系

| 维度 | k8s-workspace | xpod-research |
|------|--------------|---------------|
| 角色 | 基础设施层 | 研究主体 |
| 内容 | 集群部署、VM 定义、调度器安装 | xPod 调度算法、实验代码、论文数据 |
| 部署目标 | 真实物理集群 | dry-run 模拟器(论文)+ 未来真实部署 |

详见 [xpod-research/README.md](./xpod-research/README.md)。

## 数据与镜像不入库

`images/`(KubeVirt VM 磁盘,~120GB)、`backups/`、`certs/` 等不放入仓库,通过 `.gitignore` 排除。VM 磁盘镜像由 `scripts/vm/` 中的脚本本地生成。

## 许可

待定(建议 Apache 2.0 或 MIT)。
