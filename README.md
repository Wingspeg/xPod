# k8s-workspace

Kubernetes infrastructure layer for the xPod research project. This repository provides the cluster deployment, KubeVirt VMs, Volcano scheduler, GPU device plugin, and other operational configuration that underlies the [xpod-research](./xpod-research/) research code.

For the research background and algorithm implementation, see [xpod-research/README.md](./xpod-research/README.md).

## Repository Structure

```
k8s-workspace/
├── README.md                      # This file
├── xpod-research/                 # ⭐ xPod research core (standalone Python project; see its README)
├── manifests/                     # Kubernetes resource manifests
│   ├── base/                      # Base resources
│   │   ├── kubevirt/              # KubeVirt VM definitions
│   │   │   ├── multus.yaml             # Multus CNI networking
│   │   │   ├── bridge-conf.yaml        # Linux Bridge configuration
│   │   │   ├── gpu-node-1.yaml         # GPU node VM (PCI passthrough, A10)
│   │   │   ├── cpu-node-{1,2}.yaml     # CPU node VMs
│   │   │   ├── data-node-{1,2}.yaml    # Data node VMs
│   │   │   └── algo-node-{1,2,3}.yaml  # Algorithm node VMs (horizontally scalable)
│   │   ├── deployments/           # Application deployments
│   │   ├── services/              # Service definitions
│   │   ├── configmaps/            # ConfigMaps
│   │   ├── secrets/               # Secrets (no real credentials in this repo)
│   │   ├── namespaces/            # Namespaces
│   │   ├── pv-pvc/                # Persistent volumes
│   │   └── ingress/               # Ingress rules
│   ├── overlays/                  # Kustomize overlays (dev/prod differences)
│   └── volcano/                   # Volcano scheduler install manifests
├── clusters/                      # Cluster configuration
│   ├── dev/                       # Dev cluster config
│   ├── prod/                      # Production cluster config (template)
│   └── backup/                    # Cluster backup config
├── scripts/                       # Operational scripts
│   └── vm/                        # VM management scripts
│       ├── create-vms.sh                # Bulk VM creation
│       ├── setup-gpu-passthrough.sh     # GPU PCI passthrough setup
│       └── debs/                        # Offline deb packages (kubeadm/kubelet/cri-tools)
├── k8s-device-plugin/             # NVIDIA device plugin (vendored build source)
├── docs/                          # Operational documentation (deployment, troubleshooting)
├── certs/                         # Cluster certificates (no real certs in this repo)
├── volcano-crds.yaml              # Volcano CRDs
├── volcano-development.yaml       # Volcano deployment manifests
└── test-volcano-job.yaml          # Volcano smoke-test Job
```

## Hardware and Software Environment

### Host

| Item | Configuration |
|------|---------------|
| Motherboard | JGINYUE X99-8D4/2.5G |
| CPU | 2× Intel Xeon E5-2696 v3 (72 vCPU) |
| Memory | 251 GiB DDR4 |
| Storage | 1.8 TB SSD |
| GPU | 1× NVIDIA A10 (24 GB), PCI passthrough to gpu-node-1 |
| OS | Ubuntu 24.04.3 LTS, Kernel 6.8.0-110-generic |

### Software Stack

- **Kubernetes**: v1.31.14 (deployed via kubeadm)
- **Container runtime**: containerd
- **CNI**: Multus + Linux Bridge (`br0`)
- **Virtualization**: KubeVirt (8 VMs; see topology below)
- **Batch scheduling**: Volcano v1.9.0
- **GPU sharing**: HAMi-style vGPU (simulated at the scheduling-abstract layer; see xpod-research/README.md)

### Cluster Topology

| Role | VM Name | IP | Resources |
|------|---------|----|-----------|
| Master (physical) | leosuek8s | — | Host |
| GPU node | gpu-node-1 | 192.168.1.121 | 8c / 32G + A10 GPU |
| CPU node | cpu-node-1, cpu-node-2 | .131, .132 | 4c / 8G |
| Data node | data-node-1, data-node-2 | .141, .142 | 4c / 8G |
| Algorithm node | algo-node-1, algo-node-2, algo-node-3 | .151, .152, .153 | 4c / 8G |

## Deployment Steps

```bash
# 1. Host preparation: KVM + KubeVirt + Multus + Volcano (see docs/ for details)

# 2. Start all VMs
kubectl apply -f manifests/base/kubevirt/multus.yaml
kubectl apply -f manifests/base/kubevirt/bridge-conf.yaml
kubectl apply -f manifests/base/kubevirt/

# 3. Verify VM readiness
kubectl get vmi -A

# 4. Install Volcano
kubectl apply -f volcano-crds.yaml
kubectl apply -f volcano-development.yaml

# 5. Smoke test
kubectl apply -f test-volcano-job.yaml
```

## Relationship with xpod-research

| Dimension | k8s-workspace | xpod-research |
|-----------|---------------|---------------|
| Role | Infrastructure layer | Research core |
| Content | Cluster deployment, VM definitions, scheduler install | xPod scheduling algorithm, experiment code, paper data |
| Deployment target | Real physical cluster | Dry-run simulator (paper) + future real deployment |

See [xpod-research/README.md](./xpod-research/README.md) for details.

## Data and Images Not Tracked

`images/` (KubeVirt VM disks, ~120 GB), `backups/`, `certs/`, and similar artifacts are excluded from the repository via `.gitignore`. VM disk images are generated locally by the scripts in `scripts/vm/`.

## License

[MIT](./LICENSE) © 2026 Wingspeg.
