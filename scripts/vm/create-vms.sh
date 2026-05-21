#!/bin/bash
# =============================================================================
# create-vms.sh — xPod K8s 集群 VM 创建脚本
# 技术栈: KVM + cloud-init + Ubuntu 22.04 cloud image
#
# VM 规划:
#   VM1  192.168.100.21  gpu-node-1    8核 32G 100G  PCIe Passthrough A10
#   VM2  192.168.100.31  cpu-node-1    4核  8G  40G
#   VM3  192.168.100.32  cpu-node-2    4核  8G  40G
#   VM4  192.168.100.41  data-node-1   4核  8G  40G
#   VM5  192.168.100.42  data-node-2   4核  8G  40G
#   VM6  192.168.100.51  algo-node-1   4核  8G  40G
# =============================================================================
set -euo pipefail

# =============================================================================
# 全局配置
# =============================================================================
CLOUD_IMAGE_URL="https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img"
BASE_IMAGE_NAME="ubuntu-22.04-base.img"

IMAGE_DIR="/var/lib/libvirt/images"
CLOUD_INIT_DIR="/tmp/cloud-init"

BRIDGE="br0"
GATEWAY="192.168.100.1"
DNS="192.168.100.1"
NETMASK="24"

VM_USER="ubuntu"
VM_PASSWORD="xpod2024"   # 生产环境改用 SSH key only
SSH_PUB_KEY=""

# =============================================================================
# VM 定义: "名称|IP|CPU|内存(MB)|磁盘(GB)|类型"
# =============================================================================
declare -a VMS=(
    "gpu-node-1|192.168.100.21|8|32768|100|gpu"
    "cpu-node-1|192.168.100.31|4|8192|40|normal"
    "cpu-node-2|192.168.100.32|4|8192|40|normal"
    "data-node-1|192.168.100.41|4|8192|40|normal"
    "data-node-2|192.168.100.42|4|8192|40|normal"
    "algo-node-1|192.168.100.51|4|8192|40|normal"
)

# =============================================================================
# 工具函数
# =============================================================================
log()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

check_deps() {
    log "检查依赖..."
    local deps=(virsh virt-install qemu-img cloud-localds wget openssl)
    for dep in "${deps[@]}"; do
        command -v "$dep" &>/dev/null || \
            err "缺少依赖: $dep — 请执行: apt install -y libvirt-clients virtinst qemu-utils cloud-image-utils"
    done
    systemctl is-active --quiet libvirtd || \
        err "libvirtd 未运行，请执行: systemctl start libvirtd"
    log "依赖检查通过 ✓"
}

# =============================================================================
# 网桥检测 & 自动创建
# =============================================================================
check_bridge() {
    log "检查网桥 $BRIDGE ..."
    if ip link show "$BRIDGE" &>/dev/null; then
        log "网桥 $BRIDGE 已存在 ✓"
        return
    fi
    warn "网桥 $BRIDGE 不存在，尝试自动创建..."
    local NIC
    NIC=$(ip route | awk '/default/{print $5; exit}')
    [[ -z "$NIC" ]] && err "无法检测主网卡，请手动创建网桥 $BRIDGE"
    log "在网卡 $NIC 上创建网桥 $BRIDGE ..."
    cat > /etc/netplan/99-xpod-bridge.yaml <<EOF
network:
  version: 2
  ethernets:
    ${NIC}:
      dhcp4: false
  bridges:
    ${BRIDGE}:
      interfaces: [${NIC}]
      dhcp4: true
      parameters:
        stp: false
        forward-delay: 0
EOF
    netplan apply && sleep 3
    ip link show "$BRIDGE" &>/dev/null || err "网桥创建失败，请手动检查 /etc/netplan/"
    log "网桥 $BRIDGE 创建成功 ✓"
}

# =============================================================================
# 下载 base image
# =============================================================================
download_base_image() {
    local base_path="$IMAGE_DIR/$BASE_IMAGE_NAME"
    mkdir -p "$IMAGE_DIR"
    if [[ -f "$base_path" ]]; then
        log "Base image 已存在，跳过下载: $base_path"
        return
    fi
    log "下载 Ubuntu 22.04 cloud image..."
    wget -c "$CLOUD_IMAGE_URL" -O "$base_path" || \
        err "下载失败，请检查网络，或手动下载至: $base_path"
    log "下载完成 ✓"
}

# =============================================================================
# 生成 cloud-init ISO
# =============================================================================
gen_cloud_init() {
    local vm_name="$1"
    local ip="$2"
    local ci_dir="$CLOUD_INIT_DIR/$vm_name"
    mkdir -p "$ci_dir"

    local hashed_pass
    hashed_pass=$(openssl passwd -6 "$VM_PASSWORD")

    # user-data
    cat > "$ci_dir/user-data" <<EOF
#cloud-config
hostname: ${vm_name}
manage_etc_hosts: true

users:
  - name: ${VM_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    passwd: ${hashed_pass}
$(if [[ -n "$SSH_PUB_KEY" ]]; then
echo "    ssh_authorized_keys:"
echo "      - ${SSH_PUB_KEY}"
fi)

write_files:
  - path: /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
    content: "network: {config: disabled}"
  - path: /etc/modules-load.d/k8s.conf
    content: |
      overlay
      br_netfilter
  - path: /etc/sysctl.d/k8s.conf
    content: |
      net.bridge.bridge-nf-call-iptables  = 1
      net.bridge.bridge-nf-call-ip6tables = 1
      net.ipv4.ip_forward                 = 1

package_update: true
packages:
  - qemu-guest-agent
  - curl
  - vim
  - net-tools
  - apt-transport-https
  - ca-certificates

runcmd:
  - systemctl enable --now qemu-guest-agent
  - timedatectl set-timezone Asia/Shanghai
  - swapoff -a
  - sed -i '/\bswap\b/d' /etc/fstab
  - modprobe overlay
  - modprobe br_netfilter
  - sysctl --system

final_message: "VM ${vm_name} cloud-init 完成，耗时 \$UPTIME 秒"
EOF

    # network-config
    cat > "$ci_dir/network-config" <<EOF
version: 2
ethernets:
  enp1s0:
    dhcp4: false
    addresses:
      - ${ip}/${NETMASK}
    gateway4: ${GATEWAY}
    nameservers:
      addresses: [${DNS}, 8.8.8.8]
EOF

    # meta-data
    cat > "$ci_dir/meta-data" <<EOF
instance-id: ${vm_name}
local-hostname: ${vm_name}
EOF

    cloud-localds \
        --network-config="$ci_dir/network-config" \
        "$ci_dir/cloud-init.iso" \
        "$ci_dir/user-data" \
        "$ci_dir/meta-data"

    log "[$vm_name] cloud-init ISO 生成 ✓"
}

# =============================================================================
# 创建 VM 磁盘（基于 base image 的 CoW clone）
# =============================================================================
create_disk() {
    local vm_name="$1"
    local disk_gb="$2"
    local disk_path="$IMAGE_DIR/${vm_name}.qcow2"

    if [[ -f "$disk_path" ]]; then
        warn "[$vm_name] 磁盘已存在，跳过: $disk_path"
        return
    fi
    log "[$vm_name] 创建 ${disk_gb}G 磁盘（CoW clone）..."
    qemu-img create \
        -f qcow2 \
        -b "$IMAGE_DIR/$BASE_IMAGE_NAME" \
        -F qcow2 \
        "$disk_path" \
        "${disk_gb}G"
    log "[$vm_name] 磁盘创建 ✓"
}

# =============================================================================
# virt-install 公共参数
# =============================================================================
_virt_install_common() {
    local vm_name="$1"
    local cpu="$2"
    local mem="$3"
    local extra_args=("${@:4}")

    local disk_path="$IMAGE_DIR/${vm_name}.qcow2"
    local ci_iso="$CLOUD_INIT_DIR/${vm_name}/cloud-init.iso"

    virt-install \
        --name        "$vm_name"               \
        --memory      "$mem"                   \
        --vcpus       "$cpu"                   \
        --cpu         host-passthrough         \
        --disk        path="$disk_path",format=qcow2,bus=virtio \
        --disk        path="$ci_iso",device=cdrom               \
        --network     bridge="$BRIDGE",model=virtio             \
        --os-variant  ubuntu22.04              \
        --graphics    none                     \
        --noautoconsole                        \
        --import                               \
        --boot        hd                       \
        --channel     unix,target_type=virtio,name=org.qemu.guest_agent.0 \
        "${extra_args[@]}"
}

create_normal_vm() {
    local vm_name="$1" ip="$2" cpu="$3" mem="$4" disk_gb="$5"
    log "[$vm_name] 创建普通 VM (${cpu}核 $((mem/1024))G ${disk_gb}G) IP=$ip"
    _virt_install_common "$vm_name" "$cpu" "$mem"
    log "[$vm_name] VM 创建成功 ✓"
}

create_gpu_vm() {
    local vm_name="$1" ip="$2" cpu="$3" mem="$4" disk_gb="$5"
    log "[$vm_name] 创建 GPU VM (${cpu}核 $((mem/1024))G ${disk_gb}G) IP=$ip"
    # q35 + kvm_hidden 是 PCIe passthrough 的前置条件
    _virt_install_common "$vm_name" "$cpu" "$mem" \
        --machine      q35              \
        --features     kvm_hidden=on
    log "[$vm_name] GPU VM 创建成功 ✓（PCIe passthrough 待 setup-gpu-passthrough.sh 配置）"
}

# =============================================================================
# 主流程
# =============================================================================
main() {
    log "============================================="
    log "  xPod K8s 集群 VM 创建脚本"
    log "============================================="

    [[ $EUID -ne 0 ]] && err "请以 root 执行此脚本"

    check_deps
    check_bridge
    download_base_image

    mkdir -p "$CLOUD_INIT_DIR"

    # 自动读取宿主机 SSH 公钥
    for key_file in /root/.ssh/id_ed25519.pub /root/.ssh/id_rsa.pub; do
        if [[ -f "$key_file" ]]; then
            SSH_PUB_KEY=$(cat "$key_file")
            log "检测到宿主机公钥: $key_file，将注入所有 VM ✓"
            break
        fi
    done
    [[ -z "$SSH_PUB_KEY" ]] && \
        warn "未找到宿主机 SSH 公钥，VM 将使用密码登录 (密码: $VM_PASSWORD)"

    # 遍历创建
    local created=0 skipped=0
    for vm_def in "${VMS[@]}"; do
        IFS='|' read -r vm_name ip cpu mem disk_gb vm_type <<< "$vm_def"

        if virsh dominfo "$vm_name" &>/dev/null; then
            warn "[$vm_name] 已存在，跳过"
            warn "  如需重建: virsh destroy $vm_name && virsh undefine $vm_name --remove-all-storage"
            ((skipped++))
            continue
        fi

        gen_cloud_init "$vm_name" "$ip"
        create_disk    "$vm_name" "$disk_gb"

        if [[ "$vm_type" == "gpu" ]]; then
            create_gpu_vm    "$vm_name" "$ip" "$cpu" "$mem" "$disk_gb"
        else
            create_normal_vm "$vm_name" "$ip" "$cpu" "$mem" "$disk_gb"
        fi
        ((created++))
    done

    log ""
    log "============================================="
    log "  完成: 创建 $created 个，跳过 $skipped 个"
    log "============================================="
    log ""
    log "查看 VM 状态:  virsh list --all"
    log "进入 VM 控制台: virsh console <vm-name>  (Ctrl+] 退出)"
    log ""
    log "cloud-init 首次初始化约需 2~3 分钟"
    log "初始化完成后可 SSH: ssh ${VM_USER}@192.168.100.21"
    log ""
    log "下一步 → 配置 GPU Passthrough:"
    log "  bash setup-gpu-passthrough.sh"
}

main "$@"
