#!/bin/bash
# =============================================================================
# setup-gpu-passthrough.sh — NVIDIA A10 PCIe Passthrough 配置脚本
#
# 流程:
#   Step 1: 宿主机开启 IOMMU
#   Step 2: 绑定 A10 到 vfio-pci 驱动
#   Step 3: 将 A10 attach 到 gpu-node-1
#   Step 4: 验证
#
# 前置条件:
#   - BIOS/UEFI 已开启 Intel VT-d 或 AMD IOMMU
#   - gpu-node-1 VM 已由 create-vms.sh 创建（q35 + kvm_hidden）
#   - 宿主机已安装 nvidia 驱动（用于识别设备 ID，passthrough 后 VM 内安装）
# =============================================================================
set -euo pipefail

TARGET_VM="gpu-node-1"
VFIO_CONF="/etc/modprobe.d/vfio.conf"
GRUB_CFG="/etc/default/grub"
IOMMU_CONF="/etc/modprobe.d/iommu.conf"

log()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
sep()  { echo -e "\033[1;34m--- $* ---\033[0m"; }

# =============================================================================
# Step 0: 前置检查
# =============================================================================
preflight_check() {
    sep "Step 0: 前置检查"
    [[ $EUID -ne 0 ]] && err "请以 root 执行"

    # 检查 CPU 虚拟化支持
    if grep -qE '(vmx|svm)' /proc/cpuinfo; then
        log "CPU 虚拟化支持 ✓"
    else
        err "CPU 不支持硬件虚拟化 (vmx/svm)，无法使用 PCIe Passthrough"
    fi

    # 检查 IOMMU 是否已在内核启用
    if dmesg | grep -qi "IOMMU enabled"; then
        log "IOMMU 已启用（内核检测）✓"
        IOMMU_ALREADY_ENABLED=true
    else
        warn "IOMMU 未启用，将在 Step 1 配置（需重启）"
        IOMMU_ALREADY_ENABLED=false
    fi

    # 检查 VM 是否存在
    virsh dominfo "$TARGET_VM" &>/dev/null || \
        err "VM $TARGET_VM 不存在，请先运行 create-vms.sh"

    log "前置检查通过 ✓"
}

# =============================================================================
# Step 1: 宿主机开启 IOMMU
# =============================================================================
enable_iommu() {
    sep "Step 1: 开启 IOMMU"

    if [[ "$IOMMU_ALREADY_ENABLED" == "true" ]]; then
        log "IOMMU 已启用，跳过此步骤"
        return
    fi

    # 检测 CPU 厂商
    local iommu_param
    if grep -q "GenuineIntel" /proc/cpuinfo; then
        iommu_param="intel_iommu=on iommu=pt"
        log "检测到 Intel CPU，使用参数: $iommu_param"
    else
        iommu_param="amd_iommu=on iommu=pt"
        log "检测到 AMD CPU，使用参数: $iommu_param"
    fi

    # 备份 grub 配置
    cp "$GRUB_CFG" "${GRUB_CFG}.bak.$(date +%Y%m%d%H%M%S)"
    log "已备份 GRUB 配置"

    # 注入 IOMMU 参数
    if grep -q "intel_iommu\|amd_iommu" "$GRUB_CFG"; then
        warn "GRUB 已有 IOMMU 相关参数，请手动确认: $GRUB_CFG"
    else
        sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"${iommu_param} /" "$GRUB_CFG"
        log "GRUB 参数已更新"
    fi

    # 加载 vfio 模块
    cat > "$IOMMU_CONF" <<EOF
# xPod GPU Passthrough
options vfio-pci disable_idle_d3=1
EOF

    # 确保 vfio 模块开机加载
    cat >> /etc/modules <<'EOF'
vfio
vfio_iommu_type1
vfio_pci
vfio_virqfd
EOF

    # 更新 grub & initramfs
    update-grub 2>/dev/null || grub2-mkconfig -o /boot/grub2/grub.cfg
    update-initramfs -u -k all

    warn "================================================================"
    warn "  IOMMU 参数已写入 GRUB，需要【重启宿主机】后继续！"
    warn ""
    warn "  重启后重新执行此脚本（Step 1 会自动跳过）:"
    warn "    bash setup-gpu-passthrough.sh"
    warn "================================================================"
    read -rp "  现在重启宿主机? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        reboot
    else
        warn "请手动重启后再次执行此脚本"
        exit 0
    fi
}

# =============================================================================
# Step 2: 探测 A10 PCI 地址 & 绑定 vfio-pci
# =============================================================================
bind_vfio() {
    sep "Step 2: 绑定 A10 到 vfio-pci"

    # 探测 NVIDIA A10 的 PCI 地址
    log "扫描 NVIDIA 设备..."
    local pci_list
    pci_list=$(lspci -nn | grep -i "NVIDIA" || true)
    if [[ -z "$pci_list" ]]; then
        err "未检测到 NVIDIA 设备，请确认 A10 已正确安装"
    fi
    log "检测到 NVIDIA 设备:"
    echo "$pci_list"

    # 过滤出 A10（VGA/3D controller）
    # A10 PCI Device ID: 10de:2236
    local gpu_line
    gpu_line=$(echo "$pci_list" | grep -i "A10\|2236\|VGA\|3D controller" | head -1)
    [[ -z "$gpu_line" ]] && gpu_line=$(echo "$pci_list" | head -1)

    local pci_addr vendor_device
    pci_addr=$(echo "$gpu_line" | awk '{print $1}')       # 例: 01:00.0
    vendor_device=$(echo "$gpu_line" | grep -oP '\[\K[0-9a-f]{4}:[0-9a-f]{4}(?=\])' | head -1)

    log "GPU PCI 地址: $pci_addr"
    log "Vendor:Device ID: $vendor_device"

    # 检查是否在独立的 IOMMU group（必须）
    local iommu_group
    iommu_group=$(find /sys/kernel/iommu_groups/*/devices/ -name "${pci_addr}*" \
        -exec dirname {} \; 2>/dev/null | head -1 || true)

    if [[ -z "$iommu_group" ]]; then
        err "A10 未在任何 IOMMU group 中，请确认 IOMMU 已启用且 BIOS 开启 VT-d"
    fi

    local group_id
    group_id=$(echo "$iommu_group" | grep -oP 'iommu_groups/\K[0-9]+')
    log "IOMMU Group: $group_id"

    # 列出同 group 内所有设备（需全部绑定 vfio）
    log "IOMMU Group $group_id 内所有设备:"
    local group_devices
    group_devices=$(ls /sys/kernel/iommu_groups/${group_id}/devices/ 2>/dev/null)
    echo "$group_devices"

    # 逐个绑定到 vfio-pci
    for dev in $group_devices; do
        local dev_vendor_device
        dev_vendor_device=$(lspci -n -s "$dev" | awk '{print $3}')
        [[ -z "$dev_vendor_device" ]] && continue

        log "绑定 $dev ($dev_vendor_device) 到 vfio-pci ..."

        # 解绑当前驱动
        local driver_path="/sys/bus/pci/devices/0000:${dev}/driver"
        if [[ -L "$driver_path" ]]; then
            local current_driver
            current_driver=$(readlink "$driver_path" | xargs basename)
            if [[ "$current_driver" != "vfio-pci" ]]; then
                echo "0000:${dev}" > "/sys/bus/pci/devices/0000:${dev}/driver/unbind"
                log "  已从 $current_driver 解绑"
            fi
        fi

        # 绑定 vfio-pci
        echo "$dev_vendor_device" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
        echo "0000:${dev}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        log "  已绑定到 vfio-pci ✓"
    done

    # 持久化配置（重启后生效）
    local vfio_ids
    vfio_ids=$(for dev in $group_devices; do
        lspci -n -s "$dev" | awk '{print $3}'
    done | tr '\n' ',' | sed 's/,$//')

    cat > "$VFIO_CONF" <<EOF
# xPod GPU Passthrough — NVIDIA A10
# 生成时间: $(date)
options vfio-pci ids=${vfio_ids}
softdep nvidia pre: vfio-pci
EOF

    update-initramfs -u -k all
    log "vfio-pci 持久化配置已写入: $VFIO_CONF ✓"

    # 导出供 Step 3 使用
    echo "$pci_addr" > /tmp/xpod-gpu-pci-addr
    log "GPU PCI 地址已保存: /tmp/xpod-gpu-pci-addr"
}

# =============================================================================
# Step 3: 将 A10 Attach 到 gpu-node-1
# =============================================================================
attach_gpu_to_vm() {
    sep "Step 3: Attach A10 → $TARGET_VM"

    local pci_addr
    if [[ -f /tmp/xpod-gpu-pci-addr ]]; then
        pci_addr=$(cat /tmp/xpod-gpu-pci-addr)
    else
        log "请输入 A10 的 PCI 地址 (例: 01:00.0):"
        read -rp "  PCI 地址: " pci_addr
    fi

    # 解析 domain:bus:slot.function
    # lspci 输出格式: bus:slot.func (例 01:00.0)，添加 domain 0000
    local domain="0x0000"
    local bus func slot
    bus=$(echo "$pci_addr"  | cut -d: -f1)
    slot=$(echo "$pci_addr" | cut -d: -f2 | cut -d. -f1)
    func=$(echo "$pci_addr" | cut -d. -f2)

    # 检查 VM 是否运行
    local vm_state
    vm_state=$(virsh domstate "$TARGET_VM")
    if [[ "$vm_state" == "running" ]]; then
        warn "VM $TARGET_VM 正在运行，将进行热插拔（cold attach 更稳定）"
        warn "建议先关机: virsh shutdown $TARGET_VM"
        read -rp "  继续热插拔? [y/N] " confirm
        [[ ! "$confirm" =~ ^[Yy]$ ]] && { log "已取消，请关机后重试"; exit 0; }
    fi

    # 生成 hostdev XML
    local xml_path="/tmp/xpod-gpu-hostdev.xml"
    cat > "$xml_path" <<EOF
<hostdev mode='subsystem' type='pci' managed='yes'>
  <driver name='vfio'/>
  <source>
    <address domain='${domain}' bus='0x${bus}' slot='0x${slot}' function='0x${func}'/>
  </source>
  <address type='pci' domain='0x0000' bus='0x09' slot='0x00' function='0x0'/>
</hostdev>
EOF

    log "hostdev XML:"
    cat "$xml_path"

    # Attach
    if [[ "$vm_state" == "running" ]]; then
        virsh attach-device "$TARGET_VM" "$xml_path" --live --config
    else
        virsh attach-device "$TARGET_VM" "$xml_path" --config
    fi

    log "A10 已 attach 到 $TARGET_VM ✓"
    rm -f "$xml_path"
}

# =============================================================================
# Step 4: 验证
# =============================================================================
verify() {
    sep "Step 4: 验证"

    log "验证 VM XML 中 hostdev 配置:"
    virsh dumpxml "$TARGET_VM" | grep -A8 "hostdev" | head -30 || \
        warn "未找到 hostdev 配置"

    log ""
    log "验证 vfio-pci 绑定:"
    lspci -nnk | grep -A3 -i "NVIDIA\|A10" || true

    log ""
    log "============================================="
    log "  配置完成！下一步："
    log "============================================="
    log ""
    log "1. 启动 VM（如未运行）:"
    log "   virsh start $TARGET_VM"
    log ""
    log "2. SSH 进入 VM 安装 NVIDIA 驱动:"
    log "   ssh ubuntu@192.168.100.21"
    log "   sudo apt install -y linux-headers-\$(uname -r)"
    log "   # 从 NVIDIA 官网下载 A10 驱动安装包"
    log "   sudo sh NVIDIA-Linux-x86_64-*.run"
    log ""
    log "3. 安装 nvidia-container-toolkit（用于 HAMi）:"
    log "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    log "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    log "   sudo apt update && sudo apt install -y nvidia-container-toolkit"
    log ""
    log "4. 验证 GPU 可见（VM 内）:"
    log "   nvidia-smi"
}

# =============================================================================
# 主流程
# =============================================================================
main() {
    log "============================================="
    log "  xPod GPU Passthrough 配置脚本"
    log "  目标: NVIDIA A10 → $TARGET_VM"
    log "============================================="

    preflight_check
    enable_iommu   # 若已开启则自动跳过
    bind_vfio
    attach_gpu_to_vm
    verify
}

main "$@"
