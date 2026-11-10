#!/usr/bin/env bash
# 在厂商服务端进程内设置匹配的 DDS 运行库，默认只检查动态链接。
set -euo pipefail

usage() {
    printf '用法: %s [--check|--run]\n' "$0"
    printf '%s\n' '默认 --check，只检查动态库。--run 在前台启动服务端。'
    printf '%s\n' 'RM75B_GRIPPER_VENDOR 可覆盖厂商文件目录，RM75B_GRIPPER_NETWORK 可覆盖网卡。'
}

mode="${1:---check}"
if [[ "$#" -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --check|--run) ;;
    *) usage >&2; exit 2 ;;
esac

vendor="${RM75B_GRIPPER_VENDOR:-/home/scl/work/FastUMI_Data/ros2_ws/install/unitree_gripper/share/unitree_gripper/vendor}"
network="${RM75B_GRIPPER_NETWORK:-wlP1p1s0}"
server="$vendor/dex1_1_gripper_server"
if [[ ! -f "$server" || ! -x "$server" || ! -d "$vendor/lib" ]]; then
    printf '夹爪服务端或配套库目录不存在，请检查 RM75B_GRIPPER_VENDOR: %s\n' "$vendor" >&2
    exit 1
fi

if [[ "$mode" == --run ]]; then
    # 运行态把独占锁描述符传给 exec 后的服务端，手动与自动启动使用同一锁。
    exec 9>>/tmp/lerobot-rm75b-gripper-server.lock
    if ! flock -n 9; then
        printf '%s\n' '夹爪服务已运行或正在启动，请勿重复启动。' >&2
        exit 1
    fi
fi

# /usr/local/lib 提供配套的 CycloneDDS C/C++ 库，Humble 目录提供 Iceoryx 运行库。
# Python DDS 客户端的私有 C 库配置与厂商服务端隔离。
library_path="$vendor/lib:/usr/local/lib:/opt/ros/humble/lib/aarch64-linux-gnu"
if ! report="$(timeout --foreground 10s env -u CYCLONEDDS_HOME LC_ALL=C LD_LIBRARY_PATH="$library_path" ldd -r "$server" 2>&1)"; then
    printf '夹爪服务端动态库检查失败或超时:\n%s\n' "$report" >&2
    exit 1
fi
if [[ "$report" == *"not found"* || "$report" == *"undefined symbol"* ]]; then
    printf '夹爪服务端存在未解析的动态库或符号:\n%s\n' "$report" >&2
    exit 1
fi
printf '%s\n' "$report"
printf '%s\n' '夹爪服务端动态库检查通过。'

if [[ "$mode" == --run ]]; then
    exec env -u CYCLONEDDS_HOME LD_LIBRARY_PATH="$library_path" "$server" -n "$network"
fi
