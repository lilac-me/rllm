#!/bin/bash
set -euo pipefail
set -x

source setup_env.sh

if [ "$RAY_MASTER_HOSTNAME" = "$CURRENT_HOSTNAME" ]; then
  # 主节点启动
  ray start --head \
    --port $RAY_MASTER_PORT \
    --dashboard-host="$CURRENT_IP" \
    --node-ip-address="$CURRENT_IP" \
    --dashboard-port=8260 \
    --resources="{\"NPU\": $NPUS_PER_NODE}"

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash "$DEFAULT_SH" 2>&1 | tee logs/qwen3-30b_int8_npu.log
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册 ray 直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start \
        --address="$RAY_MASTER_HOSTNAME:$RAY_MASTER_PORT" \
        --resources="{\"NPU\": $NPUS_PER_NODE}" \
        --node-ip-address="$CURRENT_IP"

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
