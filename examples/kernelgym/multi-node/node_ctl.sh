#!/usr/bin/env bash
set -euo pipefail
# set -x

# 预备处理Ascend环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0-beta.2/share/info/ascendnpu-ir/bin/set_env.sh
export PATH=/usr/local/python3.11.15/bin/:$PATH

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

ACTION=${1:?need action}
shift || true

setup_common_env

# echo "****************"$PATH
# which python3

cleanup() {
  echo "[cleanup] stopping ray..."
  ray stop --force || true
  pkill -f examples.kernelgym.train_kernelgym || true
  pkill -f reward_server || true
  rm -rf /tmp/ray/* || true
  # pkill -9 python3 || true
}

start_head() {
  local master_addr=${MASTER_ADDR:?}
  local ray_port=${RAY_PORT:-6379}
  local dashboard_port=${DASHBOARD_PORT:-8265}

  set -x

  ray stop --force || true
  rm -rf /tmp/ray/* || true
  sleep 2

  echo ***************
  ps aux
  echo ***************

  ray start --head \
    --port "${ray_port}" \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${dashboard_port}" \
    --disable-usage-stats \
    --node-ip-address "${master_addr}"
}

start_worker() {
  local master_addr=${MASTER_ADDR:?}
  local ray_port=${RAY_PORT:-6379}
  local node_ip=${NODE_IP:?}

  ray stop --force || true
  rm -rf /tmp/ray/* || true
  sleep 2

  ray start \
    --address "${master_addr}:${ray_port}" \
    --node-ip-address "${node_ip}" \
    --disable-usage-stats
}

start_reward() {
  local port=${REWARD_PORT:-8002}
  nohup python3 -m examples.kernelgym.reward_server \
    --host 0.0.0.0 \
    --port "${port}" \
    > /tmp/reward_server.log 2>&1 &
}

set -u

case "${ACTION}" in
  cleanup) cleanup ;;
  start-head) start_head ;;
  start-worker) start_worker ;;
  start-reward) start_reward ;;
  restart-head) cleanup; start_head ;;
  restart-worker) cleanup; start_worker ;;
  *)
    echo "unknown action: ${ACTION}"
    exit 1
    ;;
esac