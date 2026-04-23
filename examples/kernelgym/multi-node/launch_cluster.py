#!/usr/bin/env python3

###########################################################################
#             集群启动脚本，根据集群配置，通过SSH在远端启动Ray
#   用法: 修改 conf/cluster.yaml 文件, 而后执行 
#         python3 launch_cluster.py conf/cluster.yaml
#   原理: 依次会在本地(远端)启动下面的命令
#         - node_ctl.sh cleanup
#         - node_ctl.sh start-head
#         - node_ctl.sh start-worker
###########################################################################

import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor

def run(cmd):
    print("[local]", cmd)
    subprocess.run(cmd, shell=True, check=True)

def ssh(user, host, cmd, port=22):
    # cmd = 'source /home/g00841271/rllm-071/examples/kernelgym/multi-node/set_base_env.sh &&' + cmd
    full = f"ssh -o StrictHostKeyChecking=no -p {port} {user}@{host} '{cmd}'"
    print("[ssh]", full)
    subprocess.run(full, shell=True, check=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def remote_env(cluster):
    g = cluster["global"]
    ray = cluster["ray"]
    svc = cluster["services"]["reward"]
    return {
        "MASTER_ADDR": g["master_addr"],
        "MASTER_PORT": str(g["master_port"]),
        "RAY_PORT": str(ray["head_port"]),
        "DASHBOARD_PORT": str(ray["dashboard_port"]),
        "REWARD_PORT": str(svc["port"]),
    }

def env_prefix(env):
    return " ".join(f'{k}="{v}"' for k, v in env.items())

def main():
    cfg = load_yaml(sys.argv[1])
    user = cfg["ssh"]["user"]
    port = cfg["ssh"].get("port", 22)
    workdir = cfg["global"]["workdir"]

    env = remote_env(cfg)
    nodes = cfg["nodes"]
    head = next(x for x in nodes if x["role"] == "head")
    reward_host = cfg["services"]["reward"]["host"]

    def do_cleanup(node):
        cmd = f"cd {workdir} && {env_prefix(env)} bash examples/kernelgym/multi-node/node_ctl.sh cleanup"
        ssh(user, node["host"], cmd, port)

    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        list(ex.map(do_cleanup, nodes))
    
    print("do_cleanup done!!!")

    head_cmd = f"cd {workdir} && {env_prefix(env)} bash examples/kernelgym/multi-node/node_ctl.sh start-head"
    ssh(user, head["host"], head_cmd, port)

    print("start_head done!!!")

    time.sleep(5)

    workers = [n for n in nodes if n["role"] == "worker"]
    def start_worker(node):
        cmd = (
            f'cd {workdir} && '
            f'{env_prefix(env)} NODE_IP="{node["host"]}" '
            f'bash examples/kernelgym/multi-node/node_ctl.sh start-worker'
        )
        ssh(user, node["host"], cmd, port)

    with ThreadPoolExecutor(max_workers=max(1, len(workers))) as ex:
        list(ex.map(start_worker, workers))
    
    print("start_worker done!!!")

    # reward_cmd = f"cd {workdir} && {env_prefix(env)} bash examples/kernelgym/multi-node/node_ctl.sh start-reward"
    # ssh(user, reward_host, reward_cmd, port)

    print("cluster launched.")

if __name__ == "__main__":
    main()