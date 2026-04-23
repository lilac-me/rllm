#!/usr/bin/env python3

###########################################################################
#                            训练任务启动提交
#   用法: 修改 conf/cluster.yaml 文件, 而后执行 
#         python3 submit_job.py conf/cluster.yaml conf/kernelgym_grpo.yaml
#   原理: 加载 kernel_gym 内的 envs 之后，通过 ray submit 提交任务.
###########################################################################

import subprocess
import sys
import yaml
import json

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def stringify(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

def main():
    cluster = load_yaml(sys.argv[1])
    train = load_yaml(sys.argv[2])

    master_addr = cluster["global"]["master_addr"]
    dashboard_port = cluster["ray"]["dashboard_port"]
    reward_host = cluster["services"]["reward"]["host"]
    reward_port = cluster["services"]["reward"]["port"]
    nnodes = len(cluster["nodes"])
    npus_per_node = cluster["global"]["npus_per_node"]
    workdir = cluster["global"]["workdir"]

    env = train.get("env", {})
    args = train.get("args", {}).copy()

    #! 手动嵌入其他相关的参数!
    args["trainer.nnodes"] = nnodes
    args["trainer.n_gpus_per_node"] = npus_per_node
    args["reward_model.server_url"] = f"http://{reward_host}:{reward_port}"

    hydra_args = []
    for k, v in args.items():
        hydra_args.append(f"{k}={stringify(v)}")

    env_cmd = " ".join(f'{k}="{v}"' for k, v in env.items())
    arg_cmd = " ".join(f'"{x}"' for x in hydra_args)

    runtime_env_json = {
        "working_dir": workdir,
        "env_vars": env
    }
    runtime_env_json = json.dumps(runtime_env_json)
    runtime_env_json = runtime_env_json.replace('"','\\\"')

    cmd = (
        # f'ray job submit --address="http://{master_addr}:{dashboard_port}" --runtime-env-json="{runtime_env_json}" --working-dir="{workdir}" -- '
        f'cd {workdir} && '
        f'python3 -m examples.kernelgym.train_kernelgym {arg_cmd}'
    )

    print(cmd)
    # subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()