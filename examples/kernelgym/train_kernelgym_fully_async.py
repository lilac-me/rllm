"""
Fully Async KernelGym Training Script

This script combines rllm's KernelGym agent with verl's fully-async training.
It separates rollout and training onto different GPU resources for maximum throughput.
"""

import hydra
import os
import socket
import ray
from omegaconf import OmegaConf
from pprint import pprint

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv

# Fully async imports
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.ray_trainer import Role
from verl.utils.fs import copy_to_local
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available


def _load_or_register(name: str, split: str, fallback_path: str) -> Dataset:
    """Load dataset from registry; if missing, load from file and auto-register."""
    ds = DatasetRegistry.load_dataset(name, split)
    if ds is not None:
        return ds

    # Fallback: load raw data file and register it so verl parquet is created
    raw = Dataset.load_data(fallback_path)
    ds = DatasetRegistry.register_dataset(
        name,
        raw.get_data(),
        split,
        source="local-file",
        description=f"Auto-registered from {fallback_path}",
        category="code",
    )
    return ds


@ray.remote(num_cpus=1)
class FullyAsyncKernelGymTaskRunner:
    """
    Ray remote class for executing fully-async KernelGym training.
    Combines rllm's agent/env pattern with verl's fully-async training.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.agent_class = None
        self.env_class = None
        self.agent_args = None
        self.env_args = None

    def set_agent_env(
        self,
        agent_class,
        env_class,
        agent_args=None,
        env_args=None,
    ):
        """Set the agent and environment classes for training."""
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async KernelGym training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        from concurrent.futures import ThreadPoolExecutor

        print("[ASYNC MAIN] Creating FullyAsyncRollouter and FullyAsyncTrainer in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Trainer first to allocate resources
            trainer_future = executor.submit(self._create_trainer, config)
            trainer_future.result()

            rollouter_future = executor.submit(self._create_rollouter, config)
            rollouter_future.result()

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Setting up parameter synchronization...")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")

        # Import agent loop for kernelgym
        from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

        # For kernelgym, we need to inject agent/env into the rollouter
        # Note: This is a simplified version - full integration would require
        # deeper modifications to support agent loop in fully-async mode

        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        import threading
        import asyncio

        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer_megatron", version_base=None)
def main(config):
    # Ensure fully async config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config - use ++async_training.staleness_threshold=0.6")

    assert config.async_training.get("use_trainer_do_validate", False) is False, \
        "use_trainer_do_validate is not ready to use."

    # Ensure rollout config is properly set
    if not hasattr(config, "rollout"):
        raise RuntimeError("must set rollout config - use ++rollout.nnodes=1 ++rollout.n_gpus_per_node=8")

    # Load datasets
    train_fallback = config.get("data", {}).get("train_files", "data/kernelbench_train.jsonl")
    val_fallback = config.get("data", {}).get("val_files", "data/kernelbench_val.jsonl")

    train_dataset = _load_or_register("kernelbench", "train", train_fallback)
    test_dataset = _load_or_register("kernelbench", "test", val_fallback)

    # Update config with dataset paths
    config.data.train_files = train_dataset.get_verl_data_path()
    config.data.val_files = test_dataset.get_verl_data_path()

    # Agent and env configuration
    agent_args = {
        "system_prompt": (
            "You are an expert GPU kernel engineer. Your task is to write a "
            "high-performance CUDA or Triton kernel that is functionally equivalent "
            "to the given PyTorch reference implementation, but runs faster.\n\n"
            "Instructions:\n"
            "1. Study the reference PyTorch implementation carefully.\n"
            "2. Implement a custom kernel as a Python class named `ModelNew`.\n"
            "3. Your implementation must pass correctness checks.\n"
            "4. Optimise for speed.\n"
            "5. Wrap your final code inside <kernel> ... </kernel> tags."
        ),
    }

    kernel_cfg = config.get("kernel", {})
    env_args = {
        "kernel_server_url": kernel_cfg.get("server_url", "http://localhost:8000"),
        "max_turns": config.get("rllm", {}).get("agent", {}).get("max_steps", 3),
        "backend": kernel_cfg.get("backend", "cuda"),
        "toolkit": kernel_cfg.get("toolkit", "kernelbench"),
        "backend_adapter": kernel_cfg.get("toolkit", "kernelbench"),
        "use_ray": kernel_cfg.get("use_ray", False),
        "num_correct_trials": kernel_cfg.get("num_correct_trials", 5),
        "num_perf_trials": kernel_cfg.get("num_perf_trials", 100),
        "timeout": kernel_cfg.get("timeout", 300),
        "reward_func_name": kernel_cfg.get("reward_func_name", "calculate_reward_like_kernel"),
        "reward_config": {
            k: v for k, v in {
                "rate_limit": kernel_cfg.get("rate_limit"),
                "max_concurrent": kernel_cfg.get("max_concurrent"),
                "acquire_timeout": kernel_cfg.get("acquire_timeout"),
                "enable_profiling": kernel_cfg.get("enable_profiling"),
                "detect_decoy_kernel": kernel_cfg.get("detect_decoy_kernel"),
                "reference_backend": kernel_cfg.get("reference_backend"),
            }.items() if v is not None
        },
    }

    # Initialize Ray with runtime env
    from rllm.trainer.ray_init_utils import get_ray_init_settings

    if not ray.is_initialized():
        ray_init_settings = get_ray_init_settings(config)
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    # Run fully async training
    from time import time

    start_time = time()

    # Create and run the fully async task runner
    runner_cls = ray.remote(num_cpus=1)(FullyAsyncKernelGymTaskRunner)

    if is_cuda_available and config.trainer.get("profile_steps") is not None and \
            len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = runner_cls.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = runner_cls.remote()

    # Set agent and env
    ray.get(runner.set_agent_env.remote(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args=agent_args,
        env_args=env_args,
    ))

    # Run training
    ray.get(runner.run.remote(config))

    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
