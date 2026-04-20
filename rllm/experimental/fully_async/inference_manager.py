# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import subprocess

import ray
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils.net_utils import get_free_port


@ray.remote(num_cpus=10, max_concurrency=100)
class InferenceManager:
    """
    Manages inference servers (vLLM or SGLang) for async training (standalone rollout).

    Rollout replicas and ``CheckpointEngineWorker`` processes are created by
    ``AgentLoopManager`` (verl 0.7.1+), not by a hybrid ``RayWorkerGroup``.

    Does NOT handle:
    - Dataset loading (owned by RolloutExecutor)
    - Staleness/queue sizing (owned by RolloutExecutor)
    - Parameter sync (owned by ``FullyAsyncTrainer`` + ``CheckpointEngineManager``)
    """

    def __init__(self, config, tokenizer, processor=None, device_name=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.device_name = device_name if device_name else config.trainer.device
        self.async_rollout_manager = None
        self.router_process = None
        self.router_url = None

        assert not config.actor_rollout_ref.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, "trigger_parameter_sync_step must larger than 1"

        self._validate_config()

    def _validate_config(self):
        if not hasattr(self.config, "async_training"):
            raise ValueError("[InferenceManager] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Launch standalone rollout replicas (``AgentLoopManager``, ``worker_group=None``)."""
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        partial = getattr(self.config.async_training, "partial_rollout", False)
        if partial:
            from rllm.experimental.fully_async.async_agent_loop import FullyAsyncAgentLoopManager

            self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
                config=self.config,
                worker_group=None,
            )
        else:
            self.async_rollout_manager = await AgentLoopManager.create(
                config=self.config,
                worker_group=None,
            )

    def get_replicas(self):
        """Rollout replicas for ``CheckpointEngineManager``."""
        if self.async_rollout_manager is None:
            raise RuntimeError("async_rollout_manager not initialized. Call init_workers() first.")
        return self.async_rollout_manager.rollout_replicas

    def get_rollout_name(self) -> str:
        return getattr(self.config.actor_rollout_ref.rollout, "name", "sglang")

    def launch_router(self, port: int = 30000):
        """Launch inference entry point: SGLang router or vLLM direct URL."""
        if self.async_rollout_manager is None:
            raise RuntimeError("async_rollout_manager not initialized. Call init_workers() first.")

        server_addresses = self.async_rollout_manager.server_addresses
        urls = [f"http://{addr}" for addr in server_addresses]
        rollout_name = self.get_rollout_name()

        if rollout_name == "sglang":
            ip = ray.util.get_node_ip_address()
            actual_port, sock = get_free_port(ip)
            sock.close()
            print(f"[InferenceManager] Launching sglang_router on port {actual_port} with server URLs: {urls}")

            cmd = [
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--worker-urls",
                *urls,
                "--port",
                str(actual_port),
                "--policy",
                "cache_aware",
                "--log-level",
                "warn",
            ]
            self.router_process = subprocess.Popen(cmd)
            self.router_url = f"http://{ip}:{actual_port}"
        else:
            self.router_process = None
            self.router_url = urls[0]
            if len(urls) > 1:
                print(
                    f"[InferenceManager] WARNING: {rollout_name} has {len(urls)} replicas but only "
                    f"the first ({self.router_url}) is used. Multi-replica load balancing requires "
                    f"an external L7 proxy."
                )
            print(f"[InferenceManager] Using {rollout_name} server directly at {self.router_url}")

        return self.router_url

    async def abort_all_rollout_requests(self):
        replicas = self.async_rollout_manager.rollout_replicas
        await asyncio.gather(*[replica.abort_all_requests() for replica in replicas])

    async def resume_all_rollout_generation(self):
        replicas = self.async_rollout_manager.rollout_replicas
        await asyncio.gather(*[replica.resume_generation() for replica in replicas])

    async def clear_kv_cache(self):
        await self.async_rollout_manager.clear_kv_cache()
