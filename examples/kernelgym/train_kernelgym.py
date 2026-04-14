import hydra

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _load_or_register(name: str, split: str, fallback_path: str) -> Dataset:
    """Load dataset from registry; if missing, load from file and auto-register.

    The verl training backend requires a ``_verl.parquet`` companion file.
    ``DatasetRegistry.register_dataset`` generates this automatically, whereas
    plain ``Dataset.load_data`` does not.  By auto-registering the JSONL/parquet
    data when the registry entry is absent, we ensure the verl data pipeline
    always has a valid parquet path.
    """
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

# @hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer_megatron", version_base=None)
def main(config):
    train_fallback = config.get("data", {}).get("train_files", "data/kernelbench_train.jsonl")
    val_fallback = config.get("data", {}).get("val_files", "data/kernelbench_val.jsonl")

    train_dataset = _load_or_register("kernelbench", "train", train_fallback)
    test_dataset = _load_or_register("kernelbench", "test", val_fallback)

    # agent_args = {
    #     "system_prompt": (
    #         "You are an expert NPU Triton kernel engineer. Your task is to write a "
    #         "high-performance CUDA or Triton kernel that is functionally equivalent "
    #         "to the given PyTorch reference implementation, but runs faster.\n\n"
    #         "Instructions:\n"
    #         "1. Study the reference PyTorch implementation carefully.\n"
    #         "2. Implement a custom kernel as a Python class named `ModelNew`.\n"
    #         "3. Your implementation must pass correctness checks.\n"
    #         "4. Optimise for speed.\n"
    #         "5. Wrap your final code inside <kernel> ... </kernel> tags."
    #     ),
    # }
    
    skills = """

        你是 **Triton-Ascend 算子代码生成专家**，负责将 PyTorch 参考实现转换为高性能 Triton-Ascend 算子代码。

        ---

        ## 生成思路

        1. 仔细阅读参考 `Model.forward()` 的 PyTorch 实现
        2. 理解算子的数学逻辑和计算模式
        3. 判断算子类型（elementwise / reduce / matmul / attention / 复合）
        4. 选择合适的并行化策略和内存访问模式
        5. 生成 kernel 函数和 `ModelNew` 类
        ---

        ## 禁止 PyTorch 退化

        `ModelNew.forward()` 中**所有核心计算**必须在 `@triton.jit` kernel 中实现。

        ### forward() 中禁止的操作

        - `torch.matmul / torch.relu / torch.sum` 等计算函数
        - `F.softmax / F.linear` 等 `torch.nn.functional`
        - `x.sum() / x.mean() / x @ w` 等 tensor 方法/运算符
        - `self.conv(x) / self.linear(x)` 等 nn.Module 调用

        ### forward() 中允许的操作

        - buffer 分配：`torch.empty / torch.zeros / torch.ones`
        - 形状操作：`.view / .reshape / .permute / .transpose / .contiguous`
        - 元信息：`.shape / .dtype / .device / .numel()`
        - kernel 启动：`kernel[grid](...)`

        ### PyTorch 退化子类型

        | 子类型 | 含义 | 修复建议 |
        |--------|------|---------|
        | Type1 | 完全无 @triton.jit kernel | 必须创建 @triton.jit kernel，使用 tl.load/tl.store 实现核心计算 |
        | Type2 | 有 kernel 定义但 forward() 未调用 | 在 forward() 中通过 kernel[grid](...) 启动 kernel |
        | Type3 | forward() 调用了 kernel 但部分计算仍用 PyTorch | 将禁止的 PyTorch 计算移入 kernel |

        ---

        ## 常见错误

        | 子类型 | error 特征 | 修复方向 |
        |--------|-----------|---------|
        | PyTorch 退化 Type 1 | 完全无 `@triton.jit` kernel | 必须创建 @triton.jit kernel，使用 tl.load/tl.store 实现核心计算 |
        | PyTorch 退化 Type 2 | 有 kernel 但 `forward()` 未调用 | 在 forward() 中通过 kernel[grid](...) 启动 kernel |
        | PyTorch 退化 Type 3 | `forward()` 部分计算仍用 PyTorch | 将禁止的 PyTorch 计算移入 kernel |
        | 输出不一致 | 数值精度差异、算法实现与参考不同 | 检查算法逻辑、精度处理 |
        | 语法/类型错误 | SyntaxError、TypeError、IndentationError | 修复语法 |
        | 形状不匹配 | Tensor shape mismatch、维度错误 | 检查 tensor 维度计算 |
        | Kernel 参数错误 | BLOCK_SIZE 不合理、grid 配置错误 | 调整分块参数和网格 |
        | DSL API 使用错误 | Triton API 参数错误、不支持的操作 | 查阅 Triton-Ascend API |

        ---

        ## 约束

        | 约束 | 说明 |
        |------|------|
        | 禁止 PyTorch 退化 | forward() 中禁止 torch.*/F.* 计算操作 |
        | 类名 `ModelNew` | 必须使用 `ModelNew`，不能是 `Model` |
        | 接口一致 | `__init__` 和 `forward` 签名与原 `Model` 完全一致 |
        | 输出一致 | 输出的形状、数据类型必须与原 `Model` 一致 |
        | 自包含 | 所有 kernel 和辅助函数定义在同一文件内 |
        | 无测试代码 | 不生成测试代码、`if __name__` 块或 `print` 语句 |
        | 数值正确性优先 | 正确性与性能冲突时，始终选择正确性 |

    """
    
    agent_args = {
        "system_prompt": (
            f"{skills}\n\n"
            "## 任务执行指令\n"
            "1. 严格遵守上述“全局约定”进行算子开发。\n"
            "2. 深入研究给出的 PyTorch 参考实现，确保数学逻辑完全一致。\n"
            "3. 优化计算效率与内存访问模式，确保在 Ascend 后端高性能运行。\n"
            "4. 必须将最终生成的完整代码包裹在 <kernel> ... </kernel> 标签内。"
        ),
    }

    kernel_cfg = config.get("kernel", {})
    env_args = {
        "kernel_server_url": kernel_cfg.get("server_url", "http://localhost:8000"),
        "max_turns": config.get("rllm", {}).get("agent", {}).get("max_steps", 3),
        "backend": kernel_cfg.get("backend", "triton"),
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

    trainer = AgentTrainer(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

