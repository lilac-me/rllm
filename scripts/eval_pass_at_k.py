"""
PASS@K Evaluation Script for KernelGym.

This script evaluates model performance on KernelBench level1 dataset with:
- Multi-turn feedback iterations (N rounds)
- Multiple rollouts per problem (M rollouts)
- PASS@K accuracy and speedup metrics
- SQLite database for interaction history
- Per-problem and dataset-level analysis
- Visualization and plotting

Usage:
    python scripts/eval_pass_at_k.py \
        --vllm-url http://localhost:8000/v1 \
        --kernelgym-url http://localhost:8002 \
        --output-dir results/pass_at_k \
        --num-rollouts 10 \
        --max-turns 3 \
        --k-values 1,5,10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx
import numpy as np
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_pass_at_k")


@dataclass
class TurnResult:
    turn: int
    compiled: bool = False
    correctness: bool = False
    speedup: float = 0.0
    reward: float = 0.0
    error: str | None = None
    kernel_code: str | None = None
    response: str | None = None
    server_result: dict | None = None


@dataclass
class RolloutResult:
    rollout_id: str
    problem_id: str
    turns: list[TurnResult] = field(default_factory=list)
    best_speedup: float = 0.0
    best_turn: int = -1
    final_correct: bool = False
    final_compiled: bool = False
    total_reward: float = 0.0


@dataclass
class ProblemStats:
    problem_id: str
    num_rollouts: int = 0
    num_passed: int = 0
    pass_at_k: dict[int, float] = field(default_factory=dict)
    best_speedup: float = 0.0
    avg_speedup: float = 0.0
    avg_reward: float = 0.0


_SYSTEM_PROMPT = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton. You need to create a Triton version with the `ModelNew`. This triton version must be execution on Ascend NPU platforms.

Please firstly analyze this code and think hard how you can optimize it. YOU MUST wrap your final code in a ```triton ... ``` code block. No other code block markers are acceptable.

**Please output and show your thinking, plan,
analysis etc., before your coding, which should be as
more as possible.**

Here's the PyTorch code:

"""

_INITIAL_USER_TEMPLATE = """
```python
{reference_code}
```
"""

_REVISION_USER_TEMPLATE = """\
Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.
"""


def extract_kernel_code(response: str) -> str | None:
    patterns = [
        re.compile(r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
        re.compile(r"```python\s*#\s*Kernel\s*\n(.*?)```", re.I | re.S),
        re.compile(r"#\s*Your\s+implementation:\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
        re.compile(r"#\s*Generated\s+kernel:\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
    ]
    
    for pat in patterns:
        m = pat.search(response)
        if m:
            return m.group(1).strip()
    
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", response, re.S)
    if code_blocks:
        return code_blocks[-1].strip()
    return None


class KernelGymClient:
    def __init__(self, base_url: str, timeout: int = 600, poll_interval: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=float(timeout), write=10.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )

    def evaluate(
        self,
        reference_code: str,
        kernel_code: str,
        entry_point: str = "Model",
        task_timeout: int = 300,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        verbose_errors: bool = True,
        enable_profiling: bool = False,
    ) -> dict[str, Any]:
        task_id = f"passatk_{uuid4().hex[:12]}"
        payload = {
            "task_id": task_id,
            "reference_code": reference_code,
            "kernel_code": kernel_code,
            "backend": "triton",
            "entry_point": entry_point,
            "timeout": task_timeout,
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "verbose_errors": verbose_errors,
            "enable_profiling": enable_profiling,
        }

        try:
            resp = self._client.post(f"{self.base_url}/evaluate", json=payload)
            if resp.status_code != 200:
                return {"status": "failed", "error": f"submit HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

        deadline = time.time() + task_timeout + 60
        while time.time() < deadline:
            try:
                s = self._client.get(f"{self.base_url}/status/{task_id}")
                if s.status_code == 200:
                    status = s.json().get("status", "unknown")
                    if status in ("completed", "failed", "timeout", "cancelled"):
                        if status == "completed":
                            r = self._client.get(f"{self.base_url}/results/{task_id}")
                            if r.status_code == 200:
                                result = r.json()
                                result["status"] = status
                                return result
                            return {"status": status, "error": f"fetch results HTTP {r.status_code}"}
                        return {"status": status, "error": s.json().get("error_message", status)}
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval)

        return {"status": "timeout", "error": "client-side polling timeout"}


class InteractionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollout_id TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                best_speedup REAL,
                final_correct INTEGER,
                final_compiled INTEGER,
                total_reward REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollout_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                compiled INTEGER,
                correctness INTEGER,
                speedup REAL,
                reward REAL,
                error TEXT,
                kernel_code TEXT,
                response TEXT,
                server_result TEXT,
                FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problem_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL UNIQUE,
                num_rollouts INTEGER,
                num_passed INTEGER,
                pass_at_1 REAL,
                pass_at_5 REAL,
                pass_at_10 REAL,
                best_speedup REAL,
                avg_speedup REAL,
                avg_reward REAL
            )
        """)
        
        conn.commit()
        conn.close()

    def save_rollout(self, result: RolloutResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rollouts (rollout_id, problem_id, timestamp, best_speedup, final_correct, final_compiled, total_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.rollout_id,
            result.problem_id,
            datetime.now().isoformat(),
            result.best_speedup,
            int(result.final_correct),
            int(result.final_compiled),
            result.total_reward,
        ))
        
        for turn in result.turns:
            cursor.execute("""
                INSERT INTO turns (rollout_id, turn, compiled, correctness, speedup, reward, error, kernel_code, response, server_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.rollout_id,
                turn.turn,
                int(turn.compiled),
                int(turn.correctness),
                turn.speedup,
                turn.reward,
                turn.error,
                turn.kernel_code,
                turn.response,
                json.dumps(turn.server_result) if turn.server_result else None,
            ))
        
        conn.commit()
        conn.close()

    def save_problem_stats(self, stats: ProblemStats):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO problem_stats 
            (problem_id, num_rollouts, num_passed, pass_at_1, pass_at_5, pass_at_10, best_speedup, avg_speedup, avg_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stats.problem_id,
            stats.num_rollouts,
            stats.num_passed,
            stats.pass_at_k.get(1, 0.0),
            stats.pass_at_k.get(5, 0.0),
            stats.pass_at_k.get(10, 0.0),
            stats.best_speedup,
            stats.avg_speedup,
            stats.avg_reward,
        ))
        
        conn.commit()
        conn.close()

    def get_all_rollouts(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rollouts")
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_problem_rollouts(self, problem_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rollouts WHERE problem_id = ?", (problem_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider
    
    Returns:
        pass@k probability
    """
    if n - k < 0:
        return 0.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)) if n > c else 1.0


def run_single_rollout(
    problem_id: str,
    reference_code: str,
    entry_point: str,
    llm: OpenAI,
    gym: KernelGymClient,
    model_name: str,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    task_timeout: int,
) -> RolloutResult:
    rollout_id = f"{problem_id}_{uuid4().hex[:8]}"
    result = RolloutResult(rollout_id=rollout_id, problem_id=problem_id)
    
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _INITIAL_USER_TEMPLATE.format(reference_code=reference_code)},
    ]
    
    for turn_idx in range(max_turns):
        try:
            completion = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error("[%s] Turn %d LLM error: %s", problem_id, turn_idx + 1, e)
            result.turns.append(TurnResult(turn=turn_idx, error=f"LLM error: {e}"))
            break
        
        kernel_code = extract_kernel_code(response_text)
        if not kernel_code:
            logger.warning("[%s] Turn %d: no code extracted", problem_id, turn_idx + 1)
            result.turns.append(TurnResult(
                turn=turn_idx, 
                error="no code block found",
                response=response_text,
            ))
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _REVISION_USER_TEMPLATE.format(
                feedback=json.dumps({"status": "failed", "error_message": "No valid Python code block found."})
            )})
            continue
        
        kernel_code = "import triton \nimport triton.language as tl\nimport torch\nimport torch.nn as nn\n" + kernel_code
        if "class ModelNew:" in kernel_code:
            kernel_code = kernel_code.replace("class ModelNew:", "class ModelNew(nn.Module):")
        
        eval_result = gym.evaluate(
            reference_code=reference_code,
            kernel_code=kernel_code,
            entry_point=entry_point,
            task_timeout=task_timeout,
        )
        
        compiled = bool(eval_result.get("compiled", False))
        correctness = bool(eval_result.get("correctness", False))
        speedup = float(eval_result.get("speedup") or 0.0)
        error = eval_result.get("error_message") or eval_result.get("error")
        
        reward = 0.0
        if not compiled:
            reward = -0.5
        elif not correctness:
            reward = -0.3
        else:
            if speedup >= 3.0:
                reward = 1.0
            elif speedup >= 2.0:
                reward = 0.8
            elif speedup >= 1.5:
                reward = 0.6
            elif speedup >= 1.2:
                reward = 0.4
            elif speedup >= 1.0:
                reward = 0.2
            else:
                reward = -0.1
        
        tr = TurnResult(
            turn=turn_idx,
            compiled=compiled,
            correctness=correctness,
            speedup=speedup,
            reward=reward,
            error=error,
            kernel_code=kernel_code,
            response=response_text,
            server_result=eval_result,
        )
        result.turns.append(tr)
        result.total_reward += reward
        
        logger.info(
            "[%s] Turn %d: compiled=%s correct=%s speedup=%.2fx reward=%.2f",
            problem_id, turn_idx + 1, compiled, correctness, speedup, reward,
        )
        
        if correctness and speedup > result.best_speedup:
            result.best_speedup = speedup
            result.best_turn = turn_idx
            result.final_correct = True
        
        if compiled:
            result.final_compiled = True
        
        if turn_idx < max_turns - 1:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _REVISION_USER_TEMPLATE.format(
                feedback=json.dumps(eval_result, default=str)
            )})
    
    return result


def load_kernelbench_data(data_path: str, hf_split: str = "level_1") -> list[dict]:
    """Load KernelBench level1 data."""
    tasks = []
    path = Path(data_path)
    
    if path.suffix in (".parquet", ".jsonl") or path.exists():
        import pandas as pd
        if path.suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        else:
            df = pd.read_parquet(data_path)
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            if "task" in row_dict:
                task_data = row_dict["task"]
                tasks.append({
                    "problem_id": task_data.get("problem_id", f"task_{idx}"),
                    "reference_code": task_data.get("reference_code", ""),
                    "entry_point": task_data.get("entry_point", "Model"),
                })
            elif "code" in row_dict:
                level = row_dict.get("level", "")
                pid = row_dict.get("problem_id", idx)
                name = row_dict.get("name", "")
                tasks.append({
                    "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                    "reference_code": row_dict["code"],
                    "entry_point": "Model",
                })
            elif "reference_code" in row_dict:
                tasks.append({
                    "problem_id": row_dict.get("problem_id", f"task_{idx}"),
                    "reference_code": row_dict["reference_code"],
                    "entry_point": row_dict.get("entry_point", "Model"),
                })
    else:
        from datasets import load_dataset
        ds = load_dataset("ScalingIntelligence/KernelBench", split=hf_split)
        for idx, row in enumerate(ds):
            level = row.get("level", "")
            pid = row.get("problem_id", idx)
            name = row.get("name", "")
            tasks.append({
                "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                "reference_code": row["code"],
                "entry_point": "Model",
            })
    
    logger.info("Loaded %d tasks from %s", len(tasks), data_path)
    return tasks


def compute_problem_stats(
    problem_id: str,
    rollout_results: list[RolloutResult],
    k_values: list[int],
) -> ProblemStats:
    """Compute statistics for a single problem."""
    stats = ProblemStats(problem_id=problem_id)
    stats.num_rollouts = len(rollout_results)
    
    passed_rollouts = [r for r in rollout_results if r.final_correct]
    stats.num_passed = len(passed_rollouts)
    
    for k in k_values:
        stats.pass_at_k[k] = calculate_pass_at_k(stats.num_rollouts, stats.num_passed, k)
    
    if passed_rollouts:
        speedups = [r.best_speedup for r in passed_rollouts]
        stats.best_speedup = max(speedups)
        stats.avg_speedup = np.mean(speedups)
    
    all_rewards = [r.total_reward for r in rollout_results]
    stats.avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    
    return stats


def plot_results(
    problem_stats: list[ProblemStats],
    output_dir: str,
    k_values: list[int],
):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    for k in k_values:
        pass_rates = [s.pass_at_k.get(k, 0.0) for s in problem_stats]
        ax1.hist(pass_rates, bins=20, alpha=0.5, label=f'Pass@{k}')
    ax1.set_xlabel('Pass Rate')
    ax1.set_ylabel('Number of Problems')
    ax1.set_title('Distribution of Pass@K Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    problem_ids = [s.problem_id[:20] for s in problem_stats[:20]]
    pass_at_1 = [s.pass_at_k.get(1, 0.0) for s in problem_stats[:20]]
    pass_at_5 = [s.pass_at_k.get(5, 0.0) for s in problem_stats[:20]]
    x = np.arange(len(problem_ids))
    width = 0.35
    ax2.bar(x - width/2, pass_at_1, width, label='Pass@1')
    ax2.bar(x + width/2, pass_at_5, width, label='Pass@5')
    ax2.set_xlabel('Problem ID')
    ax2.set_ylabel('Pass Rate')
    ax2.set_title('Pass@K by Problem (First 20)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_ids, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    speedups = [s.best_speedup for s in problem_stats if s.best_speedup > 0]
    if speedups:
        ax3.hist(speedups, bins=20, color='green', alpha=0.7)
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Baseline (1.0x)')
        ax3.set_xlabel('Best Speedup')
        ax3.set_ylabel('Number of Problems')
        ax3.set_title('Distribution of Best Speedups')
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    rewards = [s.avg_reward for s in problem_stats]
    ax4.hist(rewards, bins=20, color='purple', alpha=0.7)
    ax4.set_xlabel('Average Reward')
    ax4.set_ylabel('Number of Problems')
    ax4.set_title('Distribution of Average Rewards')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_at_k_analysis.png'), dpi=150)
    plt.close()
    
    fig2, ax = plt.subplots(figsize=(10, 6))
    overall_pass_at_k = {}
    for k in k_values:
        total_rollouts = sum(s.num_rollouts for s in problem_stats)
        total_passed = sum(s.num_passed for s in problem_stats)
        overall_pass_at_k[k] = calculate_pass_at_k(total_rollouts, total_passed, k)
    
    ax.bar([f'Pass@{k}' for k in k_values], [overall_pass_at_k[k] for k in k_values], color='steelblue')
    ax.set_ylabel('Pass Rate')
    ax.set_title('Overall Pass@K Performance')
    ax.set_ylim(0, 1)
    for i, k in enumerate(k_values):
        ax.text(i, overall_pass_at_k[k] + 0.02, f'{overall_pass_at_k[k]:.2%}', ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'overall_pass_at_k.png'), dpi=150)
    plt.close()
    
    logger.info("Plots saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="PASS@K Evaluation for KernelGym")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--vllm-api-key", default="EMPTY", help="API key for LLM")
    parser.add_argument("--model-name", default="default", help="Model name")
    parser.add_argument("--kernelgym-url", default="http://localhost:8002", help="KernelGym server URL")
    parser.add_argument("--data-path", default="data/kernelbench_train.jsonl", help="Path to KernelBench data")
    parser.add_argument("--hf-split", default="level_1", help="HuggingFace split name")
    parser.add_argument("--output-dir", default="results/pass_at_k", help="Output directory")
    parser.add_argument("--num-rollouts", type=int, default=10, help="Number of rollouts per problem")
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum turns per rollout")
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated K values for Pass@K")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per response")
    parser.add_argument("--task-timeout", type=int, default=300, help="Task timeout in seconds")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit-problems", type=int, default=None, help="Limit number of problems (for testing)")
    
    args = parser.parse_args()
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    
    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, "interactions.db")
    db = InteractionDatabase(db_path)
    
    llm = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
    gym = KernelGymClient(args.kernelgym_url, timeout=args.task_timeout + 60)
    
    tasks = load_kernelbench_data(args.data_path, args.hf_split)
    if args.limit_problems:
        tasks = tasks[:args.limit_problems]
    
    all_results: dict[str, list[RolloutResult]] = {}
    
    logger.info("Starting PASS@K evaluation: %d problems, %d rollouts each", len(tasks), args.num_rollouts)
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for task in tasks:
            for rollout_idx in range(args.num_rollouts):
                future = executor.submit(
                    run_single_rollout,
                    task["problem_id"],
                    task["reference_code"],
                    task["entry_point"],
                    llm,
                    gym,
                    args.model_name,
                    args.max_turns,
                    args.temperature,
                    args.max_tokens,
                    args.task_timeout,
                )
                futures.append((task["problem_id"], future))
        
        for problem_id, future in futures:
            try:
                result = future.result(timeout=args.task_timeout * args.max_turns + 60)
                if problem_id not in all_results:
                    all_results[problem_id] = []
                all_results[problem_id].append(result)
                db.save_rollout(result)
                logger.info("[%s] Rollout completed: correct=%s speedup=%.2f", 
                           problem_id, result.final_correct, result.best_speedup)
            except Exception as e:
                logger.error("[%s] Rollout failed: %s", problem_id, e)
    
    problem_stats: list[ProblemStats] = []
    for problem_id, results in all_results.items():
        stats = compute_problem_stats(problem_id, results, k_values)
        problem_stats.append(stats)
        db.save_problem_stats(stats)
        logger.info(
            "[%s] Stats: Pass@1=%.2f%% Pass@5=%.2f%% BestSpeedup=%.2fx",
            problem_id,
            stats.pass_at_k.get(1, 0) * 100,
            stats.pass_at_k.get(5, 0) * 100,
            stats.best_speedup,
        )
    
    total_rollouts = sum(s.num_rollouts for s in problem_stats)
    total_passed = sum(s.num_passed for s in problem_stats)
    overall_pass_at_k = {}
    for k in k_values:
        overall_pass_at_k[k] = calculate_pass_at_k(total_rollouts, total_passed, k)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_problems": len(tasks),
            "num_rollouts_per_problem": args.num_rollouts,
            "max_turns": args.max_turns,
            "k_values": k_values,
        },
        "overall_metrics": {
            "total_rollouts": total_rollouts,
            "total_passed": total_passed,
            "pass_at_k": {f"pass@{k}": v for k, v in overall_pass_at_k.items()},
        },
        "problem_stats": [asdict(s) for s in problem_stats],
    }
    
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)
    
    plot_results(problem_stats, args.output_dir, k_values)
    
    print("\n" + "=" * 60)
    print("PASS@K EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Problems: {len(tasks)}")
    print(f"Rollouts per Problem: {args.num_rollouts}")
    print(f"Max Turns: {args.max_turns}")
    print("-" * 60)
    for k in k_values:
        print(f"Overall Pass@{k}: {overall_pass_at_k[k]:.2%}")
    print("-" * 60)
    print(f"Total Passed Rollouts: {total_passed}/{total_rollouts}")
    print(f"Database: {db_path}")
    print(f"Plots: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
