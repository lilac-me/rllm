# rLLM Development Guide

## Overview

rLLM is an open-source framework for training AI agents with reinforcement learning. It supports multiple agent frameworks (LangGraph, SmolAgent, Strands, OpenAI Agents SDK, etc.) with minimal code changes.

## Core Architecture

rLLM follows a modular pipeline: **run your agent → collect traces → compute rewards → update the model**.

### Key Components

1. **SDK (`rllm/sdk/`)**: Automatic LLM trace collection with session contexts and trajectory decorators
   - Two session backends: `contextvar` (in-process) and `opentelemetry` (distributed)
   - Core decorators: `@rollout`, `@evaluator`, `@trajectory`
   - Chat clients: `get_chat_client()`, `get_chat_client_async()`

2. **Agents & Environments (`rllm/agents/`, `rllm/environments/`)**:
   - `BaseAgent`: Extend to create custom agents
   - `BaseEnv`: Extend to create custom environments
   - Data structures: `Episode`, `Trajectory`, `Step`, `Action`

3. **Workflow Engine (`rllm/workflows/`)**:
   - `AgentExecutionEngine`: High-performance parallel trajectory rollout
   - `AgentWorkflowEngine`: Orchestrates episode-level workflows
   - Workflows: `simple_workflow`, `multi_turn_workflow`, `eval_protocol_workflow`

4. **Training (`rllm/trainer/`, `rllm/experimental/`)**:
   - `AgentTrainer`: Legacy trainer using verl backend
   - `UnifiedTrainer`: New trainer supporting both `verl` and `tinker` backends
   - RL algorithms: GRPO, PPO, REINFORCE, RLOO
   - Backends: `verl` (distributed multi-GPU), `tinker` (single-machine/CPU)

5. **Model Gateway (`rllm-model-gateway/`)**:
   - Lightweight FastAPI gateway for capturing token IDs and logprobs
   - Session-sticky routing to vLLM workers
   - Zero code changes required for agents

## Common Development Tasks

### Installation

```bash
# Install with Tinker backend (Python 3.11+)
uv pip install -e .[tinker]

# Install with verl backend (GPU required)
uv pip install -e .[verl]

# Install all extras
uv pip install -e .[all,dev]
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_specific_file.py

# Run a specific test function
pytest tests/test_file.py::test_function_name

# Run with verbose output
pytest tests/ -v

# Run tests matching a pattern
pytest tests/ -k "math"
```

### Code Formatting and Linting

```bash
# Install pre-commit hooks (one-time)
pre-commit install

# Run pre-commit manually
pre-commit run

# Run ruff directly
ruff check .
ruff format .

# Fix issues automatically
ruff check . --fix
```

### Building Documentation

```bash
# Build documentation
./build_docs.sh

# Serve documentation locally with live reload
./build_docs.sh serve

# Or manually
cd docs
mkdocs build
mkdocs serve
```

### Running Examples

```bash
# CLI-based evaluation
rllm eval gsm8k

# CLI-based training
rllm train gsm8k

# Run a specific example (check examples/ directory)
cd examples/math_tinker
python train.py
```

### Launching LiteLLM Proxy

```bash
# Use the provided script
./scripts/launch_litellm.sh

# Or manually
litellm --model huggingface/Qwen/Qwen2.5-7B-Instruct --port 4000
```

## Project Structure

```
rllm-071/
├── rllm/                          # Main package
│   ├── agents/                   # Agent base classes and implementations
│   ├── environments/             # Environment base classes and implementations
│   ├── sdk/                      # Trace collection SDK
│   │   ├── chat/                # Chat client implementations
│   │   ├── session/             # Session backends (contextvar, opentelemetry)
│   │   ├── proxy/               # LiteLLM proxy integration
│   │   ├── tracers/             # Tracer implementations
│   │   └── store/               # Storage backends
│   ├── workflows/                # Workflow implementations
│   ├── engine/                   # Execution engine
│   ├── trainer/                  # Training implementations
│   │   ├── tinker/              # Tinker backend integration
│   │   └── verl/                # Verl backend integration
│   ├── rewards/                  # Reward functions
│   ├── tools/                    # Tool implementations
│   ├── parser/                   # Tool parsers
│   ├── data/                     # Data processing
│   └── experimental/             # Experimental features
│       ├── unified_trainer.py    # New unified trainer
│       ├── cli/                  # CLI implementation
│       └── eval/                 # Evaluation protocol
├── rllm-model-gateway/           # Model gateway package
├── tests/                        # Test suite
├── examples/                     # Example implementations
├── cookbooks/                    # Working examples
├── docs/                         # Documentation source
└── scripts/                      # Utility scripts
```

## Key Design Patterns

### 1. Session-Based Tracing

The SDK uses context variables for automatic trace propagation:

```python
from rllm.sdk import session, get_chat_client

llm = get_chat_client(api_key="sk-...")

with session(experiment="v1", task="math") as sess:
    response = llm.chat.completions.create(...)
    # All traces automatically captured with metadata
```

### 2. Trajectory Decorator

Wrap agent functions to automatically collect trajectories:

```python
from rllm.sdk import trajectory

@trajectory(name="solver")
def solve(task: Task, config: AgentConfig) -> Episode:
    # Each LLM call becomes a step
    response = client.chat.completions.create(...)
    return Episode(trajectories=[Trajectory(name="solver", steps=[])])
```

### 3. Unified Trainer

The new `UnifiedTrainer` supports multiple backends:

```python
from rllm.experimental.unified_trainer import AgentTrainer

trainer = AgentTrainer(
    backend="tinker",  # or "verl"
    agent_flow=solve,
    evaluator=score,
    config=config,
    train_dataset=dataset,
)
trainer.train()
```

### 4. Agent-Environment Pattern

For traditional RL training:

```python
engine = AgentExecutionEngine(
    agent_class=YourAgent,
    env_class=YourEnv,
    n_parallel_agents=64,
    # ... other config
)
results = asyncio.run(engine.execute_tasks(tasks))
```

## Configuration

### SDK Configuration (`rllm/sdk/config.yaml`)

```yaml
session_backend: "contextvar"  # or "opentelemetry"
```

### Trainer Configuration

rLLM uses Hydra for configuration management. Default configs are in `rllm/trainer/config/`.

```python
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer")
def main(config):
    trainer = AgentTrainer(...)
    trainer.train()
```

## Testing Guidelines

### Test Structure

Tests are organized by module in `tests/`:
- `tests/agents/`: Agent tests
- `tests/envs/`: Environment tests
- `tests/rewards/`: Reward function tests
- `tests/sdk/`: SDK tests
- `tests/trainer/`: Training tests
- `tests/unified_trainer/`: Unified trainer tests

### Test Fixtures

`tests/conftest.py` provides:
- Stub modules for heavy dependencies (torch, transformers, ray)
- Mock objects for API clients
- Shared test utilities

### Running Specific Tests

```bash
# Test SDK functionality
pytest tests/sdk/ -v

# Test unified trainer
pytest tests/unified_trainer/ -v

# Test with coverage
pytest tests/ --cov=rllm --cov-report=html
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Sessions

```python
from rllm.sdk import get_current_session, get_current_metadata

session = get_current_session()
if session:
    print(f"Session: {session.name}")
    print(f"Traces: {len(session.llm_calls)}")
    print(f"Metadata: {get_current_metadata()}")
```

### Check Gateway Logs

The model gateway logs traces to SQLite by default. Check:
- Gateway logs for request/response details
- SQLite database at `gateway_store.sqlite`

## Common Issues

### Import Errors

Ensure the package is installed in editable mode:
```bash
uv pip install -e .
```

### Missing Token IDs/Logprobs

Verify:
1. Model gateway is running
2. `base_url` points to the gateway
3. Gateway can reach the inference server

### Training Failures

Check:
1. GPU availability (for verl backend)
2. Configuration YAML syntax
3. Dataset format matches expectations

## Documentation Resources

- **Main docs**: `docs/` directory, built with MkDocs Material
- **SDK docs**: `rllm/sdk/README.md`
- **Gateway docs**: `rllm-model-gateway/README.md`
- **Examples**: `examples/` and `cookbooks/` directories
- **API reference**: Auto-generated from docstrings via mkdocstrings

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Run `pre-commit run` to format code
4. Run `pytest tests/` to verify tests pass
5. Submit a pull request

See `docs/contributing.md` for detailed contribution guidelines.
