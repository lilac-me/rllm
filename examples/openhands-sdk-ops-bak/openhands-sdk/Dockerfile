# Dockerfile for the rllm OpenHands sandbox worker (NPU operator generation).
#
# This image runs the rllm worker_server.py and the OpenHands Python app
# (openhands_agent.py) inside the outer rllm sandbox container.
#
# OpenHands itself will spawn a SEPARATE per-rollout Docker container
# (the "runtime" container) for isolated code execution.  For this to work,
# /var/run/docker.sock must be mounted into this container at runtime:
#
#   docker run --rm \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     rllm-openhands-sandbox ...
#
# The SandboxOrchestrator handles this via SandboxConfig.extra["docker_volumes"].
#
# Build:
#   docker build -t rllm-openhands-sandbox -f openhands-sdk/Dockerfile .
#
# The orchestrator will:
#   1. Upload agent code to /app/agent
#   2. Upload worker_server.py to /app/runner
#   3. Install rllm[sdk] if configured
#   4. Start worker_server.py as the agent process
#
FROM python:3.11-slim

# System dependencies:
#   - git: for repo cloning (if needed by tasks)
#   - curl: health-check probes inside the container
#   - build-essential: C extensions for Python packages, and g++ for mock compilation checks
#   - docker.io: Docker CLI (for DooD — OpenHands spawns runtime containers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# Python packages:
#   - openhands-ai: the OpenHands app runs IN THIS CONTAINER in-process.
#                   It manages the runtime container via /var/run/docker.sock.
#   - aiohttp: required by worker_server.py
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
        "openhands-sdk==1.15.0"
        aiohttp \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# Working directory for agent code (uploaded by orchestrator)
WORKDIR /app/agent

# Default command — overridden by orchestrator's start_agent_process
CMD ["bash"]
