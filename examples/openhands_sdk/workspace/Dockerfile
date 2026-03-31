# =============================================================================
# Custom OpenHands container for rllm training
#
# Extends the official OpenHands image with the rllm entrypoint script
# (workspace/entrypoint.py) that uses the new OpenHands Python SDK:
#   LLM  →  Agent (AgentContext / Skill)  →  Conversation
#
# No inner Docker sandbox is created — this container is the execution
# environment. The agent uses TerminalTool and FileEditorTool directly.
#
# Build (from repo root):
#   docker build -t rllm-openhands \
#       -f examples/openhands-sdk/workspace/Dockerfile \
#       examples/openhands-sdk/workspace
#
# Run (example):
#   docker run --rm \
#     -e LLM_BASE_URL="http://host.docker.internal:4000/meta/rllm1:.../v1" \
#     -e LLM_MODEL="openai/openhands-model" \
#     -e LLM_API_KEY="EMPTY" \
#     -e TASK_INSTRUCTION="Fix the failing test in src/utils.py" \
#     -e WORKSPACE_BASE=/opt/workspace \
#     -e MAX_ITERATIONS=30 \
#     -v /path/to/repo:/opt/workspace \
#     --add-host host.docker.internal:host-gateway \
#     rllm-openhands
# =============================================================================

FROM ghcr.io/all-hands-ai/openhands:0.28

# ---------------------------------------------------------------------------
# Copy rllm entrypoint
# ---------------------------------------------------------------------------
COPY entrypoint.py /app/rllm_entrypoint.py

# ---------------------------------------------------------------------------
# Default workspace (overridden at runtime via WORKSPACE_BASE env var)
# ---------------------------------------------------------------------------
ENV WORKSPACE_BASE=/opt/workspace
RUN mkdir -p /opt/workspace

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "/app/rllm_entrypoint.py"]
