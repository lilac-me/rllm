python -m vllm.entrypoints.openai.api_server \
    --model /home/g00841271/Qwen3-8B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1
