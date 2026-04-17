python eval_baseline.py \
    --vllm-url "http://80.48.5.52:30000" \
    --kernelgym-url "http://80.48.5.52:8002" \
    --data "/home/g00841271/datasets/level_1-00000-of-00001.parquet" \
    --max-turns 5 \
    --max-tasks 100 \
    --max-tokens 16384 \