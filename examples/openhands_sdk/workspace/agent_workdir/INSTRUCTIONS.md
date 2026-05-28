# Current Task

This file is overwritten by the host for every rollout.

The generated task will include:

- operator name
- target Ascend architecture
- full PyTorch reference code
- required implementation path
- validation command

The agent should implement `ModelNew` in the required implementation file and
run `bash tools/operator_pipeline.sh --op_name <op_name>` until `metrics.json`
is produced.
