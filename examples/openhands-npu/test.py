# /home/g00841271/rllm-071/examples/openhands-sdk-ops/openhands-sdk/mock_npu_data.parquet

import pandas as pd
file_path = "/home/g00841271/rllm-071/examples/openhands-sdk-ops/openhands-sdk/mock_npu_data.parquet"
data = pd.read_parquet(file_path)

print(data)