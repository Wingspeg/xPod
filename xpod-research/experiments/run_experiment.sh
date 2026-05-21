#!/bin/bash
MODE=${1:-advanced}
LIMIT=${2:-100}
RUN_ID=$(date +%Y%m%d_%H%M%S)

python3 -m experiments.replay.replay_workload \
  --input datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv \
  --mode $MODE \
  --run-id $RUN_ID \
  --limit $LIMIT \
  --data-nodes "data-node-1,data-node-2" \
  --gpu-nodes "gpu-node-1" \
  --cpu-nodes "cpu-node-1,cpu-node-2" \
  --apply

sleep 60

python3 -m experiments.collect.collect_metrics --run-id $RUN_ID
python3 -m experiments.collect.summarize --run-id $RUN_ID
python3 -m experiments.plot.plot_results --run-id $RUN_ID

echo "实验完成，run_id: $RUN_ID"
echo "结果目录: results/raw/$RUN_ID/"
echo "汇总目录: results/summary/$RUN_ID/"
echo "图表目录: results/figures/$RUN_ID/"
