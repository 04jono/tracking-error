#!/bin/bash

# Navigate to the working directory
WORKING_DIR=/home/jqc3/tracking-error
cd $WORKING_DIR

# Load inputs.env if present
if [ -f "$WORKING_DIR/inputs.env" ]; then
    set -o allexport
    source "$WORKING_DIR/inputs.env"
    set +o allexport
fi

# Set up python environment
source .venv/bin/activate

date

# Source: https://veldaio.substack.com/p/running-vllm-on-slurm-clusters-a

# Get the node name where this job is running
NODE_NAME=$(hostname)
# Find a random free port
export PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Determine tensor parallelism from allocated GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    N_GPUS=$SLURM_GPUS_ON_NODE
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi
N_GPUS=${N_GPUS:-1}

echo "=========================================="
echo "vLLM Server Starting"
echo "=========================================="
echo "Node: $NODE_NAME"
echo "Port: $PORT"
echo "Endpoint: http://${NODE_NAME}:${PORT}"
echo "Tensor parallelism: $N_GPUS GPU(s)"
echo "=========================================="
echo ""

# Write port to file so pipeline.py can discover it
echo $PORT > $WORKING_DIR/vllm.port
echo $$ > $WORKING_DIR/vllm.pid

# Start the vLLM server

vllm serve ${MODEL:-/share/j_sun/jqc3/Qwen3.6-27B} \
  --gpu-memory-utilization 0.75 \
  --tensor-parallel-size $N_GPUS \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"video": 12}' \
  --allowed-local-media-path $WORKING_DIR \
  --port $PORT \
  2>&1 | tee $WORKING_DIR/vllm.log
