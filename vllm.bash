#!/bin/bash

# Navigate to the working directory
WORKING_DIR=/home/jqc3/tracking-error
cd $WORKING_DIR
    
# Set up python environment
source .venv/bin/activate

date

# Source: https://veldaio.substack.com/p/running-vllm-on-slurm-clusters-a

# Get the node name where this job is running
NODE_NAME=$(hostname)
# Find a random free port
export PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "=========================================="
echo "vLLM Server Starting"
echo "=========================================="
echo "Node: $NODE_NAME"
echo "Port: $PORT"
echo "Endpoint: http://${NODE_NAME}:${PORT}"
echo "=========================================="
echo ""

# Start the vLLM server

vllm serve /share/j_sun/jqc3/Qwen3.5-27B-FP8 \
  --gpu-memory-utilization 0.75 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"video": 4}' \
  --allowed-local-media-path $WORKING_DIR \
  --port $PORT \
  2>&1 | tee $WORKING_DIR/vllm.log
