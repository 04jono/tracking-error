# APT Improved Tracking Error Detection

This project uses the Python package manager `uv`.



# Startup

Setup Python environment:

`source .venv/bin/activate`

Allocate interactive node:

`salloc --partition=jjs533-interactive,jjs533,gpu --mem=60g --gres=gpu:nvidia_h100_nvl:1 --ntasks=8 --time=48:00:00`

Start the vLLM server (do this in a separate tmux session):

`bash vllm.bash`



