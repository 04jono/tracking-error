# APT Improved Tracking Error Detection

This project uses the Python package manager `uv`.



# Startup

Allocate interactive node:

`salloc --partition=jjs533-interactive,jjs533,gpu --mem=60g --gres=gpu:nvidia_h100_nvl:1 --ntasks=8 --time=48:00:00`

Setup Python environment:

`source .venv/bin/activate`

Start the vLLM server (do this in a separate tmux session):

`bash vllm.bash`



# Example workflow:

`python make_subclips.py --video data/fly_bubble/cx_GMR_SS00030_CsChr_RigC_20150826T144616/movie.ufmf --mat data/fly_bubble/cx_GMR_SS00030_CsChr_RigC_20150826T144616/ctrax_results.mat --black-bg`



