# APT Improved Tracking Error Detection

This project uses the Python package manager `uv`.



# Startup

Allocate interactive node:

`salloc --partition=jjs533-interactive,jjs533,gpu --mem=60g --gres=gpu:nvidia_h100_nvl:1 --ntasks=8 --time=48:00:00`

Setup Python environment:

`source .venv/bin/activate`

Start the vLLM server (do this in a separate tmux session):

`bash vllm.bash`


# Dev setup on Unicorn

`tmux`

`salloc --partition=jjs533-interactive,jjs533,gpu --mem=60g --gres=gpu:nvidia_rtx_6000_ada_generation:2 --ntasks=8 --time=48:00:00`

`mkdir -p /tmp/tmux-1640796 && chmod 700 /tmp/tmux-1640796`

`tmux`

`source .venv/bin/activate`

`bash vllm.bash`



# Example workflow:

`python make_subclips.py --video data/fly_bubble/VNC2_Line1_RigC_20230621T102506/movie.ufmf --mat data/fly_bubble/VNC2_Line1_RigC_20230621T102506/flytracker_trx.mat --black-bg`

`python main.py --clips-dir subclips --port PORT --output vnc_line1.out`








