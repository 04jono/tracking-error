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


# Pipeline (`pipeline.py`)

Runs the full end-to-end pipeline: subclip generation → VLM inference → evaluation → rule-based detection. Paths are read from `inputs.env`.

```bash
python pipeline.py --start-server            # start vllm server, run all steps
python pipeline.py --port 8080               # use already-running server
python pipeline.py --start-server --few-shot --report report.txt
python pipeline.py --port 8080 --skip-inference  # re-evaluate existing results
```

### Server
| Flag | Description |
|---|---|
| `--start-server` | Launch vLLM script and wait for it to be ready |
| `--port PORT` | Connect to an already-running vLLM server |
| `--vllm-script SCRIPT` | vLLM launch script to use (default: `vllm_qwen.bash`) |

### Step control
| Flag | Description |
|---|---|
| `--skip-subclips` | Skip clip generation even if manifest is missing |
| `--skip-inference` | Skip VLM inference, re-evaluate existing results only |
| `--skip-eval` | Skip VLM evaluation |
| `--skip-rules` | Skip rule-based detection step |

### Subclip generation
| Flag | Description |
|---|---|
| `--n N` | Number of clips to produce |
| `--workers N` | Parallel ffmpeg workers (default: 4) |

### VLM inference
| Flag | Description |
|---|---|
| `--concurrency N` | Max concurrent VLM requests (default: 4) |
| `--log FILE` | Mirror VLM inference stdout to a log file |
| `--thinking` | Enable thinking/reasoning mode |
| `--few-shot` | Prepend in-context examples from `assets/examples/` |

### Evaluation
| Flag | Description |
|---|---|
| `--include-unknown` | Treat UNKNOWN verdicts as errors |
| `--report FILE` | Save evaluation report to file |

### Rule-based detection
| Flag | Description |
|---|---|
| `--rule-events FILE` | Output TSV for rule events (default: `rule_events.tsv`) |
| `--hybrid-report FILE` | Save hybrid report (default: `hybrid_<report>` if `--report` is set) |








