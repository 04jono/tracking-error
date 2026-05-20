#!/usr/bin/env python3
"""End-to-end pipeline: subclips → VLM inference → evaluation.

Reads paths from inputs.env. Steps:
  1. make_subclips.py  — skipped if manifest.json already exists in CLIPS_DIR
  2. main.py           — VLM inference on the subclips
  3. compare_to_vlm.py — precision/recall evaluation against tracking diff

Usage:
    python pipeline.py --start-server            # start vllm_qwen.bash, run all steps
    python pipeline.py --port 8080               # use already-running vllm server
    python pipeline.py --start-server --skip-subclips
    python pipeline.py --port 8080 --skip-inference   # re-evaluate existing results
    python pipeline.py --start-server --n 50
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

_vllm_pid = None  # set after server starts so the signal handler can kill it


def _shutdown(signum, frame):
    if _vllm_pid is not None:
        print(f"\nInterrupted — stopping vLLM server (pid {_vllm_pid})...", file=sys.stderr)
        try:
            os.killpg(_vllm_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(1)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def _load_inputs_env():
    p = Path.cwd()
    for _ in range(4):
        candidate = p / "inputs.env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return str(candidate)
        p = p.parent
    return None


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(f"\nERROR: '{label}' exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n[{label} done in {elapsed:.1f}s]")


def start_vllm_server(script="vllm.bash", port_file="vllm.port", pid_file="vllm.pid",
                      startup_timeout=600, poll_interval=5):
    """
    Launch vllm.bash detached, wait for it to be healthy, and return the port.

    vllm.bash writes its PID to vllm.pid and its port to vllm.port before
    starting the server. We poll /health until the server responds.
    """
    port_path = Path(port_file)
    pid_path  = Path(pid_file)
    port_path.unlink(missing_ok=True)

    global _vllm_pid
    print(f"Starting vLLM server (detached) via {script}...")
    proc = subprocess.Popen(
        ["bash", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,   # detach from this process group
    )
    _vllm_pid = proc.pid
    print(f"  bash pid: {proc.pid}  (server pid will be in {pid_file})")

    # Wait for vllm.bash to write the port
    print(f"  Waiting for {port_file}...", end="", flush=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if port_path.exists() and port_path.read_text().strip():
            break
        print(".", end="", flush=True)
        time.sleep(2)
    else:
        sys.exit(f"\nERROR: vllm.bash did not write {port_file} within 2 minutes")

    port = int(port_path.read_text().strip())
    print(f"\n  Port: {port}")

    # Poll /health until the model is loaded
    url = f"http://localhost:{port}/health"
    print(f"  Waiting for {url} (up to {startup_timeout}s)...", end="", flush=True)
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    else:
        sys.exit(f"\nERROR: vLLM server did not become healthy within {startup_timeout}s")

    print(f"\n  vLLM server ready on port {port}")
    return port


def main():
    env_path = _load_inputs_env()
    if env_path:
        print(f"Loaded {env_path}")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    server_group = parser.add_mutually_exclusive_group()
    server_group.add_argument("--start-server", action="store_true",
                              help="Launch vllm script detached and wait for it to be ready")
    parser.add_argument("--vllm-script", default="vllm_qwen.bash",
                        help="vLLM launch script to use (default: vllm_qwen.bash)")
    server_group.add_argument("--port", type=int,
                              help="Port of an already-running vLLM server")

    # Step control
    parser.add_argument("--skip-subclips", action="store_true",
                        help="Skip make_subclips.py even if manifest is missing")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip VLM inference (re-evaluate existing results)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation (just run subclips + inference)")

    # Forwarded to make_subclips.py
    parser.add_argument("--n", type=int, default=None,
                        help="Number of clips to produce (make_subclips.py --n)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg workers for make_subclips.py (default: 4)")

    # Forwarded to main.py
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent VLM requests (default: 4)")
    parser.add_argument("--log", default=None,
                        help="Mirror VLM inference stdout to this log file")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Gemma thinking mode")
    parser.add_argument("--few-shot", action="store_true",
                        help="Prepend in-context examples from assets/examples/")

    # Forwarded to compare_to_vlm.py
    parser.add_argument("--include-unknown", action="store_true",
                        help="Treat UNKNOWN verdicts as errors in evaluation")
    parser.add_argument("--report", default=None,
                        help="Save evaluation report to this file (also printed to stdout)")

    # Rules
    parser.add_argument("--skip-rules", action="store_true",
                        help="Skip rule-based error detection step")
    parser.add_argument("--rule-events", default="rule_events.tsv",
                        help="Output TSV for rule events (default: rule_events.tsv)")
    parser.add_argument("--hybrid-report", default=None,
                        help="Save hybrid (VLM OR rule) evaluation report (default: hybrid_<report>)")

    args = parser.parse_args()

    if not args.start_server and args.port is None and not args.skip_inference:
        parser.error("either --start-server or --port is required (or use --skip-inference)")

    # Resolve paths from env
    clips_dir   = os.environ.get("CLIPS_DIR", "subclips")
    vlm_results = os.environ.get("VLM_RESULTS", "results.out")
    video       = os.environ.get("VIDEO")
    mat         = os.environ.get("MAT")
    trk         = os.environ.get("TRK")
    flytracker  = os.environ.get("FLYTRACKER")
    fixed       = os.environ.get("FIXED")

    py = sys.executable

    # ── Step 1: Subclips ──────────────────────────────────────────────────────
    manifest_path = Path(clips_dir) / "manifest.json"
    if not args.skip_subclips and manifest_path.exists():
        print(f"Skipping subclips — manifest already exists: {manifest_path}")
        args.skip_subclips = True

    if not args.skip_subclips:
        if not video:
            sys.exit("ERROR: VIDEO not set in inputs.env")
        cmd = [py, "make_subclips.py", "--output-dir", clips_dir, "--black-bg"]
        if mat:
            cmd += ["--mat", mat]
        if trk:
            cmd += ["--trk", trk]
        if args.n is not None:
            cmd += ["--n", str(args.n)]
        cmd += ["--workers", str(args.workers)]
        run(cmd, "make_subclips")

    # ── Start server (if requested) ───────────────────────────────────────────
    port = args.port
    if args.start_server and not args.skip_inference:
        port = start_vllm_server(script=args.vllm_script)

    # ── Step 2: VLM inference ─────────────────────────────────────────────────
    if not args.skip_inference:
        cmd = [py, "main.py",
               "--port", str(port),
               "--clips-dir", clips_dir,
               "--output", vlm_results,
               "--concurrency", str(args.concurrency)]
        if args.log:
            cmd += ["--log", args.log]
        if args.thinking:
            cmd += ["--thinking"]
        if args.few_shot:
            cmd += ["--few-shot"]
        run(cmd, "VLM inference")

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    if not args.skip_eval:
        if not flytracker or not fixed:
            print("Skipping evaluation — FLYTRACKER or FIXED not set in inputs.env")
        elif not Path(vlm_results).exists():
            print(f"Skipping evaluation — results file not found: {vlm_results}")
        else:
            cmd = [py, "diff/compare_to_vlm.py", flytracker, fixed, vlm_results]
            if args.include_unknown:
                cmd.append("--include-unknown")
            if args.report:
                cmd += ["--report", args.report]
            run(cmd, "evaluation")

    # ── Step 4: Rule-based detection + hybrid evaluation ─────────────────────
    if not args.skip_rules:
        if not flytracker:
            print("Skipping rules — FLYTRACKER not set in inputs.env")
        else:
            hybrid_report = args.hybrid_report
            if hybrid_report is None and args.report:
                p = Path(args.report)
                hybrid_report = str(p.parent / f"hybrid_{p.name}")

            cmd = [py, "rules/run_rules.py",
                   "--tracking", flytracker,
                   "-o", args.rule_events]
            if fixed and Path(vlm_results).exists():
                cmd += ["--vlm-results", vlm_results,
                        "--gt-flytracker", flytracker,
                        "--gt-fixed", fixed]
                if args.include_unknown:
                    cmd.append("--include-unknown")
                if hybrid_report:
                    cmd += ["--report", hybrid_report]
            run(cmd, "rule-based detection")

    # ── Shut down server ──────────────────────────────────────────────────────
    if _vllm_pid is not None:
        print(f"Stopping vLLM server (pid {_vllm_pid})...")
        try:
            os.killpg(_vllm_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
