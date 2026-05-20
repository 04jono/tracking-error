#!/usr/bin/env python3
"""Compare tracking diff (flytracker vs fixed) against VLM-detected errors.

Evaluates every clip span in the VLM results file against the ground truth
derived from the tracking diff (flytracker vs fixed). Reports a full confusion
matrix and precision/recall/F1 at the clip-span level.

  TP: VLM=ERROR  and tracking diff exists in that span/quadrant
  FP: VLM=ERROR  but no tracking diff
  FN: VLM=CORRECT/UNKNOWN  but tracking diff exists
  TN: VLM=CORRECT/UNKNOWN  and no tracking diff

Usage:
    python compare_to_vlm.py flytracker_trx.mat fixed_trx.mat results.out
    python compare_to_vlm.py flytracker_trx.mat fixed_trx.mat results.out \\
        --manifest subclips/manifest.json --include-unknown
    python compare_to_vlm.py flytracker_trx.mat fixed_trx.mat results.out \\
        --diffs-tsv diffs.out   # skip recompute if already have TSV
"""

import argparse
import contextlib
import csv
import io
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_body_tracking import (
    diff_tracking, write_output,
    DIST_THRESHOLD, ANGLE_THRESHOLD, ID_SWITCH_DIST_THRESHOLD, JITTER_JUMP_THRESHOLD,
)
from result_parsing import (
    parse_entries, load_manifest, entry_frame_span, autodetect_manifest,
    load_inputs_env,
)


def load_diffs_tsv(path):
    """Load a pre-computed diffs TSV (produced by diff_body_tracking --output)."""
    diffs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            def _f(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            diffs.append({
                "frame":     int(row["frame"]),
                "fly":       int(row["fly"]),
                "type":      row["type"],
                "x":         _f(row.get("x")),
                "y":         _f(row.get("y")),
                "dist":      _f(row.get("dist")),
                "angle_deg": _f(row.get("angle_deg")),
            })
    return diffs


def episode_hits_span(episode, f0, f1, crop):
    """True if any frame of an episode falls within the span and crop region."""
    if episode["f1"] < f0 or episode["f0"] > f1:
        return False
    if crop:
        x, y = episode.get("x"), episode.get("y")
        if x is not None and y is not None:
            if not (crop["x"] <= x <= crop["x"] + crop["w"] and
                    crop["y"] <= y <= crop["y"] + crop["h"]):
                return False
    return True


def cluster_diffs(diffs, gap=10):
    """
    Cluster per-fly diff events into continuous error episodes.

    Groups events by fly, merging frames within `gap` of each other into one
    episode. Returns a list of episode dicts with f0, f1, fly, types, n_frames.
    """
    from collections import defaultdict
    by_fly = defaultdict(list)
    for d in diffs:
        by_fly[d["fly"]].append(d)

    episodes = []
    for fly_id, fly_diffs in by_fly.items():
        fly_diffs = sorted(fly_diffs, key=lambda d: d["frame"])
        run = [fly_diffs[0]]
        for d in fly_diffs[1:]:
            if d["frame"] <= run[-1]["frame"] + gap:
                run.append(d)
            else:
                episodes.append(run)
                run = [d]
        episodes.append(run)

    result = []
    for run in episodes:
        # Use the median event's position as a representative location
        mid = run[len(run) // 2]
        result.append({
            "fly":      run[0]["fly"],
            "f0":       run[0]["frame"],
            "f1":       run[-1]["frame"],
            "n_frames": len(run),
            "types":    sorted(set(d["type"] for d in run)),
            "x":        mid.get("x"),
            "y":        mid.get("y"),
        })
    return sorted(result, key=lambda e: (e["f0"], e["fly"]))


def type_summary(episodes):
    counts = Counter(t for ep in episodes for t in ep["types"])
    return "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def compare(all_entries, manifest, diffs, target_verdicts, episode_gap=10):
    """
    Evaluate all VLM clip spans against tracking diff episodes.

    Returns:
        spans: list of dicts — one per entry, with vlm_positive, has_diff,
               and matching_episodes (list of clustered error episodes)
        uncovered_episodes: episodes not overlapping any evaluated clip span
    """
    episodes = cluster_diffs(diffs, gap=episode_gap)

    spans = []
    for e in all_entries:
        f0, f1, crop, quadrant = entry_frame_span(e, manifest)
        spans.append({
            "entry":             e,
            "f0":                f0,
            "f1":                f1,
            "crop":              crop,
            "quadrant":          quadrant,
            "vlm_positive":      e["verdict"] in target_verdicts,
            "matching_episodes": [],
        })

    covered = set()
    for span in spans:
        for idx, ep in enumerate(episodes):
            if episode_hits_span(ep, span["f0"], span["f1"], span["crop"]):
                span["matching_episodes"].append(ep)
                covered.add(idx)

    for span in spans:
        span["has_diff"] = bool(span["matching_episodes"])

    uncovered_episodes = [ep for i, ep in enumerate(episodes) if i not in covered]
    return spans, uncovered_episodes


def print_report(spans, uncovered_episodes):
    tp = [s for s in spans if     s["vlm_positive"] and     s["has_diff"]]
    fp = [s for s in spans if     s["vlm_positive"] and not s["has_diff"]]
    fn = [s for s in spans if not s["vlm_positive"] and     s["has_diff"]]
    tn = [s for s in spans if not s["vlm_positive"] and not s["has_diff"]]

    n_total   = len(spans)
    n_correct = sum(1 for s in spans if s["entry"]["verdict"] == "CORRECT")
    n_error   = sum(1 for s in spans if s["entry"]["verdict"] == "ERROR")
    n_unknown = sum(1 for s in spans if s["entry"]["verdict"] == "UNKNOWN")

    print(f"VLM results: {n_total} total  {n_correct} CORRECT  {n_error} ERROR  {n_unknown} UNKNOWN")
    print()

    def ep_str(ep):
        return f"fly {ep['fly']} frames {ep['f0']}-{ep['f1']} ({ep['n_frames']}fr) [{','.join(ep['types'])}]"

    print(f"=== TRUE POSITIVES — VLM error confirmed by tracking diff  ({len(tp)}) ===")
    for s in tp:
        e = s["entry"]
        eps = s["matching_episodes"]
        print(f"  {e['clip']}  frames {s['f0']}-{s['f1']}  {s['quadrant']}  →  {len(eps)} episode(s)")
        for ep in eps:
            print(f"    {ep_str(ep)}")
    print()

    print(f"=== FALSE POSITIVES — VLM flagged error, no tracking diff  ({len(fp)}) ===")
    for s in fp:
        e = s["entry"]
        print(f"  {e['clip']}  frames {s['f0']}-{s['f1']}  {s['quadrant']}")
    print()

    print(f"=== FALSE NEGATIVES — VLM missed, tracking diff exists  ({len(fn)}) ===")
    for s in fn:
        e = s["entry"]
        eps = s["matching_episodes"]
        print(f"  {e['clip']}  frames {s['f0']}-{s['f1']}  {s['quadrant']}  verdict={e['verdict']}  →  {len(eps)} episode(s)")
        for ep in eps:
            print(f"    {ep_str(ep)}")
    print()

    print(f"=== TRUE NEGATIVES — VLM correct, no tracking diff  ({len(tn)}) ===")
    print()

    if uncovered_episodes:
        print(f"=== Tracking episodes outside any VLM-evaluated span  ({len(uncovered_episodes)}) ===")
        for ep in uncovered_episodes:
            print(f"  {ep_str(ep)}")
        print()

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall    = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (len(tp) + len(tn)) / n_total if n_total else 0.0

    print("=== Metrics (clip-span level) ===")
    print(f"  TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  TN={len(tn)}")
    print(f"  Precision : {precision:.3f}   (of VLM errors flagged, fraction that are real)")
    print(f"  Recall    : {recall:.3f}   (of real errors, fraction caught by VLM)")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")
    if uncovered_episodes:
        print(f"  Note: {len(uncovered_episodes)} episode(s) outside evaluated clip spans (not counted in FN)")


def main():
    env_path = load_inputs_env()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("flytracker", nargs="?", default=os.environ.get("FLYTRACKER"),
                        help="Raw flytracker_trx.mat (unfixed). Default: $FLYTRACKER from inputs.env")
    parser.add_argument("fixed", nargs="?", default=os.environ.get("FIXED"),
                        help="Human-fixed fixed_trx.mat. Default: $FIXED from inputs.env")
    parser.add_argument("vlm_results", nargs="?", default=os.environ.get("VLM_RESULTS"),
                        help="VLM results .out file. Default: $VLM_RESULTS from inputs.env")
    parser.add_argument("--manifest", default=os.environ.get("MANIFEST"),
                        help="Path to manifest.json (default: auto-detect)")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Treat UNKNOWN verdicts as positive predictions (errors)")
    parser.add_argument("--diffs-tsv", default=None, metavar="TSV",
                        help="Use a pre-computed diff TSV instead of recomputing from .mat files")
    parser.add_argument("--save-diffs", default=None, metavar="FILE",
                        help="Save computed tracking diffs to a TSV for later reuse")
    parser.add_argument("--threshold", type=float, default=DIST_THRESHOLD,
                        help=f"Position diff threshold in pixels (default: {DIST_THRESHOLD})")
    parser.add_argument("--angle-threshold", type=float, default=ANGLE_THRESHOLD,
                        help=f"Orientation diff threshold in degrees (default: {ANGLE_THRESHOLD})")
    parser.add_argument("--id-switch-threshold", type=float, default=ID_SWITCH_DIST_THRESHOLD,
                        help=f"ID switch distance threshold (default: {ID_SWITCH_DIST_THRESHOLD})")
    parser.add_argument("--jitter-threshold", type=float, default=JITTER_JUMP_THRESHOLD,
                        help=f"Jitter jump threshold in pixels (default: {JITTER_JUMP_THRESHOLD})")
    parser.add_argument("--report", default=os.environ.get("REPORT"),
                        help="Save evaluation report to this file (also printed to stdout)")
    args = parser.parse_args()

    if not args.vlm_results:
        parser.error("vlm_results is required (or set VLM_RESULTS in inputs.env)")

    # Load tracking diffs
    if args.diffs_tsv:
        print(f"Loading pre-computed diffs from {args.diffs_tsv}...", file=sys.stderr)
        diffs = load_diffs_tsv(args.diffs_tsv)
        print(f"  {len(diffs)} diff events loaded", file=sys.stderr)
    else:
        if not args.flytracker or not args.fixed:
            parser.error("flytracker and fixed positional arguments are required unless --diffs-tsv is given")
        print("Computing tracking diff...", file=sys.stderr)
        diffs = diff_tracking(
            args.flytracker, args.fixed,
            dist_threshold=args.threshold,
            angle_threshold_deg=args.angle_threshold,
            id_switch_dist_threshold=args.id_switch_threshold,
            jitter_jump_threshold=args.jitter_threshold,
        )
        print(f"  {len(diffs)} diff events total", file=sys.stderr)
        if args.save_diffs:
            write_output(diffs, args.save_diffs)

    # Load VLM results
    manifest_path = args.manifest or autodetect_manifest(args.vlm_results)
    manifest = load_manifest(manifest_path)
    if manifest:
        print(f"Using manifest: {manifest_path}", file=sys.stderr)

    all_entries = parse_entries(args.vlm_results)
    target_verdicts = {"ERROR"}
    if args.include_unknown:
        target_verdicts.add("UNKNOWN")

    spans, uncovered_diffs = compare(all_entries, manifest, diffs, target_verdicts)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(spans, uncovered_diffs)
    report_text = buf.getvalue()
    print(report_text, end="")

    if args.report:
        Path(args.report).write_text(report_text)
        print(f"Report written to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
