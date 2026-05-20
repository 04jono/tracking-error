#!/usr/bin/env python3
"""
Apply rule-based tracking error detectors to a tracking file.

Each rule in this directory exposes a detect(TH, X, Y) function that scans
the heading/position arrays frame-by-frame and returns a list of event dicts.

Rules available:
  orientation_flip  — flag frame-to-frame ~180° heading reversals

Outputs a TSV of flagged events (same schema as diff/diff_body_tracking.py output)
so results can be fed into compare_to_vlm.py via --diffs-tsv.

Optional hybrid evaluation: combine rule events with VLM results and measure
precision/recall against a ground-truth diff TSV or tracking .mat pair.

Usage:
    python rules/run_rules.py --tracking flytracker_trx.mat
    python rules/run_rules.py --tracking flytracker_trx.mat -o rule_events.tsv
    python rules/run_rules.py --tracking flytracker_trx.mat \\
        --vlm-results results.out \\
        --gt-flytracker flytracker_trx.mat --gt-fixed fixed_trx.mat \\
        --report hybrid_report.txt
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

# Allow importing from both rules/ and diff/
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "diff"))
sys.path.insert(0, str(_ROOT / "rules"))

from diff_body_tracking import (
    load_trx, write_output,
    DIST_THRESHOLD, ANGLE_THRESHOLD, ID_SWITCH_DIST_THRESHOLD, JITTER_JUMP_THRESHOLD,
)
from result_parsing import (
    parse_entries, load_manifest, entry_frame_span, autodetect_manifest, load_inputs_env,
)
from compare_to_vlm import (
    cluster_diffs, episode_hits_span, load_diffs_tsv, print_report, compare,
)

import heading_bias as rule_heading_bias


ALL_RULES = [
    ("heading_bias", rule_heading_bias),
]


def run_rules(X, Y, TH,
              threshold_deg=rule_heading_bias.HEADING_BIAS_THRESHOLD_DEG,
              min_velocity=rule_heading_bias.MIN_VELOCITY_PX,
              min_consecutive=rule_heading_bias.MIN_CONSECUTIVE):
    """Apply all rules to the tracking arrays. Returns a combined list of events."""
    events = []
    for name, module in ALL_RULES:
        result = module.detect(TH, X=X, Y=Y, threshold_deg=threshold_deg,
                               min_velocity=min_velocity,
                               min_consecutive=min_consecutive)
        print(f"  {name}: {len(result)} events", file=sys.stderr)
        events.extend(result)
    return events


def print_rules_summary(events):
    from collections import Counter
    counts = Counter(e["type"] for e in events)
    print(f"Total rule events: {len(events)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


def hybrid_compare(all_entries, manifest, rule_events, gt_diffs, target_verdicts, episode_gap=10):
    """
    Evaluate hybrid (VLM OR rule) against ground-truth diff episodes.

    A span is considered positive if VLM flagged it OR a rule event falls in the span.
    Returns spans list (same structure as compare_to_vlm.compare) with hybrid_positive set.
    """
    gt_episodes = cluster_diffs(gt_diffs, gap=episode_gap)
    rule_episodes = cluster_diffs(rule_events, gap=episode_gap)

    spans = []
    for e in all_entries:
        f0, f1, crop, quadrant = entry_frame_span(e, manifest)
        vlm_pos = e["verdict"] in target_verdicts
        rule_pos = any(episode_hits_span(ep, f0, f1, crop) for ep in rule_episodes)
        hybrid_pos = vlm_pos or rule_pos
        spans.append({
            "entry":             e,
            "f0":                f0,
            "f1":                f1,
            "crop":              crop,
            "quadrant":          quadrant,
            "vlm_positive":      hybrid_pos,  # print_report reads this field
            "rule_positive":     rule_pos,
            "matching_episodes": [],
        })

    covered = set()
    for span in spans:
        for idx, ep in enumerate(gt_episodes):
            if episode_hits_span(ep, span["f0"], span["f1"], span["crop"]):
                span["matching_episodes"].append(ep)
                covered.add(idx)
    for span in spans:
        span["has_diff"] = bool(span["matching_episodes"])

    uncovered = [ep for i, ep in enumerate(gt_episodes) if i not in covered]
    return spans, uncovered


def main():
    load_inputs_env()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tracking", default=os.environ.get("FLYTRACKER"),
                        help="Tracking .mat file to scan (default: $FLYTRACKER from inputs.env)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write rule events to this TSV file (default: stdout)")
    parser.add_argument("--threshold", type=float,
                        default=rule_heading_bias.HEADING_BIAS_THRESHOLD_DEG,
                        help=f"Heading-velocity misalignment threshold in degrees (default: {rule_heading_bias.HEADING_BIAS_THRESHOLD_DEG})")
    parser.add_argument("--min-velocity", type=float,
                        default=rule_heading_bias.MIN_VELOCITY_PX,
                        help=f"Minimum px/frame speed to use velocity signal (default: {rule_heading_bias.MIN_VELOCITY_PX})")
    parser.add_argument("--min-consecutive", type=int,
                        default=rule_heading_bias.MIN_CONSECUTIVE,
                        help=f"Minimum consecutive anti-aligned frames to emit (default: {rule_heading_bias.MIN_CONSECUTIVE})")

    eval_group = parser.add_argument_group("hybrid evaluation (all three required)")
    eval_group.add_argument("--vlm-results", default=os.environ.get("VLM_RESULTS"),
                            help="VLM results .out file")
    eval_group.add_argument("--gt-flytracker", default=os.environ.get("FLYTRACKER"),
                            help="Ground-truth raw flytracker .mat (default: $FLYTRACKER)")
    eval_group.add_argument("--gt-fixed", default=os.environ.get("FIXED"),
                            help="Ground-truth human-fixed .mat (default: $FIXED)")
    eval_group.add_argument("--gt-diffs-tsv", default=None,
                            help="Pre-computed ground-truth diffs TSV (skips recomputing from .mat)")
    eval_group.add_argument("--manifest", default=os.environ.get("MANIFEST"),
                            help="Path to manifest.json (default: auto-detect)")
    eval_group.add_argument("--include-unknown", action="store_true",
                            help="Treat UNKNOWN VLM verdicts as positive")
    eval_group.add_argument("--report", default=None,
                            help="Write hybrid evaluation report to this file")

    args = parser.parse_args()

    if not args.tracking:
        parser.error("--tracking is required (or set FLYTRACKER in inputs.env)")

    # Load tracking and run rules
    print(f"Loading tracking from {args.tracking}...", file=sys.stderr)
    X, Y, TH = load_trx(args.tracking)
    print(f"  {TH.shape[1]} flies, {TH.shape[0]} frames", file=sys.stderr)

    print("Applying rules...", file=sys.stderr)
    events = run_rules(X, Y, TH, threshold_deg=args.threshold,
                       min_velocity=args.min_velocity,
                       min_consecutive=args.min_consecutive)

    # Output events TSV
    if args.output:
        write_output(events, args.output)
        print(f"Rule events written to {args.output}", file=sys.stderr)
    else:
        print_rules_summary(events)
        if not args.vlm_results:
            # Print TSV to stdout if no eval requested
            print("frame\tfly\ttype\tx\ty\tdist\tangle_deg")
            for e in sorted(events, key=lambda x: (x["frame"], x["fly"])):
                x = e["x"] if e.get("x") is not None else "nan"
                y = e["y"] if e.get("y") is not None else "nan"
                dist = e.get("dist", "nan")
                if dist is None:
                    dist = "nan"
                angle = e.get("angle_deg", "nan")
                print(f"{e['frame']}\t{e['fly']}\t{e['type']}\t{x}\t{y}\t{dist}\t{angle}")

    # Hybrid evaluation
    if not args.vlm_results:
        return

    # Load ground-truth diffs
    if args.gt_diffs_tsv:
        print(f"Loading ground-truth diffs from {args.gt_diffs_tsv}...", file=sys.stderr)
        gt_diffs = load_diffs_tsv(args.gt_diffs_tsv)
    elif args.gt_flytracker and args.gt_fixed:
        from diff_body_tracking import diff_tracking
        print("Computing ground-truth diff...", file=sys.stderr)
        gt_diffs = diff_tracking(args.gt_flytracker, args.gt_fixed)
        print(f"  {len(gt_diffs)} diff events", file=sys.stderr)
    else:
        print("Skipping hybrid evaluation — no ground-truth source provided.", file=sys.stderr)
        return

    manifest_path = args.manifest or autodetect_manifest(args.vlm_results)
    manifest = load_manifest(manifest_path)
    if manifest:
        print(f"Using manifest: {manifest_path}", file=sys.stderr)

    all_entries = parse_entries(args.vlm_results)
    target_verdicts = {"ERROR"}
    if args.include_unknown:
        target_verdicts.add("UNKNOWN")

    # VLM-only baseline
    vlm_spans, vlm_uncovered = compare(all_entries, manifest, gt_diffs, target_verdicts)

    # Hybrid: VLM OR rule
    hybrid_spans, hybrid_uncovered = hybrid_compare(
        all_entries, manifest, events, gt_diffs, target_verdicts
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("=" * 60)
        print("VLM-ONLY BASELINE")
        print("=" * 60)
        print_report(vlm_spans, vlm_uncovered)
        print()
        print("=" * 60)
        print("HYBRID (VLM OR RULE)")
        print("=" * 60)
        print_report(hybrid_spans, hybrid_uncovered)

    report_text = buf.getvalue()
    print(report_text, end="")

    if args.report:
        Path(args.report).write_text(report_text)
        print(f"Report written to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
