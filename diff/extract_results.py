#!/usr/bin/env python3
"""Extract ERROR (and optionally UNKNOWN) entries from a results .out file,
enriched with frame range, timestamps, quadrant, and crop from manifest.json.

Usage:
    python extract_results.py vnc_line1.out
    python extract_results.py vnc_line1.out --manifest subclips/manifest.json
    python extract_results.py vnc_line1.out --include-unknown
    python extract_results.py vnc_line1.out --summary-only
    python extract_results.py vnc_line1.out -o errors.out
    python extract_results.py vnc_line1.out --copy-to error_clips/
"""

import argparse
import shutil
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_parsing import (
    QUADRANT_NAMES, fmt_time, parse_entries, load_manifest,
    enrich, entry_frame_span, format_entry, autodetect_manifest,
    load_inputs_env,
)


def main():
    load_inputs_env()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=os.environ.get("VLM_RESULTS"),
                        help="Results .out file to parse. Default: $VLM_RESULTS from inputs.env")
    parser.add_argument("--manifest", default=os.environ.get("MANIFEST"),
                        help="Path to manifest.json (default: auto-detect)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write extracted entries to this file (default: stdout)")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Also include UNKNOWN verdicts")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print header + location info only, no body text")
    parser.add_argument("--copy-to", default=None, metavar="DIR",
                        help="Copy error clip videos (overlay, raw, black_bg) into DIR")
    parser.add_argument("--clips-dir", default=os.environ.get("CLIPS_DIR"),
                        help="Base clips directory for --copy-to. Default: $CLIPS_DIR from inputs.env")
    args = parser.parse_args()

    if not args.input:
        parser.error("input is required (or set VLM_RESULTS in inputs.env)")

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = autodetect_manifest(args.input)

    manifest = load_manifest(manifest_path)
    if manifest:
        print(f"Using manifest: {manifest_path}", file=sys.stderr)

    entries = parse_entries(args.input)

    target_verdicts = {"ERROR"}
    if args.include_unknown:
        target_verdicts.add("UNKNOWN")

    errors = [e for e in entries if e["verdict"] in target_verdicts]

    total     = len(entries)
    n_correct = sum(1 for e in entries if e["verdict"] == "CORRECT")
    n_error   = sum(1 for e in entries if e["verdict"] == "ERROR")
    n_unknown = sum(1 for e in entries if e["verdict"] == "UNKNOWN")

    out_lines = []
    out_lines.append(
        f"# {Path(args.input).name}  —  "
        f"{total} entries: {n_correct} CORRECT  {n_error} ERROR  {n_unknown} UNKNOWN"
    )
    out_lines.append(
        f"# Showing {len(errors)} {'entries' if len(errors) != 1 else 'entry'} "
        f"({', '.join(sorted(target_verdicts))})"
    )
    out_lines.append("")

    for e in errors:
        meta = enrich(e, manifest)
        out_lines.append(format_entry(e, meta, args.summary_only))
        out_lines.append("")

    output = "\n".join(out_lines)

    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"Wrote {len(errors)} entries to {args.output}", file=sys.stderr)
    else:
        print(output)

    if args.copy_to:
        clips_base = Path(args.clips_dir) if args.clips_dir else (
            Path(manifest_path).parent if manifest_path else Path("subclips")
        )
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for e in errors:
            clip = e["clip"]
            clip_idx = clip.split("_")[1]
            candidates = [
                ("overlay",   clips_base / "overlay"   / f"{clip}.mp4"),
                ("raw",       clips_base / "raw"       / f"{clip}.mp4"),
                ("black_bg",  clips_base / "black_bg"  / f"{clip}.mp4"),
            ]
            for subdir, src in candidates:
                if src.exists():
                    out_subdir = dest / subdir
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, out_subdir / src.name)
                    copied += 1
                else:
                    print(f"  missing: {src}", file=sys.stderr)
                    skipped += 1

        print(f"Copied {copied} files to {dest}/  ({skipped} missing)", file=sys.stderr)

        spans_lines = []
        for e in errors:
            f0, f1, crop, quadrant = entry_frame_span(e, manifest)
            t0 = enrich(e, manifest).get("start_time_s")
            t1 = enrich(e, manifest).get("end_time_s")
            t0_str = fmt_time(t0) if t0 is not None else e["t0"]
            t1_str = fmt_time(t1) if t1 is not None else e["t1"]
            spans_lines.append(
                f"{e['clip']}  frames {f0}-{f1}  {t0_str} – {t1_str}  {quadrant}"
            )

        (dest / "frame_spans.out").write_text("\n".join(spans_lines) + "\n")
        print(f"Written {dest}/frame_spans.out", file=sys.stderr)


if __name__ == "__main__":
    main()
