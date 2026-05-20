"""Shared utilities for parsing VLM results .out files and manifest.json."""

import json
import os
import re
import sys
from pathlib import Path


def load_inputs_env(filename="inputs.env"):
    """Load inputs.env from CWD or up to 4 parent directories. CLI args override."""
    from dotenv import load_dotenv
    p = Path.cwd()
    for _ in range(4):
        candidate = p / filename
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return str(candidate)
        p = p.parent
    return None

HEADER_RE = re.compile(
    r"^\[(?P<verdict>CORRECT|ERROR|UNKNOWN)\s*\|\s*"
    r"(?P<clip>clip_\d+_tile_\d+_\d+)\s*\|\s*"
    r"frames\s+(?P<f0>\d+)-(?P<f1>\d+)\s*\|\s*"
    r"(?P<t0>\d{2}:\d{2}:\d{2})-(?P<t1>\d{2}:\d{2}:\d{2})\]$"
)

QUADRANT_NAMES = {
    (0, 0): "top-left",
    (0, 1): "top-right",
    (1, 0): "bottom-left",
    (1, 1): "bottom-right",
}


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def parse_entries(path):
    """Parse a VLM results .out file into a list of entry dicts."""
    entries = []
    current = None
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.rstrip()
        m = HEADER_RE.match(line)
        if m:
            if current is not None:
                current["body"] = current["body"].strip()
                entries.append(current)
            current = {**m.groupdict(), "body": ""}
        elif current is not None:
            current["body"] += line + "\n"
    if current is not None:
        current["body"] = current["body"].strip()
        entries.append(current)
    return entries


def load_manifest(manifest_path):
    """Load manifest.json, returning {} if not found."""
    if manifest_path is None:
        return {}
    p = Path(manifest_path)
    if not p.exists():
        print(f"Warning: manifest not found at {p}", file=sys.stderr)
        return {}
    return json.loads(p.read_text())


def enrich(entry, manifest):
    """Return manifest metadata for a result entry, or {} if not found."""
    key = f"overlay/{entry['clip']}.mp4"
    return manifest.get(key, {})


def entry_frame_span(entry, manifest):
    """Return (f0, f1, crop, quadrant) for an entry, enriched from manifest."""
    meta = enrich(entry, manifest)
    f0 = meta.get("start_frame", int(entry["f0"]))
    f1 = meta.get("end_frame", int(entry["f1"]))
    crop = meta.get("crop", {})
    row = meta.get("tile", {}).get("row")
    col = meta.get("tile", {}).get("col")
    quadrant = QUADRANT_NAMES.get((row, col), f"row{row}_col{col}") if row is not None else "unknown"
    return f0, f1, crop, quadrant


def format_entry(e, meta, summary_only):
    """Format a result entry as a human-readable string."""
    row = meta.get("tile", {}).get("row")
    col = meta.get("tile", {}).get("col")
    quadrant = QUADRANT_NAMES.get((row, col), f"row{row}_col{col}") if row is not None else "unknown"
    crop = meta.get("crop", {})

    f0 = meta.get("start_frame", e["f0"])
    f1 = meta.get("end_frame",   e["f1"])
    t0 = meta.get("start_time_s")
    t1 = meta.get("end_time_s")
    t0_str = fmt_time(t0) if t0 is not None else e["t0"]
    t1_str = fmt_time(t1) if t1 is not None else e["t1"]

    lines = [f"[{e['verdict']} | {e['clip']} | frames {f0}-{f1} | {t0_str} – {t1_str}]"]
    detail = (
        f"  quadrant : {quadrant}  (row {row}, col {col})\n"
        f"  frames   : {f0} – {f1}\n"
        f"  time     : {t0_str} – {t1_str}"
    )
    if crop:
        detail += f"\n  crop     : x={crop['x']}  y={crop['y']}  w={crop['w']}  h={crop['h']}"
    lines.append(detail)

    if not summary_only:
        lines.append(e["body"])

    return "\n".join(lines)


def autodetect_manifest(input_path, subclips_dir="subclips"):
    """Try to find manifest.json next to input or in subclips/."""
    candidates = [
        Path(input_path).parent / "manifest.json",
        Path(subclips_dir) / "manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
