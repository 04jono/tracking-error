"""
Rule: detect sustained heading-velocity anti-alignment (head-tail orientation flip).

A fly's heading should roughly align with its direction of motion. When heading
is consistently anti-aligned (> 90° from velocity direction) across several
consecutive moving frames, this indicates the tracker has the head and tail
swapped for that stretch.

Only fires when the fly is moving (velocity >= MIN_VELOCITY_PX), so stationary
periods produce no signal.
"""

import numpy as np

HEADING_BIAS_THRESHOLD_DEG = 90.0  # misalignment above which a frame is anti-aligned
MIN_VELOCITY_PX = 2.0              # minimum px/frame to use velocity as a signal
MIN_CONSECUTIVE = 3                # minimum anti-aligned moving frames to emit events
FRAME_GAP = 5                      # max frame gap to bridge when grouping runs


def _angle_diff(a, b):
    """Absolute angular difference, wrapped to [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def detect(TH, X, Y, threshold_deg=HEADING_BIAS_THRESHOLD_DEG,
           min_velocity=MIN_VELOCITY_PX, min_consecutive=MIN_CONSECUTIVE):
    """
    Scan heading angles and positions for sustained heading-velocity anti-alignment.

    Args:
        TH: (n_frames, n_flies) heading angles in radians
        X:  (n_frames, n_flies) x positions
        Y:  (n_frames, n_flies) y positions
        threshold_deg:   misalignment in degrees above which a frame is anti-aligned
        min_velocity:    minimum px/frame speed to use velocity direction signal
        min_consecutive: minimum consecutive anti-aligned frames before emitting events

    Returns:
        List of event dicts {frame, fly, type, angle_deg, x, y}.
        frame is 1-indexed.
    """
    if X is None or Y is None:
        return []

    threshold_rad = np.radians(threshold_deg)
    n_frames, n_flies = TH.shape
    events = []

    for i in range(n_flies):
        # Collect (frame_1idx, misalignment_deg, x, y) for anti-aligned moving frames
        candidates = []
        for f in range(1, n_frames):
            if (np.isnan(TH[f, i]) or
                    np.isnan(X[f, i]) or np.isnan(Y[f, i]) or
                    np.isnan(X[f - 1, i]) or np.isnan(Y[f - 1, i])):
                continue
            dx = X[f, i] - X[f - 1, i]
            dy = Y[f, i] - Y[f - 1, i]
            if np.hypot(dx, dy) < min_velocity:
                continue
            vel_dir = np.arctan2(dy, dx)
            diff = float(_angle_diff(TH[f, i], vel_dir))
            if diff > threshold_rad:
                candidates.append((f + 1, float(np.degrees(diff)),
                                   float(X[f, i]), float(Y[f, i])))

        if not candidates:
            continue

        # Group into runs, bridging gaps <= FRAME_GAP (stationary periods)
        runs = [[candidates[0]]]
        for item in candidates[1:]:
            if item[0] - runs[-1][-1][0] <= FRAME_GAP:
                runs[-1].append(item)
            else:
                runs.append([item])

        for run in runs:
            if len(run) >= min_consecutive:
                for frame, angle_deg, x, y in run:
                    events.append({
                        "frame":     frame,
                        "fly":       i,
                        "type":      "heading_bias",
                        "angle_deg": angle_deg,
                        "x":         x,
                        "y":         y,
                    })

    return events
