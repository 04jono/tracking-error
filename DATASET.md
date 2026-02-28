# fly_bubble APT .trk File Structure

## File Format
- MATLAB v7.3 mat-file (HDF5 format)
- Requires `h5py` to read (NOT `scipy.io.loadmat`)

## Top-Level Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `pTrk` | (n_flies, 1) | HDF5 references to tracking data arrays |
| `pTrkConf` | (n_flies, 1) | HDF5 references to confidence scores |
| `pTrkFrm` | (n_frames, 1) | Frame numbers (1-indexed, MATLAB convention) |
| `pTrkiTgt` | (n_flies, 1) | Target IDs mapping to Ctrax fly indices (1-indexed) |
| `pTrkTS` | (n_flies, 1) | Timestamps |
| `pTrkTag` | (n_flies, 1) | Tags/labels |
| `startframes` | (n_flies, 1) | Start frame per target |
| `endframes` | (n_flies, 1) | End frame per target |
| `trkInfo` | Group | Metadata (crop_loc, model_file, etc.) |

## Data Structure

### pTrk Organization
- Contains **n_flies separate arrays**, NOT a single multi-dimensional array
- `pTrk[i]` is an HDF5 reference that must be dereferenced to access data
- Each dereferenced array has shape: **(n_frames, 2, n_keypoints)**

### Array Dimensions
```
Shape: (n_frames, 2, n_keypoints)
       │          │  │
       │          │  └─ Keypoint index (0 to n_keypoints-1)
       │          └──── Coordinate type: [0=x, 1=y]
       └─────────────── Frame number (0-indexed after loading)
```

### Accessing Coordinates
- `data[frame_num, 0, :]` → All x-coordinates for all keypoints at given frame
- `data[frame_num, 1, :]` → All y-coordinates for all keypoints at given frame
- `data[frame_num, 0, keypoint_idx]` → X-coordinate of specific keypoint
- `data[frame_num, 1, keypoint_idx]` → Y-coordinate of specific keypoint

## Coordinate System
- Origin: Top-left corner (standard image coordinates)
- Units: Pixels
- X-axis: Left to right (horizontal)
- Y-axis: Top to bottom (vertical)

## Target Mapping
- `pTrkiTgt[i]` = Ctrax fly ID (1-indexed)
- `pTrk[i]` corresponds to Ctrax fly `i` (0-indexed in Python, `i+1` in MATLAB)

## Common Dataset Parameters
- **Frames**: ~50,000
- **Flies**: 10
- **Keypoints**: 21 per fly
- **Image size**: 1024×1024 pixels
