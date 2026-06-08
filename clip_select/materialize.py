"""
Extract frames for human-selected clips after preview review.

Reads a selection CSV (video_uid, start_sec, end_sec, type, [role]) and
extracts frames via ffmpeg to disk for VGGT-Omega inference.

Selection CSV format (one row per clip you want to keep):
    video_uid,start_sec,end_sec,type,role
    abc123...,120,130,ego-motion,stress
    abc123...,450,460,ego-motion,control
    ...

Usage:
    python -m clip_select.materialize --selection /path/to/selected.csv \
        --ego4d-root /path/to/Ego4D/v2 \
        --output-dir /path/to/frames_out \
        --fps 6

Outputs (under --output-dir/<type>/<video_uid>_<start>-<end>s/):
    frame_000001.jpg, frame_000002.jpg, ...   (at --fps)
    clip_info.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys

from clip_select.common import video_duration


def extract_clip_frames(video_path: str, out_dir: str, start_sec: float,
                        end_sec: float, fps: float, image_resolution: int):
    """Extract frames for one clip window to out_dir at given fps."""
    os.makedirs(out_dir, exist_ok=True)
    duration = end_sec - start_sec

    scale_filter = (f'fps={fps},'
                    f'scale={image_resolution}:{image_resolution}:'
                    f'force_original_aspect_ratio=decrease:'
                    f'flags=lanczos')
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'warning',
        '-ss', str(start_sec),
        '-i', video_path,
        '-t', str(duration),
        '-vf', scale_filter,
        '-q:v', '2',
        os.path.join(out_dir, 'frame_%06d.jpg'),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")

    n_frames = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    return n_frames


def write_clip_info(out_dir: str, row: dict, n_frames: int, fps: float):
    info = {
        'video_uid': row['video_uid'],
        'start_sec': float(row['start_sec']),
        'end_sec': float(row['end_sec']),
        'type': row.get('type', ''),
        'role': row.get('role', ''),
        'fps': fps,
        'n_frames': n_frames,
    }
    with open(os.path.join(out_dir, 'clip_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def load_selection(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', required=True,
                   help='CSV with selected clips (video_uid, start_sec, end_sec, type, role)')
    p.add_argument('--ego4d-root', required=True,
                   help='Ego4D v2 root containing full_scale/')
    p.add_argument('--output-dir', default='select_output/frames',
                   help='Where to extract frames (default: select_output/frames)')
    p.add_argument('--fps', type=float, default=6.0,
                   help='Frame rate for extraction (default: 6.0)')
    p.add_argument('--image-resolution', type=int, default=512,
                   help='Max side length for frame resize (default: 512)')
    args = p.parse_args()

    video_dir = os.path.join(args.ego4d_root, 'full_scale')
    rows = load_selection(args.selection)
    print(f"[materialize.py] {len(rows)} clips to extract")
    print(f"                 fps={args.fps}, resolution={args.image_resolution}")
    print(f"                 output={args.output_dir}")

    n_ok = n_skip = 0
    for i, row in enumerate(rows):
        uid = row['video_uid']
        start = float(row['start_sec'])
        end = float(row['end_sec'])
        clip_type = row.get('type', 'unknown')
        role = row.get('role', '')

        video_path = os.path.join(video_dir, uid + '.mp4')
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{len(rows)}] SKIP {uid[:16]}: video not found")
            n_skip += 1
            continue

        role_tag = f'_{role}' if role else ''
        out_subdir = os.path.join(args.output_dir, clip_type,
                                  f'{uid}_{start:.0f}-{end:.0f}s{role_tag}')

        # Skip if already extracted
        if os.path.exists(os.path.join(out_subdir, 'clip_info.json')):
            print(f"  [{i+1}/{len(rows)}] SKIP {uid[:16]}: already done")
            n_ok += 1
            continue

        print(f"  [{i+1}/{len(rows)}] {uid[:16]}  {start:.0f}s-{end:.0f}s  {clip_type}/{role}", end=' ', flush=True)
        try:
            n_frames = extract_clip_frames(
                video_path, out_subdir, start, end,
                args.fps, args.image_resolution,
            )
            write_clip_info(out_subdir, row, n_frames, args.fps)
            print(f"→ {n_frames} frames")
            n_ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            n_skip += 1

    print(f"\nDone. extracted={n_ok} skipped={n_skip}")
    print(f"Frames dir: {args.output_dir}")


if __name__ == '__main__':
    main()
