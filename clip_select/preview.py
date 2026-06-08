"""
Generate MP4 preview clips for human review of scored candidates.

Uses imageio_ffmpeg (bundled binary, libx264) — same method as reconstruction.py.
Viewable directly in VS Code.

Usage:
    python -m clip_select.preview --type ego-motion --ego4d-root /path/to/Ego4D/v2
    python -m clip_select.preview --type all --ego4d-root ...

Outputs (under <base_dir>/<type>/preview/):
    <video_uid>_<start>-<end>s[_role].mp4  – 10s clip at 480x270
    index.html                              – HTML table with inline images
"""
import argparse
import csv
import os

import cv2
import imageio_ffmpeg

from clip_select.common import iter_frames

# Default top-K by type
DEFAULT_TOPK = {
    'ego-motion': 10,
    'low-texture': 30,
    'reflective': 30,
    'dynamic': 10,
    'control': 10,
}

CLIP_W = 480
CLIP_H = 270
CLIP_FPS = 6.0


# ── Candidate loading ─────────────────────────────────────────────────────────

def load_top_unique_videos(csv_path: str, topk: int) -> list[dict]:
    """Top-K candidates, one clip per video (highest score), unique videos only."""
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    best_per_video: dict = {}
    for row in rows:
        uid = row['video_uid']
        score = float(row.get('score', 0))
        if uid not in best_per_video or score > float(best_per_video[uid].get('score', 0)):
            best_per_video[uid] = row

    unique = sorted(best_per_video.values(),
                    key=lambda r: float(r.get('score', 0)), reverse=True)
    return unique[:topk]


def load_ego_motion_pairs(csv_path: str, topk: int) -> list[tuple]:
    """
    Group by video_uid, compute within-video contrast = max_score - min_score.
    Returns top-K videos sorted by contrast (desc).
    Each entry: (contrast, stress_row, control_row) from the SAME video.
    """
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_video: dict = {}
    for row in rows:
        by_video.setdefault(row['video_uid'], []).append(row)

    pairs = []
    for uid, vid_rows in by_video.items():
        sorted_rows = sorted(vid_rows, key=lambda r: float(r.get('score', 0)))
        if len(sorted_rows) < 2:
            continue
        min_row = sorted_rows[0]
        max_row = sorted_rows[-1]
        contrast = float(max_row['score']) - float(min_row['score'])
        pairs.append((contrast, max_row, min_row))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:topk]


# ── MP4 generation ────────────────────────────────────────────────────────────

def generate_clip(row: dict, video_dir: str, out_dir: str) -> dict | None:
    uid = row['video_uid']
    start = float(row['start_sec'])
    end = float(row['end_sec'])
    score = float(row.get('score', 0))
    role = row.get('role', '')

    video_path = os.path.join(video_dir, uid + '.mp4')
    if not os.path.exists(video_path):
        print(f"  SKIP {uid[:16]}: video not found")
        return None

    role_tag = f'_{role}' if role else ''
    fname = f"{uid}_{start:.0f}-{end:.0f}s{role_tag}.mp4"
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  SKIP (exists) {fname}")
        return {'video_uid': uid, 'start_sec': start, 'end_sec': end,
                'score': score, 'role': role, 'clip_file': fname}

    frames_bgr = [f for _, f in iter_frames(
        video_path, target_fps=CLIP_FPS,
        resize=(CLIP_W, CLIP_H),
        start_sec=start, end_sec=end,
    )]
    if not frames_bgr:
        print(f"  SKIP {uid[:16]}: no frames extracted")
        return None

    # Label overlay on first frame
    label = f"{uid[:12]} {start:.0f}-{end:.0f}s score={score:.3f} {role}"
    frames_bgr[0] = frames_bgr[0].copy()
    cv2.putText(frames_bgr[0], label, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Write MP4 via imageio_ffmpeg (bundled libx264, same as reconstruction.py)
    writer = imageio_ffmpeg.write_frames(
        out_path, size=(CLIP_W, CLIP_H), fps=CLIP_FPS,
        codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
    )
    writer.send(None)
    for frame in frames_bgr:
        writer.send(frame.tobytes())
    writer.close()

    print(f"  → {fname}  ({len(frames_bgr)} frames)")
    return {'video_uid': uid, 'start_sec': start, 'end_sec': end,
            'score': score, 'role': role, 'clip_file': fname}


# ── HTML index ────────────────────────────────────────────────────────────────

def write_index(entries: list[dict], out_dir: str, clip_type: str):
    html_path = os.path.join(out_dir, 'index.html')
    rows_html = ''
    for e in entries:
        if e is None:
            continue
        rows_html += (
            f'<tr>'
            f'<td>{e["video_uid"][:20]}</td>'
            f'<td>{e["start_sec"]:.0f}s – {e["end_sec"]:.0f}s</td>'
            f'<td>{e["score"]:.3f}</td>'
            f'<td>{e.get("role", "")}</td>'
            f'<td><video src="{e["clip_file"]}" width="480" height="270" controls autoplay loop muted></video></td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Preview: {clip_type}</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 10px; border: 1px solid #444; vertical-align: middle; }}
  th {{ background: #333; }}
  img {{ display: block; }}
  tr:hover {{ background: #1a1a1a; }}
</style>
</head>
<body>
<h2>Candidates: {clip_type} ({len([e for e in entries if e])} clips)</h2>
<table>
<tr><th>video_uid</th><th>window</th><th>score</th><th>role</th><th>preview</th></tr>
{rows_html}
</table>
</body></html>
"""
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Index: {html_path}")


# ── Per-type runner ───────────────────────────────────────────────────────────

ALL_TYPES = ['ego-motion', 'low-texture', 'reflective', 'dynamic', 'control']


def run_preview_for_type(clip_type: str, video_dir: str, base_dir: str, topk: int):
    in_dir = os.path.join(base_dir, clip_type)
    out_dir = os.path.join(in_dir, 'preview')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(in_dir, 'candidates.csv')
    print(f"\n[preview.py] type={clip_type}, topk={topk}")
    print(f"             input={csv_path}")
    print(f"             output={out_dir}")

    if not os.path.exists(csv_path):
        print(f"  SKIP: candidates.csv not found")
        return

    entries = []

    if clip_type == 'ego-motion':
        pairs = load_ego_motion_pairs(csv_path, topk)
        print(f"  {len(pairs)} videos with stress/control pairs (sorted by contrast)")
        for contrast, stress_row, control_row in pairs:
            stress_row['role'] = 'stress'
            control_row['role'] = 'control'
            for row in (stress_row, control_row):
                entries.append(generate_clip(row, video_dir, out_dir))
    else:
        candidates = load_top_unique_videos(csv_path, topk)
        print(f"  {len(candidates)} candidates (1 clip per video)")
        for row in candidates:
            entries.append(generate_clip(row, video_dir, out_dir))

    write_index([e for e in entries if e], out_dir, clip_type)
    n_ok = sum(1 for e in entries if e is not None)
    print(f"  Done. {n_ok} GIFs → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--type', required=True,
                   choices=ALL_TYPES + ['all'],
                   help='"all" generates previews for all 5 types at once')
    p.add_argument('--ego4d-root', required=True)
    p.add_argument('--input-dir',
                   help='Base dir containing <type>/candidates.csv (default: select_output)')
    p.add_argument('--topk', type=int, default=None,
                   help='Number of top candidates per type (type-specific default if unset)')
    args = p.parse_args()

    video_dir = os.path.join(args.ego4d_root, 'full_scale')
    base_dir = args.input_dir or 'select_output'
    active_types = ALL_TYPES if args.type == 'all' else [args.type]

    for clip_type in active_types:
        topk = args.topk if args.topk is not None else DEFAULT_TOPK[clip_type]
        run_preview_for_type(clip_type, video_dir, base_dir, topk)


if __name__ == '__main__':
    main()
