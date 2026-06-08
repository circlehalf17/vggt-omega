"""
Score Ego4D video windows for 5 stress/control types and emit candidate CSVs.

Usage:
    python -m clip_select.score --type ego-motion --ego4d-root /path/to/Ego4D/v2
    python -m clip_select.score --type low-texture --ego4d-root ...
    python -m clip_select.score --type reflective  --ego4d-root ...
    python -m clip_select.score --type dynamic     --ego4d-root ...
    python -m clip_select.score --type control     --ego4d-root ...

Outputs (under --output-dir, default ./select_output/<type>/):
    candidates.csv   – one row per scored window
    manifest.json    – run metadata + counts

Disk policy: NO frames are extracted to disk. Scoring is done entirely via
ffmpeg pipe at 1 fps (subsampled). Only the CSV is written.
"""
import argparse
import os
import sys

import numpy as np

from clip_select.common import (
    CandidateWriter,
    DYNAMIC_NOUNS,
    NarrationIndex,
    REFLECTIVE_NOUNS,
    dense_flow,
    flow_stats,
    group_windows,
    highlight_ratio,
    load_ego4d_meta,
    stream_video,
    texture_signals,
    video_duration,
    write_manifest,
)

import cv2

# ── CSV schemas ───────────────────────────────────────────────────────────────

MOTION_FIELDS = [
    'video_uid', 'start_sec', 'end_sec',
    'cam_mag_mean', 'cam_mag_max', 'cam_mag_std',
    'res_mag_mean', 'score', 'pair_id', 'role',
]

TEXTURE_FIELDS = [
    'video_uid', 'start_sec', 'end_sec',
    'kp_mean', 'lap_var_mean', 'brightness_mean', 'score',
]

REFLECTIVE_FIELDS = [
    'video_uid', 'start_sec', 'end_sec',
    'highlight_mean', 'narration_hit', 'score',
]

DYNAMIC_FIELDS = [
    'video_uid', 'start_sec', 'end_sec',
    'res_mag_mean', 'res_mag_max', 'narration_hit', 'score',
]

CONTROL_FIELDS = [
    'video_uid', 'start_sec', 'end_sec',
    'cam_mag_mean', 'kp_mean', 'lap_var_mean', 'brightness_mean',
    'highlight_mean', 'score',
]

# ── Window-level scorers ──────────────────────────────────────────────────────

def _score_window_frames(frames: list) -> dict:
    """Compute flow and texture signals over a list of BGR frames."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    cam_mags, res_mags = [], []
    for i in range(1, len(grays)):
        flow = dense_flow(grays[i - 1], grays[i])
        cam, res = flow_stats(flow)
        cam_mags.append(cam)
        res_mags.append(res)

    kp_counts, lap_vars, brightnesses, hl_ratios = [], [], [], []
    for f in frames:
        sig = texture_signals(f)
        kp_counts.append(sig['kp_count'])
        lap_vars.append(sig['laplacian_var'])
        brightnesses.append(sig['brightness'])
        hl_ratios.append(highlight_ratio(f))

    return dict(
        cam_mag_mean=float(np.mean(cam_mags)) if cam_mags else 0.0,
        cam_mag_max=float(np.max(cam_mags)) if cam_mags else 0.0,
        cam_mag_std=float(np.std(cam_mags)) if cam_mags else 0.0,
        res_mag_mean=float(np.mean(res_mags)) if res_mags else 0.0,
        res_mag_max=float(np.max(res_mags)) if res_mags else 0.0,
        kp_mean=float(np.mean(kp_counts)),
        lap_var_mean=float(np.mean(lap_vars)),
        brightness_mean=float(np.mean(brightnesses)),
        highlight_mean=float(np.mean(hl_ratios)),
    )


def score_ego_motion(signals: dict) -> float:
    return signals['cam_mag_mean']


def score_low_texture(signals: dict) -> float:
    # Low score = low texture. Negate so higher score = worse for reconstruction.
    kp = signals['kp_mean']
    lap = signals['lap_var_mean']
    # Normalize roughly: kp in [0,500], lap in [0, ~2000]
    kp_norm = kp / 500.0
    lap_norm = lap / 2000.0
    return 1.0 - (0.5 * kp_norm + 0.5 * lap_norm)


def score_reflective(signals: dict, narration_hit: bool) -> float:
    hl = signals['highlight_mean']
    nar = 1.0 if narration_hit else 0.0
    return 0.7 * hl + 0.3 * nar


def score_dynamic(signals: dict, narration_hit: bool) -> float:
    res = signals['res_mag_mean'] / 5.0  # normalize: 5px/frame ≈ high
    nar = 1.0 if narration_hit else 0.0
    return min(1.0, 0.7 * res + 0.3 * nar)


def score_control(signals: dict) -> float:
    # Composite: stable camera + rich texture + bright + not reflective
    cam = signals['cam_mag_mean']
    kp = signals['kp_mean']
    lap = signals['lap_var_mean']
    brightness = signals['brightness_mean']
    hl = signals['highlight_mean']

    motion_ok = 1.0 - min(1.0, cam / 3.0)        # cam < 3px/frame → good
    texture_ok = min(1.0, kp / 200.0)             # kp > 200 → good
    sharp_ok = min(1.0, lap / 500.0)              # lap_var > 500 → sharp
    bright_ok = 1.0 if 80 < brightness < 200 else 0.5
    no_hl = 1.0 - min(1.0, hl / 0.05)            # hl < 5% → good

    return (motion_ok * 0.35 + texture_ok * 0.25 +
            sharp_ok * 0.2 + bright_ok * 0.1 + no_hl * 0.1)


# ── Per-video processing ──────────────────────────────────────────────────────

def process_video_motion(video_uid: str, video_path: str,
                         writer: CandidateWriter, window_sec: float = 10.0):
    """Score windows, emit stress+control pair annotations within same video."""
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return 0

    windows = group_windows(ts, frames, window_sec)
    window_scores = []

    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        s = score_ego_motion(sig)
        window_scores.append((start, end, sig, s))

    if len(window_scores) < 2:
        return 0

    scores_only = [s for _, _, _, s in window_scores]
    max_s = max(scores_only) if scores_only else 1.0
    min_s = min(scores_only) if scores_only else 0.0

    # Within-video pair: top-3 stress, bottom-3 control
    # Only emit pairs if contrast ratio ≥ 3x
    contrast = max_s / max(min_s, 0.5)
    if contrast < 3.0:
        return 0

    sorted_by_score = sorted(window_scores, key=lambda x: x[3])
    stress_windows = sorted_by_score[-3:]   # top 3
    control_windows = sorted_by_score[:3]   # bottom 3

    pair_id = video_uid[:8]
    n = 0
    for start, end, sig, s in stress_windows:
        writer.write({
            'video_uid': video_uid,
            'start_sec': f'{start:.1f}',
            'end_sec': f'{end:.1f}',
            'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}',
            'cam_mag_max': f'{sig["cam_mag_max"]:.3f}',
            'cam_mag_std': f'{sig["cam_mag_std"]:.3f}',
            'res_mag_mean': f'{sig["res_mag_mean"]:.3f}',
            'score': f'{s:.4f}',
            'pair_id': pair_id,
            'role': 'stress',
        })
        n += 1

    for start, end, sig, s in control_windows:
        writer.write({
            'video_uid': video_uid,
            'start_sec': f'{start:.1f}',
            'end_sec': f'{end:.1f}',
            'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}',
            'cam_mag_max': f'{sig["cam_mag_max"]:.3f}',
            'cam_mag_std': f'{sig["cam_mag_std"]:.3f}',
            'res_mag_mean': f'{sig["res_mag_mean"]:.3f}',
            'score': f'{s:.4f}',
            'pair_id': pair_id,
            'role': 'control',
        })
        n += 1

    return n


def process_video_texture(video_uid: str, video_path: str,
                          writer: CandidateWriter, window_sec: float = 10.0,
                          score_thresh: float = 0.5):
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return 0

    windows = group_windows(ts, frames, window_sec)
    n = 0
    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        s = score_low_texture(sig)
        if s >= score_thresh:
            writer.write({
                'video_uid': video_uid,
                'start_sec': f'{start:.1f}',
                'end_sec': f'{end:.1f}',
                'kp_mean': f'{sig["kp_mean"]:.1f}',
                'lap_var_mean': f'{sig["lap_var_mean"]:.2f}',
                'brightness_mean': f'{sig["brightness_mean"]:.1f}',
                'score': f'{s:.4f}',
            })
            n += 1
    return n


def process_video_reflective(video_uid: str, video_path: str,
                              writer: CandidateWriter,
                              narr: NarrationIndex,
                              window_sec: float = 10.0,
                              score_thresh: float = 0.2):
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return 0

    windows = group_windows(ts, frames, window_sec)
    n = 0
    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        actions = narr.query(video_uid, start, end)
        nar_hit = any(noun in REFLECTIVE_NOUNS for _, noun in actions)
        s = score_reflective(sig, nar_hit)
        if s >= score_thresh:
            writer.write({
                'video_uid': video_uid,
                'start_sec': f'{start:.1f}',
                'end_sec': f'{end:.1f}',
                'highlight_mean': f'{sig["highlight_mean"]:.4f}',
                'narration_hit': '1' if nar_hit else '0',
                'score': f'{s:.4f}',
            })
            n += 1
    return n


def process_video_dynamic(video_uid: str, video_path: str,
                          writer: CandidateWriter,
                          narr: NarrationIndex,
                          window_sec: float = 10.0,
                          score_thresh: float = 0.3):
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return 0

    windows = group_windows(ts, frames, window_sec)
    n = 0
    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        actions = narr.query(video_uid, start, end)
        nar_hit = any(noun in DYNAMIC_NOUNS for _, noun in actions)
        s = score_dynamic(sig, nar_hit)
        if s >= score_thresh:
            writer.write({
                'video_uid': video_uid,
                'start_sec': f'{start:.1f}',
                'end_sec': f'{end:.1f}',
                'res_mag_mean': f'{sig["res_mag_mean"]:.3f}',
                'res_mag_max': f'{sig["res_mag_max"]:.3f}',
                'narration_hit': '1' if nar_hit else '0',
                'score': f'{s:.4f}',
            })
            n += 1
    return n


def process_video_control(video_uid: str, video_path: str,
                          writer: CandidateWriter, window_sec: float = 10.0,
                          score_thresh: float = 0.5):
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return 0

    windows = group_windows(ts, frames, window_sec)
    n = 0
    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        s = score_control(sig)
        if s >= score_thresh:
            writer.write({
                'video_uid': video_uid,
                'start_sec': f'{start:.1f}',
                'end_sec': f'{end:.1f}',
                'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}',
                'kp_mean': f'{sig["kp_mean"]:.1f}',
                'lap_var_mean': f'{sig["lap_var_mean"]:.2f}',
                'brightness_mean': f'{sig["brightness_mean"]:.1f}',
                'highlight_mean': f'{sig["highlight_mean"]:.4f}',
                'score': f'{s:.4f}',
            })
            n += 1
    return n


# ── Main ──────────────────────────────────────────────────────────────────────

FIELDS_MAP = {
    'ego-motion': MOTION_FIELDS,
    'low-texture': TEXTURE_FIELDS,
    'reflective': REFLECTIVE_FIELDS,
    'dynamic': DYNAMIC_FIELDS,
    'control': CONTROL_FIELDS,
}

THRESH_DEFAULTS = {
    'ego-motion': 0.0,
    'low-texture': 0.5,
    'reflective': 0.05,
    'dynamic': 0.3,
    'control': 0.5,
}

NEEDS_NARRATION = {'reflective', 'dynamic'}

ALL_TYPES = ['ego-motion', 'low-texture', 'reflective', 'dynamic', 'control']


def process_video_all(video_uid: str, video_path: str,
                      writers: dict, narr: NarrationIndex,
                      window_sec: float, thresholds: dict) -> dict:
    """
    Single-pass: decode video once, score all 5 types simultaneously.
    Returns {type: n_candidates}.
    """
    ts, frames = stream_video(video_path, sample_fps=1.0)
    if len(frames) < 3:
        return {t: 0 for t in ALL_TYPES}

    windows = group_windows(ts, frames, window_sec)
    counts = {t: 0 for t in ALL_TYPES}

    # ego-motion: need all windows first to compute contrast ratio
    motion_windows = []

    for start, end, wframes in windows:
        sig = _score_window_frames(wframes)
        actions = narr.query(video_uid, start, end) if narr else []

        # low-texture
        s = score_low_texture(sig)
        if s >= thresholds['low-texture']:
            writers['low-texture'].write({
                'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                'kp_mean': f'{sig["kp_mean"]:.1f}', 'lap_var_mean': f'{sig["lap_var_mean"]:.2f}',
                'brightness_mean': f'{sig["brightness_mean"]:.1f}', 'score': f'{s:.4f}',
            })
            counts['low-texture'] += 1

        # reflective
        nar_refl = any(noun in REFLECTIVE_NOUNS for _, noun in actions)
        s = score_reflective(sig, nar_refl)
        if s >= thresholds['reflective']:
            writers['reflective'].write({
                'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                'highlight_mean': f'{sig["highlight_mean"]:.4f}',
                'narration_hit': '1' if nar_refl else '0', 'score': f'{s:.4f}',
            })
            counts['reflective'] += 1

        # dynamic
        nar_dyn = any(noun in DYNAMIC_NOUNS for _, noun in actions)
        s = score_dynamic(sig, nar_dyn)
        if s >= thresholds['dynamic']:
            writers['dynamic'].write({
                'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                'res_mag_mean': f'{sig["res_mag_mean"]:.3f}', 'res_mag_max': f'{sig["res_mag_max"]:.3f}',
                'narration_hit': '1' if nar_dyn else '0', 'score': f'{s:.4f}',
            })
            counts['dynamic'] += 1

        # control
        s = score_control(sig)
        if s >= thresholds['control']:
            writers['control'].write({
                'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}', 'kp_mean': f'{sig["kp_mean"]:.1f}',
                'lap_var_mean': f'{sig["lap_var_mean"]:.2f}', 'brightness_mean': f'{sig["brightness_mean"]:.1f}',
                'highlight_mean': f'{sig["highlight_mean"]:.4f}', 'score': f'{s:.4f}',
            })
            counts['control'] += 1

        # ego-motion: accumulate for contrast check
        motion_windows.append((start, end, sig, score_ego_motion(sig)))

    # ego-motion: emit pairs only if contrast ratio >= 3x
    if len(motion_windows) >= 2:
        scores_only = [s for _, _, _, s in motion_windows]
        max_s, min_s = max(scores_only), min(scores_only)
        if max_s / max(min_s, 0.5) >= 3.0:
            pair_id = video_uid[:8]
            for start, end, sig, s in sorted(motion_windows, key=lambda x: x[3])[-3:]:
                writers['ego-motion'].write({
                    'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                    'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}', 'cam_mag_max': f'{sig["cam_mag_max"]:.3f}',
                    'cam_mag_std': f'{sig["cam_mag_std"]:.3f}', 'res_mag_mean': f'{sig["res_mag_mean"]:.3f}',
                    'score': f'{s:.4f}', 'pair_id': pair_id, 'role': 'stress',
                })
                counts['ego-motion'] += 1
            for start, end, sig, s in sorted(motion_windows, key=lambda x: x[3])[:3]:
                writers['ego-motion'].write({
                    'video_uid': video_uid, 'start_sec': f'{start:.1f}', 'end_sec': f'{end:.1f}',
                    'cam_mag_mean': f'{sig["cam_mag_mean"]:.3f}', 'cam_mag_max': f'{sig["cam_mag_max"]:.3f}',
                    'cam_mag_std': f'{sig["cam_mag_std"]:.3f}', 'res_mag_mean': f'{sig["res_mag_mean"]:.3f}',
                    'score': f'{s:.4f}', 'pair_id': pair_id, 'role': 'control',
                })
                counts['ego-motion'] += 1

    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--type', required=True,
                   choices=list(FIELDS_MAP) + ['all'],
                   help='"all" processes all 5 types in a single decode pass (5x faster)')
    p.add_argument('--ego4d-root', required=True,
                   help='Path to Ego4D v2 root (contains full_scale/ and annotations/)')
    p.add_argument('--ego4d-json',
                   help='Path to ego4d.json (default: <ego4d-root>/../ego4d.json)')
    p.add_argument('--output-dir',
                   help='Base output dir. Each type gets a subdir. (default: ./select_output)')
    p.add_argument('--window-sec', type=float, default=10.0)
    p.add_argument('--score-thresh', type=float, default=None,
                   help='Override score threshold for all types (type-specific defaults if unset)')
    p.add_argument('--max-videos', type=int, default=None,
                   help='Process at most N videos (useful for testing)')
    args = p.parse_args()

    ego4d_root = args.ego4d_root
    video_dir = os.path.join(ego4d_root, 'full_scale')
    ann_dir = os.path.join(ego4d_root, 'annotations')
    ego4d_json = args.ego4d_json or os.path.join(
        os.path.dirname(ego4d_root.rstrip('/')), 'ego4d.json')
    base_out = args.output_dir or 'select_output'

    active_types = ALL_TYPES if args.type == 'all' else [args.type]
    thresholds = {
        t: (args.score_thresh if args.score_thresh is not None else THRESH_DEFAULTS[t])
        for t in active_types
    }

    print(f"[score.py] type={args.type}, ego4d_root={ego4d_root}")
    for t in active_types:
        print(f"           {t}: thresh={thresholds[t]}, out={os.path.join(base_out, t)}")

    load_ego4d_meta(ego4d_json)  # validates json is readable
    local_mp4s = {
        os.path.splitext(f)[0]: os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    } if os.path.isdir(video_dir) else {}
    print(f"  Found {len(local_mp4s)} local MP4s")

    narr = NarrationIndex(ann_dir)  # graceful if files absent

    # Open one CandidateWriter per active type
    writers = {}
    for t in active_types:
        out_dir = os.path.join(base_out, t)
        os.makedirs(out_dir, exist_ok=True)
        writers[t] = CandidateWriter(os.path.join(out_dir, 'candidates.csv'), FIELDS_MAP[t])

    # Resume: skip videos already done in ALL active types
    done_in_all = set.intersection(*[w.done_uids for w in writers.values()]) if writers else set()
    print(f"  {len(done_in_all)} videos already processed in all types (resuming)")

    video_uids = [uid for uid in local_mp4s if uid not in done_in_all]
    if args.max_videos:
        video_uids = video_uids[:args.max_videos]

    n_processed = n_skipped = 0
    total_candidates = {t: 0 for t in active_types}

    for i, video_uid in enumerate(video_uids):
        video_path = local_mp4s[video_uid]
        print(f"  [{i+1}/{len(video_uids)}] {video_uid[:16]}… ", end='', flush=True)

        try:
            if args.type == 'all':
                counts = process_video_all(
                    video_uid, video_path, writers, narr, args.window_sec, thresholds)
                for t in active_types:
                    writers[t].done_uids.add(video_uid)
                    total_candidates[t] += counts[t]
                summary = ' '.join(f'{t[:3]}={counts[t]}' for t in active_types)
                print(f"→ {summary}")
            else:
                t = active_types[0]
                if t == 'ego-motion':
                    n = process_video_motion(video_uid, video_path, writers[t], args.window_sec)
                elif t == 'low-texture':
                    n = process_video_texture(video_uid, video_path, writers[t], args.window_sec, thresholds[t])
                elif t == 'reflective':
                    n = process_video_reflective(video_uid, video_path, writers[t], narr, args.window_sec, thresholds[t])
                elif t == 'dynamic':
                    n = process_video_dynamic(video_uid, video_path, writers[t], narr, args.window_sec, thresholds[t])
                else:
                    n = process_video_control(video_uid, video_path, writers[t], args.window_sec, thresholds[t])
                writers[t].done_uids.add(video_uid)
                total_candidates[t] += n
                print(f"→ {n} candidates")

            n_processed += 1

        except Exception as e:
            n_skipped += 1
            print(f"SKIP ({e})")
            continue

    for t, w in writers.items():
        w.close()
        out_dir = os.path.join(base_out, t)
        write_manifest(out_dir, vars(args), n_processed, n_skipped, total_candidates[t])

    print(f"\nDone. processed={n_processed} skipped={n_skipped}")
    for t in active_types:
        print(f"  {t}: {total_candidates[t]} candidates → {os.path.join(base_out, t)}/candidates.csv")


if __name__ == '__main__':
    main()
