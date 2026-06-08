"""
Shared utilities for the Ego4D stress/control clip selection pipeline.

Signals available locally:
  - Video frames (optical flow, texture, brightness, highlights)
  - fho_lta_train/val.json: verb+noun action labels with video-level timestamps
  - ego4d.json: video metadata (duration, scenarios, has_imu)

IMU: stored on S3 only → optical flow used for motion detection.
"""
import csv
import json
import os
import subprocess
import time

import cv2
import numpy as np


# ── Frame streaming ───────────────────────────────────────────────────────────

def iter_frames(video_path: str, target_fps: float = 1.0,
                resize: tuple = (320, 240),
                start_sec: float = 0.0, end_sec: float = None):
    """
    Yield (timestamp_sec, bgr_frame) by piping through ffmpeg.
    Single-pass decode: much faster than cv2 random seeking for long videos.
    """
    W, H = resize
    cmd = ['ffmpeg', '-loglevel', 'error', '-ss', str(start_sec), '-i', video_path]
    if end_sec is not None:
        cmd += ['-t', str(max(0.5, end_sec - start_sec))]
    cmd += ['-vf', f'fps={target_fps},scale={W}:{H}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = W * H * 3
    t = start_sec
    dt = 1.0 / target_fps
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).copy()
            yield t, frame
            t += dt
    finally:
        proc.stdout.close()
        proc.wait()


def video_duration(video_path: str) -> float:
    """Return duration in seconds via ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return 0.0


def stream_video(video_path: str, sample_fps: float = 1.0,
                 resize: tuple = (320, 240), max_frames: int = 800):
    """
    Stream entire video at sample_fps, capped at max_frames.
    Returns (timestamps, frames) as lists.
    Adjusts fps downward if video is very long to stay within max_frames.
    """
    dur = video_duration(video_path)
    if dur <= 0:
        return [], []

    effective_fps = min(sample_fps, max_frames / max(dur, 1.0))
    effective_fps = max(effective_fps, 0.1)

    ts_list, frame_list = [], []
    for ts, frame in iter_frames(video_path, target_fps=effective_fps, resize=resize):
        ts_list.append(ts)
        frame_list.append(frame)

    return ts_list, frame_list


# ── Optical flow ──────────────────────────────────────────────────────────────

def dense_flow(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow. Returns (H, W, 2)."""
    return cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )


def flow_stats(flow: np.ndarray) -> tuple[float, float]:
    """
    Decompose flow into camera (global) and residual (local/independent) motion.
    Uses median translation as camera motion estimate.
    Returns (camera_mag, residual_mag) in pixels/frame.
    """
    cam_dx = float(np.median(flow[:, :, 0]))
    cam_dy = float(np.median(flow[:, :, 1]))
    residual = flow - np.array([[[cam_dx, cam_dy]]])
    cam_mag = float(np.sqrt(cam_dx ** 2 + cam_dy ** 2))
    res_mag = float(np.sqrt((residual ** 2).sum(axis=-1)).mean())
    return cam_mag, res_mag


def total_flow_mag(flow: np.ndarray) -> float:
    return float(np.sqrt((flow ** 2).sum(axis=-1)).mean())


# ── Per-frame signals ─────────────────────────────────────────────────────────

_orb = None


def _get_orb():
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=500)
    return _orb


def texture_signals(frame_bgr: np.ndarray) -> dict:
    """
    Returns {'kp_count', 'laplacian_var', 'brightness'} for a frame.
    Uses ORB keypoint count as primary texture measure.
    laplacian_var is also used as blur/sharpness detector.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    kps = _get_orb().detect(gray, None)
    return {'kp_count': len(kps), 'laplacian_var': lap_var, 'brightness': brightness}


def highlight_ratio(frame_bgr: np.ndarray, v_thresh: int = 200) -> float:
    """Fraction of pixels with HSV Value > v_thresh (specular highlights)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float((hsv[:, :, 2] > v_thresh).mean())


# ── Window grouping ───────────────────────────────────────────────────────────

def group_windows(timestamps: list, frames: list,
                  window_sec: float = 10.0) -> list[tuple]:
    """
    Group streamed frames into non-overlapping windows.
    Yields (start_sec, end_sec, frame_list).
    """
    if not frames:
        return []

    result = []
    window_start = timestamps[0]
    window_frames = []

    for ts, frame in zip(timestamps, frames):
        if ts - window_start >= window_sec and window_frames:
            result.append((window_start, ts, window_frames))
            window_start = ts
            window_frames = [frame]
        else:
            window_frames.append(frame)

    # Last partial window: include if >= 5s
    if window_frames and (timestamps[-1] - window_start) >= 5.0:
        result.append((window_start, timestamps[-1], window_frames))

    return result


# ── Narration / annotation index ──────────────────────────────────────────────

REFLECTIVE_NOUNS = {
    'television', 'tv', 'screen', 'mirror', 'window', 'glass',
    'phone', 'tablet', 'laptop', 'monitor', 'display', 'computer',
}

DYNAMIC_NOUNS = {
    'person', 'people', 'child', 'kid', 'dog', 'cat', 'animal', 'ball',
}


def _norm_label(label) -> str:
    """'television_(television,_tv)' → 'television'"""
    return str(label).split('_(')[0].lower().strip()


class NarrationIndex:
    """
    Temporal index of (verb, noun) action labels per video_uid.
    Built from fho_lta_train.json and fho_lta_val.json.

    Coverage: ~26k unique videos in fho_lta. Videos not in fho_lta
    will return empty queries (graceful degradation).
    """

    def __init__(self, annotation_dir: str):
        self._idx: dict[str, list] = {}  # video_uid → [(start, end, verb, noun)]
        n_loaded = 0
        for fname in ('fho_lta_train.json', 'fho_lta_val.json'):
            fpath = os.path.join(annotation_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                data = json.load(f)
            for clip in data.get('clips', []):
                vid = clip.get('video_uid')
                if not vid:
                    continue
                parent_start = clip.get('clip_parent_start_sec') or 0.0
                a_start = clip.get('action_clip_start_sec') or 0.0
                a_end = clip.get('action_clip_end_sec') or (a_start + 8.0)
                verb = _norm_label(clip.get('verb_label') or '')
                noun = _norm_label(clip.get('noun_label') or '')
                self._idx.setdefault(vid, []).append(
                    (parent_start + a_start, parent_start + a_end, verb, noun)
                )
                n_loaded += 1
        print(f"  NarrationIndex: {n_loaded} actions across {len(self._idx)} videos")

    def query(self, video_uid: str, start_sec: float, end_sec: float) -> list[tuple]:
        """Return [(verb, noun)] for actions overlapping the window."""
        return [
            (v, n)
            for s, e, v, n in self._idx.get(video_uid, [])
            if e >= start_sec and s <= end_sec
        ]

    def has_video(self, video_uid: str) -> bool:
        return video_uid in self._idx

    def videos_with_noun(self, noun_set: set) -> set:
        return {
            vid for vid, actions in self._idx.items()
            if any(n in noun_set for _, _, _, n in actions)
        }


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_ego4d_meta(ego4d_json: str) -> dict:
    """Returns {video_uid: video_dict} for all videos."""
    with open(ego4d_json) as f:
        data = json.load(f)
    return {v['video_uid']: v for v in data['videos']}


# ── Resumable CSV writer ──────────────────────────────────────────────────────

class CandidateWriter:
    """
    Append-safe CSV writer with resume support.
    Tracks already-processed video_uids to allow interrupted jobs to continue.
    """

    def __init__(self, csv_path: str, fieldnames: list):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.done_uids: set = set()

        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    uid = row.get('video_uid')
                    if uid:
                        self.done_uids.add(uid)

        self._file = open(csv_path, 'a', newline='')
        self._writer = csv.DictWriter(
            self._file, fieldnames=fieldnames, extrasaction='ignore'
        )
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            self._writer.writeheader()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def write_many(self, rows: list):
        for row in rows:
            self._writer.writerow(row)
        self._file.flush()

    def already_done(self, video_uid: str) -> bool:
        return video_uid in self.done_uids

    def close(self):
        self._file.close()


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest(out_dir: str, params: dict,
                   n_processed: int, n_skipped: int, n_candidates: int):
    path = os.path.join(out_dir, 'manifest.json')

    # Merge with existing manifest if present (accumulate counts)
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)

    manifest = {
        **existing,
        'last_run': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': params,
        'n_processed': existing.get('n_processed', 0) + n_processed,
        'n_skipped': existing.get('n_skipped', 0) + n_skipped,
        'n_candidates': existing.get('n_candidates', 0) + n_candidates,
    }
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
