#!/usr/bin/env python3
"""
Depth map visualization for hand-selected stress/control clips.

Outputs per clip (under OUTPUT_BASE/<type>/<name>/):
  depth_frame_xxx.png   – per-frame depth map (plasma colormap, clip-normalized)
  depth_video.mp4       – depth map video
  original.mp4          – original 10s clip
"""
import gc
import glob
import os
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


# ── Clip list ─────────────────────────────────────────────────────────────────
CLIPS = [
    ('control',   '0b9ee926-00f1-4b22-9e83-b664c5e465e4', 1492, 1503, ''),
    ('control',   '1cdc92fa-50cd-4461-adf2-ece8cb2a5d31',  940,  951, ''),
    ('control',   '39011e23-5fe4-41cd-b146-f3f9e3f3941a', 3776, 3787, ''),
    ('dynamic',   '1eb500cd-cdb4-415e-8bf6-cc1289edf0ee',    0,   11, ''),
    ('dynamic',   '3c826d8a-22b9-4083-a315-2aeee7bde095',   51,   61, ''),
    ('dynamic',   '04fe8f4d-081e-437e-a56a-2d53b6233fc9',   13,   26, ''),
    ('egomotion', '1b1acfa6-ee3b-483c-9c0a-245d70d7125f',   52,   65, 'control'),
    ('egomotion', '1b1acfa6-ee3b-483c-9c0a-245d70d7125f',  902,  916, 'stress'),
    ('egomotion', '04fe8f4d-081e-437e-a56a-2d53b6233fc9', 3240, 3254, 'control'),
    ('egomotion', '04fe8f4d-081e-437e-a56a-2d53b6233fc9',  476,  489, 'stress'),
    ('egomotion', 'cfa96e76-d909-4ba5-bccb-77402fde6be7',  316,  327, 'control'),
    ('egomotion', 'cfa96e76-d909-4ba5-bccb-77402fde6be7', 1001, 1011, 'stress'),
]

VIDEO_ROOT  = '/workspace/data/Ego4D/v2/full_scale'
OUTPUT_BASE = '/workspace/outputs/renders/vggt-omega/ego4d/depth/CNN'
CHECKPOINT  = '/workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt'
IMAGE_RES   = 512
SAMPLE_FPS  = 6.0

_CMAP = plt.get_cmap('plasma')


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.to('cuda')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(image_paths: list, model: VGGTOmega) -> dict:
    images = load_and_preprocess_images(image_paths, image_resolution=IMAGE_RES).to('cuda')
    with torch.inference_mode():
        pred = model(images)
    out = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
            out[k] = v
    torch.cuda.empty_cache()
    return out


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str,
                   start_sec: float, end_sec: float, fps: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start_sec), '-i', video_path,
        '-t', str(end_sec - start_sec),
        '-vf', f'fps={fps}', '-q:v', '2',
        os.path.join(out_dir, 'frame_%06d.jpg'),
    ], check=True)
    return len(glob.glob(os.path.join(out_dir, 'frame_*.jpg')))


# ── Depth visualization ───────────────────────────────────────────────────────

def depth_to_rgb(depth_map: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Normalize depth to [0,1] with clip-level range, apply plasma colormap → BGR uint8."""
    norm = (depth_map - d_min) / (d_max - d_min + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = (_CMAP(norm) * 255).astype(np.uint8)      # H×W×4, RGBA
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def save_depth_frames(depth: np.ndarray, out_dir: str) -> list:
    """
    depth: (S, H, W, 1) — camera-space depth in world units
    Returns list of BGR frames for video writing.
    """
    d = depth[..., 0]                                 # (S, H, W)
    # Clip-level normalization with 2/98 percentile for robustness
    d_min = float(np.percentile(d, 2))
    d_max = float(np.percentile(d, 98))
    print(f'  Depth range: [{d_min:.3f}, {d_max:.3f}] (2-98 pct)')

    frames_bgr = []
    for idx in range(d.shape[0]):
        bgr = depth_to_rgb(d[idx], d_min, d_max)
        out_path = os.path.join(out_dir, f'depth_frame_{idx:03d}.png')
        cv2.imwrite(out_path, bgr)
        frames_bgr.append(bgr)
    return frames_bgr


def save_depth_video(frames_bgr: list, out_path: str, fps: float):
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    writer = imageio_ffmpeg.write_frames(
        out_path, size=(w, h), fps=fps,
        codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
    )
    writer.send(None)
    for f in frames_bgr:
        writer.send(f[:h, :w].tobytes())
    writer.close()


def save_original_clip(video_path: str, start_sec: float, end_sec: float,
                       out_path: str, fps: float = 30.0):
    """Extract original clip using imageio_ffmpeg (bundled libx264)."""
    tmp_dir = tempfile.mkdtemp()
    try:
        frame_glob = os.path.join(tmp_dir, 'frame_%06d.jpg')
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', str(start_sec), '-i', video_path,
            '-t', str(end_sec - start_sec),
            '-vf', f'fps={fps}', '-q:v', '2', frame_glob,
        ], check=True)
        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.jpg')))
        if not frame_paths:
            print('  WARN: no frames for original clip')
            return
        first = cv2.imread(frame_paths[0])
        h, w = first.shape[:2]
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1
        writer = imageio_ffmpeg.write_frames(
            out_path, size=(w, h), fps=fps,
            codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
        )
        writer.send(None)
        for p in frame_paths:
            f = cv2.imread(p)
            if f is not None:
                writer.send(f[:h, :w].tobytes())
        writer.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def clip_out_name(uid: str, start: int, end: int, role: str) -> str:
    role_tag = f'_{role}' if role else ''
    return f'{uid}_{start}-{end}s{role_tag}'


def main():
    print(f'Loading model from {CHECKPOINT}')
    model = load_model(CHECKPOINT)
    print('Model loaded.\n')

    for i, (clip_type, uid, start, end, role) in enumerate(CLIPS):
        name = clip_out_name(uid, start, end, role)
        out_dir = os.path.join(OUTPUT_BASE, clip_type, name)
        print(f'[{i+1}/{len(CLIPS)}] {clip_type}/{name}')

        depth_done = bool(glob.glob(os.path.join(out_dir, 'depth_frame_*.png')))
        vid_done   = os.path.exists(os.path.join(out_dir, 'depth_video.mp4'))
        orig_done  = os.path.exists(os.path.join(out_dir, 'original.mp4'))

        if depth_done and vid_done and orig_done:
            print('  Already done, skipping.')
            continue

        video_path = os.path.join(VIDEO_ROOT, uid + '.mp4')
        if not os.path.exists(video_path):
            print('  SKIP: video not found')
            continue

        os.makedirs(out_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f'  Extracting frames at {SAMPLE_FPS}fps ...')
            n = extract_frames(video_path, tmp_dir, start, end, SAMPLE_FPS)
            print(f'  {n} frames extracted')

            if n < 2:
                print('  SKIP: too few frames')
                continue

            image_paths = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.jpg')))

            print('  Running inference ...')
            try:
                pred = run_inference(image_paths, model)
            except Exception as e:
                print(f'  ERROR during inference: {e}')
                gc.collect(); torch.cuda.empty_cache()
                continue

        # depth: (S, H, W, 1)
        depth = pred['depth']
        print(f'  depth shape: {depth.shape}')

        if not depth_done or not vid_done:
            print('  Saving depth frames ...')
            frames_bgr = save_depth_frames(depth, out_dir)
            print(f'  {len(frames_bgr)} depth frames saved')

            if not vid_done:
                depth_vid = os.path.join(out_dir, 'depth_video.mp4')
                save_depth_video(frames_bgr, depth_vid, fps=SAMPLE_FPS)
                print(f'  Saved: {depth_vid}')

        if not orig_done:
            print('  Saving original.mp4 ...')
            orig_mp4 = os.path.join(out_dir, 'original.mp4')
            save_original_clip(video_path, start, end, orig_mp4)
            print(f'  Saved: {orig_mp4}')

        print()
        gc.collect()
        torch.cuda.empty_cache()

    print('All done.')


if __name__ == '__main__':
    main()
