"""
VGGT-Omega inference on Ego4D rotation labeling clips.

Inputs:
  data/ego4d/v2/label/rotation_low/*.mp4
  data/ego4d/v2/label/rotation_medium/*.mp4
  data/ego4d/v2/label/rotation_high/*.mp4

Outputs per clip (under outputs/renders/vggt-omega/ego4d/reconstruction/rotation_<level>/<clip_name>/):
  camera_trajectory.png   – estimated camera path
  scene.ply               – reconstructed point cloud
  predictions_pose.npz    – predicted VGGT-Omega camera pose/intrinsics
  original.mp4            – original 10s input clip
"""
import glob
import os
import tempfile

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visual_util import predictions_to_ply
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


# ── Rotation clip roots ───────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
INPUT_BASE = os.path.join(PROJECT_ROOT, 'data/ego4d/v2/label')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'outputs/renders/vggt-omega/ego4d/reconstruction')
ROTATION_LEVELS = ('rotation_low', 'rotation_medium', 'rotation_high')
CHECKPOINT  = os.path.join(PROJECT_ROOT, 'outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt')
WORKSPACE_CHECKPOINT = '/workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt'
IMAGE_RES   = 512
SAMPLE_FPS  = 6.0
CLIP_START_SEC = 0.0
CLIP_END_SEC = 10.0
CONF_THRES  = 20.0
MAX_POINTS  = 1_000_000


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> VGGTOmega:
    if not os.path.exists(checkpoint_path) and os.path.exists(WORKSPACE_CHECKPOINT):
        checkpoint_path = WORKSPACE_CHECKPOINT
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.to('cuda')


# ── Inference ─────────────────────────────────────────────────────────────────

def unproject_depth(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    S, H, W = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x = np.broadcast_to(x[None], (S, H, W))
    y = np.broadcast_to(y[None], (S, H, W))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    cam_pts = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return np.einsum('nij,nhwj->nhwi', R.transpose(0, 2, 1),
                     cam_pts - t[:, None, None, :])


def run_inference(image_dir: str, model: VGGTOmega) -> dict:
    paths = sorted(p for p in glob.glob(os.path.join(image_dir, '*'))
                   if os.path.splitext(p)[1].lower() in ('.jpg', '.jpeg', '.png'))
    images = load_and_preprocess_images(paths, image_resolution=IMAGE_RES).to('cuda')
    with torch.inference_mode():
        pred = model(images)
    extrinsic, intrinsic = encoding_to_camera(pred['pose_enc'], pred['images'].shape[-2:])
    pred['extrinsic'] = extrinsic
    pred['intrinsic'] = intrinsic
    out = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
            out[k] = v
    out['world_points_from_depth'] = unproject_depth(
        out['depth'], out['extrinsic'], out['intrinsic'])
    torch.cuda.empty_cache()
    return out


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str,
                   start_sec: float, end_sec: float, fps: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start_sec), '-i', video_path,
        '-t', str(end_sec - start_sec),
        '-vf', f'fps={fps}', '-q:v', '2',
        os.path.join(out_dir, 'frame_%06d.jpg'),
    ]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(out_dir, 'frame_*.jpg')))


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_trajectory_png(extrinsic_w2c: np.ndarray, save_path: str):
    N = len(extrinsic_w2c)
    R = extrinsic_w2c[:, :3, :3]
    t = extrinsic_w2c[:, :3, 3]
    positions = -np.einsum('nij,nj->ni', R.transpose(0, 2, 1), t)
    R_c2w = R.transpose(0, 2, 1)
    cmap_colors = plt.cm.viridis(np.linspace(0, 1, N))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', alpha=0.35, linewidth=1.2)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=cmap_colors, s=18, zorder=5)
    ax.scatter(*positions[0],  c='lime', s=140, zorder=6, marker='*', label='Start')
    ax.scatter(*positions[-1], c='red',  s=140, zorder=6, marker='*', label='End')

    # Quiver: 1/5 of original 0.05 scale
    step = max(1, N // 12)
    for i in range(0, N, step):
        fwd = -R_c2w[i, :, 2] * 0.01
        ax.quiver(*positions[i], *fwd, color='orange', alpha=0.8, linewidth=1.5)

    spread = np.abs(positions).max() * 1.1 + 1e-6
    mid = positions.mean(axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter([m - spread, m + spread])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Estimated Camera Trajectory (VGGT-Omega)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def save_ply(predictions: dict, save_path: str):
    pc = predictions_to_ply(
        predictions,
        conf_thres=CONF_THRES,
        max_points=MAX_POINTS,
        cam_sphere_pts=0,
    )
    pc.export(save_path)
    print(f'  Saved: {save_path}')


def save_prediction_pose(predictions: dict, save_path: str,
                         start_sec: float, fps: float):
    n = len(predictions['extrinsic'])
    frame_indices = np.arange(n, dtype=np.int64)
    timestamps = start_sec + frame_indices.astype(np.float64) / fps
    np.savez_compressed(
        save_path,
        extrinsic_w2c=predictions['extrinsic'],
        intrinsic=predictions['intrinsic'],
        pose_enc=predictions['pose_enc'],
        timestamps=timestamps,
        frame_indices=frame_indices,
    )
    print(f'  Saved: {save_path}')


def save_original_clip(video_path: str, start_sec: float, end_sec: float,
                       out_path: str, fps: float = 30.0):
    """Extract original clip using imageio_ffmpeg (bundled libx264)."""
    import subprocess, tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    try:
        frame_glob = os.path.join(tmp_dir, 'frame_%06d.jpg')
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', str(start_sec), '-i', video_path,
            '-t', str(end_sec - start_sec),
            '-vf', f'fps={fps}', '-q:v', '2', frame_glob,
        ]
        subprocess.run(cmd, check=True)
        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.jpg')))
        if not frame_paths:
            print(f'  WARN: no frames for original clip')
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
        print(f'  Saved: {out_path}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def list_rotation_clips():
    clips = []
    for clip_type in ROTATION_LEVELS:
        input_dir = os.path.join(INPUT_BASE, clip_type)
        video_paths = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))
        for video_path in video_paths:
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            clips.append((clip_type, clip_name, video_path))
    return clips


def main():
    print(f'Loading model from {CHECKPOINT}')
    model = load_model(CHECKPOINT)
    print('Model loaded.\n')

    clips = list_rotation_clips()
    print(f'Found {len(clips)} rotation clips under {INPUT_BASE}\n')

    for i, (clip_type, clip_name, video_path) in enumerate(clips):
        out_dir = os.path.join(OUTPUT_BASE, clip_type, clip_name)
        print(f'[{i+1}/{len(clips)}] {clip_type}/{clip_name}')

        # Skip if already done
        if (os.path.exists(os.path.join(out_dir, 'scene.ply')) and
                os.path.exists(os.path.join(out_dir, 'predictions_pose.npz'))):
            print('  Already done, skipping.')
            continue

        if not os.path.exists(video_path):
            print(f'  SKIP: clip not found: {video_path}')
            continue

        os.makedirs(out_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f'  Extracting frames at {SAMPLE_FPS}fps ...')
            n = extract_frames(
                video_path, tmp_dir, CLIP_START_SEC, CLIP_END_SEC, SAMPLE_FPS)
            print(f'  {n} frames extracted')

            if n < 2:
                print('  SKIP: too few frames')
                continue

            print('  Running inference ...')
            predictions = run_inference(tmp_dir, model)

        print('  Saving outputs ...')
        save_trajectory_png(predictions['extrinsic'],
                            os.path.join(out_dir, 'camera_trajectory.png'))
        save_prediction_pose(predictions,
                             os.path.join(out_dir, 'predictions_pose.npz'),
                             CLIP_START_SEC, SAMPLE_FPS)
        save_ply(predictions, os.path.join(out_dir, 'scene.ply'))
        save_original_clip(video_path, CLIP_START_SEC, CLIP_END_SEC,
                           os.path.join(out_dir, 'original.mp4'))
        print()

    print('All done.')


if __name__ == '__main__':
    main()
