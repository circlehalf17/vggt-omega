#!/usr/bin/env python3
"""
VGGT-Omega 3D reconstruction.

Modes:
  ego4d-json    Ego4D clips via ego4d.json + eval list (cv2 extraction by clip timestamp)
  ego4d-clips   Specific clip UIDs from mp4 files (ffmpeg extraction, fixed window)
  videos        All video files in a directory (ffmpeg extraction)
  image-dirs    Pre-extracted image directories
"""
import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visual_util import predictions_to_glb, predictions_to_ply
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── Shared utilities ──────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def run_inference(image_dir: str, model: VGGTOmega, image_resolution: int) -> dict:
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_names = [p for p in image_names
                   if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")]
    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    torch.cuda.empty_cache()
    return predictions_np


def save_camera_trajectory(extrinsic_w2c: np.ndarray, save_path: str):
    N = len(extrinsic_w2c)
    R = extrinsic_w2c[:, :3, :3]
    t = extrinsic_w2c[:, :3, 3]
    positions = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
    R_c2w = R.transpose(0, 2, 1)
    cmap_colors = plt.cm.viridis(np.linspace(0, 1, N))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "b-", alpha=0.35, linewidth=1.2)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=cmap_colors, s=18, zorder=5)
    ax.scatter(*positions[0],  c="lime", s=140, zorder=6, marker="*", label="Start")
    ax.scatter(*positions[-1], c="red",  s=140, zorder=6, marker="*", label="End")
    step = max(1, N // 12)
    for i in range(0, N, step):
        fwd = -R_c2w[i, :, 2] * 0.05
        ax.quiver(*positions[i], *fwd, color="orange", alpha=0.8, linewidth=1.5)
    spread = np.abs(positions).max() * 1.1 + 1e-6
    mid = positions.mean(axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter([m - spread, m + spread])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Estimated Camera Trajectory (VGGT-Omega)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def extract_frames_cv2(video_path: str, out_dir: str, start_sec: float,
                       end_sec: float, sample_fps: float = 1.0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps / sample_fps)), 1)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved, frame_idx = 0, start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.png"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def extract_frames_ffmpeg(video_path: str, out_dir: str, fps: float,
                          num_frames: int = None, start_sec: float = 0.0) -> int:
    """Extract frames via ffmpeg. num_frames=None extracts the entire video."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", video_path]
    if num_frames is not None:
        cmd += ["-t", str(num_frames / fps)]
    cmd += ["-vf", f"fps={fps}", "-q:v", "2",
            os.path.join(out_dir, "frame_%06d.jpg")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg"))))


def make_video_from_frames(frame_dir: str, out_path: str, fps: float):
    import imageio_ffmpeg
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    writer = imageio_ffmpeg.write_frames(out_path, size=(w, h), fps=fps,
                                         codec="libx264", pix_fmt_in="bgr24",
                                         pix_fmt_out="yuv420p")
    writer.send(None)
    for p in frame_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        frame = frame[:h, :w]
        writer.send(frame.tobytes())
    writer.close()


def save_outputs(predictions: dict, out_dir: str, args, image_dir: str = None):
    os.makedirs(out_dir, exist_ok=True)
    save_camera_trajectory(predictions["extrinsic"],
                           os.path.join(out_dir, "camera_trajectory.png"))

    ply_path = os.path.join(out_dir, "scene.ply")
    pc = predictions_to_ply(predictions, conf_thres=args.conf_thres,
                             max_points=args.max_points_k * 1000, cam_sphere_pts=300)
    pc.export(ply_path)
    print(f"  Saved: {ply_path}")

    if args.save_glb:
        glb_path = os.path.join(out_dir, "scene.glb")
        scene = predictions_to_glb(predictions, conf_thres=args.conf_thres,
                                   show_cam=True, target_dir=image_dir,
                                   max_points=args.max_points_k * 1000)
        scene.export(file_obj=glb_path)
        print(f"  Saved: {glb_path}")

    if args.save_npz:
        npz_path = os.path.join(out_dir, "predictions.npz")
        np.savez(npz_path, **predictions)
        print(f"  Saved: {npz_path}")


def _print_summary(skipped: list, total: int):
    print(f"\nDone. Processed {total - len(skipped)}/{total}.")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


# ── Mode: ego4d-json ──────────────────────────────────────────────────────────

def run_ego4d_json(args, model):
    with open(args.ego4d_json) as f:
        ego4d = json.load(f)
    clip_map = {c["clip_uid"]: c for c in ego4d["clips"]}
    eval_uids = open(args.eval_list).read().strip().split()
    os.makedirs(args.output_dir, exist_ok=True)
    skipped = []

    for i, clip_uid in enumerate(eval_uids):
        clip = clip_map.get(clip_uid)
        if clip is None:
            print(f"[{i+1}/{len(eval_uids)}] SKIP (not in ego4d.json): {clip_uid}")
            skipped.append(clip_uid); continue

        video_path = os.path.join(args.ego4d_root, "full_scale", f"{clip['video_uid']}.mp4")
        if not os.path.exists(video_path):
            print(f"[{i+1}/{len(eval_uids)}] SKIP (video missing): {clip_uid}")
            skipped.append(clip_uid); continue

        print(f"[{i+1}/{len(eval_uids)}] Processing: {clip_uid}")
        tmp_dir = os.path.join(args.tmp_dir, clip_uid)
        images_dir = os.path.join(tmp_dir, "images")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        start_sec = clip["video_start_sec"]
        end_sec = clip["video_end_sec"]
        if args.max_duration is not None:
            end_sec = min(end_sec, start_sec + args.max_duration)

        n_frames = extract_frames_cv2(video_path, images_dir, start_sec, end_sec,
                                      args.sample_fps)
        print(f"  Extracted {n_frames} frames")

        if n_frames == 0:
            print("  SKIP: no frames extracted")
            skipped.append(clip_uid)
            shutil.rmtree(tmp_dir, ignore_errors=True); continue

        try:
            predictions = run_inference(images_dir, model, args.image_resolution)
            save_outputs(predictions, os.path.join(args.output_dir, clip_uid),
                         args, image_dir=tmp_dir)
        except Exception as e:
            print(f"  ERROR: {e}"); skipped.append(clip_uid)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect(); torch.cuda.empty_cache()

    _print_summary(skipped, len(eval_uids))


# ── Mode: ego4d-clips ─────────────────────────────────────────────────────────

def run_ego4d_clips(args, model):
    if args.clip_uids:
        clip_uids = [u.strip() for u in args.clip_uids.split(",")]
    else:
        mp4s = sorted(glob.glob(os.path.join(args.data_root, "*.mp4")))
        clip_uids = [os.path.splitext(os.path.basename(p))[0] for p in mp4s]

    end_sec = args.start_sec + args.num_frames / args.fps
    os.makedirs(args.output_dir, exist_ok=True)
    skipped = []

    for idx, clip_uid in enumerate(clip_uids, 1):
        video_path = os.path.join(args.data_root, f"{clip_uid}.mp4")
        print(f"[{idx}/{len(clip_uids)}] Processing: {clip_uid}")
        if not os.path.isfile(video_path):
            print(f"  SKIP: not found at {video_path}")
            skipped.append(clip_uid); continue

        out_dir = os.path.join(args.output_dir, clip_uid)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as frame_dir:
                n = extract_frames_ffmpeg(video_path, frame_dir, args.fps,
                                          args.num_frames, args.start_sec)
                print(f"  Extracted {n} frames ({args.start_sec:.1f}s~{end_sec:.1f}s)")

                mp4_path = os.path.join(
                    out_dir,
                    f"{clip_uid}_{args.fps:.0f}fps_{args.start_sec:.0f}s-{end_sec:.0f}s.mp4")
                make_video_from_frames(frame_dir, mp4_path, fps=args.fps)
                print(f"  Saved: {mp4_path}")

                predictions = run_inference(frame_dir, model, args.image_resolution)

            save_outputs(predictions, out_dir, args)
        except Exception as e:
            print(f"  ERROR: {e}"); skipped.append(clip_uid)
        finally:
            gc.collect(); torch.cuda.empty_cache()

    _print_summary(skipped, len(clip_uids))


# ── Mode: videos ──────────────────────────────────────────────────────────────

def run_videos(args, model):
    video_paths = sorted(
        p for p in glob.glob(os.path.join(args.data_root, "*"))
        if os.path.splitext(p)[1].lower() in VIDEO_EXTS
    )
    if not video_paths:
        print(f"No video files found in {args.data_root}"); return

    os.makedirs(args.output_dir, exist_ok=True)
    skipped = []

    for idx, video_path in enumerate(video_paths, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx}/{len(video_paths)}] Processing: {video_name}")

        out_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as frame_dir:
                n = extract_frames_ffmpeg(video_path, frame_dir, args.fps)
                print(f"  Extracted {n} frames @ {args.fps}fps")

                mp4_path = os.path.join(out_dir, f"{video_name}_{args.fps}fps.mp4")
                make_video_from_frames(frame_dir, mp4_path, fps=args.fps)
                print(f"  Saved: {mp4_path}")

                predictions = run_inference(frame_dir, model, args.image_resolution)

            save_outputs(predictions, out_dir, args)
        except Exception as e:
            print(f"  ERROR: {e}"); skipped.append(video_name)
        finally:
            gc.collect(); torch.cuda.empty_cache()

    _print_summary(skipped, len(video_paths))


# ── Mode: image-dirs ──────────────────────────────────────────────────────────

def _find_image_dirs(root: str) -> list:
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    result = []
    for dirpath, _, filenames in os.walk(root):
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in filenames):
            result.append(dirpath)
    return sorted(result)


def run_image_dirs(args, model):
    image_dirs = _find_image_dirs(args.data_root)
    if not image_dirs:
        print(f"No image directories found under {args.data_root}"); return

    os.makedirs(args.output_dir, exist_ok=True)
    skipped = []

    for idx, image_dir in enumerate(image_dirs, 1):
        rel = os.path.relpath(image_dir, args.data_root)
        print(f"[{idx}/{len(image_dirs)}] Processing: {rel}")

        try:
            predictions = run_inference(image_dir, model, args.image_resolution)
            save_outputs(predictions, os.path.join(args.output_dir, rel),
                         args, image_dir=image_dir)
        except Exception as e:
            print(f"  ERROR: {e}"); skipped.append(rel)
        finally:
            gc.collect(); torch.cuda.empty_cache()

    _print_summary(skipped, len(image_dirs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VGGT-Omega 3D reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True,
                        choices=["ego4d-json", "ego4d-clips", "videos", "image-dirs"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument("--max-points-k", type=int, default=1000)
    parser.add_argument("--save-glb", action="store_true",
                        help="Also export scene.glb (requires predictions_to_glb)")
    parser.add_argument("--save-npz", action="store_true",
                        help="Also save predictions.npz")

    # ego4d-json
    parser.add_argument("--ego4d-root", help="Ego4D root dir (contains full_scale/)")
    parser.add_argument("--ego4d-json", help="Path to ego4d.json")
    parser.add_argument("--eval-list", help="Path to eval list (one clip_uid per line)")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Max seconds per clip from clip start")
    parser.add_argument("--tmp-dir", default="/tmp/vggt_frames")

    # ego4d-clips
    parser.add_argument("--clip-uids", default=None,
                        help="Comma-separated clip UIDs (default: all *.mp4 in data-root)")

    # ego4d-clips + videos
    parser.add_argument("--data-root", help="Dir with .mp4 files or image subdirs")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Extraction fps (ego4d-clips: 30, videos: 6 recommended)")
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Number of frames to extract (ego4d-clips only)")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Extraction start time in seconds (ego4d-clips only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}  [mode={args.mode}]")
    model = load_model(args.checkpoint)
    print("Model loaded.\n")

    {
        "ego4d-json":  run_ego4d_json,
        "ego4d-clips": run_ego4d_clips,
        "videos":      run_videos,
        "image-dirs":  run_image_dirs,
    }[args.mode](args, model)


if __name__ == "__main__":
    main()
