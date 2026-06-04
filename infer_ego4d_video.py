import argparse
import gc
import glob
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visual_util import predictions_to_ply
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


CLIP_UIDS = [
    "ccd5eb6c-e9cb-4777-8b3b-494cbff10733",
    "cd7e8eb9-df58-41ae-aaca-d3d92907de12",
    "cd9113d5-3d83-4aab-8207-57b0f965bb04",
    "cd0f4c06-08f6-4677-a654-0d755ede3e34",
    "cd8ee8a6-b0d7-47d5-a1eb-5008bd5c3804",
    "cd953245-5750-4040-933d-41a21691862a",
]


def load_model(checkpoint_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def extract_frames(video_path: str, out_dir: str, fps: float, num_frames: int) -> int:
    """Extract exactly num_frames frames starting from t=0 at given fps."""
    os.makedirs(out_dir, exist_ok=True)
    duration = num_frames / fps  # e.g. 300/30 = 10s
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    return len(frames)


def make_video_from_frames(frame_dir: str, out_path: str, fps: float):
    """Write frames to H.264 mp4 using imageio-ffmpeg's bundled ffmpeg."""
    import imageio_ffmpeg
    import cv2

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    writer = imageio_ffmpeg.write_frames(
        out_path,
        size=(w, h),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    for p in frame_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        frame = frame[:h, :w]
        writer.send(frame.tobytes())
    writer.close()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True,
                        help="Dir containing Ego4D mp4 files (e.g. data/Ego4D/v2/full_scale)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Number of frames to extract per clip (default: 300 = 10s @ 30fps)")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument("--max-points-k", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("Model loaded.\n")

    total = len(CLIP_UIDS)
    skipped = []

    for idx, clip_uid in enumerate(CLIP_UIDS, 1):
        video_path = os.path.join(args.data_root, f"{clip_uid}.mp4")
        print(f"[{idx}/{total}] Processing: {clip_uid}")

        if not os.path.isfile(video_path):
            print(f"  SKIP: not found at {video_path}")
            skipped.append(clip_uid)
            continue

        out_dir = os.path.join(args.output_dir, clip_uid)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as frame_dir:
                num_extracted = extract_frames(
                    video_path, frame_dir, fps=args.fps, num_frames=args.num_frames
                )
                print(f"  Extracted {num_extracted} frames "
                      f"(target: {args.num_frames} @ {args.fps}fps = "
                      f"{args.num_frames / args.fps:.1f}s)")

                mp4_path = os.path.join(out_dir, f"{clip_uid}_{args.fps:.0f}fps.mp4")
                make_video_from_frames(frame_dir, mp4_path, fps=args.fps)
                print(f"  Saved: {mp4_path}")

                predictions = run_inference(frame_dir, model, args.image_resolution)

            traj_path = os.path.join(out_dir, "camera_trajectory.png")
            save_camera_trajectory(predictions["extrinsic"], traj_path)

            ply_path = os.path.join(out_dir, "scene.ply")
            pc = predictions_to_ply(
                predictions,
                conf_thres=args.conf_thres,
                max_points=args.max_points_k * 1000,
                cam_sphere_pts=300,
            )
            pc.export(ply_path)
            print(f"  Saved: {ply_path}")
            print(f"  --> {num_extracted} frames used for reconstruction")

        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(clip_uid)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nDone. Processed {total - len(skipped)}/{total} clips.")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
