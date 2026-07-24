"""
Create 2D tracking overlay videos and 3D trajectory point clouds for Ego4D clips.

The released VGGT-Omega code in this repo does not expose an explicit tracking
head. This script therefore combines:
  - VGGT-Omega depth / camera / world point predictions
  - OpenCV KLT 2D point tracking on the VGGT-preprocessed frames

Outputs per clip:
  outputs/renders/vggt-omega/ego4d/tracking/<level>/<clip_name>/
    tracking_overlay.mp4
    tracking_trajectories.ply
    tracks.npz
    predictions_pose.npz
    metadata.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_INPUT_BASE = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/ego4d/tracking")
ROTATION_LEVELS = ("rotation_low", "rotation_medium", "rotation_high")
DEFAULT_FPS = 6.0
DEFAULT_START_SEC = 0.0
DEFAULT_END_SEC = 10.0


def list_clips(input_base: str, levels: list[str]) -> list[tuple[str, str, str]]:
    clips = []
    for level in levels:
        for video_path in sorted(glob.glob(os.path.join(input_base, level, "*.mp4"))):
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            clips.append((level, clip_name, video_path))
    return clips


def predictions_images_to_bgr(predictions: dict) -> list[np.ndarray]:
    import cv2

    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 255).clip(0, 255).astype(np.uint8)
    return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]


def select_initial_points(frame_bgr: np.ndarray, max_tracks: int, quality: float, min_distance: int) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_tracks,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=7,
    )
    if points is None:
        raise RuntimeError("Could not find trackable points in the first frame")
    return points.astype(np.float32)


def klt_tracks(
    frames_bgr: list[np.ndarray],
    max_tracks: int,
    quality: float,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    num_frames = len(frames_bgr)
    initial_points = select_initial_points(frames_bgr[0], max_tracks, quality, min_distance)
    num_tracks = len(initial_points)
    tracks = np.full((num_frames, num_tracks, 2), np.nan, dtype=np.float32)
    visible = np.zeros((num_frames, num_tracks), dtype=bool)
    tracks[0] = initial_points[:, 0, :]
    visible[0] = True

    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    prev_points = initial_points
    active_indices = np.arange(num_tracks)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    for frame_idx in range(1, num_frames):
        if len(active_indices) == 0:
            break
        gray = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        if next_points is None or status is None:
            break
        status = status.reshape(-1).astype(bool)
        good_indices = active_indices[status]
        good_points = next_points[status]
        h, w = gray.shape
        inside = (
            (good_points[:, 0, 0] >= 0)
            & (good_points[:, 0, 0] < w)
            & (good_points[:, 0, 1] >= 0)
            & (good_points[:, 0, 1] < h)
        )
        good_indices = good_indices[inside]
        good_points = good_points[inside]
        tracks[frame_idx, good_indices] = good_points[:, 0, :]
        visible[frame_idx, good_indices] = True

        prev_gray = gray
        prev_points = good_points.reshape(-1, 1, 2).astype(np.float32)
        active_indices = good_indices

    return tracks, visible


def track_colors(num_tracks: int) -> np.ndarray:
    import cv2

    hsv = np.zeros((num_tracks, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, num_tracks, endpoint=False).astype(np.uint8)
    hsv[:, 0, 1] = 220
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]


def save_overlay_video(
    frames_bgr: list[np.ndarray],
    tracks: np.ndarray,
    visible: np.ndarray,
    output_path: str,
    fps: float,
    trail: int,
) -> None:
    import cv2
    import imageio_ffmpeg

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        output_path,
        size=(w, h),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    colors = track_colors(tracks.shape[1])

    for frame_idx, frame in enumerate(frames_bgr):
        vis = frame.copy()
        start = max(0, frame_idx - trail)
        for track_idx in range(tracks.shape[1]):
            if not visible[frame_idx, track_idx]:
                continue
            color = tuple(int(x) for x in colors[track_idx])
            pts = []
            for t in range(start, frame_idx + 1):
                if visible[t, track_idx]:
                    pts.append(tuple(np.round(tracks[t, track_idx]).astype(int)))
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(vis, a, b, color, 2, cv2.LINE_AA)
            cv2.circle(vis, pts[-1], 3, color, -1, cv2.LINE_AA)
        writer.send(vis.tobytes())
    writer.close()
    print(f"  Saved: {output_path}")


def sample_world_trajectories(predictions: dict, tracks: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_points = predictions["world_points_from_depth"]
    num_frames, height, width, _ = world_points.shape
    num_tracks = tracks.shape[1]
    traj = np.full((num_frames, num_tracks, 3), np.nan, dtype=np.float32)
    traj_visible = np.zeros((num_frames, num_tracks), dtype=bool)

    for frame_idx in range(num_frames):
        for track_idx in np.where(visible[frame_idx])[0]:
            x, y = tracks[frame_idx, track_idx]
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= xi < width and 0 <= yi < height:
                xyz = world_points[frame_idx, yi, xi]
                if np.isfinite(xyz).all():
                    traj[frame_idx, track_idx] = xyz
                    traj_visible[frame_idx, track_idx] = True
    return traj, traj_visible


def save_trajectory_ply(traj: np.ndarray, visible: np.ndarray, output_path: str, samples_per_segment: int) -> None:
    import trimesh

    vertices = []
    colors = []
    bgr_colors = track_colors(traj.shape[1])
    rgb_colors = bgr_colors[:, ::-1]

    for track_idx in range(traj.shape[1]):
        frames = np.where(visible[:, track_idx])[0]
        if len(frames) < 2:
            continue
        color = rgb_colors[track_idx]
        for a, b in zip(frames[:-1], frames[1:]):
            pa = traj[a, track_idx]
            pb = traj[b, track_idx]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            for s in range(samples_per_segment):
                alpha = s / max(samples_per_segment, 1)
                vertices.append((1.0 - alpha) * pa + alpha * pb)
                colors.append(color)

    if not vertices:
        vertices = [np.zeros(3, dtype=np.float32)]
        colors = [np.array([255, 255, 255], dtype=np.uint8)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trimesh.PointCloud(
        vertices=np.asarray(vertices, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
    ).export(output_path)
    print(f"  Saved: {output_path}")


def save_metadata(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def process_clip(args: argparse.Namespace, level: str, clip_name: str, video_path: str, reconstruction, model) -> None:
    out_dir = os.path.join(args.output_root, level, clip_name)
    overlay_path = os.path.join(out_dir, "tracking_overlay.mp4")
    trajectory_path = os.path.join(out_dir, "tracking_trajectories.ply")
    if os.path.exists(overlay_path) and os.path.exists(trajectory_path) and not args.overwrite:
        print(f"  Already done, skipping: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  Extracting frames at {args.fps:g} fps ...")
        num_frames = reconstruction.extract_frames(video_path, tmp_dir, args.start_sec, args.end_sec, args.fps)
        if num_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {num_frames}")

        print("  Running VGGT-Omega inference ...")
        predictions = reconstruction.run_inference(tmp_dir, model)

    frames_bgr = predictions_images_to_bgr(predictions)
    print("  Tracking 2D points with KLT ...")
    tracks, visible = klt_tracks(
        frames_bgr,
        max_tracks=args.max_tracks,
        quality=args.feature_quality,
        min_distance=args.min_distance,
    )

    print("  Saving 2D tracking overlay video ...")
    save_overlay_video(frames_bgr, tracks, visible, overlay_path, args.fps, args.trail)

    print("  Sampling VGGT world points and saving 3D trajectories ...")
    traj, traj_visible = sample_world_trajectories(predictions, tracks, visible)
    save_trajectory_ply(traj, traj_visible, trajectory_path, args.samples_per_segment)

    reconstruction.save_prediction_pose(
        predictions,
        os.path.join(out_dir, "predictions_pose.npz"),
        start_sec=args.start_sec,
        fps=args.fps,
    )
    np.savez_compressed(
        os.path.join(out_dir, "tracks.npz"),
        tracks_2d=tracks,
        visible_2d=visible,
        trajectories_3d=traj,
        visible_3d=traj_visible,
    )
    save_metadata(
        os.path.join(out_dir, "metadata.json"),
        {
            "level": level,
            "clip_name": clip_name,
            "video_path": video_path,
            "num_frames": int(len(frames_bgr)),
            "fps": args.fps,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "max_tracks": args.max_tracks,
            "note": "2D tracks are KLT tracks; 3D trajectories are sampled from VGGT-Omega world_points_from_depth at KLT positions.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base", default=DEFAULT_INPUT_BASE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--levels", nargs="+", choices=ROTATION_LEVELS, default=["rotation_high"])
    parser.add_argument("--clip-name", help="Only process one clip basename")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", help="VGGT-Omega checkpoint path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC)
    parser.add_argument("--max-tracks", type=int, default=300)
    parser.add_argument("--feature-quality", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--trail", type=int, default=20)
    parser.add_argument("--samples-per-segment", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = list_clips(args.input_base, args.levels)
    if args.clip_name:
        clips = [clip for clip in clips if clip[1] == args.clip_name]
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        print("No clips found.")
        return

    import reconstruction

    if args.checkpoint is None:
        args.checkpoint = reconstruction.CHECKPOINT

    print(f"Loading model from {args.checkpoint}")
    model = reconstruction.load_model(args.checkpoint)
    print("Model loaded.")
    print(f"Found {len(clips)} clips.")

    for index, (level, clip_name, video_path) in enumerate(clips, start=1):
        print(f"[{index}/{len(clips)}] {level}/{clip_name}")
        process_clip(args, level, clip_name, video_path, reconstruction, model)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
