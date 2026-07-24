"""
Compare two SOTA point trackers on Ego4D rotation clips.

Pipeline per clip:
  1. Extract 60 frames at 6 fps.
  2. Generate the same query point grid on frame 0.
  3. Run CoTracker3.
  4. Run BootsTAPIR via an external runner command.
  5. Save a 2D overlay video for each tracker.
  6. Save visibility, active track count, and lifetime plots.
  7. Optionally lift 2D tracks to 3D by sampling VGGT-Omega world_points_from_depth.

CoTracker3 is run directly with torch.hub:
  torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

BootsTAPIR is connected through a small command-line contract because local
installations differ between JAX/PyTorch/checkpoint variants. The runner command
must accept:
  --frames-npy <T,H,W,3 uint8 RGB>
  --queries-npy <N,3 float32 rows: query_frame, x, y>
  --output-npz
and write:
  tracks <T,N,2 float32>
  visibility <T,N bool/float>

Outputs:
  outputs/renders/vggt-omega/ego4d/tracking_compare/<level>/<clip_name>/
    frames/
    queries.npy
    cotracker3/
      overlay.mp4
      tracks.npz
      active_tracks.png
      track_lifetime_hist.png
      trajectories_3d.ply    optional
    bootstapir/
      overlay.mp4
      tracks.npz
      active_tracks.png
      track_lifetime_hist.png
      trajectories_3d.ply    optional
    vggt_predictions_pose.npz optional
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import tempfile

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_INPUT_BASE = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/ego4d/tracking_compare")
ROTATION_LEVELS = ("rotation_low", "rotation_medium", "rotation_high")
DEFAULT_FPS = 6.0
DEFAULT_START_SEC = 0.0
DEFAULT_END_SEC = 10.0
DEFAULT_NUM_FRAMES = 60


def list_clips(input_base: str, levels: list[str]) -> list[tuple[str, str, str]]:
    clips = []
    for level in levels:
        for video_path in sorted(glob.glob(os.path.join(input_base, level, "*.mp4"))):
            clip_name = os.path.splitext(os.path.basename(video_path))[0]
            clips.append((level, clip_name, video_path))
    return clips


def extract_frames(video_path: str, out_dir: str, start_sec: float, end_sec: float, fps: float, num_frames: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(num_frames),
        "-q:v",
        "2",
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(out_dir, "frame_*.jpg")))


def image_paths(image_dir: str, max_frames: int | None = None) -> list[str]:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    if max_frames is not None:
        paths = paths[:max_frames]
    return paths


def load_frames_rgb(image_dir: str, resize_width: int = 0, max_frames: int | None = None) -> np.ndarray:
    import cv2

    frames = []
    for path in image_paths(image_dir, max_frames=max_frames):
        frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Could not read frame: {path}")
        if resize_width > 0 and frame_bgr.shape[1] != resize_width:
            scale = resize_width / frame_bgr.shape[1]
            new_h = max(1, int(round(frame_bgr.shape[0] * scale)))
            frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in {image_dir}")
    return np.stack(frames, axis=0)


def make_query_grid(width: int, height: int, grid_size: int, margin: int) -> np.ndarray:
    xs = np.linspace(margin, width - 1 - margin, grid_size, dtype=np.float32)
    ys = np.linspace(margin, height - 1 - margin, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    query_t = np.zeros_like(xx, dtype=np.float32)
    return np.stack([query_t, xx, yy], axis=-1).reshape(-1, 3)


def run_cotracker3(frames_rgb: np.ndarray, queries: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.from_numpy(queries)[None].float().to(device)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    with torch.inference_mode():
        tracks, visibility = model(video, queries=query_tensor)
    tracks_np = tracks[0].detach().float().cpu().numpy()
    vis_np = visibility[0].detach().float().cpu().numpy()
    if vis_np.ndim == 3:
        vis_np = vis_np[..., 0]
    return tracks_np.astype(np.float32), (vis_np > 0.5)


def run_bootstapir_runner(
    frames_rgb: np.ndarray,
    queries: np.ndarray,
    runner_command: str,
    work_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not runner_command:
        raise ValueError(
            "BootsTAPIR requires --bootstapir-runner. "
            "The runner must accept --frames-npy, --queries-npy, and --output-npz."
        )

    frames_path = os.path.join(work_dir, "frames_rgb.npy")
    queries_path = os.path.join(work_dir, "queries.npy")
    output_path = os.path.join(work_dir, "bootstapir_tracks.npz")
    np.save(frames_path, frames_rgb)
    np.save(queries_path, queries.astype(np.float32))

    cmd = shlex.split(runner_command) + [
        "--frames-npy",
        frames_path,
        "--queries-npy",
        queries_path,
        "--output-npz",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    with np.load(output_path) as data:
        tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"]
    if visibility.ndim == 3:
        visibility = visibility[..., 0]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"BootsTAPIR runner returned invalid tracks shape: {tracks.shape}")
    if tracks.shape[0] != frames_rgb.shape[0] or tracks.shape[1] != queries.shape[0]:
        raise ValueError(
            "BootsTAPIR runner must return tracks with shape [T,N,2]. "
            f"Got tracks={tracks.shape}, frames={frames_rgb.shape[0]}, queries={queries.shape[0]}"
        )
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            "BootsTAPIR runner must return visibility with shape [T,N]. "
            f"Got visibility={visibility.shape}, tracks={tracks.shape}"
        )
    return tracks, visibility.astype(bool)


def track_colors(num_tracks: int) -> np.ndarray:
    import cv2

    hsv = np.zeros((num_tracks, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, num_tracks, endpoint=False).astype(np.uint8)
    hsv[:, 0, 1] = 220
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0, :]


def save_overlay_video(frames_rgb: np.ndarray, tracks: np.ndarray, visibility: np.ndarray, output_path: str, fps: float, trail: int) -> None:
    import cv2
    import imageio_ffmpeg

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        output_path,
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    colors_rgb = track_colors(tracks.shape[1])
    colors_bgr = colors_rgb[:, ::-1]

    for frame_idx in range(len(frames_rgb)):
        vis = cv2.cvtColor(frames_rgb[frame_idx], cv2.COLOR_RGB2BGR)
        start = max(0, frame_idx - trail)
        for track_idx in range(tracks.shape[1]):
            if not visibility[frame_idx, track_idx]:
                continue
            pts = []
            for t in range(start, frame_idx + 1):
                if visibility[t, track_idx]:
                    pts.append(tuple(np.round(tracks[t, track_idx]).astype(int)))
            if not pts:
                continue
            color = tuple(int(x) for x in colors_bgr[track_idx])
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(vis, a, b, color, 2, cv2.LINE_AA)
            cv2.circle(vis, pts[-1], 3, color, -1, cv2.LINE_AA)
        writer.send(vis.tobytes())
    writer.close()
    print(f"    Saved: {output_path}")


def save_track_plots(visibility: np.ndarray, output_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    active = visibility.sum(axis=1)
    lifetime = visibility.sum(axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(active)), active)
    plt.xlabel("Frame")
    plt.ylabel("Active tracks")
    plt.title("Active Track Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "active_tracks.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(lifetime, bins=30)
    plt.xlabel("Track lifetime (frames)")
    plt.ylabel("Count")
    plt.title("Track Lifetime Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "track_lifetime_hist.png"), dpi=150)
    plt.close()


def save_track_metrics(visibility: np.ndarray, output_dir: str) -> None:
    active_count = visibility.sum(axis=1).astype(np.int32)
    lifetime = visibility.sum(axis=0).astype(np.int32)
    np.savez_compressed(
        os.path.join(output_dir, "metrics.npz"),
        visibility=visibility.astype(bool),
        active_track_count=active_count,
        track_lifetime=lifetime,
    )
    summary = {
        "num_frames": int(visibility.shape[0]),
        "num_tracks": int(visibility.shape[1]),
        "active_track_count_mean": float(active_count.mean()),
        "active_track_count_min": int(active_count.min()),
        "active_track_count_max": int(active_count.max()),
        "track_lifetime_mean": float(lifetime.mean()),
        "track_lifetime_median": float(np.median(lifetime)),
        "track_lifetime_min": int(lifetime.min()),
        "track_lifetime_max": int(lifetime.max()),
    }
    save_metadata(os.path.join(output_dir, "metrics.json"), summary)


def save_tracks_npz(path: str, tracks: np.ndarray, visibility: np.ndarray, queries: np.ndarray) -> None:
    np.savez_compressed(path, tracks=tracks, visibility=visibility, queries=queries)
    print(f"    Saved: {path}")


def run_vggt_predictions(image_dir: str, args: argparse.Namespace) -> dict:
    import reconstruction

    model = reconstruction.load_model(args.vggt_checkpoint or reconstruction.CHECKPOINT)
    return reconstruction.run_inference(image_dir, model)


def sample_world_trajectories(
    predictions: dict,
    tracks: np.ndarray,
    visibility: np.ndarray,
    track_frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    world_points = predictions["world_points_from_depth"]
    num_frames, height, width, _ = world_points.shape
    track_width, track_height = track_frame_size
    scale_x = width / float(track_width)
    scale_y = height / float(track_height)
    traj = np.full((num_frames, tracks.shape[1], 3), np.nan, dtype=np.float32)
    traj_visible = np.zeros((num_frames, tracks.shape[1]), dtype=bool)
    for frame_idx in range(min(num_frames, tracks.shape[0])):
        for track_idx in np.where(visibility[frame_idx])[0]:
            x, y = tracks[frame_idx, track_idx]
            xi = int(round(float(x) * scale_x))
            yi = int(round(float(y) * scale_y))
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
    rgb_colors = track_colors(traj.shape[1])
    for track_idx in range(traj.shape[1]):
        frames = np.where(visible[:, track_idx])[0]
        if len(frames) < 2:
            continue
        for a, b in zip(frames[:-1], frames[1:]):
            pa = traj[a, track_idx]
            pb = traj[b, track_idx]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            for s in range(samples_per_segment):
                alpha = s / max(samples_per_segment, 1)
                vertices.append((1 - alpha) * pa + alpha * pb)
                colors.append(rgb_colors[track_idx])
    if not vertices:
        vertices = [np.zeros(3, dtype=np.float32)]
        colors = [np.array([255, 255, 255], dtype=np.uint8)]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trimesh.PointCloud(np.asarray(vertices, dtype=np.float32), colors=np.asarray(colors, dtype=np.uint8)).export(output_path)
    print(f"    Saved: {output_path}")


def save_metadata(path: str, metadata: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_algorithm(
    name: str,
    frames_rgb: np.ndarray,
    queries: np.ndarray,
    out_dir: str,
    args: argparse.Namespace,
    work_dir: str,
    vggt_predictions: dict | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tracks_path = os.path.join(out_dir, "tracks.npz")
    overlay_path = os.path.join(out_dir, "overlay.mp4")
    if os.path.exists(tracks_path) and os.path.exists(overlay_path) and not args.overwrite:
        print(f"  {name}: already done, skipping")
        return

    print(f"  Running {name} ...")
    if name == "cotracker3":
        tracks, visibility = run_cotracker3(frames_rgb, queries, args.device)
    elif name == "bootstapir":
        tracks, visibility = run_bootstapir_runner(frames_rgb, queries, args.bootstapir_runner, work_dir)
    else:
        raise ValueError(name)

    save_tracks_npz(tracks_path, tracks, visibility, queries)
    save_overlay_video(frames_rgb, tracks, visibility, overlay_path, args.fps, args.trail)
    save_track_metrics(visibility, out_dir)
    save_track_plots(visibility, out_dir)

    if args.lift_3d:
        if vggt_predictions is None:
            raise ValueError("--lift-3d requested but VGGT predictions are missing")
        track_height, track_width = frames_rgb.shape[1:3]
        traj, traj_visible = sample_world_trajectories(
            vggt_predictions,
            tracks,
            visibility,
            track_frame_size=(track_width, track_height),
        )
        np.savez_compressed(os.path.join(out_dir, "trajectories_3d.npz"), trajectories_3d=traj, visible_3d=traj_visible)
        save_trajectory_ply(traj, traj_visible, os.path.join(out_dir, "trajectories_3d.ply"), args.samples_per_segment)


def process_clip(level: str, clip_name: str, video_path: str, args: argparse.Namespace) -> None:
    clip_out = os.path.join(args.output_root, level, clip_name)
    frames_dir = os.path.join(clip_out, "frames")
    os.makedirs(clip_out, exist_ok=True)

    existing_frame_count = len(image_paths(frames_dir, max_frames=args.num_frames))
    if existing_frame_count < args.num_frames or args.overwrite_frames:
        print("  Extracting frames ...")
        n = extract_frames(video_path, frames_dir, args.start_sec, args.end_sec, args.fps, args.num_frames)
        print(f"  Extracted {n} frames")
    frames_rgb = load_frames_rgb(frames_dir, resize_width=args.track_width, max_frames=args.num_frames)
    height, width = frames_rgb.shape[1:3]
    queries = make_query_grid(width, height, args.grid_size, args.query_margin)
    np.save(os.path.join(clip_out, "queries.npy"), queries)

    vggt_predictions = None
    if args.lift_3d:
        print("  Running VGGT-Omega for 3D lifting ...")
        vggt_predictions = run_vggt_predictions(frames_dir, args)
        import reconstruction

        reconstruction.save_prediction_pose(
            vggt_predictions,
            os.path.join(clip_out, "vggt_predictions_pose.npz"),
            start_sec=args.start_sec,
            fps=args.fps,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for algorithm in args.algorithms:
            process_algorithm(
                algorithm,
                frames_rgb,
                queries,
                os.path.join(clip_out, algorithm),
                args,
                tmp_dir,
                vggt_predictions,
            )

    save_metadata(
        os.path.join(clip_out, "metadata.json"),
        {
            "level": level,
            "clip_name": clip_name,
            "video_path": video_path,
            "num_frames": int(len(frames_rgb)),
            "frame_size": [int(width), int(height)],
            "fps": args.fps,
            "grid_size": args.grid_size,
            "num_queries": int(len(queries)),
            "algorithms": args.algorithms,
            "lift_3d": args.lift_3d,
            "bootstapir_runner": args.bootstapir_runner,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base", default=DEFAULT_INPUT_BASE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--levels", nargs="+", choices=ROTATION_LEVELS, default=list(ROTATION_LEVELS))
    parser.add_argument("--clip-name", help="Only process one clip basename")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--algorithms", nargs="+", choices=("cotracker3", "bootstapir"), default=["cotracker3", "bootstapir"])
    parser.add_argument("--bootstapir-runner", help="Command for BootsTAPIR adapter script")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--query-margin", type=int, default=24)
    parser.add_argument("--track-width", type=int, default=512, help="Resize frames for trackers; 0 keeps original size")
    parser.add_argument("--trail", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lift-3d", action="store_true")
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--samples-per-segment", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--skip-failed", action="store_true", help="Continue batch processing and save failed clips to failures.json")
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
    if "bootstapir" in args.algorithms and not args.bootstapir_runner:
        raise ValueError("Requested bootstapir but --bootstapir-runner was not provided.")

    print(f"Found {len(clips)} clips")
    failures = []
    for index, (level, clip_name, video_path) in enumerate(clips, start=1):
        print(f"[{index}/{len(clips)}] {level}/{clip_name}")
        try:
            process_clip(level, clip_name, video_path, args)
        except Exception as exc:
            failure = {
                "level": level,
                "clip_name": clip_name,
                "video_path": video_path,
                "error": repr(exc),
            }
            failures.append(failure)
            print(f"  FAILED: {exc}")
            if not args.skip_failed:
                raise
        print()
    if failures:
        os.makedirs(args.output_root, exist_ok=True)
        save_metadata(os.path.join(args.output_root, "failures.json"), failures)
        print(f"Finished with {len(failures)} failed clips. See failures.json.")
    print("All done.")


if __name__ == "__main__":
    main()
