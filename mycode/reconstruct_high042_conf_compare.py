"""
Compare rotation_high_042 reconstruction with and without confidence percentile filtering.

This runs VGGT-Omega inference once on 60 frames sampled at 6 fps, then writes:

  original_conf20/scene.ply
    Existing behavior: remove the lowest 20% confidence points.

  no_conf_percentile_filter/scene.ply
    Removes only invalid/non-positive-confidence points and optional depth-edge
    points, but does not remove the lowest confidence percentile.
  no_conf_percentile_filter/per_frame/frame_000001.ply ...
    Per-frame reconstruction without the lowest-20% confidence percentile filter.

Outputs:
  outputs/renders/vggt-omega/ego4d/high042_conf_compare/<clip_name>/
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CLIP_NAME = "rotation_high_042__b493e8e3-4e18-4480-8e63-c21e25bf99f7"
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label/rotation_high", f"{CLIP_NAME}.mp4")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/renders/vggt-omega/ego4d/high042_conf_compare",
    CLIP_NAME,
)
DEFAULT_FPS = 6.0
DEFAULT_START_SEC = 0.0
DEFAULT_END_SEC = 10.0


def images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def save_ply_without_conf_percentile_filter(
    predictions: dict,
    save_path: str,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> int:
    import trimesh
    from visual_util import depth_edge

    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0

    vertices = points.reshape(-1, 3)
    colors = images_to_rgb(predictions["images"]).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    mask &= conf > 1e-5

    vertices = vertices[mask].astype(np.float32)
    colors = colors[mask]
    vertices, colors = limit_points(vertices, colors, max_points)

    if vertices.size == 0:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trimesh.PointCloud(vertices=vertices, colors=colors).export(save_path)
    print(f"  Saved: {save_path} ({len(vertices)} points)")
    return int(len(vertices))


def slice_predictions(predictions: dict, indices: list[int]) -> dict:
    n = len(predictions["extrinsic"])
    sliced = {}
    for key, value in predictions.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            sliced[key] = value[indices]
        else:
            sliced[key] = value
    return sliced


def save_per_frame_plys_without_conf_percentile_filter(
    predictions: dict,
    output_dir: str,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> list[int]:
    os.makedirs(output_dir, exist_ok=True)
    point_counts = []
    num_frames = len(predictions["extrinsic"])
    for frame_idx in range(num_frames):
        frame_predictions = slice_predictions(predictions, [frame_idx])
        out_path = os.path.join(output_dir, f"frame_{frame_idx + 1:06d}.ply")
        point_count = save_ply_without_conf_percentile_filter(
            frame_predictions,
            out_path,
            max_points=max_points,
            filter_depth_edges=filter_depth_edges,
            depth_edge_rtol=depth_edge_rtol,
        )
        point_counts.append(point_count)
    return point_counts


def save_metadata(
    save_path: str,
    mode: str,
    clip_name: str,
    video: str,
    num_frames: int,
    fps: float,
    start_sec: float,
    end_sec: float,
    image_resolution: int,
    checkpoint: str,
    max_points: int,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    num_saved_points: int | None = None,
    per_frame_point_counts: list[int] | None = None,
) -> None:
    metadata = {
        "mode": mode,
        "clip_name": clip_name,
        "video": video,
        "num_frames": num_frames,
        "fps": fps,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "image_resolution": image_resolution,
        "checkpoint": checkpoint,
        "max_points": max_points,
        "filter_depth_edges": filter_depth_edges,
        "depth_edge_rtol": depth_edge_rtol,
        "num_saved_points": num_saved_points,
        "per_frame_point_counts": per_frame_point_counts,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", help="VGGT-Omega checkpoint path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--no-filter-depth-edges", action="store_true")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    import reconstruction

    if args.checkpoint is None:
        args.checkpoint = reconstruction.CHECKPOINT

    original_dir = os.path.join(args.output_dir, "original_conf20")
    no_filter_dir = os.path.join(args.output_dir, "no_conf_percentile_filter")
    original_scene = os.path.join(original_dir, "scene.ply")
    no_filter_scene = os.path.join(no_filter_dir, "scene.ply")
    no_filter_per_frame_dir = os.path.join(no_filter_dir, "per_frame")
    if (
        os.path.exists(original_scene)
        and os.path.exists(no_filter_scene)
        and os.path.exists(os.path.join(no_filter_per_frame_dir, "frame_000001.ply"))
        and not args.overwrite
    ):
        print(f"Already done, skipping: {args.output_dir}")
        return

    print(f"Loading model from {args.checkpoint}")
    model = reconstruction.load_model(args.checkpoint)
    print("Model loaded.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Extracting frames from {args.video}")
        num_frames = reconstruction.extract_frames(
            args.video,
            tmp_dir,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            fps=args.fps,
        )
        print(f"Extracted {num_frames} frames at {args.fps} fps")
        if num_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {num_frames}")

        print("Running inference ...")
        predictions = reconstruction.run_inference(tmp_dir, model)

    print("Saving original confidence-filtered reconstruction ...")
    os.makedirs(original_dir, exist_ok=True)
    reconstruction.save_trajectory_png(predictions["extrinsic"], os.path.join(original_dir, "camera_trajectory.png"))
    reconstruction.save_prediction_pose(predictions, os.path.join(original_dir, "predictions_pose.npz"), args.start_sec, args.fps)
    reconstruction.save_ply(predictions, original_scene)
    save_metadata(
        os.path.join(original_dir, "metadata.json"),
        mode="original_conf20",
        clip_name=CLIP_NAME,
        video=args.video,
        num_frames=num_frames,
        fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        image_resolution=reconstruction.IMAGE_RES,
        checkpoint=args.checkpoint,
        max_points=reconstruction.MAX_POINTS,
        filter_depth_edges=True,
        depth_edge_rtol=0.03,
    )

    print("Saving reconstruction without confidence percentile filtering ...")
    os.makedirs(no_filter_dir, exist_ok=True)
    reconstruction.save_trajectory_png(predictions["extrinsic"], os.path.join(no_filter_dir, "camera_trajectory.png"))
    reconstruction.save_prediction_pose(predictions, os.path.join(no_filter_dir, "predictions_pose.npz"), args.start_sec, args.fps)
    num_saved_points = save_ply_without_conf_percentile_filter(
        predictions,
        no_filter_scene,
        max_points=args.max_points,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
    )
    print("Saving per-frame reconstructions without confidence percentile filtering ...")
    per_frame_point_counts = save_per_frame_plys_without_conf_percentile_filter(
        predictions,
        no_filter_per_frame_dir,
        max_points=args.max_points,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
    )
    save_metadata(
        os.path.join(no_filter_dir, "metadata.json"),
        mode="no_conf_percentile_filter",
        clip_name=CLIP_NAME,
        video=args.video,
        num_frames=num_frames,
        fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        image_resolution=reconstruction.IMAGE_RES,
        checkpoint=args.checkpoint,
        max_points=args.max_points,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        num_saved_points=num_saved_points,
        per_frame_point_counts=per_frame_point_counts,
    )
    print("All done.")


if __name__ == "__main__":
    main()
