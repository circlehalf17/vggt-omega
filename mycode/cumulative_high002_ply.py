"""
Create per-frame and cumulative PLY files for Ego4D rotation_high_002.

For 6 fps frames, this writes:
  per_frame/frame_000001.ply = P1
  per_frame/frame_000002.ply = P2
  per_frame/frame_000003.ply = P3
  ...

  cumulative/cumulative_000001.ply = P1
  cumulative/cumulative_000002.ply = P1 union P2
  cumulative/cumulative_000003.ply = P1 union P2 union P3
  ...

Default input is the original high_002 video, sampled from 0s to 10s at 6 fps.
Default output:
  outputs/renders/vggt-omega/ego4d/cumulative/high_002/<clip_name>/

It also writes subset visualizations under without_blur/:
  all_60_frames.ply        = P1 union ... union P60
  high_copy_1_48_frames.ply = union of frames listed in data/ego4d/image/high copy 1/<clip_name>/
  high_12_frames.ply        = union of frames listed in data/ego4d/image/high/<clip_name>/
  high_copy_1_48_reconstruction.ply = reconstruction from only the 48 selected frames
  high_12_reconstruction.ply        = reconstruction from only the 12 selected frames
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile

import numpy as np

try:
    import trimesh
except ModuleNotFoundError:
    trimesh = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CLIP_NAME = "rotation_high_002__057670b3-f243-416c-b41b-82d309fd2b88"
CHECKPOINT = os.path.join(PROJECT_ROOT, "outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt")
SAMPLE_FPS = 6.0
CLIP_START_SEC = 0.0
CLIP_END_SEC = 10.0
CONF_THRES = 20.0
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label/rotation_high", f"{CLIP_NAME}.mp4")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/renders/vggt-omega/ego4d/cumulative/high_002",
    CLIP_NAME,
)
DEFAULT_HIGH_COPY_1_DIR = os.path.join(PROJECT_ROOT, "data/ego4d/image/high copy 1", CLIP_NAME)
DEFAULT_HIGH_DIR = os.path.join(PROJECT_ROOT, "data/ego4d/image/high", CLIP_NAME)


def require_trimesh() -> None:
    if trimesh is None:
        raise ModuleNotFoundError("This script requires trimesh. Install it in the VGGT-Omega environment.")


def images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def frame_index_from_path(path: str) -> int:
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    if not stem.startswith("frame_"):
        raise ValueError(f"Expected frame_XXXXXX image name, got: {basename}")
    return int(stem.split("_", 1)[1]) - 1


def image_frame_indices(image_dir: str) -> list[int]:
    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    return [frame_index_from_path(path) for path in image_paths]


def frame_point_clouds(
    predictions: dict,
    conf_thres: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    max_points_per_frame: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0

    rgb = images_to_rgb(predictions["images"])
    colors_all = (rgb * 255).clip(0, 255).astype(np.uint8)

    frame_clouds = []
    conf_thres = max(0.0, float(conf_thres))
    for frame_idx in range(len(points)):
        vertices = points[frame_idx].reshape(-1, 3)
        colors = colors_all[frame_idx].reshape(-1, 3)
        frame_conf = conf[frame_idx].reshape(-1)

        mask = np.isfinite(vertices).all(axis=1) & np.isfinite(frame_conf)
        if conf_thres > 0 and np.any(mask):
            threshold = np.percentile(frame_conf[mask], conf_thres)
            mask &= frame_conf >= threshold
        mask &= frame_conf > 1e-5

        vertices = vertices[mask].astype(np.float32)
        colors = colors[mask]
        vertices, colors = limit_points(vertices, colors, max_points_per_frame)
        frame_clouds.append((vertices, colors))

    return frame_clouds


def save_cumulative_plys(
    frame_clouds: list[tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    max_points_per_ply: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cumulative_vertices = []
    cumulative_colors = []

    for frame_idx, (vertices, colors) in enumerate(frame_clouds, start=1):
        cumulative_vertices.append(vertices)
        cumulative_colors.append(colors)

        out_vertices = np.concatenate(cumulative_vertices, axis=0)
        out_colors = np.concatenate(cumulative_colors, axis=0)
        out_vertices, out_colors = limit_points(out_vertices, out_colors, max_points_per_ply)

        out_path = os.path.join(output_dir, f"cumulative_{frame_idx:06d}.ply")
        trimesh.PointCloud(vertices=out_vertices, colors=out_colors).export(out_path)
        print(f"  [{frame_idx:03d}/{len(frame_clouds):03d}] Saved {out_path} ({len(out_vertices)} points)")


def save_per_frame_plys(
    frame_clouds: list[tuple[np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx, (vertices, colors) in enumerate(frame_clouds, start=1):
        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.ply")
        trimesh.PointCloud(vertices=vertices, colors=colors).export(out_path)
        print(f"  [{frame_idx:03d}/{len(frame_clouds):03d}] Saved {out_path} ({len(vertices)} points)")


def save_frame_union_ply(
    frame_clouds: list[tuple[np.ndarray, np.ndarray]],
    frame_indices: list[int],
    output_path: str,
    max_points: int,
) -> None:
    valid_indices = [idx for idx in frame_indices if 0 <= idx < len(frame_clouds)]
    if not valid_indices:
        raise ValueError(f"No valid frame indices for {output_path}")

    vertices = np.concatenate([frame_clouds[idx][0] for idx in valid_indices], axis=0)
    colors = np.concatenate([frame_clouds[idx][1] for idx in valid_indices], axis=0)
    vertices, colors = limit_points(vertices, colors, max_points)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trimesh.PointCloud(vertices=vertices, colors=colors).export(output_path)
    print(f"  Saved {output_path} ({len(vertices)} points from {len(valid_indices)} frames)")


def save_without_blur_visualizations(
    frame_clouds: list[tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    high_copy_1_dir: str,
    high_dir: str,
    max_points_per_ply: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    subsets = [
        ("all_60_frames.ply", list(range(len(frame_clouds)))),
        ("high_copy_1_48_frames.ply", image_frame_indices(high_copy_1_dir)),
        ("high_12_frames.ply", image_frame_indices(high_dir)),
    ]
    for filename, frame_indices in subsets:
        save_frame_union_ply(
            frame_clouds,
            frame_indices,
            os.path.join(output_dir, filename),
            max_points=max_points_per_ply,
        )


def save_reconstruction_from_image_dir(
    image_dir: str,
    model,
    output_path: str,
    conf_thres: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    max_points_per_frame: int,
    max_points_per_ply: int,
    run_inference,
) -> None:
    n = sorted_image_count(image_dir)
    if n < 1:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Running subset-only reconstruction from {image_dir} ({n} frames) ...")
    predictions = run_inference(image_dir, model)
    frame_clouds = frame_point_clouds(
        predictions,
        conf_thres=conf_thres,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points_per_frame=max_points_per_frame,
    )
    save_frame_union_ply(
        frame_clouds,
        list(range(len(frame_clouds))),
        output_path,
        max_points=max_points_per_ply,
    )


def save_subset_only_reconstructions(
    model,
    run_inference,
    output_dir: str,
    high_copy_1_dir: str,
    high_dir: str,
    conf_thres: float,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    max_points_per_frame: int,
    max_points_per_ply: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_reconstruction_from_image_dir(
        high_copy_1_dir,
        model,
        os.path.join(output_dir, "high_copy_1_48_reconstruction.ply"),
        conf_thres=conf_thres,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points_per_frame=max_points_per_frame,
        max_points_per_ply=max_points_per_ply,
        run_inference=run_inference,
    )
    save_reconstruction_from_image_dir(
        high_dir,
        model,
        os.path.join(output_dir, "high_12_reconstruction.ply"),
        conf_thres=conf_thres,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points_per_frame=max_points_per_frame,
        max_points_per_ply=max_points_per_ply,
        run_inference=run_inference,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--image-dir", help="Use an existing image directory instead of extracting from --video")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-copy-1-dir", default=DEFAULT_HIGH_COPY_1_DIR)
    parser.add_argument("--high-dir", default=DEFAULT_HIGH_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--start-sec", type=float, default=CLIP_START_SEC)
    parser.add_argument("--end-sec", type=float, default=CLIP_END_SEC)
    parser.add_argument("--conf-thres", type=float, default=CONF_THRES)
    parser.add_argument("--no-filter-depth-edges", action="store_true")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03)
    parser.add_argument("--max-points-per-frame", type=int, default=0)
    parser.add_argument(
        "--max-points-per-ply",
        type=int,
        default=0,
        help="0 keeps the full cumulative union. Use e.g. 1000000 to cap each PLY size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_image_count(image_dir: str) -> int:
    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    return len(image_paths)


def main() -> None:
    args = parse_args()
    require_trimesh()
    from reconstruction import extract_frames, load_model, run_inference, save_prediction_pose

    cumulative_dir = os.path.join(args.output_dir, "cumulative")
    per_frame_dir = os.path.join(args.output_dir, "per_frame")
    without_blur_dir = os.path.join(args.output_dir, "without_blur")
    first_outputs = [
        os.path.join(cumulative_dir, "cumulative_000001.ply"),
        os.path.join(per_frame_dir, "frame_000001.ply"),
        os.path.join(without_blur_dir, "all_60_frames.ply"),
        os.path.join(without_blur_dir, "high_copy_1_48_frames.ply"),
        os.path.join(without_blur_dir, "high_12_frames.ply"),
        os.path.join(without_blur_dir, "high_copy_1_48_reconstruction.ply"),
        os.path.join(without_blur_dir, "high_12_reconstruction.ply"),
    ]
    if any(os.path.exists(path) for path in first_outputs) and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to rewrite: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("Model loaded.")

    if args.image_dir:
        image_dir = args.image_dir
        n = sorted_image_count(image_dir)
        print(f"Using existing image directory: {image_dir} ({n} frames)")
        if n < 1:
            raise ValueError(f"No images found in {image_dir}")
        print("Running inference ...")
        predictions = run_inference(image_dir, model)
    else:
        if not os.path.exists(args.video):
            raise FileNotFoundError(args.video)
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Extracting frames from {args.video}")
            n = extract_frames(args.video, tmp_dir, args.start_sec, args.end_sec, args.fps)
            print(f"Extracted {n} frames at {args.fps} fps")
            if n < 1:
                raise ValueError("No frames extracted")
            print("Running inference ...")
            predictions = run_inference(tmp_dir, model)

    print("Saving prediction pose metadata ...")
    save_prediction_pose(
        predictions,
        os.path.join(args.output_dir, "predictions_pose.npz"),
        args.start_sec,
        args.fps,
    )

    print("Building per-frame point clouds ...")
    frame_clouds = frame_point_clouds(
        predictions,
        conf_thres=args.conf_thres,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        max_points_per_frame=args.max_points_per_frame,
    )

    print(f"Saving {len(frame_clouds)} per-frame PLY files to {per_frame_dir}")
    save_per_frame_plys(frame_clouds, per_frame_dir)

    print(f"Saving {len(frame_clouds)} cumulative PLY files to {cumulative_dir}")
    save_cumulative_plys(frame_clouds, cumulative_dir, args.max_points_per_ply)

    print(f"Saving without-blur subset visualizations to {without_blur_dir}")
    save_without_blur_visualizations(
        frame_clouds,
        without_blur_dir,
        high_copy_1_dir=args.high_copy_1_dir,
        high_dir=args.high_dir,
        max_points_per_ply=args.max_points_per_ply,
    )

    print(f"Saving subset-only reconstructions to {without_blur_dir}")
    save_subset_only_reconstructions(
        model,
        run_inference,
        without_blur_dir,
        high_copy_1_dir=args.high_copy_1_dir,
        high_dir=args.high_dir,
        conf_thres=args.conf_thres,
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        max_points_per_frame=args.max_points_per_frame,
        max_points_per_ply=args.max_points_per_ply,
    )
    print("All done.")


if __name__ == "__main__":
    main()
