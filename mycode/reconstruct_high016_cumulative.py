"""
Create per-frame and cumulative PLY files for Ego4D rotation_high_016.

This samples the clip at 6 fps for 0s..10s, runs VGGT-Omega once on all
60 frames, then writes:

  per_frame/frame_000001.ply = P1
  per_frame/frame_000002.ply = P2
  ...

  cumulative/cumulative_000001.ply = P1
  cumulative/cumulative_000002.ply = P1 union P2
  cumulative/cumulative_000003.ply = P1 union P2 union P3
  ...

Default output:
  outputs/renders/vggt-omega/ego4d/cumulative/high_016/<clip_name>/
"""

from __future__ import annotations

import argparse
import os
import tempfile

from cumulative_high002_ply import (
    CONF_THRES,
    PROJECT_ROOT,
    SAMPLE_FPS,
    CLIP_START_SEC,
    CLIP_END_SEC,
    frame_point_clouds,
    require_trimesh,
    save_cumulative_plys,
    save_per_frame_plys,
)


CLIP_NAME = "rotation_high_016__3bba24a5-f80c-4798-96a8-96687b4aba60"
CHECKPOINT = os.path.join(PROJECT_ROOT, "outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt")
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label/rotation_high", f"{CLIP_NAME}.mp4")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/renders/vggt-omega/ego4d/cumulative/high_016",
    CLIP_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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


def main() -> None:
    args = parse_args()
    require_trimesh()
    from reconstruction import extract_frames, load_model, run_inference, save_prediction_pose

    cumulative_dir = os.path.join(args.output_dir, "cumulative")
    per_frame_dir = os.path.join(args.output_dir, "per_frame")
    first_outputs = [
        os.path.join(cumulative_dir, "cumulative_000001.ply"),
        os.path.join(per_frame_dir, "frame_000001.ply"),
    ]
    if any(os.path.exists(path) for path in first_outputs) and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to rewrite: {args.output_dir}")

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("Model loaded.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Extracting frames from {args.video}")
        n = extract_frames(args.video, tmp_dir, args.start_sec, args.end_sec, args.fps)
        print(f"Extracted {n} frames at {args.fps} fps")
        if n < 2:
            raise ValueError(f"Need at least 2 frames, got {n}")

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
    print("All done.")


if __name__ == "__main__":
    main()
