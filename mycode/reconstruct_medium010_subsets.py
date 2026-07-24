"""
Reconstruct rotation_medium_010 with two frame sets:

1. all_60: frames 1..60 extracted at 6 fps from the 0s..10s clip
2. frames_037_060: frames 37..60 only, reconstructed by running VGGT-Omega
   on those 24 frames only

Outputs:
  outputs/renders/vggt-omega/ego4d/medium010_subsets/<clip_name>/all_60/
  outputs/renders/vggt-omega/ego4d/medium010_subsets/<clip_name>/frames_037_060/
    scene.ply
    predictions_pose.npz
    camera_trajectory.png
    metadata.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CLIP_NAME = "rotation_medium_010__1e5cd1c1-7023-4800-b743-16a9c2753ac2"
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label/rotation_medium", f"{CLIP_NAME}.mp4")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/renders/vggt-omega/ego4d/medium010_subsets",
    CLIP_NAME,
)
DEFAULT_FPS = 6.0
DEFAULT_START_SEC = 0.0
DEFAULT_END_SEC = 10.0


def image_paths(image_dir: str) -> list[str]:
    return sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )


def copy_frame_subset(src_dir: str, dst_dir: str, start_frame: int, end_frame: int) -> list[int]:
    """Copy 1-based inclusive frame range and return 0-based original frame indices."""
    os.makedirs(dst_dir, exist_ok=True)
    original_indices = []
    for frame_number in range(start_frame, end_frame + 1):
        src = os.path.join(src_dir, f"frame_{frame_number:06d}.jpg")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(dst_dir, f"frame_{frame_number:06d}.jpg")
        shutil.copy2(src, dst)
        original_indices.append(frame_number - 1)
    return original_indices


def save_prediction_pose_with_indices(
    predictions: dict,
    save_path: str,
    original_frame_indices: list[int],
    start_sec: float,
    fps: float,
) -> None:
    frame_indices = np.asarray(original_frame_indices, dtype=np.int64)
    timestamps = start_sec + frame_indices.astype(np.float64) / fps
    np.savez_compressed(
        save_path,
        extrinsic_w2c=predictions["extrinsic"],
        intrinsic=predictions["intrinsic"],
        pose_enc=predictions["pose_enc"],
        timestamps=timestamps,
        frame_indices=frame_indices,
    )
    print(f"  Saved: {save_path}")


def save_metadata(
    save_path: str,
    clip_name: str,
    video: str,
    subset_name: str,
    num_frames: int,
    original_frame_indices: list[int],
    fps: float,
    start_sec: float,
    end_sec: float,
    image_resolution: int,
    checkpoint: str,
) -> None:
    metadata = {
        "clip_name": clip_name,
        "video": video,
        "subset_name": subset_name,
        "num_frames": num_frames,
        "original_frame_indices": original_frame_indices,
        "original_frame_numbers_1based": [idx + 1 for idx in original_frame_indices],
        "fps": fps,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "image_resolution": image_resolution,
        "checkpoint": checkpoint,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {save_path}")


def reconstruct_subset(
    reconstruction,
    model,
    image_dir: str,
    out_dir: str,
    subset_name: str,
    original_frame_indices: list[int],
    args: argparse.Namespace,
) -> None:
    scene_path = os.path.join(out_dir, "scene.ply")
    pose_path = os.path.join(out_dir, "predictions_pose.npz")
    if os.path.exists(scene_path) and os.path.exists(pose_path) and not args.overwrite:
        print(f"  Already done, skipping: {out_dir}")
        return

    paths = image_paths(image_dir)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(paths)} in {image_dir}")
    if len(paths) != len(original_frame_indices):
        raise ValueError(
            f"Frame count mismatch for {subset_name}: "
            f"images={len(paths)}, indices={len(original_frame_indices)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Running inference for {subset_name} ({len(paths)} frames) ...")
    predictions = reconstruction.run_inference(image_dir, model)

    print(f"  Saving {subset_name} outputs ...")
    reconstruction.save_trajectory_png(predictions["extrinsic"], os.path.join(out_dir, "camera_trajectory.png"))
    save_prediction_pose_with_indices(
        predictions,
        pose_path,
        original_frame_indices=original_frame_indices,
        start_sec=args.start_sec,
        fps=args.fps,
    )
    reconstruction.save_ply(predictions, scene_path)
    save_metadata(
        os.path.join(out_dir, "metadata.json"),
        clip_name=CLIP_NAME,
        video=args.video,
        subset_name=subset_name,
        num_frames=len(paths),
        original_frame_indices=original_frame_indices,
        fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        image_resolution=reconstruction.IMAGE_RES,
        checkpoint=args.checkpoint,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", help="VGGT-Omega checkpoint path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC)
    parser.add_argument("--subset-start-frame", type=int, default=37)
    parser.add_argument("--subset-end-frame", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    import reconstruction

    if args.checkpoint is None:
        args.checkpoint = reconstruction.CHECKPOINT

    print(f"Loading model from {args.checkpoint}")
    model = reconstruction.load_model(args.checkpoint)
    print("Model loaded.")

    with tempfile.TemporaryDirectory() as tmp_root:
        all_frames_dir = os.path.join(tmp_root, "all_60_frames")
        subset_name = f"frames_{args.subset_start_frame:03d}_{args.subset_end_frame:03d}"
        subset_dir = os.path.join(tmp_root, subset_name)

        print(f"Extracting frames from {args.video}")
        num_frames = reconstruction.extract_frames(
            args.video,
            all_frames_dir,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            fps=args.fps,
        )
        print(f"Extracted {num_frames} frames at {args.fps} fps")
        if num_frames < args.subset_end_frame:
            raise ValueError(
                f"Need at least {args.subset_end_frame} extracted frames, got {num_frames}"
            )

        all_indices = list(range(num_frames))
        subset_indices = copy_frame_subset(
            all_frames_dir,
            subset_dir,
            start_frame=args.subset_start_frame,
            end_frame=args.subset_end_frame,
        )

        reconstruct_subset(
            reconstruction,
            model,
            image_dir=all_frames_dir,
            out_dir=os.path.join(args.output_dir, "all_60"),
            subset_name="all_60",
            original_frame_indices=all_indices,
            args=args,
        )
        reconstruct_subset(
            reconstruction,
            model,
            image_dir=subset_dir,
            out_dir=os.path.join(args.output_dir, subset_name),
            subset_name=subset_name,
            original_frame_indices=subset_indices,
            args=args,
        )

    print("All done.")


if __name__ == "__main__":
    main()
