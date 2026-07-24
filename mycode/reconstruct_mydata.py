"""
Run VGGT-Omega reconstruction on images under data/mydata.

Default input:
  data/mydata/*.jpg

Default output:
  outputs/renders/vggt-omega/mydata/reconstruction/
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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/mydata")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/mydata/reconstruction")
DEFAULT_FPS = 1.0


def image_paths(image_dir: str) -> list[str]:
    return sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )


def save_metadata(
    save_path: str,
    image_dir: str,
    output_dir: str,
    num_images: int,
    fps: float,
    image_resolution: int,
    checkpoint: str,
) -> None:
    metadata = {
        "image_dir": image_dir,
        "output_dir": output_dir,
        "num_images": num_images,
        "fps_for_pose_timestamps": fps,
        "image_resolution": image_resolution,
        "checkpoint": checkpoint,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", help="VGGT-Omega checkpoint path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Only used for saved pose timestamps")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = image_paths(args.image_dir)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images, got {len(paths)} in {args.image_dir}")

    import reconstruction

    if args.checkpoint is None:
        args.checkpoint = reconstruction.CHECKPOINT

    scene_path = os.path.join(args.output_dir, "scene.ply")
    pose_path = os.path.join(args.output_dir, "predictions_pose.npz")
    if os.path.exists(scene_path) and os.path.exists(pose_path) and not args.overwrite:
        print(f"Already done, skipping: {args.output_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(paths)} images under {args.image_dir}")
    print(f"Loading model from {args.checkpoint}")
    model = reconstruction.load_model(args.checkpoint)
    print("Model loaded.")

    print("Running inference ...")
    predictions = reconstruction.run_inference(args.image_dir, model)

    print("Saving outputs ...")
    reconstruction.save_trajectory_png(
        predictions["extrinsic"],
        os.path.join(args.output_dir, "camera_trajectory.png"),
    )
    reconstruction.save_prediction_pose(
        predictions,
        pose_path,
        start_sec=0.0,
        fps=args.fps,
    )
    reconstruction.save_ply(predictions, scene_path)
    save_metadata(
        os.path.join(args.output_dir, "metadata.json"),
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_images=len(paths),
        fps=args.fps,
        image_resolution=reconstruction.IMAGE_RES,
        checkpoint=args.checkpoint,
    )
    print("All done.")


if __name__ == "__main__":
    main()
