"""
VGGT-Omega reconstruction for paired Ego4D image folders.

Inputs:
  data/ego4d/image/high/<clip_name>/frame_*.jpg
  data/ego4d/image/high copy/<clip_name>/frame_*.jpg

Outputs:
  outputs/renders/vggt-omega/ego4d/compare/high/<clip_name>/
  outputs/renders/vggt-omega/ego4d/compare/high_copy/<clip_name>/
    camera_trajectory.png
    predictions_pose.npz
    scene.ply
    metadata.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_IMAGE_ROOT = os.path.join(PROJECT_ROOT, "data/ego4d/image")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/ego4d/compare")
DEFAULT_IMAGE_SETS = ("high", "high copy")
DEFAULT_FPS = 6.0


def output_set_name(image_set: str) -> str:
    return image_set.replace(" ", "_")


def list_image_dirs(image_root: str, image_sets: tuple[str, ...]) -> list[tuple[str, str, str]]:
    items = []
    for image_set in image_sets:
        set_dir = os.path.join(image_root, image_set)
        if not os.path.isdir(set_dir):
            print(f"Missing image set, skipping: {set_dir}")
            continue
        for entry in sorted(os.scandir(set_dir), key=lambda x: x.name):
            if not entry.is_dir():
                continue
            image_paths = sorted(
                p
                for p in glob.glob(os.path.join(entry.path, "*"))
                if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
            )
            if image_paths:
                items.append((image_set, entry.name, entry.path))
    return items


def save_metadata(
    save_path: str,
    image_set: str,
    clip_name: str,
    image_dir: str,
    num_frames: int,
    fps: float,
    image_resolution: int,
    checkpoint: str,
) -> None:
    metadata = {
        "image_set": image_set,
        "clip_name": clip_name,
        "image_dir": image_dir,
        "num_frames": num_frames,
        "fps": fps,
        "image_resolution": image_resolution,
        "checkpoint": checkpoint,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {save_path}")


def reconstruct_image_dir(
    reconstruction,
    model,
    image_set: str,
    clip_name: str,
    image_dir: str,
    output_root: str,
    fps: float,
    overwrite: bool,
) -> None:
    out_dir = os.path.join(output_root, output_set_name(image_set), clip_name)
    scene_path = os.path.join(out_dir, "scene.ply")
    pose_path = os.path.join(out_dir, "predictions_pose.npz")

    if os.path.exists(scene_path) and os.path.exists(pose_path) and not overwrite:
        print("  Already done, skipping.")
        return

    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    if len(image_paths) < 2:
        print(f"  SKIP: too few images ({len(image_paths)})")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"  Running inference on {len(image_paths)} images ...")
    predictions = reconstruction.run_inference(image_dir, model)

    print("  Saving outputs ...")
    reconstruction.save_trajectory_png(
        predictions["extrinsic"],
        os.path.join(out_dir, "camera_trajectory.png"),
    )
    reconstruction.save_prediction_pose(
        predictions,
        pose_path,
        start_sec=0.0,
        fps=fps,
    )
    reconstruction.save_ply(predictions, scene_path)
    save_metadata(
        os.path.join(out_dir, "metadata.json"),
        image_set=image_set,
        clip_name=clip_name,
        image_dir=image_dir,
        num_frames=len(image_paths),
        fps=fps,
        image_resolution=reconstruction.IMAGE_RES,
        checkpoint=reconstruction.CHECKPOINT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-sets", nargs="+", default=list(DEFAULT_IMAGE_SETS))
    parser.add_argument("--clip-name", help="Only reconstruct one clip basename")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = list_image_dirs(args.image_root, tuple(args.image_sets))
    if args.clip_name:
        items = [item for item in items if item[1] == args.clip_name]

    print(f"Found {len(items)} image folders under {args.image_root}")
    print(f"Writing compare reconstructions to {args.output_root}")
    if not items:
        return

    import reconstruction

    print(f"Loading model from {reconstruction.CHECKPOINT}")
    model = reconstruction.load_model(reconstruction.CHECKPOINT)
    print("Model loaded.\n")

    for index, (image_set, clip_name, image_dir) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {image_set}/{clip_name}")
        reconstruct_image_dir(
            reconstruction,
            model,
            image_set=image_set,
            clip_name=clip_name,
            image_dir=image_dir,
            output_root=args.output_root,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        print()

    print("All done.")


if __name__ == "__main__":
    main()
