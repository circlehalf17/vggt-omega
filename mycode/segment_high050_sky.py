"""
Run sky segmentation for rotation_high_050 using visual_util.py.

This extracts 60 frames at 6 fps from the 0s..10s clip, then uses
visual_util.segment_sky to create masks.

Important: visual_util.segment_sky writes a keep-mask used by apply_sky_mask:
  - sky pixels:     0
  - non-sky pixels: 255

Outputs:
  outputs/renders/vggt-omega/ego4d/skyseg/rotation_high_050__.../
    images/frame_000001.jpg ...
    sky_masks/frame_000001.jpg ...
    masked_images/frame_000001.jpg ...
    sky_masks_strict/frame_000001.jpg ...
    masked_images_strict/frame_000001.jpg ...
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CLIP_NAME = "rotation_high_050__dc708402-b918-4bef-b2ea-88bf22a7fb1f"
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label/rotation_high", f"{CLIP_NAME}.mp4")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/renders/vggt-omega/ego4d/skyseg",
    CLIP_NAME,
)
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "skyseg.onnx")
DEFAULT_FPS = 6.0
DEFAULT_START_SEC = 0.0
DEFAULT_END_SEC = 10.0


def extract_frames(video_path: str, image_dir: str, start_sec: float, end_sec: float, fps: float) -> int:
    os.makedirs(image_dir, exist_ok=True)
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
        str(end_sec - start_sec),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        os.path.join(image_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(image_dir, "frame_*.jpg")))


def image_paths(image_dir: str) -> list[str]:
    return sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
    )


def ensure_skyseg_model(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    from visual_util import download_file_from_url

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    print(f"Downloading skyseg model to {model_path}")
    download_file_from_url(
        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
        model_path,
    )
    return model_path


def save_masked_image(image_path: str, mask_path: str, output_path: str) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked = (image.astype(np.float32) * (mask[..., None].astype(np.float32) / 255.0)).clip(0, 255)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, masked.astype(np.uint8))


def save_strict_sky_mask(
    image_path: str,
    base_mask_path: str,
    output_mask_path: str,
    dilate_px: int,
    upper_ratio: float,
) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    keep_mask = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if keep_mask is None:
        raise ValueError(f"Could not read mask: {base_mask_path}")
    if keep_mask.shape != image.shape[:2]:
        keep_mask = cv2.resize(keep_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Base convention: sky=0, non-sky=255. Convert to remove-mask.
    sky_remove = keep_mask < 128
    h, w = sky_remove.shape

    if dilate_px > 0:
        kernel_size = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        sky_remove = cv2.dilate(sky_remove.astype(np.uint8) * 255, kernel) > 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    b, g, r = cv2.split(image)
    y = np.arange(h)[:, None]
    upper = y < int(h * upper_ratio)

    # Aggressive sky/glare candidates: blue sky, bright sunset/orange sky, and blown-out glare.
    blue_sky = (hue >= 85) & (hue <= 130) & (sat >= 25) & (val >= 90)
    sunset_sky = ((hue <= 28) | (hue >= 165)) & (sat >= 30) & (val >= 115) & (r > b)
    bright_glare = (val >= 205) & (sat <= 120)
    reddish_glare = (r >= 180) & (g >= 100) & (r > b + 20)
    strict_extra = upper & (blue_sky | sunset_sky | bright_glare | reddish_glare)

    sky_remove |= strict_extra
    strict_keep = np.where(sky_remove, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    cv2.imwrite(output_mask_path, strict_keep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to skyseg.onnx")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    parser.add_argument("--end-sec", type=float, default=DEFAULT_END_SEC)
    parser.add_argument("--skip-extract", action="store_true", help="Reuse existing output-dir/images")
    parser.add_argument("--strict-dilate-px", type=int, default=9)
    parser.add_argument("--strict-upper-ratio", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    image_dir = os.path.join(args.output_dir, "images")
    mask_dir = os.path.join(args.output_dir, "sky_masks")
    masked_image_dir = os.path.join(args.output_dir, "masked_images")
    strict_mask_dir = os.path.join(args.output_dir, "sky_masks_strict")
    strict_masked_image_dir = os.path.join(args.output_dir, "masked_images_strict")
    first_mask = os.path.join(mask_dir, "frame_000001.jpg")
    first_masked_image = os.path.join(masked_image_dir, "frame_000001.jpg")
    first_strict_mask = os.path.join(strict_mask_dir, "frame_000001.jpg")
    first_strict_masked_image = os.path.join(strict_masked_image_dir, "frame_000001.jpg")
    if (
        os.path.exists(first_mask)
        and os.path.exists(first_masked_image)
        and os.path.exists(first_strict_mask)
        and os.path.exists(first_strict_masked_image)
        and not args.overwrite
    ):
        print(f"Already segmented, skipping: {args.output_dir}")
        return

    if not args.skip_extract:
        print(f"Extracting frames from {args.video}")
        n = extract_frames(args.video, image_dir, args.start_sec, args.end_sec, args.fps)
        print(f"Extracted {n} frames at {args.fps} fps")
        if n < 1:
            raise ValueError("No frames extracted")

    paths = image_paths(image_dir)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    import onnxruntime
    from visual_util import segment_sky

    model_path = ensure_skyseg_model(args.model)
    print(f"Loading skyseg model from {model_path}")
    session = onnxruntime.InferenceSession(model_path)

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_image_dir, exist_ok=True)
    os.makedirs(strict_mask_dir, exist_ok=True)
    os.makedirs(strict_masked_image_dir, exist_ok=True)
    print(f"Writing masks to {mask_dir}")
    for index, image_path in enumerate(paths, start=1):
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, image_name)
        masked_image_path = os.path.join(masked_image_dir, image_name)
        strict_mask_path = os.path.join(strict_mask_dir, image_name)
        strict_masked_image_path = os.path.join(strict_masked_image_dir, image_name)
        if not os.path.exists(mask_path) or args.overwrite:
            segment_sky(image_path, session, mask_path)
            print(f"  [{index:03d}/{len(paths):03d}] Saved: {mask_path}")
        else:
            print(f"  [{index:03d}/{len(paths):03d}] Reusing: {mask_path}")

        if not os.path.exists(masked_image_path) or args.overwrite:
            save_masked_image(image_path, mask_path, masked_image_path)
            print(f"  [{index:03d}/{len(paths):03d}] Saved: {masked_image_path}")

        if not os.path.exists(strict_mask_path) or args.overwrite:
            save_strict_sky_mask(
                image_path,
                mask_path,
                strict_mask_path,
                dilate_px=args.strict_dilate_px,
                upper_ratio=args.strict_upper_ratio,
            )
            print(f"  [{index:03d}/{len(paths):03d}] Saved: {strict_mask_path}")

        if not os.path.exists(strict_masked_image_path) or args.overwrite:
            save_masked_image(image_path, strict_mask_path, strict_masked_image_path)
            print(f"  [{index:03d}/{len(paths):03d}] Saved: {strict_masked_image_path}")

    print("All done.")


if __name__ == "__main__":
    main()
