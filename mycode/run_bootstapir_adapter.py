"""
BootsTAPIR adapter for mycode/tracking_compare_sota.py.

Input contract:
  --frames-npy  : npy with shape [T,H,W,3], uint8 RGB
  --queries-npy : npy with shape [N,3], float32 rows [query_frame, x, y]
  --output-npz  : output npz path

Output contract:
  tracks     : [T,N,2], float32, rows [x, y] in the input frame coordinate system
  visibility : [T,N], bool

Install dependency/checkpoint first:
  bash mycode/install_bootstapir.sh
"""

from __future__ import annotations

import argparse
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints/tapnet/bootstapir_checkpoint_v2.pt")


def resize_video_and_queries(
    frames: np.ndarray,
    queries_txy: np.ndarray,
    model_height: int,
    model_width: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    import cv2

    input_height, input_width = frames.shape[1:3]
    scale_x = model_width / float(input_width)
    scale_y = model_height / float(input_height)

    if input_height == model_height and input_width == model_width:
        resized = frames
    else:
        resized = np.stack(
            [
                cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_AREA)
                for frame in frames
            ],
            axis=0,
        )

    queries_tyx = np.empty_like(queries_txy, dtype=np.float32)
    queries_tyx[:, 0] = queries_txy[:, 0]
    queries_tyx[:, 1] = queries_txy[:, 2] * scale_y
    queries_tyx[:, 2] = queries_txy[:, 1] * scale_x
    return resized, queries_tyx, (scale_x, scale_y)


def postprocess_occlusions(occlusions, expected_dist):
    import torch.nn.functional as F

    return (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5


def run_bootstapir(
    frames: np.ndarray,
    queries_txy: np.ndarray,
    checkpoint: str,
    device: str,
    model_height: int,
    model_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from tapnet.torch import tapir_model

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"BootsTAPIR checkpoint not found: {checkpoint}\n"
            "Run: bash mycode/install_bootstapir.sh"
        )

    frames_model, queries_tyx, (scale_x, scale_y) = resize_video_and_queries(
        frames,
        queries_txy,
        model_height=model_height,
        model_width=model_width,
    )

    model = tapir_model.TAPIR(pyramid_level=1)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    frames_tensor = torch.from_numpy(frames_model).to(device).float()
    frames_tensor = frames_tensor / 255.0 * 2.0 - 1.0
    queries_tensor = torch.from_numpy(queries_tyx).to(device).float()

    with torch.inference_mode():
        outputs = model(frames_tensor[None], queries_tensor[None])
        tracks_yx = outputs["tracks"][0].detach()
        visibility = postprocess_occlusions(
            outputs["occlusion"][0],
            outputs["expected_dist"][0],
        ).detach()

    # BootsTAPIR receives query points as [t, y, x] and returns tracks as
    # [N, T, 2] in [y, x] order. The comparison script expects [T, N, 2]
    # in [x, y] order.
    tracks_yx_np = tracks_yx.float().cpu().numpy()
    visibility_np = visibility.bool().cpu().numpy()

    if tracks_yx_np.ndim != 3:
        raise ValueError(f"Expected BootsTAPIR tracks with shape [N,T,2], got {tracks_yx_np.shape}")
    if tracks_yx_np.shape[-1] != 2:
        raise ValueError(f"Expected last track dimension to be 2, got {tracks_yx_np.shape}")

    tracks_xy_nt = np.empty_like(tracks_yx_np, dtype=np.float32)
    tracks_xy_nt[..., 0] = tracks_yx_np[..., 1] / scale_x
    tracks_xy_nt[..., 1] = tracks_yx_np[..., 0] / scale_y
    tracks_xy_tn = np.transpose(tracks_xy_nt, (1, 0, 2))

    if visibility_np.shape == tracks_yx_np.shape[:2]:
        visibility_tn = np.transpose(visibility_np, (1, 0))
    elif visibility_np.shape == tracks_xy_tn.shape[:2]:
        visibility_tn = visibility_np
    else:
        raise ValueError(
            f"Visibility shape {visibility_np.shape} does not match tracks shape {tracks_yx_np.shape}"
        )

    return tracks_xy_tn, visibility_tn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-npy", required=True)
    parser.add_argument("--queries-npy", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-height", type=int, default=512)
    parser.add_argument("--model-width", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = np.load(args.frames_npy)
    queries = np.load(args.queries_npy).astype(np.float32)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T,H,W,3], got {frames.shape}")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"Expected queries shape [N,3], got {queries.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    tracks, visibility = run_bootstapir(
        frames=frames,
        queries_txy=queries,
        checkpoint=args.checkpoint,
        device=args.device,
        model_height=args.model_height,
        model_width=args.model_width,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, tracks=tracks, visibility=visibility)
    print(f"Saved BootsTAPIR tracks: {args.output_npz}")


if __name__ == "__main__":
    main()
