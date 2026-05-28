import argparse
import gc
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

from visual_util import predictions_to_glb, predictions_to_ply
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def load_model(checkpoint_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def extract_frames(video_path: str, out_dir: str, start_sec: float, end_sec: float, sample_fps: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = max(int(round(fps / sample_fps)), 1)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.png"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def run_inference(target_dir: str, model: VGGTOmega, image_resolution: int) -> dict:
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    torch.cuda.empty_cache()
    return predictions_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ego4d-root", required=True, help="Path to Ego4D v2 root (contains full_scale/)")
    parser.add_argument("--ego4d-json", required=True, help="Path to ego4d.json")
    parser.add_argument("--eval-list", required=True, help="Path to eval_50seqs.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=None, help="Max seconds from clip start to use (e.g. 10.0)")
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument("--max-points-k", type=int, default=1000)
    parser.add_argument("--tmp-dir", default="/tmp/vggt_ego4d_frames")
    args = parser.parse_args()

    with open(args.ego4d_json) as f:
        ego4d = json.load(f)
    clip_map = {c["clip_uid"]: c for c in ego4d["clips"]}

    eval_uids = open(args.eval_list).read().strip().split()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("Model loaded.")

    skipped = []
    for i, clip_uid in enumerate(eval_uids):
        out_path = os.path.join(args.output_dir, clip_uid, "scene.glb")
        if os.path.exists(out_path):
            print(f"[{i+1}/50] SKIP (exists): {clip_uid}")
            continue

        clip = clip_map.get(clip_uid)
        if clip is None:
            print(f"[{i+1}/50] SKIP (not in ego4d.json): {clip_uid}")
            skipped.append(clip_uid)
            continue

        video_uid = clip["video_uid"]
        video_path = os.path.join(args.ego4d_root, "full_scale", f"{video_uid}.mp4")
        if not os.path.exists(video_path):
            print(f"[{i+1}/50] SKIP (video missing): {clip_uid} -> {video_uid}")
            skipped.append(clip_uid)
            continue

        print(f"[{i+1}/50] Processing: {clip_uid}")

        tmp_dir = os.path.join(args.tmp_dir, clip_uid)
        images_dir = os.path.join(tmp_dir, "images")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        start_sec = clip["video_start_sec"]
        end_sec = clip["video_end_sec"]
        if args.max_duration is not None:
            end_sec = min(end_sec, start_sec + args.max_duration)

        n_frames = extract_frames(
            video_path, images_dir,
            start_sec, end_sec,
            args.sample_fps,
        )
        print(f"  Extracted {n_frames} frames")

        if n_frames == 0:
            print(f"  SKIP: no frames extracted")
            skipped.append(clip_uid)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        try:
            predictions = run_inference(tmp_dir, model, args.image_resolution)

            clip_out_dir = os.path.join(args.output_dir, clip_uid)
            os.makedirs(clip_out_dir, exist_ok=True)

            np.savez(os.path.join(clip_out_dir, "predictions.npz"), **predictions)

            scene = predictions_to_glb(
                predictions,
                conf_thres=args.conf_thres,
                show_cam=True,
                target_dir=tmp_dir,
                max_points=args.max_points_k * 1000,
            )
            scene.export(file_obj=out_path)

            ply_path = os.path.join(clip_out_dir, "scene.ply")
            pc = predictions_to_ply(
                predictions,
                conf_thres=args.conf_thres,
                max_points=args.max_points_k * 1000,
            )
            pc.export(ply_path)
            print(f"  Saved: {out_path}, {ply_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(clip_uid)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nDone. Skipped {len(skipped)}/50 sequences:")
    for uid in skipped:
        print(f"  {uid}")


if __name__ == "__main__":
    main()
