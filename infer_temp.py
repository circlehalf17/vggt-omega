import argparse
import gc
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visual_util import predictions_to_glb, predictions_to_ply
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

DATASETS = [
    "bear",
    "bike-packing",
    "blackswan",
    "bear_blur",
    "bike-packing_blur",
    "blackswan_blur",
]


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


def run_inference(image_dir: str, model: VGGTOmega, image_resolution: int) -> dict:
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_names = [p for p in image_names if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")]
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


def save_camera_trajectory(extrinsic_w2c: np.ndarray, save_path: str):
    N = len(extrinsic_w2c)
    R = extrinsic_w2c[:, :3, :3]
    t = extrinsic_w2c[:, :3, 3]

    positions = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
    R_c2w = R.transpose(0, 2, 1)

    cmap_colors = plt.cm.viridis(np.linspace(0, 1, N))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "b-", alpha=0.35, linewidth=1.2)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=cmap_colors, s=18, zorder=5)
    ax.scatter(*positions[0],  c="lime", s=140, zorder=6, marker="*", label="Start")
    ax.scatter(*positions[-1], c="red",  s=140, zorder=6, marker="*", label="End")

    step = max(1, N // 12)
    for i in range(0, N, step):
        fwd = -R_c2w[i, :, 2] * 0.05
        ax.quiver(*positions[i], *fwd, color="orange", alpha=0.8, linewidth=1.5)

    spread = np.abs(positions).max() * 1.1 + 1e-6
    mid = positions.mean(axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter([m - spread, m + spread])

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Estimated Camera Trajectory (VGGT-Omega)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Root dir containing dataset folders (e.g. wbLee/data/temp)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-thres", type=float, default=20.0)
    parser.add_argument("--max-points-k", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("Model loaded.")

    skipped = []
    for i, dataset in enumerate(DATASETS):
        image_dir = os.path.join(args.data_root, dataset)
        if not os.path.isdir(image_dir):
            print(f"[{i+1}/{len(DATASETS)}] SKIP (directory not found): {image_dir}")
            skipped.append(dataset)
            continue

        print(f"[{i+1}/{len(DATASETS)}] Processing: {dataset}")

        try:
            predictions = run_inference(image_dir, model, args.image_resolution)

            out_dir = os.path.join(args.output_dir, dataset)
            os.makedirs(out_dir, exist_ok=True)

            np.savez(os.path.join(out_dir, "predictions.npz"), **predictions)

            traj_path = os.path.join(out_dir, "camera_trajectory.png")
            save_camera_trajectory(predictions["extrinsic"], traj_path)

            glb_path = os.path.join(out_dir, "scene.glb")
            scene = predictions_to_glb(
                predictions,
                conf_thres=args.conf_thres,
                show_cam=True,
                target_dir=image_dir,
                max_points=args.max_points_k * 1000,
            )
            scene.export(file_obj=glb_path)

            ply_path = os.path.join(out_dir, "scene.ply")
            pc = predictions_to_ply(
                predictions,
                conf_thres=args.conf_thres,
                max_points=args.max_points_k * 1000,
                cam_sphere_pts=300,
            )
            pc.export(ply_path)
            print(f"  Saved: {glb_path}, {ply_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(dataset)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nDone. Skipped {len(skipped)}/{len(DATASETS)} datasets:")
    for d in skipped:
        print(f"  {d}")


if __name__ == "__main__":
    main()
