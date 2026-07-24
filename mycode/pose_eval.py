"""
Qualitative pose evaluation for VGGT-Omega reconstructions.

This script overlays camera centers as sphere point clouds on a reconstructed scene:
  - pseudo-GT poses: green spheres
  - VGGT-Omega predicted poses: red spheres

Typical usage:
  # Batch mode: process rotation_low/medium/high and write to outputs/.../pose_eval
  python mycode/pose_eval.py

  # Single-clip mode
  python mycode/pose_eval.py \
      --reconstruction-dir outputs/renders/vggt-omega/ego4d/reconstruction/rotation_low/clip_name \
      --pseudo-gt-poses data/ego4d/pose_annotations/clip_name/gt_poses_tum.txt

Expected prediction file under --reconstruction-dir:
  predictions_pose.npz with key "extrinsic_w2c" or "extrinsic".

Pseudo-GT pose formats:
  - TUM txt: timestamp tx ty tz qx qy qz qw
  - .npy:   (N, 4, 4) or (N, 3, 4)
  - .npz:   one of c2w, poses, gt_c2w, T_c2w, extrinsic_w2c, extrinsic
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
try:
    import trimesh
except ModuleNotFoundError:
    trimesh = None


PRED_COLOR = np.array([255, 32, 32, 255], dtype=np.uint8)
GT_COLOR = np.array([32, 220, 64, 255], dtype=np.uint8)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_RECONSTRUCTION_ROOT = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/ego4d/reconstruction")
DEFAULT_ANNOTATION_ROOT = os.path.join(PROJECT_ROOT, "data/ego4d/pose_annotations")
DEFAULT_BATCH_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs/renders/vggt-omega/ego4d/pose_eval")
ROTATION_LEVELS = ("rotation_low", "rotation_medium", "rotation_high")


def require_trimesh() -> None:
    if trimesh is None:
        raise ModuleNotFoundError(
            "pose_eval.py requires trimesh. Install the VGGT-Omega demo extras "
            'with `pip install -e ".[demo]"`, or install `trimesh` directly.'
        )


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class BatchItem:
    level: str
    clip_name: str
    reconstruction_dir: str
    pred_poses: str
    pseudo_gt_poses: str
    output: str


@dataclass
class PoseData:
    poses: np.ndarray
    timestamps: np.ndarray | None = None
    frame_indices: np.ndarray | None = None


def as_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected poses shaped (N,3,4) or (N,4,4), got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses.copy()

    out = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    out[:, :3, :4] = poses
    return out


def invert_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(as_homogeneous(poses))


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    x, y, z, w = np.moveaxis(quat, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mats = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mats[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[..., 0, 1] = 2.0 * (xy - wz)
    mats[..., 0, 2] = 2.0 * (xz + wy)
    mats[..., 1, 0] = 2.0 * (xy + wz)
    mats[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[..., 1, 2] = 2.0 * (yz - wx)
    mats[..., 2, 0] = 2.0 * (xz - wy)
    mats[..., 2, 1] = 2.0 * (yz + wx)
    mats[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats


def read_tum_pose_data(path: str) -> PoseData:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"TUM pose rows must have 8 columns: timestamp tx ty tz qx qy qz qw. "
                    f"Bad row in {path}: {line}"
                )
            rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"No poses found in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    poses = np.tile(np.eye(4, dtype=np.float64), (len(arr), 1, 1))
    poses[:, :3, :3] = quat_xyzw_to_matrix(arr[:, 4:8])
    poses[:, :3, 3] = arr[:, 1:4]
    return PoseData(poses=poses, timestamps=arr[:, 0])


def read_tum_poses(path: str) -> np.ndarray:
    return read_tum_pose_data(path).poses


def pick_npz_pose_array(npz: np.lib.npyio.NpzFile, preferred_keys: tuple[str, ...]) -> tuple[str, np.ndarray]:
    for key in preferred_keys:
        if key in npz:
            return key, npz[key]
    raise KeyError(f"Could not find any pose key in {npz.files}. Tried: {preferred_keys}")


def load_pose_data(path: str, pose_type: str, preferred_keys: tuple[str, ...]) -> PoseData:
    ext = os.path.splitext(path)[1].lower()
    timestamps = None
    frame_indices = None
    if ext == ".txt":
        data = read_tum_pose_data(path)
        poses = data.poses
        timestamps = data.timestamps
    elif ext == ".npy":
        poses = np.load(path)
    elif ext == ".npz":
        with np.load(path) as data:
            key, poses = pick_npz_pose_array(data, preferred_keys)
            if pose_type == "auto":
                pose_type = "w2c" if "w2c" in key or key == "extrinsic" else "c2w"
            if "timestamps" in data:
                timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            if "frame_indices" in data:
                frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported pose format: {path}")

    poses = as_homogeneous(poses)
    if pose_type == "w2c":
        poses = invert_poses(poses)
    elif pose_type != "c2w":
        raise ValueError(f"pose_type must be c2w, w2c, or auto. Got {pose_type}")
    return PoseData(poses=poses, timestamps=timestamps, frame_indices=frame_indices)


def load_pose_file(path: str, pose_type: str, preferred_keys: tuple[str, ...]) -> np.ndarray:
    return load_pose_data(path, pose_type, preferred_keys).poses


def load_prediction_poses(path: str, pose_type: str) -> np.ndarray:
    return load_prediction_pose_data(path, pose_type).poses


def load_prediction_pose_data(path: str, pose_type: str) -> PoseData:
    return load_pose_data(
        path,
        pose_type=pose_type,
        preferred_keys=(
            "extrinsic_w2c",
            "extrinsic",
            "pred_extrinsic_w2c",
            "pred_extrinsic",
            "c2w",
            "poses",
        ),
    )


def load_gt_poses(path: str, pose_type: str) -> np.ndarray:
    return load_gt_pose_data(path, pose_type).poses


def load_gt_pose_data(path: str, pose_type: str) -> PoseData:
    return load_pose_data(
        path,
        pose_type=pose_type,
        preferred_keys=(
            "c2w",
            "gt_c2w",
            "T_c2w",
            "poses",
            "extrinsic_w2c",
            "extrinsic",
        ),
    )


def match_pose_data(pred: PoseData, gt: PoseData, timestamp_tolerance: float) -> tuple[np.ndarray, np.ndarray, str]:
    pred_poses = pred.poses
    gt_poses = gt.poses
    if pred.timestamps is not None and gt.timestamps is not None:
        pred_ts = np.asarray(pred.timestamps, dtype=np.float64)
        gt_ts = np.asarray(gt.timestamps, dtype=np.float64)
        pred_indices = []
        gt_indices = []
        used_pred = set()
        for gt_idx, gt_time in enumerate(gt_ts):
            pred_idx = int(np.argmin(np.abs(pred_ts - gt_time)))
            if pred_idx in used_pred:
                continue
            if abs(pred_ts[pred_idx] - gt_time) <= timestamp_tolerance:
                used_pred.add(pred_idx)
                pred_indices.append(pred_idx)
                gt_indices.append(gt_idx)
        if pred_indices:
            return pred_poses[pred_indices], gt_poses[gt_indices], "timestamp"

    if pred.frame_indices is not None and gt.frame_indices is not None:
        pred_map = {int(frame_idx): idx for idx, frame_idx in enumerate(pred.frame_indices)}
        pred_indices = []
        gt_indices = []
        for gt_idx, frame_idx in enumerate(gt.frame_indices):
            pred_idx = pred_map.get(int(frame_idx))
            if pred_idx is not None:
                pred_indices.append(pred_idx)
                gt_indices.append(gt_idx)
        if pred_indices:
            return pred_poses[pred_indices], gt_poses[gt_indices], "frame_index"

    n = min(len(pred_poses), len(gt_poses))
    return pred_poses[:n], gt_poses[:n], "prefix"


def camera_centers(c2w: np.ndarray) -> np.ndarray:
    return as_homogeneous(c2w)[:, :3, 3]


def estimate_sim3(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Sim3:
    """Estimate y = scale * rotation @ x + translation."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both have shape (N,3)")
    if len(source_xyz) < 3:
        return Sim3(scale=1.0, rotation=np.eye(3), translation=target_xyz.mean(0) - source_xyz.mean(0))

    mu_x = source_xyz.mean(axis=0)
    mu_y = target_xyz.mean(axis=0)
    x = source_xyz - mu_x
    y = target_xyz - mu_y
    cov = (y.T @ x) / len(source_xyz)
    u, singular_values, vt = np.linalg.svd(cov)

    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    s_mat = np.diag(sign)
    rotation = u @ s_mat @ vt
    var_x = np.mean(np.sum(x * x, axis=1))
    scale = float(np.sum(singular_values * sign) / max(var_x, 1e-12))
    translation = mu_y - scale * (rotation @ mu_x)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def apply_sim3_to_c2w(poses_c2w: np.ndarray, sim3: Sim3) -> np.ndarray:
    poses_c2w = as_homogeneous(poses_c2w)
    out = poses_c2w.copy()
    out[:, :3, :3] = sim3.rotation[None] @ poses_c2w[:, :3, :3]
    out[:, :3, 3] = sim3.scale * (poses_c2w[:, :3, 3] @ sim3.rotation.T) + sim3.translation
    return out


def make_pose_sphere_cloud(
    poses_c2w: np.ndarray,
    color: np.ndarray,
    scene_scale: float,
    sphere_radius_scale: float,
    sphere_points: int,
    stride: int,
    max_frames: int,
) -> trimesh.PointCloud:
    poses_c2w = as_homogeneous(poses_c2w)
    stride = max(1, int(stride))
    indices = np.arange(0, len(poses_c2w), stride)
    if max_frames > 0 and len(indices) > max_frames:
        indices = np.linspace(0, len(poses_c2w) - 1, max_frames).round().astype(np.int64)

    radius = max(scene_scale * sphere_radius_scale, 1e-6)
    sphere_points = max(8, int(sphere_points))
    rng = np.random.default_rng(0)
    vertices = []
    colors = []
    for idx in indices:
        center = poses_c2w[idx, :3, 3]
        dirs = rng.standard_normal((sphere_points, 3)).astype(np.float64)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        vertices.append(center[None] + dirs * radius)
        colors.append(np.tile(color, (sphere_points, 1)))

    if not vertices:
        return trimesh.PointCloud(vertices=np.zeros((0, 3)), colors=np.zeros((0, 4), dtype=np.uint8))
    return trimesh.PointCloud(vertices=np.concatenate(vertices, axis=0), colors=np.concatenate(colors, axis=0))


def point_cloud_arrays(geom: object) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(getattr(geom, "vertices", np.zeros((0, 3))), dtype=np.float64)
    if vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.uint8)
    vertices = vertices.reshape(-1, 3)

    colors = getattr(geom, "colors", None)
    if colors is None and hasattr(geom, "visual"):
        colors = getattr(geom.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        colors = np.tile(np.array([220, 220, 220, 255], dtype=np.uint8), (len(vertices), 1))
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape[1] == 3:
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=1)
    return vertices, colors[:, :4]


def load_base_point_cloud(scene_ply: str | None) -> tuple[np.ndarray, np.ndarray]:
    if scene_ply is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.uint8)

    geom = trimesh.load(scene_ply, process=False)
    vertices = []
    colors = []
    if isinstance(geom, trimesh.Scene):
        geometries = geom.geometry.values()
    else:
        geometries = (geom,)

    for item in geometries:
        item_vertices, item_colors = point_cloud_arrays(item)
        if len(item_vertices) == 0:
            continue
        vertices.append(item_vertices)
        colors.append(item_colors)

    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.uint8)
    return np.concatenate(vertices, axis=0), np.concatenate(colors, axis=0)


def scene_extent(base_vertices: np.ndarray, pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> float:
    points = [camera_centers(pred_c2w), camera_centers(gt_c2w)]
    if len(base_vertices) > 0:
        points.append(base_vertices)
    xyz = np.concatenate([p.reshape(-1, 3) for p in points], axis=0)
    if len(xyz) == 0:
        return 1.0
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    extent = float(np.linalg.norm(hi - lo))
    return extent if extent > 1e-9 else 1.0


def resolve_paths(args: argparse.Namespace) -> tuple[str | None, str, str, str]:
    scene_ply = args.scene_ply
    pred_poses = args.pred_poses
    output = args.output

    if args.reconstruction_dir:
        if scene_ply is None:
            scene_ply = os.path.join(args.reconstruction_dir, "scene.ply")
        if pred_poses is None:
            pred_poses = os.path.join(args.reconstruction_dir, "predictions_pose.npz")
        if output is None:
            output = os.path.join(args.reconstruction_dir, "pose_eval.ply")

    if scene_ply is not None and not os.path.exists(scene_ply):
        raise FileNotFoundError(f"Scene file not found: {scene_ply}")
    if pred_poses is None:
        raise ValueError("Provide --pred-poses or --reconstruction-dir")
    if not os.path.exists(pred_poses):
        raise FileNotFoundError(
            f"Prediction pose file not found: {pred_poses}. "
            "Save VGGT extrinsics as predictions_pose.npz first."
        )
    if args.pseudo_gt_poses is None:
        raise ValueError("Provide --pseudo-gt-poses")
    if not os.path.exists(args.pseudo_gt_poses):
        raise FileNotFoundError(f"Pseudo-GT pose file not found: {args.pseudo_gt_poses}")
    if output is None:
        output = "pose_eval.ply"
    return scene_ply, pred_poses, args.pseudo_gt_poses, output


def find_pseudo_gt_pose(annotation_root: str, level: str, clip_name: str) -> str | None:
    candidates = []
    for base in (
        os.path.join(annotation_root, clip_name),
        os.path.join(annotation_root, level, clip_name),
    ):
        candidates.extend(
            [
                os.path.join(base, "gt_poses_tum.txt"),
                os.path.join(base, "poses_tum.txt"),
                os.path.join(base, "gt_poses.npy"),
                os.path.join(base, "gt_poses.npz"),
                os.path.join(base, "poses.npy"),
                os.path.join(base, "poses.npz"),
            ]
        )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def iter_batch_items(args: argparse.Namespace) -> list[BatchItem]:
    items = []
    levels = args.levels or ROTATION_LEVELS
    for level in levels:
        level_dir = os.path.join(args.reconstruction_root, level)
        if not os.path.isdir(level_dir):
            continue
        for entry in sorted(os.scandir(level_dir), key=lambda x: x.name):
            if not entry.is_dir():
                continue
            clip_name = entry.name
            reconstruction_dir = entry.path
            scene_ply = os.path.join(reconstruction_dir, "scene.ply")
            pred_poses = os.path.join(reconstruction_dir, "predictions_pose.npz")
            pseudo_gt_poses = find_pseudo_gt_pose(args.annotation_root, level, clip_name)
            if not os.path.exists(scene_ply) or not os.path.exists(pred_poses) or pseudo_gt_poses is None:
                continue
            output = os.path.join(args.batch_output_root, level, clip_name, "pose_eval.ply")
            items.append(
                BatchItem(
                    level=level,
                    clip_name=clip_name,
                    reconstruction_dir=reconstruction_dir,
                    pred_poses=pred_poses,
                    pseudo_gt_poses=pseudo_gt_poses,
                    output=output,
                )
            )
    return items


def make_single_item_args(args: argparse.Namespace, item: BatchItem) -> argparse.Namespace:
    single = argparse.Namespace(**vars(args))
    single.reconstruction_dir = item.reconstruction_dir
    single.scene_ply = os.path.join(item.reconstruction_dir, "scene.ply")
    single.pred_poses = item.pred_poses
    single.pseudo_gt_poses = item.pseudo_gt_poses
    single.output = item.output
    return single


def run_single(args: argparse.Namespace) -> dict[str, float]:
    _, _, _, output_path = resolve_paths(args)
    scene, stats = build_pose_eval_scene(args)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scene.export(output_path)
    print(f"Saved pose evaluation scene: {output_path}")
    print(f"Matched poses: {int(stats['num_matched_poses'])}")
    print(f"Pose match mode: {stats['match_mode']}")
    print(f"GT alignment scale: {stats['gt_alignment_scale']:.6g}")
    print(f"Scene scale: {stats['scene_scale']:.6g}")
    print("Colors: pseudo-GT=green spheres, VGGT prediction=red spheres")
    return stats


def run_batch(args: argparse.Namespace) -> None:
    items = iter_batch_items(args)
    print(f"Found {len(items)} clips with scene.ply, predictions_pose.npz, and pseudo-GT poses.")
    print(f"Reconstruction root: {args.reconstruction_root}")
    print(f"Annotation root: {args.annotation_root}")
    print(f"Output root: {args.batch_output_root}")
    if not items:
        print("No outputs written. Each clip needs scene.ply, predictions_pose.npz, and a pseudo-GT pose file.")
        return

    done = 0
    skipped_existing = 0
    failed = 0
    for index, item in enumerate(items, start=1):
        if os.path.exists(item.output) and not args.overwrite:
            skipped_existing += 1
            print(f"[{index}/{len(items)}] SKIP existing: {item.level}/{item.clip_name}")
            continue

        print(f"[{index}/{len(items)}] {item.level}/{item.clip_name}")
        try:
            run_single(make_single_item_args(args, item))
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")

    print(f"Batch done. saved={done}, skipped_existing={skipped_existing}, failed={failed}")


def build_pose_eval_scene(args: argparse.Namespace) -> tuple[trimesh.PointCloud, dict[str, float]]:
    require_trimesh()
    scene_ply, pred_path, gt_path, output_path = resolve_paths(args)
    del output_path

    pred_data = load_prediction_pose_data(pred_path, args.pred_pose_type)
    gt_data = load_gt_pose_data(gt_path, args.gt_pose_type)
    pred_c2w, gt_c2w, match_mode = match_pose_data(pred_data, gt_data, args.timestamp_tolerance)

    n = len(pred_c2w)
    if n < 2:
        raise ValueError(
            f"Need at least 2 matched poses, got matched={n}, "
            f"pred={len(pred_data.poses)}, gt={len(gt_data.poses)}"
        )

    if args.align_gt:
        sim3 = estimate_sim3(camera_centers(gt_c2w), camera_centers(pred_c2w))
        gt_vis_c2w = apply_sim3_to_c2w(gt_c2w, sim3)
    else:
        sim3 = Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))
        gt_vis_c2w = gt_c2w

    base_vertices, base_colors = load_base_point_cloud(scene_ply)
    scale = scene_extent(base_vertices, pred_c2w, gt_vis_c2w)

    gt_spheres = make_pose_sphere_cloud(
        gt_vis_c2w,
        GT_COLOR,
        scene_scale=scale,
        sphere_radius_scale=args.sphere_radius_scale,
        sphere_points=args.sphere_points,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    pred_spheres = make_pose_sphere_cloud(
        pred_c2w,
        PRED_COLOR,
        scene_scale=scale,
        sphere_radius_scale=args.sphere_radius_scale,
        sphere_points=args.sphere_points,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    clouds = [
        (base_vertices, base_colors),
        point_cloud_arrays(gt_spheres),
        point_cloud_arrays(pred_spheres),
    ]
    vertices = np.concatenate([v for v, _ in clouds if len(v) > 0], axis=0)
    colors = np.concatenate([c for v, c in clouds if len(v) > 0], axis=0)
    scene = trimesh.PointCloud(vertices=vertices, colors=colors)

    stats = {
        "num_matched_poses": float(n),
        "gt_alignment_scale": float(sim3.scale),
        "scene_scale": float(scale),
        "match_mode": match_mode,
    }
    return scene, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-dir", help="Directory containing scene.ply and predictions_pose.npz")
    parser.add_argument("--scene-ply", help="Background reconstructed scene PLY")
    parser.add_argument("--pred-poses", help="VGGT pose file (.npz/.npy/.txt)")
    parser.add_argument("--pseudo-gt-poses", help="Pseudo-GT pose file (.txt TUM, .npy, or .npz)")
    parser.add_argument("--output", help="Output scene path, e.g. pose_eval.ply or pose_eval.glb")
    parser.add_argument("--reconstruction-root", default=DEFAULT_RECONSTRUCTION_ROOT)
    parser.add_argument("--annotation-root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--batch-output-root", default=DEFAULT_BATCH_OUTPUT_ROOT)
    parser.add_argument("--levels", nargs="+", choices=ROTATION_LEVELS, default=list(ROTATION_LEVELS))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing batch outputs")
    parser.add_argument("--pred-pose-type", choices=("w2c", "c2w", "auto"), default="w2c")
    parser.add_argument("--gt-pose-type", choices=("c2w", "w2c", "auto"), default="c2w")
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-4)
    parser.add_argument("--align-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stride", type=int, default=1, help="Draw every Nth pose")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of poses to draw; <=0 draws all")
    parser.add_argument("--sphere-radius-scale", type=float, default=0.002, help="Pose sphere radius as a fraction of scene extent")
    parser.add_argument("--sphere-points", type=int, default=300, help="Number of sampled surface points per pose sphere")
    return parser.parse_args()


def is_single_clip_mode(args: argparse.Namespace) -> bool:
    return any(
        [
            args.reconstruction_dir,
            args.scene_ply,
            args.pred_poses,
            args.pseudo_gt_poses,
            args.output,
        ]
    )


def main() -> None:
    args = parse_args()
    if is_single_clip_mode(args):
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
