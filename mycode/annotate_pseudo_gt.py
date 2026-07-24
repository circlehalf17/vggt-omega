"""
Create pseudo-GT camera pose annotations for Ego4D rotation clips with COLMAP.

Inputs:
  data/ego4d/v2/label/rotation_low/*.mp4
  data/ego4d/v2/label/rotation_medium/*.mp4
  data/ego4d/v2/label/rotation_high/*.mp4

Outputs per clip:
  data/ego4d/pose_annotations/<level>/<clip_name>/
    gt_poses_tum.txt        timestamp tx ty tz qx qy qz qw, camera-to-world
    gt_poses.npz            timestamps, c2w, frame_indices, image_names
    quality.json            registration stats and command status
    colmap_text/            exported COLMAP model in text format

This is pseudo-GT, not sensor ground truth. Treat clips with low registration
ratio or obviously bad reconstructions as failed annotations.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
INPUT_BASE = os.path.join(PROJECT_ROOT, "data/ego4d/v2/label")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data/ego4d/pose_annotations")
ROTATION_LEVELS = ("rotation_low", "rotation_medium", "rotation_high")
SAMPLE_FPS = 6.0
CLIP_START_SEC = 0.0
CLIP_END_SEC = 10.0


@dataclass
class Clip:
    level: str
    name: str
    video_path: str


@dataclass
class RegisteredImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    image_name: str
    num_points2d: int


def run_command(cmd: list[str], log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write("\n")


def extract_frames(video_path: str, out_dir: str, start_sec: float, end_sec: float, fps: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
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
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(out_dir, "frame_*.jpg")))


def list_rotation_clips(input_base: str, levels: tuple[str, ...]) -> list[Clip]:
    clips = []
    for level in levels:
        for video_path in sorted(glob.glob(os.path.join(input_base, level, "*.mp4"))):
            name = os.path.splitext(os.path.basename(video_path))[0]
            clips.append(Clip(level=level, name=name, video_path=video_path))
    return clips


def qvec_wxyz_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)


def colmap_w2c_to_c2w(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    rot_w2c = qvec_wxyz_to_rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))
    trans_w2c = np.array([tx, ty, tz], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_w2c.T
    c2w[:3, 3] = -rot_w2c.T @ trans_w2c
    return c2w


def frame_index_from_name(image_name: str) -> int:
    match = re.search(r"frame_(\d+)", image_name)
    if not match:
        raise ValueError(f"Cannot parse frame index from image name: {image_name}")
    return int(match.group(1)) - 1


def read_images_txt(path: str) -> list[RegisteredImage]:
    images = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue
            points_line = f.readline().strip()
            images.append(
                RegisteredImage(
                    image_id=int(parts[0]),
                    qw=float(parts[1]),
                    qx=float(parts[2]),
                    qy=float(parts[3]),
                    qz=float(parts[4]),
                    tx=float(parts[5]),
                    ty=float(parts[6]),
                    tz=float(parts[7]),
                    camera_id=int(parts[8]),
                    image_name=parts[9],
                    num_points2d=len(points_line.split()) // 3 if points_line else 0,
                )
            )
    return sorted(images, key=lambda img: frame_index_from_name(img.image_name))


def write_pose_outputs(
    registered: list[RegisteredImage],
    out_dir: str,
    fps: float,
    start_sec: float,
) -> dict:
    timestamps = []
    frame_indices = []
    image_names = []
    c2w_list = []

    for image in registered:
        frame_idx = frame_index_from_name(image.image_name)
        timestamps.append(start_sec + frame_idx / fps)
        frame_indices.append(frame_idx)
        image_names.append(image.image_name)
        c2w_list.append(colmap_w2c_to_c2w(image.qw, image.qx, image.qy, image.qz, image.tx, image.ty, image.tz))

    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    frame_indices_np = np.asarray(frame_indices, dtype=np.int64)
    c2w_np = np.asarray(c2w_list, dtype=np.float64)

    tum_path = os.path.join(out_dir, "gt_poses_tum.txt")
    with open(tum_path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw camera-to-world\n")
        for timestamp, pose in zip(timestamps_np, c2w_np):
            quat = rotmat_to_quat_xyzw(pose[:3, :3])
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = quat
            f.write(f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

    npz_path = os.path.join(out_dir, "gt_poses.npz")
    np.savez_compressed(
        npz_path,
        timestamps=timestamps_np,
        c2w=c2w_np,
        frame_indices=frame_indices_np,
        image_names=np.asarray(image_names),
    )
    return {"tum_path": tum_path, "npz_path": npz_path}


def find_best_sparse_model(sparse_dir: str) -> str | None:
    candidates = []
    for path in sorted(Path(sparse_dir).iterdir()):
        if not path.is_dir():
            continue
        images_bin = path / "images.bin"
        images_txt = path / "images.txt"
        if images_bin.exists() or images_txt.exists():
            candidates.append(str(path))
    if not candidates:
        return None
    return candidates[0]


def run_colmap(
    colmap_bin: str,
    image_dir: str,
    work_dir: str,
    matcher: str,
    camera_model: str,
    sequential_overlap: int,
    log_path: str,
) -> str:
    database_path = os.path.join(work_dir, "database.db")
    sparse_dir = os.path.join(work_dir, "sparse")
    text_dir = os.path.join(work_dir, "colmap_text")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    run_command(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            image_dir,
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            camera_model,
        ],
        log_path,
    )

    if matcher == "sequential":
        run_command(
            [
                colmap_bin,
                "sequential_matcher",
                "--database_path",
                database_path,
                "--SequentialMatching.overlap",
                str(sequential_overlap),
            ],
            log_path,
        )
    elif matcher == "exhaustive":
        run_command([colmap_bin, "exhaustive_matcher", "--database_path", database_path], log_path)
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    run_command(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            image_dir,
            "--output_path",
            sparse_dir,
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "1",
        ],
        log_path,
    )

    best_model = find_best_sparse_model(sparse_dir)
    if best_model is None:
        raise RuntimeError("COLMAP mapper did not produce a sparse model")

    run_command(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            best_model,
            "--output_path",
            text_dir,
            "--output_type",
            "TXT",
        ],
        log_path,
    )
    return text_dir


def annotate_clip(clip: Clip, args: argparse.Namespace) -> None:
    out_dir = os.path.join(args.output_root, clip.level, clip.name)
    quality_path = os.path.join(out_dir, "quality.json")
    tum_path = os.path.join(out_dir, "gt_poses_tum.txt")

    if os.path.exists(tum_path) and not args.overwrite:
        print("  Already annotated, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "colmap.log")
    if os.path.exists(log_path) and args.overwrite:
        os.remove(log_path)

    work_parent = out_dir if args.keep_workdir else tempfile.mkdtemp(prefix="pseudo_gt_")
    work_dir = os.path.join(work_parent, "work")
    image_dir = os.path.join(work_dir, "images")
    os.makedirs(work_dir, exist_ok=True)

    status = "failed"
    n_frames = 0
    registered = []
    error = None

    try:
        print(f"  Extracting frames at {args.fps:g}fps ...")
        n_frames = extract_frames(clip.video_path, image_dir, args.start_sec, args.end_sec, args.fps)
        if n_frames < 2:
            raise RuntimeError(f"Too few frames extracted: {n_frames}")

        print("  Running COLMAP ...")
        text_dir = run_colmap(
            args.colmap_bin,
            image_dir,
            work_dir,
            args.matcher,
            args.camera_model,
            args.sequential_overlap,
            log_path,
        )

        final_text_dir = os.path.join(out_dir, "colmap_text")
        if os.path.exists(final_text_dir):
            shutil.rmtree(final_text_dir)
        shutil.copytree(text_dir, final_text_dir)

        registered = read_images_txt(os.path.join(final_text_dir, "images.txt"))
        if not registered:
            raise RuntimeError("No registered images found in COLMAP model")

        outputs = write_pose_outputs(registered, out_dir, args.fps, args.start_sec)
        status = "ok"
        print(f"  Saved: {outputs['tum_path']}")
        print(f"  Saved: {outputs['npz_path']}")
    except Exception as exc:
        error = str(exc)
        print(f"  FAILED: {error}")
    finally:
        registered_ratio = (len(registered) / n_frames) if n_frames else 0.0
        quality = {
            "status": status,
            "clip_level": clip.level,
            "clip_name": clip.name,
            "video_path": clip.video_path,
            "num_extracted_frames": n_frames,
            "num_registered_frames": len(registered),
            "registered_ratio": registered_ratio,
            "mean_observations_per_registered_image": float(np.mean([x.num_points2d for x in registered])) if registered else 0.0,
            "fps": args.fps,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "matcher": args.matcher,
            "camera_model": args.camera_model,
            "error": error,
        }
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2)
        print(f"  Saved: {quality_path}")

        if not args.keep_workdir:
            shutil.rmtree(work_parent, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-base", default=INPUT_BASE)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--levels", nargs="+", choices=ROTATION_LEVELS, default=list(ROTATION_LEVELS))
    parser.add_argument("--clip-name", help="Annotate only one clip basename")
    parser.add_argument("--limit", type=int, help="Annotate at most N clips")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep COLMAP database/images under each annotation dir")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--start-sec", type=float, default=CLIP_START_SEC)
    parser.add_argument("--end-sec", type=float, default=CLIP_END_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which(args.colmap_bin) is None:
        raise FileNotFoundError(
            f"COLMAP binary not found: {args.colmap_bin}. "
            "Install COLMAP or pass --colmap-bin /path/to/colmap."
        )

    clips = list_rotation_clips(args.input_base, tuple(args.levels))
    if args.clip_name:
        clips = [clip for clip in clips if clip.name == args.clip_name]
    if args.limit is not None:
        clips = clips[: args.limit]

    print(f"Found {len(clips)} clips under {args.input_base}")
    print(f"Writing pseudo-GT annotations to {args.output_root}\n")
    for index, clip in enumerate(clips, start=1):
        print(f"[{index}/{len(clips)}] {clip.level}/{clip.name}")
        annotate_clip(clip, args)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
