#!/bin/bash
#SBATCH --job-name=vggt_recon
#SBATCH --partition=h200q          # ego4d-clips(h200q), others(l40sq)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=240:00:00            # ego4d-json(24h), others(6h)
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-02      # ego4d-clips → iREMB-C-02

# ── 수정할 부분 ───────────────────────────────────────────────────────────────
#  RECON_MODE: ego4d-json | ego4d-clips | videos | image-dirs
RECON_MODE="ego4d-json"
EXPERIMENT_NAME="ego4d_recon"

# ego4d-json / attention 공통
EGO4D_ROOT="/workspace/data/Ego4D/v2"
EGO4D_JSON="/workspace/data/Ego4D/ego4d.json"
EVAL_LIST="/workspace/data/Ego4D/eval_50seqs.txt"
SAMPLE_FPS="1.0"
MAX_DURATION="10.0"

# ego4d-clips / videos 공통
DATA_ROOT="/workspace/data/mydata"   # mp4 파일 디렉토리 또는 이미지 디렉토리
FPS="30.0"
NUM_FRAMES="300"
START_SEC="0.0"
# CLIP_UIDS=""   # 특정 UID만 처리할 때: "uid1,uid2,..."

# 출력 옵션 (필요한 것만 주석 해제)
EXTRA_FLAGS=""
# EXTRA_FLAGS="--save-glb --save-npz"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"

JOBDIR="/scratch/mip25/wbLee/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}"
JOBDIR_WS="/workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}"
mkdir -p "$JOBDIR"

exec 1>"$JOBDIR/out.log"
exec 2>"$JOBDIR/error.log"

module purge
module load Singularity/4.3.4

echo "=== Job Info ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Mode:       $RECON_MODE"
echo "Log dir:    $JOBDIR"
echo "================"

srun --mpi=pmix singularity exec --nv \
    --bind /scratch/mip25/wbLee:/workspace \
    --bind /usr/lib64/libXext.so.6:/usr/lib64/libXext.so.6 \
    --bind /usr/lib64/libXext.so.6.4.0:/usr/lib64/libXext.so.6.4.0 \
    --pwd /workspace/repo/vggt-omega \
    "$SIF_IMAGE" \
    bash -c "
        set -euo pipefail
        export PATH=/workspace/envs/vggt-omega/bin:\$PATH
        export PYTHONPATH=/workspace/repo/vggt-omega:\${PYTHONPATH:-}
        export LD_LIBRARY_PATH=/usr/lib64:\${LD_LIBRARY_PATH:-}

        nvidia-smi

        python reconstruction.py \
            --mode          ${RECON_MODE} \
            --checkpoint    /workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt \
            --output-dir    /workspace/outputs/renders/vggt-omega/${EXPERIMENT_NAME} \
            --image-resolution 512 \
            --conf-thres    20.0 \
            --max-points-k  1000 \
            --ego4d-root    ${EGO4D_ROOT} \
            --ego4d-json    ${EGO4D_JSON} \
            --eval-list     ${EVAL_LIST} \
            --sample-fps    ${SAMPLE_FPS} \
            --max-duration  ${MAX_DURATION} \
            --tmp-dir       /tmp/vggt_frames \
            --data-root     ${DATA_ROOT} \
            --fps           ${FPS} \
            --num-frames    ${NUM_FRAMES} \
            --start-sec     ${START_SEC} \
            ${EXTRA_FLAGS} \
            2>&1 | tee \"$JOBDIR_WS/recon.log\"
    " 2>&1 | tee "$JOBDIR/train.log"

echo "Exit code: $?"
echo "End: $(date)"
