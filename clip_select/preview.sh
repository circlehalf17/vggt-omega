#!/bin/bash
#SBATCH --job-name=ego4d_preview
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-02

# ── 수정할 부분 ───────────────────────────────────────────────────────────────
# PREVIEW_TYPE: all | ego-motion | low-texture | reflective | dynamic | control
#   "all" = 5개 type 한꺼번에 preview 생성 (권장)
PREVIEW_TYPE="all"
TOPK=""      # 비워두면 type별 기본값 사용 (ego/dyn/con=10, low/ref=15)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"
EXPERIMENT_NAME="ego4d_preview_${PREVIEW_TYPE}"

JOBDIR="/scratch/mip25/wbLee/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}"
mkdir -p "$JOBDIR"

exec 1>"$JOBDIR/out.log"
exec 2>"$JOBDIR/error.log"

module purge
module load Singularity/4.3.4

echo "=== Job Info ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Type:       $PREVIEW_TYPE  TopK=$TOPK"
echo "Log dir:    $JOBDIR"
echo "================"

TOPK_FLAG=""
if [ -n "${TOPK}" ]; then TOPK_FLAG="--topk ${TOPK}"; fi

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

        python -m clip_select.preview \
            --type          ${PREVIEW_TYPE} \
            --ego4d-root    /workspace/data/Ego4D/v2 \
            --input-dir     /workspace/outputs/select \
            ${TOPK_FLAG} \
            2>&1 | tee /workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/preview.log
    "

echo "Exit code: $?"
echo "End: $(date)"
