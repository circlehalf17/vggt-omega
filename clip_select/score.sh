#!/bin/bash
#SBATCH --job-name=ego4d_score
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-02

# ── 수정할 부분 ───────────────────────────────────────────────────────────────
# SCORE_TYPE: all | ego-motion | low-texture | reflective | dynamic | control
#   "all" = 영상 1회 decode로 5개 type 동시 처리 (5배 빠름, 권장)
SCORE_TYPE="all"
WINDOW_SEC="10.0"
# SCORE_THRESH=""    # 비워두면 type별 기본값 사용
# MAX_VIDEOS=""      # 테스트시: "--max-videos 50"
MAX_VIDEOS_FLAG=""
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"
EXPERIMENT_NAME="ego4d_select_${SCORE_TYPE}"

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
echo "Type:       $SCORE_TYPE"
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

        python -m clip_select.score \
            --type          ${SCORE_TYPE} \
            --ego4d-root    /workspace/data/Ego4D/v2 \
            --ego4d-json    /workspace/data/Ego4D/ego4d.json \
            --output-dir    /workspace/outputs/select \
            --window-sec    ${WINDOW_SEC} \
            ${MAX_VIDEOS_FLAG} \
            2>&1 | tee /workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/score.log
    "

echo "Exit code: $?"
echo "End: $(date)"
