#!/bin/bash
#SBATCH --job-name=ego4d_materialize
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
# preview 검토 후 선택한 클립 목록 CSV 경로
SELECTION_CSV="/workspace/outputs/select/selected.csv"
OUTPUT_DIR="/workspace/outputs/select/frames"
EXTRACT_FPS="6.0"
IMAGE_RESOLUTION="512"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"
EXPERIMENT_NAME="ego4d_materialize"

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
echo "Selection:  $SELECTION_CSV"
echo "Output:     $OUTPUT_DIR"
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

        python -m clip_select.materialize \
            --selection         ${SELECTION_CSV} \
            --ego4d-root        /workspace/data/Ego4D/v2 \
            --output-dir        ${OUTPUT_DIR} \
            --fps               ${EXTRACT_FPS} \
            --image-resolution  ${IMAGE_RESOLUTION} \
            2>&1 | tee /workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/materialize.log
    "

echo "Exit code: $?"
echo "End: $(date)"
