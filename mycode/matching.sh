#!/bin/bash
#SBATCH --job-name=vggt_attn_match
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-02

set -euo pipefail

PROJECT_NAME="vggt-omega"
EXPERIMENT_NAME="attn_matching"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"

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

        # ── Clip 8: egomotion/04fe8f4d_3240-3254s_control (~78 frames at 6fps) ──
        # head 9 (globally best by Borda-count), all 19 global layers
        python render_matching_attention.py \
            --clip_idx   8 \
            --pairs      '0,10 5,30 15,60' \
            --head       9 \
            --alpha      0.55 \
            --output_dir /workspace/outputs/renders/vggt-omega/ego4d/matching \
            2>&1 | tee /workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/matching.log

        echo '--- Done clip 8 ---'

        # ── Clip 6: egomotion/1b1acfa6_52-65s_control ────────────────────────
        python render_matching_attention.py \
            --clip_idx   6 \
            --pairs      '0,5 3,20 8,40' \
            --head       9 \
            --alpha      0.55 \
            --output_dir /workspace/outputs/renders/vggt-omega/ego4d/matching \
            2>&1 | tee -a /workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/matching.log

        echo '--- Done clip 6 ---'
    "

echo "Exit code: $?"
echo "End: $(date)"
