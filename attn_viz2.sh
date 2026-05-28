#!/bin/bash
#SBATCH --job-name=vggt_attn2
#SBATCH --partition=l40sq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-08

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"
EXPERIMENT_NAME="ego4d_attn2"

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

        python attn_viz2.py \
            --checkpoint  /workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt \
            --ego4d-root  /workspace/data/Ego4D/v2 \
            --ego4d-json  /workspace/data/Ego4D/ego4d.json \
            --eval-list   /workspace/data/Ego4D/eval_50seqs.txt \
            --output-dir  /workspace/outputs/renders/vggt-omega/ego4d \
            --image-resolution 512 \
            --sample-fps 6.0 \
            --max-duration 10.0 \
            --tmp-dir /tmp/vggt_ego4d_attn2_frames \
            2>&1 | tee \"$JOBDIR_WS/attn_viz2.log\"
    " 2>&1 | tee "$JOBDIR/train.log"

echo "Exit code: $?"
echo "End: $(date)"
