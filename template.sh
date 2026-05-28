#!/bin/bash
#SBATCH --job-name=vggt-omega
#SBATCH --partition=l40sq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=iREMB-C-08

# 코드는 (2,18,20,21,51,52)줄만 수정

set -euo pipefail

PROJECT_NAME="vggt-omega"
SIF_IMAGE="/scratch/mip25/wbLee/pytorch.sif"
EXPERIMENT_NAME=""
CONFIG_NAME=""

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
echo "Config:     $CONFIG_NAME"
echo "Log dir:    $JOBDIR"
echo "================"

# .bak 파일 충돌 방지 (known issue)
rm -f "/scratch/mip25/wbLee/outputs/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"/*.bak 2>/dev/null || true

srun --mpi=pmix singularity exec --nv \
    --bind /scratch/mip25/wbLee:/workspace \
    --bind /usr/lib64/libXext.so.6:/usr/lib64/libXext.so.6 \
    --bind /usr/lib64/libXext.so.6.4.0:/usr/lib64/libXext.so.6.4.0 \
    --pwd /workspace \
    "$SIF_IMAGE" \
    bash -c "
        set -euo pipefail
        export PATH=/workspace/envs/vggt-omega/bin:\$PATH
        export PYTHONPATH=/workspace/repo/${PROJECT_NAME}:\${PYTHONPATH:-}
        export LD_LIBRARY_PATH=/usr/lib64:\${LD_LIBRARY_PATH:-}

        nvidia-smi

        LOG_FILE=\"/workspace/outputs/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}/job_${SLURM_JOB_ID}/train.log\"
        mkdir -p \"\$(dirname \$LOG_FILE)\"
        MAX_RETRIES=40
        RETRY_COUNT=0

        # 포트 충돌 방지: 사용 가능한 포트 자동 탐색
        MASTER_PORT=\$(python3 -c \"
import socket
for port in range(29500, 29600):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', port))
        s.close()
        print(port)
        break
    except OSError:
        pass
\")
        echo \"Using MASTER_PORT=\$MASTER_PORT\"

        TRAINING_CMD=\"torchrun --nproc_per_node=1 \
            --master_addr=127.0.0.1 --master_port=\$MASTER_PORT \
            /workspace/${PROJECT_NAME}/training/launch_gra.py --config ${CONFIG_NAME}\"

        cd /workspace/${PROJECT_NAME}/training

        echo 'Starting training (max \$MAX_RETRIES retries)' | tee -a \"\$LOG_FILE\"

        while [ \$RETRY_COUNT -le \$MAX_RETRIES ]; do
            echo \"=== ATTEMPT \$((RETRY_COUNT + 1)) AT \$(date) ===\" | tee -a \"\$LOG_FILE\"
            eval \$TRAINING_CMD 2>&1 | tee -a \"\$LOG_FILE\"
            EXIT_CODE=\${PIPESTATUS[0]}
            if [ \$EXIT_CODE -eq 0 ]; then
                echo 'Training completed successfully!' | tee -a \"\$LOG_FILE\"
                break
            else
                if [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; then
                    echo \"Training failed (exit \$EXIT_CODE). Restarting in 10s...\" | tee -a \"\$LOG_FILE\"
                    sleep 10
                    RETRY_COUNT=\$((RETRY_COUNT + 1))
                else
                    echo 'Training failed after all attempts.' | tee -a \"\$LOG_FILE\"
                    break
                fi
            fi
        done
    " 2>&1 | tee "$JOBDIR/train.log"

echo "Exit code: $?"
echo "End: $(date)"
