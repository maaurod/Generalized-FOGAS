#!/bin/bash
# Submit the resumable ContinuousSBEED Pendulum grid on a GPU node.
# Usage: bash jobs/submit_continuous_sbeed_pendulum_grid.sh [time] [gpu_count] [partition]

set -euo pipefail

TIME=${1:-"48:00:00"}
GPU_COUNT=${2:-6}
PARTITION=${3:-"frida"}
JOB_NAME="sbeed_pendulum_grid"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/data/results/sbeed/logs"
CSV_PATH="$REPO_ROOT/data/results/sbeed/continuous_pendulum_grid.csv"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

echo "Submitting ContinuousSBEED Pendulum grid"
echo "  partition : $PARTITION"
echo "  GPUs      : $GPU_COUNT"
echo "  time      : $TIME"
echo "  CSV       : $CSV_PATH"
echo "  log       : $LOG_FILE"

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --mem=64G
#SBATCH --cpus-per-task=$((GPU_COUNT * 4))
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

set -euo pipefail

cd "$REPO_ROOT"

if [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate fogas
elif command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    conda activate fogas
fi

echo "Job started at: \$(date)"
echo "Node: \$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python experiments/sbeed/scripts/grid_search_continuous_sbeed_pendulum.py \\
  --workers 0 \\
  --eval-every-episodes 10 \\
  --csv "$CSV_PATH"

echo ""
echo "Job finished at: \$(date)"
SBATCH_EOF

echo ""
echo "Submitted. Monitor with: squeue -u \$(whoami)"
echo "Latest log: tail -f \$(ls -t "$LOG_DIR"/${JOB_NAME}_*.log | head -n 1)"
echo "Cancel with: scancel -n $JOB_NAME -u \$(whoami)"
