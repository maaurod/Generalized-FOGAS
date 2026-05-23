#!/bin/bash
# Submit FOGAS 10-grid Tabular Grid Search as a Slurm batch job
# Usage: bash submit_grid_10_tabular.sh [time] [gpu_count]
#
# Defaults:  time = 24:00:00,  gpu_count = 1
#
# Examples:
#   bash submit_grid_10_tabular.sh            # default 24 h, 1 GPU
#   bash submit_grid_10_tabular.sh 12:00:00   # 12 h, 1 GPU
#   bash submit_grid_10_tabular.sh 08:00:00 2 # 8 h, 2 GPUs

TIME=${1:-"24:00:00"}
GPU_COUNT=${2:-1}
JOB_NAME="fogas_gs_10grid"

LOG_DIR="/shared/home/mauro.diaz/logs/fogas"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

echo "📤 Submitting FOGAS 10-grid Tabular Grid Search to Slurm …"
echo "   Time limit : $TIME"
echo "   GPUs       : $GPU_COUNT × A100_80GB"
echo "   Log        : $LOG_FILE"
echo ""

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --gres=gpu:A100_80GB:$GPU_COUNT
#SBATCH --partition=frida
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

cd /shared/home/mauro.diaz/work/FOGAS
source venv/bin/activate
export PATH="/shared/home/mauro.diaz/work/FOGAS/venv/bin:\$PATH"

echo "✅ Job started at: \$(date)"
echo "🖥️  Running on node: \$(hostname)"
echo "🐍 Python: \$(which python3)"
echo "🎮 GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

/shared/home/mauro.diaz/work/FOGAS/venv/bin/python3 /shared/home/mauro.diaz/work/FOGAS/experiments/fogas_clean/scripts/grid_10_tabular.py --resume --no-progress

echo ""
echo "✅ Job finished at: \$(date)"
SBATCH_EOF

echo ""
echo "✅ Job submitted!  Monitor with:"
echo "   squeue -u \$(whoami)"
echo ""
echo "📋 Tail the log in real time:"
echo "   tail -f \$(ls -t $LOG_DIR/${JOB_NAME}_*.log | head -n 1)"
echo ""
echo "❌ To cancel:"
echo "   scancel -n $JOB_NAME -u \$(whoami)"
