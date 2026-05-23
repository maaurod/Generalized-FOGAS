#!/bin/bash
# General-purpose FOGAS batch GPU job (excludes L4 node)
# Usage: bash submit_batch_gpu.sh <script.py> [extra_args...] [-- time gpu_count mem]
#
# The script path is REQUIRED (first argument).
# Any arguments before "--" are forwarded to the Python script.
# Arguments after "--" configure Slurm (positional: time, gpu_count, mem).
#
# Examples:
#   bash submit_batch_gpu.sh experiments/fogas_clean/scripts/grid_10_tabular.py --resume --no-progress
#   bash submit_batch_gpu.sh my_script.py --resume -- 12:00:00 2
#   bash submit_batch_gpu.sh my_script.py -- 08:00:00 1 64G

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "❌ Usage: bash submit_batch_gpu.sh <script.py> [script_args...] [-- time gpu_count mem]"
    exit 1
fi

SCRIPT_PATH="$1"; shift

# Separate python args from slurm args (split on --)
PYTHON_ARGS=()
SLURM_ARGS=()
PAST_SEP=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        PAST_SEP=true
        continue
    fi
    if $PAST_SEP; then
        SLURM_ARGS+=("$arg")
    else
        PYTHON_ARGS+=("$arg")
    fi
done

TIME="${SLURM_ARGS[0]:-24:00:00}"
GPU_COUNT="${SLURM_ARGS[1]:-1}"
MEM="${SLURM_ARGS[2]:-48G}"

# Resolve script path relative to project root
PROJECT_ROOT="/shared/home/mauro.diaz/work/FOGAS"
if [[ "$SCRIPT_PATH" = /* ]]; then
    FULL_SCRIPT="$SCRIPT_PATH"
else
    FULL_SCRIPT="$PROJECT_ROOT/$SCRIPT_PATH"
fi

if [ ! -f "$FULL_SCRIPT" ]; then
    echo "❌ Script not found: $FULL_SCRIPT"
    exit 1
fi

# Job name from script filename (without extension)
JOB_NAME="fogas_$(basename "$SCRIPT_PATH" .py)"

LOG_DIR="/shared/home/mauro.diaz/logs/fogas"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

PORTABLE_PY="/shared/home/mauro.diaz/portable-python/python"
VENV="/shared/home/mauro.diaz/work/FOGAS/venv"
PY="$PORTABLE_PY/bin/python3"

echo "📤 Submitting batch GPU job to Slurm …"
echo "   Script    : $FULL_SCRIPT"
echo "   Args      : ${PYTHON_ARGS[*]:-<none>}"
echo "   Time limit: $TIME"
echo "   GPUs      : $GPU_COUNT (excluding L4)"
echo "   Memory    : $MEM"
echo "   Log       : $LOG_FILE"
echo ""

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --partition=frida
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=8
#SBATCH --exclude=apl,aga
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

# ── Environment setup ──
# We use a portable, self-contained Python 3.10 to ensure binary compatibility
# across all nodes (especially axa, which natively runs Python 3.12).
export PYTHONHOME="$PORTABLE_PY"
export PYTHONPATH="$VENV/lib/python3.10/site-packages"
export PATH="$PORTABLE_PY/bin:$VENV/bin:\$PATH"
export VIRTUAL_ENV="$VENV"

cd /shared/home/mauro.diaz/work/FOGAS

echo "✅ Job started at: \$(date)"
echo "🖥️  Running on node: \$(hostname)"
echo "🐍 Python: $PY → \$($PY --version)"
echo "🎮 GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "📜 Script: $FULL_SCRIPT"
echo "📎 Args: ${PYTHON_ARGS[*]:-<none>}"
echo ""

$PY "$FULL_SCRIPT" ${PYTHON_ARGS[*]:-}

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
