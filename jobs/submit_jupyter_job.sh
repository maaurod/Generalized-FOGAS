#!/bin/bash
# Alternative: Submit Jupyter as a batch job on GPU node
# This will run in the background and you can connect later
# Usage: ./submit_jupyter_job.sh [time] [gpu_count] [port]

TIME=${1:-"12:00:00"}
GPU_COUNT=${2:-1}
PORT=${3:-8888}
JOB_NAME="fogas-jupyter"
LOG_FILE="jupyter_${PORT}.log"

echo "📤 Submitting Jupyter Lab job to GPU node..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

cd /shared/home/mauro.diaz/work/FOGAS
source venv/bin/activate

# Get the actual hostname and username
COMPUTE_HOSTNAME=\$(hostname)
USERNAME=\$(whoami)

echo "✅ Job started at: \$(date)"
echo "🖥️  Running on node: \$COMPUTE_HOSTNAME"
echo "🎮 GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "📓 Starting Jupyter Lab on \$COMPUTE_HOSTNAME:$PORT"
echo "🔗 To access, create an SSH tunnel from your local machine:"
echo "   ssh -L $PORT:\$COMPUTE_HOSTNAME:$PORT \$USERNAME@login-frida"
echo ""
echo "Then open in your browser: http://localhost:$PORT"
echo ""

jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser
EOF

echo ""
echo "✅ Job submitted! Monitor it with: squeue -u \$(whoami)"
echo "📋 Check the log file: tail -f $LOG_FILE"
echo "🛑 To cancel: scancel \$(squeue -u \$(whoami) -n $JOB_NAME -h -o %i)"
