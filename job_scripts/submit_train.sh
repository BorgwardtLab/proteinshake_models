#!/bin/bash
#SBATCH --job-name="proteinshake_eval"
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --gpus-per-node=h100_pcie_2g.20gb:1
#SBATCH --exclude=hpcl9101
#SBATCH --time=10:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output slurm_logs/train_%A_%a.log
#SBATCH --array=0-3


hostname; date
echo $CUDA_VISIBLE_DEVICES
source ~/.bashrc
mamba activate shakesat

cd ..

python experiments/train.py task=enzyme_class representation=graph task.split=random seed=${SLURM_ARRAY_TASK_ID}

exit 0
