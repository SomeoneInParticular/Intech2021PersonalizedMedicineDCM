#!/bin/bash
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time 8:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node 1

module load cuda/10.0.130
source activate PyTorchEnv

echo "Preparations complete! Running code!"

python "/home/kalum.ost/intechChapter/dense_progressive_batched.py" -o "/home/kalum.ost/intechChapter/outputs/batched/dense_progressive" -et 300 -ep 10 -mp 20 -b 64
