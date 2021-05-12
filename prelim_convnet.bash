#!/bin/bash
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time 8:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node 1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=kalum.ost

module load cuda/10.0.130
source activate PyTorchEnv

echo "Preparations complete! Running code!"

python "/home/kalum.ost/intechChapter/convnet.py" -o "/home/kalum.ost/intechChapter/outputs/full_set/simple" -sc 10 -pc 0 -e 90 -b 256
