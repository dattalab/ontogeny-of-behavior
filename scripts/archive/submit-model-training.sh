#!/bin/env bash
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=14G
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:30:00
#SBATCH --output=/n/scratch3/users/w/wg41/train-sn-model-%j.out

source $HOME/.bashrc
conda activate aging
# module load gcc/9.2.0
# module load cuda/12.1
if [ "$#" -eq 1 ]; then
	python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py $1
elif [ "$#" -eq 2 ]; then
	python /home/wg41/code/ontogeny/scripts/03-train-size-norm.py $1 --checkpoint $2
fi
