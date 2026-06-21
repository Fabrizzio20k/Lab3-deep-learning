#!/bin/bash
#SBATCH --job-name=forest_pretrain
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pretrain_%j.log
#SBATCH --error=logs/pretrain_%j.err

mkdir -p logs
module load python3/3.13.2
source lab3/bin/activate

python pretrain.py \
    --data data/unlabeled.h5 \
    --epochs 30 \
    --batch 512 \
    --lr 3e-4 \
    --embed 256 \
    --out pretrained.pt
