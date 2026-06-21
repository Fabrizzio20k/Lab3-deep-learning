#!/bin/bash
#SBATCH --job-name=forest_train
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

mkdir -p logs
module load python3/3.13.2
source .venv/bin/activate

python train.py \
    --train_h5 data/train.h5 \
    --train_csv data/train.csv \
    --pretrained pretrained.pt \
    --epochs 60 \
    --batch 32 \
    --lr 1e-3 \
    --embed 256 \
    --out model_best.pt \
    --workers 8
