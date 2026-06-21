#!/bin/bash
#SBATCH --job-name=train_effnet
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

mkdir -p logs
module load python3/3.13.2
source .venv/bin/activate

python train.py \
    --train_h5 data/train.h5 \
    --train_csv data/train.csv \
    --epochs 80 \
    --batch 32 \
    --lr 3e-4 \
    --lr_backbone 1e-5 \
    --freeze-epochs 10 \
    --embed 256 \
    --backbone efficientnet_b0 \
    --out model_effnet.pt \
    --no-val \
    --workers 2
