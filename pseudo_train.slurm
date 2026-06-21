#!/bin/bash
#SBATCH --job-name=pseudo_train
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pseudo_%j.log
#SBATCH --error=logs/pseudo_%j.err

mkdir -p logs
module load python3/3.13.2
source .venv/bin/activate

python pseudo_train.py \
    --train_h5 data/train.h5 \
    --train_csv data/train.csv \
    --unlabeled_h5 data/unlabeled.h5 \
    --pretrained pretrained.pt \
    --epochs 60 \
    --pseudo_epochs 40 \
    --batch 64 \
    --lr 5e-4 \
    --embed 256 \
    --out model_pseudo.pt \
    --workers 2
