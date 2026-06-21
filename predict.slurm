#!/bin/bash
#SBATCH --job-name=predict_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/predict_%j.log
#SBATCH --error=logs/predict_%j.err

mkdir -p logs
module load python3/3.13.2
source .venv/bin/activate

python predict.py \
    --unlabeled_h5 data/unlabeled.h5 \
    --test_ids data/testIDs.csv \
    --model model_pseudo.pt \
    --batch 64 \
    --embed 256 \
    --out submission_pseudo.csv \
    --workers 2
