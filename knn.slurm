#!/bin/bash
#SBATCH --job-name=knn_predict
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/knn_%j.log
#SBATCH --error=logs/knn_%j.err

mkdir -p logs
module load python3/3.13.2
source .venv/bin/activate

python knn_predict.py \
    --train_h5 data/train.h5 \
    --train_csv data/train.csv \
    --unlabeled_h5 data/unlabeled.h5 \
    --test_ids data/testIDs.csv \
    --model model_effnet.pt \
    --embed 256 \
    --k 15 \
    --batch 128 \
    --workers 2 \
    --out submission_knn.csv
