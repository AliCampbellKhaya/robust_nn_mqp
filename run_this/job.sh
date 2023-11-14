#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -C A100
#SBATCH -t 12:00:00
#SBATCH --mem 24G
#SBATCH -p short
#SBATCH --job-name="traffic_train"

source activate robustnn_mqp
python TRAFFIC_CNN_1.py --train_dqn