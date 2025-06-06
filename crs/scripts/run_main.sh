#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/gomi/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

python crs/main.py --use_yaml --yaml_path crs/config/config.yaml
