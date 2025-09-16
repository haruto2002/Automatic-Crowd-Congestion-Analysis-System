#!/bin/bash
# PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P gaa50073
#PBS -e /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/
#PBS -o /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

EXP_NAME=${EXP_NAME:-$1}
VIDEO_PATH=${VIDEO_PATH:-$2}

python run_job_hydra.py \
    experiment_name=${EXP_NAME} \
    settings.video_path=${VIDEO_PATH} \
    job_id=${PBS_JOBID}

