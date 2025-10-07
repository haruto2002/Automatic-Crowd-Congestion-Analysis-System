#!/bin/bash
#PBS -l select=1
#PBS -q rt_HF
#PBS -l walltime=1:00:00
#PBS -P gaa50073
#PBS -e /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/
#PBS -o /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

IO_INFO_FILE=${IO_INFO_FILE:-$1}
IMG_DIR_NAME=${IMG_DIR_NAME:-$2}
IMG_EXTENSION=${IMG_EXTENSION:-$3}
NODE_TYPE=${NODE_TYPE:-$4}
LOG_LEVEL=${LOG_LEVEL:-$5}

python preprocess/get_all_frame_parallel.py \
        --io_info_file $IO_INFO_FILE \
        --img_dir_name $IMG_DIR_NAME \
        --img_extension $IMG_EXTENSION \
        --node_type $NODE_TYPE \
        --log_level $LOG_LEVEL
