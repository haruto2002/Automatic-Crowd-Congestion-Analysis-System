#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

WEIGHT_PATH=${WEIGHT_PATH:-$1}
IO_INFO_FILE=${IO_INFO_FILE:-$2}
IMG_SUB_DIR=${IMG_SUB_DIR:-$3}
OUT_SUB_DIR=${OUT_SUB_DIR:-$4}
FULL_DET_SUB_DIR=${FULL_DET_SUB_DIR:-$5}
NODE_TYPE=${NODE_TYPE:-$6}
LOG_LEVEL=${LOG_LEVEL:-$7}

python scripts/p2pnet/submit_fullsize_inference_ddp_multi_video.py \
    --weight_path $WEIGHT_PATH \
    --io_info_file $IO_INFO_FILE \
    --img_sub_dir $IMG_SUB_DIR \
    --out_sub_dir $OUT_SUB_DIR \
    --full_det_sub_dir $FULL_DET_SUB_DIR \
    --node_type $NODE_TYPE \
    --log_level $LOG_LEVEL
