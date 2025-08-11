#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/log/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/log/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

# IMG_DIR=demo/img
# SOURCE_DIR=demo/crowd_risk_score/debug
# SAVE_DIR=demo/risk_heatmap
# FRAME_RANGE="1 151"
# CROP_AREA="180 350 460 630"
# SCALE=5

IMG_DIR=demo_5min_8K/img
SOURCE_DIR=demo_5min_8K/crowd_risk_score/debug
SAVE_DIR=demo_5min_8K/risk_heatmap
FRAME_RANGE="1 1000"
CROP_AREA="0 0 1500 1100"
SCALE=5

python visualize/crowd_risk/heatmap.py \
        --save_base_dir ${SAVE_DIR} \
        --img_dir ${IMG_DIR} \
        --source_dir ${SOURCE_DIR} \
        --frame_range ${FRAME_RANGE} \
        --crop_area ${CROP_AREA} \
        --scale ${SCALE} \
        # --smoothing

echo "Done"
echo "Time: $SECONDS seconds"