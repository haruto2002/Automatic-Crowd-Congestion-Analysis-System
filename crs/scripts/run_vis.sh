#!/bin/bash
#SBATCH -p gpu2
#SBATCH -o /homes/hnakayama/P2P/P2PNet/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/P2P/P2PNet/gomi/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/P2P/P2PNet"
cd $WORKDIR

HEATMAP_DIR="danger/results/0306_div/WorldPorters_noon/1_3500/10_5_1_50_0.1_False_True_75_3/v2/each_result/danger_heatmap"
BACK_IMG_DIR="danger/vec_vis/WorldPorters_noon/frames_with_map"
SAVE_DIR="danger/results/0306_div/WorldPorters_noon/1_3500/10_5_1_50_0.1_False_True_75_3/v2/visualize"
SAVE_MOV_NAME="heatmap_on_vec_map"

python danger/main_vis_crop.py \
--display_on_map \
--heatmap_dir $HEATMAP_DIR \
--back_img_dir $BACK_IMG_DIR \
--save_dir $SAVE_DIR \
--save_mov_name $SAVE_MOV_NAME \
--movie_fps 15

HEATMAP_DIR="danger/results/0306_curl/WorldPorters_noon/1_3500/10_5_1_50_0.1_False_True_75_3/curl_v2/each_result/danger_heatmap"
BACK_IMG_DIR="danger/vec_vis/WorldPorters_noon/frames_with_map"
SAVE_DIR="danger/results/0306_curl/WorldPorters_noon/1_3500/10_5_1_50_0.1_False_True_75_3/curl_v2/visualize"
SAVE_MOV_NAME="heatmap_on_vec_map"

python danger/main_vis_crop.py \
--display_on_map \
--heatmap_dir $HEATMAP_DIR \
--back_img_dir $BACK_IMG_DIR \
--save_dir $SAVE_DIR \
--save_mov_name $SAVE_MOV_NAME \
--movie_fps 15

HEATMAP_DIR="danger/results/0306_Cd/WorldPorters_noon/1_3500/10_10_1_50_0.1_False_True_75_3/v2/each_result/danger_heatmap"
BACK_IMG_DIR="danger/vec_vis/WorldPorters_noon/frames_with_map"
SAVE_DIR="danger/results/0306_Cd/WorldPorters_noon/1_3500/10_10_1_50_0.1_False_True_75_3/v2/visualize"
SAVE_MOV_NAME="heatmap_on_vec_map"

python danger/main_vis_crop.py \
--display_on_map \
--heatmap_dir $HEATMAP_DIR \
--back_img_dir $BACK_IMG_DIR \
--save_dir $SAVE_DIR \
--save_mov_name $SAVE_MOV_NAME \
--movie_fps 15