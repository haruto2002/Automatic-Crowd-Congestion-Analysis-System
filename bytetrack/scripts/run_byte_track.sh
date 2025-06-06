#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/gomi/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

PLACE=$1
if [ $PLACE == "WorldPorters_noon" ]; then
    VIDEO_NAME="DSC_7484"
elif [ $PLACE == "WorldPorters_night" ]; then
    VIDEO_NAME="DSC_7512"
elif [ $PLACE == "WorldPorters_night2" ]; then
    VIDEO_NAME="DSC_7510"
fi
TRACK_SAVE_DIR=byte_track/track_data/${PLACE}

echo "Running byte track for ${PLACE}(${VIDEO_NAME})..."

python byte_track/run_point_tracker.py \
        --save_dir ${TRACK_SAVE_DIR} \
        --source_dir byte_track/full_detection/${PLACE} \
        --img_h_size 4320 \
        --img_w_size 7680 \
        --track_thresh 0.5 \
        --track_buffer 30 \
        --match_thresh 10.0 \
        --distance_metric euclidean

echo "Converting track data to frame data..."
python byte_track/convert_track_data.py \
        --json_path ${TRACK_SAVE_DIR}/interpolated_tracklets.json \
        --save_dir ${TRACK_SAVE_DIR}/interpolated_frame_data


# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_noon
# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_night
# sbatch byte_track/scripts/run_byte_track.sh WorldPorters_night2
