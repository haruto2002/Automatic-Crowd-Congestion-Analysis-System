#!/bin/bash
#SBATCH -p cpu1
#SBATCH -o /homes/hnakayama/congestion_analysis/gomi/%x-%j.out
#SBATCH -e /homes/hnakayama/congestion_analysis/gomi/%x-%j.out

source ~/miniconda3/bin/activate hnakayama2

WORKDIR="/homes/hnakayama/congestion_analysis"
cd $WORKDIR

PLACE=$1
TRACK_SAVE_DIR=byte_track/track_data/${PLACE}

# echo "Converting track data to frame data..."
# python byte_track/convert_track_data.py \
#         --json_path ${TRACK_SAVE_DIR}/interpolated_tracklets.json \
#         --save_dir ${TRACK_SAVE_DIR}/interpolated_frame_data

echo "Displaying track data..."
python byte_track/display_track.py \
        --place ${PLACE}

echo "Done"
echo "Time: $SECONDS seconds"

# sbatch byte_track/scripts/run_vis_track.sh WorldPorters_noon
# sbatch byte_track/scripts/run_vis_track.sh WorldPorters_night
# sbatch byte_track/scripts/run_vis_track.sh WorldPorters_night2