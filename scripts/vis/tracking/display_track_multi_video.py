import subprocess
import os
import argparse
import json
from tqdm import tqdm


def display_track_multi(
    io_info_file,
    img_sub_dir,
    track_sub_dir,
    track_vis_sub_dir,
    freq=1,
    node_type=None,
    log_level=None,
):
    with open(io_info_file, "r") as f:
        io_info = json.load(f)
    for video_path, save_dir in tqdm(
        io_info.items(), desc="Displaying Track of Multi Video", leave=True, position=0
    ):
        img_dir = os.path.join(save_dir, img_sub_dir)
        track_dir = os.path.join(save_dir, track_sub_dir)
        save_dir = os.path.join(save_dir, track_vis_sub_dir)
        subprocess.run(
            [
                "bash",
                "scripts/vis/tracking/display_track.sh",
                track_dir,
                img_dir,
                save_dir,
                str(freq),
                node_type,
                log_level,
                "True",
            ]
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--io_info_file", type=str, required=True)
    parser.add_argument("--img_sub_dir", type=str, required=True)
    parser.add_argument("--track_sub_dir", type=str, required=True)
    parser.add_argument("--track_vis_sub_dir", type=str, required=True)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--node_type", type=str, required=True)
    parser.add_argument("--log_level", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    display_track_multi(
        args.io_info_file,
        args.img_sub_dir,
        args.track_sub_dir,
        args.track_vis_sub_dir,
        args.freq,
        args.node_type,
        args.log_level,
    )


if __name__ == "__main__":
    main()
