import subprocess
import os
import argparse
import json
from tqdm import tqdm


def plot_detection_multi(
    io_info_file,
    img_sub_dir,
    detection_sub_dir,
    detection_vis_sub_dir,
    freq=1,
    node_type=None,
    log_level=None,
):
    with open(io_info_file, "r") as f:
        io_info = json.load(f)
    for video_path, save_dir in tqdm(
        io_info.items(),
        desc="Plotting Detection of Multi Video",
        leave=True,
        position=0,
    ):
        img_dir = os.path.join(save_dir, img_sub_dir)
        detection_dir = os.path.join(save_dir, detection_sub_dir)
        save_dir = os.path.join(save_dir, detection_vis_sub_dir)
        subprocess.run(
            [
                "bash",
                "scripts/vis/detection/plot_detection.sh",
                img_dir,
                detection_dir,
                save_dir,
                str(freq),
                "True",
                node_type,
                log_level,
            ]
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--io_info_file", type=str, required=True)
    parser.add_argument("--img_sub_dir", type=str, required=True)
    parser.add_argument("--detection_sub_dir", type=str, required=True)
    parser.add_argument("--detection_vis_sub_dir", type=str, required=True)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--node_type", type=str, required=True)
    parser.add_argument("--log_level", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    plot_detection_multi(
        args.io_info_file,
        args.img_sub_dir,
        args.detection_sub_dir,
        args.detection_vis_sub_dir,
        args.freq,
        args.node_type,
        args.log_level,
    )


if __name__ == "__main__":
    main()
