import argparse
import glob
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--PARENT_DIR", type=str, default="outputs")
    parser.add_argument("--SAVE_DIR", type=str, default="outputs/IO_info")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def create_IO_info_file(parent_dir, save_dir):
    IO_setting_info = {}
    # dir_list = sorted(glob.glob("samples/*/*"))
    dir_parent_list = [
        "samples/2023_07_31_smart_festival",
        "samples/2024_0602_Yokohama_Kaikosai",
        "samples/2024_0805_Yokohama",
        "samples/2025_0602_Yokohama_Kaikosai",
    ]
    dir_list = []
    for dir in dir_parent_list:
        dir_list.extend(glob.glob(f"{dir}/*"))

    for dir in dir_list:
        video_list = sorted(glob.glob(f"{dir}/*.MOV"))
        if len(video_list) == 0:
            video_list = sorted(glob.glob(f"{dir}/*.MP4"))
        for video_path in video_list:
            event_name = video_path.split("/")[-3]
            location_name = video_path.split("/")[-2]
            video_name = video_path.split("/")[-1].split(".")[0]
            dir_structure = f"{parent_dir}/{event_name}/{location_name}/{video_name}"
            IO_setting_info[video_path] = dir_structure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "IO_info.json")
    with open(save_path, "w") as f:
        json.dump(IO_setting_info, f, indent=4)
    logger.info(f"IO_info.json saved to {save_path}")
    logger.info(f"Number of videos: {len(IO_setting_info)}")


def main():
    parser = get_args()
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    create_IO_info_file(args.PARENT_DIR, args.SAVE_DIR)


if __name__ == "__main__":
    main()
