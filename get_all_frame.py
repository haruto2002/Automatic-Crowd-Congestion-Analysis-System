import subprocess
import os
import glob
import time
import argparse


def get_frame(path2video, save_dir):
    command = f"ffmpeg -i {path2video} -vcodec png {save_dir}/%04d.png"
    subprocess.run(command, shell=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2video", type=str, default="demo.mov")
    parser.add_argument("--save_dir", type=str, default="demo/img")
    return parser.parse_args()


def main():
    args = get_args()
    path2video = args.path2video
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    get_frame(path2video, save_dir)
    frame_end = time.time()
    with open(save_dir + "/time_get_frame.txt", "w") as f:
        text = f"get frame:{frame_end-start}\n"
        f.write(text)


if __name__ == "__main__":
    main()
