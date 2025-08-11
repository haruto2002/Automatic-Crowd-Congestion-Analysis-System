import datetime
import glob
import os
import subprocess
from multiprocessing import Pool
import ffmpeg
from tqdm import tqdm
import argparse


def main(save_dir, video_dir, freq):
    video_list_MOV = sorted(glob.glob(f"{video_dir}/*.MOV"))
    assert len(video_list_MOV) > 0, f"No MOV files found in {video_dir}"
    video_list = video_list_MOV

    pool_list = []
    for path2video in video_list:
        pool_list.append([path2video, save_dir, freq])

    pool_size = os.cpu_count()
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(parallel_cutter, pool_list), total=len(pool_list)))


def parallel_cutter(pool_list):
    path2video, save_dir, freq = pool_list

    video_name = path2video.split("/")[-1].split(".")[0]
    creation_time, start_time, duration = get_time_info(path2video)
    start_sec = (
        start_time - creation_time
    ).seconds  # 15:40:45>>15:41:00になるようにするための秒数
    for sec in range(start_sec, int(duration), freq):
        timestamp = creation_time + datetime.timedelta(seconds=sec)
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")

        output_img_path = f"{save_dir}/{video_name}_{timestamp_str}.png"
        command = f"ffmpeg -loglevel quiet -ss {sec} -i {path2video} -frames:v 1 -f image2 -q:v 1 {output_img_path}"
        subprocess.run(command, shell=True)


def get_time_info(cap_path):
    metadata = ffmpeg.probe(cap_path)
    creation_time = metadata["format"]["tags"]["creation_time"]
    creation_time = creation_time.replace("T", "_")
    creation_time = creation_time.replace("Z", "")
    creation_time = datetime.datetime.strptime(creation_time, "%Y-%m-%d_%H:%M:%S.%f")
    creation_time = creation_time + datetime.timedelta(hours=9)

    start_minute = creation_time.replace(second=0) + datetime.timedelta(minutes=1)

    duration = float(metadata["format"]["duration"])

    return creation_time, start_minute, duration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="yokohama")
    parser.add_argument("--video_dir", type=str, default="/homes/SHARE/Hanabi/20250602_Yokohama/8K/WorldPorter")
    parser.add_argument("--freq", type=int, default=60)  # second
    return parser.parse_args()


def run_main():
    args = get_args()
    os.makedirs(args.save_dir)
    main(args.save_dir, args.video_dir, args.freq)


if __name__ == "__main__":
    run_main()