import cv2
import glob
from tqdm import tqdm
import argparse
import numpy as np
import os
from multiprocessing import Pool
import logging


def plot_det_parallel(
    detection_dir,
    img_dir,
    save_dir,
    threshold=0.5,
    freq=1,
    node_type=None,
    disable_tqdm=False,
    tqdm_pos=0,
):
    path2detection_list = sorted(glob.glob(f"{detection_dir}/*.txt"))
    if freq > 1:
        path2detection_list = path2detection_list[::freq]
    pool_list = []
    for path2detection in path2detection_list:
        frame_name = path2detection.split("/")[-1].split(".")[0]
        pool_list.append([path2detection, img_dir, save_dir, frame_name, threshold])

    if node_type == "rt_HF":
        pool_size = 192
    elif node_type == "rt_HG":
        pool_size = 16
    elif node_type == "rt_HC":
        pool_size = 32
    else:
        pool_size = os.cpu_count()
    with Pool(processes=pool_size) as p:
        list(
            tqdm(
                p.imap_unordered(plot_det, pool_list),
                total=len(pool_list),
                desc="Plotting Detection",
                leave=False,
                position=tqdm_pos,
                disable=disable_tqdm,
            )
        )


def plot_det(pool_list):
    path2detection, img_dir, save_dir, frame_name, threshold = pool_list
    img_extension = glob.glob(f"{img_dir}/*")[0].split(".")[-1]
    img_path = f"{img_dir}/{frame_name}.{img_extension}"
    img = cv2.imread(img_path)
    det = np.loadtxt(path2detection)
    det = det[det[:, 2] > threshold]
    for d in det:
        x, y, _ = d
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imwrite(f"{save_dir}/{frame_name}.{img_extension}", img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_dir", type=str, default="demo/full_detection")
    parser.add_argument("--img_dir", type=str, default="demo/img")
    parser.add_argument("--save_dir", type=str, default="demo/detection_plot")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--node_type", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--multi_video_mode", type=bool, default=False)
    return parser.parse_args()


def main():
    args = get_args()
    detection_dir = args.detection_dir
    img_dir = args.img_dir
    save_dir = args.save_dir
    threshold = args.threshold
    freq = args.freq
    node_type = args.node_type
    log_level = args.log_level

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    disable_tqdm = logger.level >= logging.ERROR

    if args.multi_video_mode:
        tqdm_pos = 1
    else:
        tqdm_pos = 0

    os.makedirs(save_dir, exist_ok=True)
    plot_det_parallel(
        detection_dir,
        img_dir,
        save_dir,
        threshold,
        freq,
        node_type,
        disable_tqdm,
        tqdm_pos,
    )


if __name__ == "__main__":
    main()
