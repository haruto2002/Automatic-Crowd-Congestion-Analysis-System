import cv2
import glob
from tqdm import tqdm
import argparse
import numpy as np
import os
from multiprocessing import Pool


def plot_det_parallel(detection_dir, img_dir, save_dir, threshold=0.5):
    path2detection_list = sorted(glob.glob(f"{detection_dir}/*.txt"))
    pool_list = []
    for path2detection in path2detection_list:
        frame_name = path2detection.split("/")[-1].split(".")[0]
        pool_list.append([path2detection, img_dir, save_dir, frame_name, threshold])

    pool_size = os.cpu_count()
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(plot_det, pool_list), total=len(pool_list)))


def plot_det(pool_list):
    path2detection, img_dir, save_dir, frame_name, threshold = pool_list
    img_path = f"{img_dir}/{frame_name}.png"
    img = cv2.imread(img_path)
    det = np.loadtxt(path2detection)
    det = det[det[:, 2] > threshold]
    for d in det:
        x, y, _ = d
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imwrite(f"{save_dir}/{frame_name}.png", img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_dir", type=str, default="demo/full_detection")
    parser.add_argument("--img_dir", type=str, default="demo/img")
    parser.add_argument("--save_dir", type=str, default="demo/detection_plot")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()
    detection_dir = args.detection_dir
    img_dir = args.img_dir
    save_dir = args.save_dir
    threshold = args.threshold
    os.makedirs(save_dir, exist_ok=True)
    plot_det_parallel(detection_dir, img_dir, save_dir, threshold)


def plot_custom_det(frame):
    # source_dir = "demo_5min_8K"
    source_dir = "demo"

    frame_name = f"{frame:04d}"
    path2detection = f"{source_dir}/full_detection/{frame_name}.txt"
    img_dir = f"{source_dir}/img"
    save_dir = f"{source_dir}/detection_plot_custom"
    threshold = 0.1
    inputs = [path2detection, img_dir, save_dir, frame_name, threshold]
    os.makedirs(save_dir, exist_ok=True)
    plot_det(inputs)


if __name__ == "__main__":
    # main()
    plot_custom_det(1)
