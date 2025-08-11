import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
import os
import glob
from tqdm import tqdm
import argparse


def moving_average(map_data, space_window_size=3, time_window_size=10):
    if space_window_size > 0 and time_window_size > 0:
        print("Smoothing map data by space")
        ma_map_data = space_moving_average(map_data, window_size=space_window_size)
        print("Smoothing map data by time")
        ma_map_data = time_moving_average(ma_map_data, window_size=time_window_size)
    elif time_window_size > 0 and space_window_size == 0:
        print("Smoothing map data by time")
        ma_map_data = time_moving_average(map_data, window_size=time_window_size)
    elif space_window_size > 0 and time_window_size == 0:
        print("Smoothing map data by space")
        ma_map_data = space_moving_average(map_data, window_size=space_window_size)

    return ma_map_data


def time_moving_average(map_data, window_size=10):
    # map_data.shape >> (num_frame, y_size, x_size)
    ma_hm_data = uniform_filter1d(map_data, size=window_size, axis=0, mode="nearest")
    return ma_hm_data


def space_moving_average(map_data, window_size=3):
    # map_data.shape >> (num_frame, y_size, x_size)
    kernel = np.ones((1, window_size, window_size)) / (window_size * window_size)
    ma_hm_data = np.zeros_like(map_data)

    for i in range(len(map_data)):
        ma_hm_data[i] = cv2.filter2D(map_data[i], -1, kernel[0])

    return ma_hm_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--space_window_size", type=int, required=True)
    parser.add_argument("--time_window_size", type=int, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    source_dir = args.source_dir
    save_dir = args.save_dir
    space_window_size = args.space_window_size
    time_window_size = args.time_window_size
    os.makedirs(save_dir, exist_ok=True)
    map_data = [
        np.loadtxt(file)
        for file in sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    ]
    file_name_list = [
        os.path.basename(file)
        for file in sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    ]
    ma_map_data = moving_average(map_data, space_window_size, time_window_size)

    print("Saving smoothed map data")
    for i, map_data in enumerate(tqdm(ma_map_data)):
        np.savetxt(os.path.join(save_dir, file_name_list[i]), map_data)


def main_single():
    source_dir = "demo_5min_8K/crowd_risk_score/debug/each_result/danger_score"
    save_dir = "demo_5min_8K/crowd_risk_score/debug/each_result/danger_score_smoothed"
    space_window_size = 5
    time_window_size = 10

    os.makedirs(save_dir, exist_ok=True)
    map_data = [
        np.loadtxt(file)
        for file in sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    ]
    file_name_list = [
        os.path.basename(file)
        for file in sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    ]
    ma_map_data = moving_average(map_data, space_window_size, time_window_size)

    print("Saving smoothed map data")
    for i, map_data in enumerate(tqdm(ma_map_data)):
        np.savetxt(os.path.join(save_dir, file_name_list[i]), map_data)


if __name__ == "__main__":
    # main()
    main_single()
