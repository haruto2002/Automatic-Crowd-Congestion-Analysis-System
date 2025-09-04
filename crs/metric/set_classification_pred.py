import glob
import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import yaml
from utils import get_cropped_danger_score, get_cropped_density, get_bev_points


def process_target_info(args):
    """Function to process one target information (for parallel processing)"""
    i, info, danger_score_path_list, vec_data_path_list, grid_size, freq, bev_dir = args

    video_path = info["video_path"]

    raw_video_info_path = video_path.replace("videos/video", "infos/info").replace(
        ".mp4", ".json"
    )

    with open(raw_video_info_path, "r") as f:
        raw_video_info = json.load(f)

    frame_range = raw_video_info["frame_range"]
    crop_area = raw_video_info["crop_points"]
    target_danger_score_path_list = danger_score_path_list[
        frame_range[0] // freq : frame_range[1] // freq - 1
    ]
    target_vec_data_path_list = vec_data_path_list[
        frame_range[0] // freq : frame_range[1] // freq - 1
    ]
    bev_crop_data = get_bev_points(bev_dir, crop_area)
    # map_crop_area = bev_tl_point, bev_tr_point, bev_br_point, bev_bl_point / grid_size

    danger_scores = []
    clip_danger_scores = []
    densities = []

    for danger_score_path in target_danger_score_path_list:
        map_danger_score = np.loadtxt(danger_score_path)
        danger_score, clip_danger_score = get_cropped_danger_score(
            map_danger_score, bev_crop_data, grid_size
        )
        danger_scores.append(danger_score)
        clip_danger_scores.append(clip_danger_score)

    for vec_data_path in target_vec_data_path_list:
        vec_data = np.loadtxt(vec_data_path)
        density = get_cropped_density(vec_data, bev_crop_data)
        densities.append(density)

    danger_scores = np.array(danger_scores)
    clip_danger_scores = np.array(clip_danger_scores)
    densities = np.array(densities)

    danger_score_mean = np.mean(danger_scores)
    clip_danger_score_mean = np.mean(clip_danger_scores)
    density_mean = np.mean(densities)

    return i, {
        "danger_score": danger_score_mean,
        "danger_score_clip": clip_danger_score_mean,
        "density": density_mean,
    }


def set_classification_pred(dataset_name, pred_dir, dataset_dir):
    path2cfg = os.path.join(pred_dir, "config.yaml")
    with open(path2cfg, "r") as f:
        cfg = yaml.safe_load(f)
    grid_size = cfg["grid_size"]
    bev_dir = cfg["bev_dir"]
    freq = cfg["freq"]
    danger_score_dir = os.path.join(pred_dir, "each_result", "danger_score")
    vec_data_dir = os.path.join(pred_dir, "each_result", "vec_data")
    danger_score_path_list = sorted(glob.glob(os.path.join(danger_score_dir, "*.txt")))
    vec_data_path_list = sorted(glob.glob(os.path.join(vec_data_dir, "*.txt")))

    print("Setting data for targets")

    target_info_path = f"{dataset_dir}/all_info.csv"

    df_info = pd.read_csv(target_info_path)
    info_list = df_info[["video_path", "density"]].to_dict(orient="records")

    print(f"Found {len(df_info)} target info files")

    # 並列処理用の引数リストを作成
    args_list = [
        (i, info, danger_score_path_list, vec_data_path_list, grid_size, freq, bev_dir)
        for i, info in enumerate(info_list)
    ]

    # プロセス数の設定（CPU数を利用）
    num_processes = os.cpu_count()
    print(f"Using {num_processes} processes for parallel processing")

    # 並列処理の実行
    estimate_data = {}
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(args_list), desc="Processing targets") as pbar:
            for i, result in pool.imap_unordered(process_target_info, args_list):
                estimate_data[i] = result
                pbar.update()

    # 結果の保存
    save_dir = f"{pred_dir}/pred_data"
    os.makedirs(save_dir, exist_ok=True)
    output_path = f"{save_dir}/{dataset_name}_pred_data.json"
    with open(output_path, "w") as f:
        json.dump(estimate_data, f)

    print(f"Results saved to {output_path}")
