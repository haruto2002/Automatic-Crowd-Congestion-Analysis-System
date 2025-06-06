import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
def main(place):
    print(f"Counting ID for {place}...")
    if place == "WorldPorters_night2":
        tracking_results_dir = f"tracking_results/{place}/tracking3/results"
    else:
        tracking_results_dir = f"tracking_results/{place}/tracking/results"
    path2track_list = glob.glob(os.path.join(tracking_results_dir, "*.csv"))
    id_list = []
    for path2track in tqdm(path2track_list):
        df = pd.read_csv(path2track)
        id_list.append(df["id"].unique())
    id_list = np.concatenate(id_list)
    print("simple algorithm:", len(np.unique(id_list)))

    path2track = f"byte_track/track_data_pre/{place}/interpolated_tracklets.json"
    with open(path2track, "r") as f:
        data = json.load(f)
    print("byte track:", len(data.keys()))

def main_set_range(place):
    print(f"Counting ID for {place}...")
    if place == "WorldPorters_night2":
        tracking_results_dir = f"tracking_results/{place}/tracking3/results"
    else:
        tracking_results_dir = f"tracking_results/{place}/tracking/results"
    path2track_list = glob.glob(os.path.join(tracking_results_dir, "*.csv"))
    path2track_list = path2track_list[:1500]
    id_list = []
    for path2track in tqdm(path2track_list):
        df = pd.read_csv(path2track)
        id_list.append(df["id"].unique())
    id_list = np.concatenate(id_list)
    print("simple algorithm:", len(np.unique(id_list)))

    tracking_results_dir = f"byte_track/track_data/{place}/interpolated_frame_data"
    path2track_list = glob.glob(os.path.join(tracking_results_dir, "*.txt"))
    path2track_list = path2track_list[:1500]
    id_list = []
    for path2track in tqdm(path2track_list):
        df = pd.read_csv(path2track, header=None)
        id_list.append(df[0].unique())
    id_list = np.concatenate(id_list)
    print("byte track:", len(np.unique(id_list)))

if __name__ == "__main__":
    # main("WorldPorters_noon")
    # main("WorldPorters_night")
    # main("WorldPorters_night2")
    main_set_range("WorldPorters_noon")