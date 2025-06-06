import json
import numpy as np
import os
from tqdm import tqdm
import cv2
import argparse

def load_tracklets(path2tracklets):
    with open(path2tracklets, "r") as f:
        tracklets = json.load(f)
    
    for track_id, tracklet in tracklets.items():
        tracklet["frame_dict"] = {frame: idx for idx, frame in enumerate(tracklet["frames"])}
    
    return tracklets


def get_frame_data(tracklets, current_frame):
    frame_points = []
    for track_id, tracklet in tracklets.items():
        if current_frame in tracklet["frame_dict"]:
            index = tracklet["frame_dict"][current_frame]
            frame_point = tracklet["points"][index]
            frame_data = [int(track_id), frame_point[0], frame_point[1]]
            frame_points.append(frame_data)
    return frame_points

def convert_tracking_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,default="byte_track/track_all_data/WorldPorters_noon/interpolated_tracklets.json")
    parser.add_argument("--save_dir", type=str, required=True,default="byte_track/track_all_data/WorldPorters_noon/frame_data")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    tracklets = load_tracklets(args.json_path)
    max_frame = max(tracklet["frames"][-1] for tracklet in tracklets.values())
    for frame in tqdm(range(1, max_frame + 1)):
        output_path=args.save_dir + f"/{frame:04d}.txt"
        frame_points = get_frame_data(tracklets, frame)
        np.savetxt(output_path, np.array(frame_points), delimiter=",")

def debug():
    dets = np.loadtxt("byte_track/track_all_data/WorldPorters_noon/frame_data/0001.txt", delimiter=",")
    img=cv2.imread("tracking_results/WorldPorters_noon/img/DSC_7484_0001.png")
    for det in dets:
        cv2.circle(img, (int(det[1]), int(det[2])), 5, (0, 0, 255), -1)
    cv2.imwrite("0001.png", img)

if __name__ == "__main__":
    convert_tracking_json()
    # debug()