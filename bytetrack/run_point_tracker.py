import argparse
import numpy as np
from track_utils.byte_tracker import BYTETracker
import json
import pandas as pd
import os
import glob
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Point Tracker")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="byte_track/track_data",
        help="save directory",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="byte_track/full_detection/WorldPorters_noon",
        help="source directory",
    )
    parser.add_argument(
        "--img_h_size", type=int, default=4320, help="image height size"
    )
    parser.add_argument(
        "--img_w_size", type=int, default=7680, help="image width size"
    )
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=10.0,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["maha", "euclidean"],
        help="distance metric to use for matching (maha: マハラノビス距離, euclidean: ユークリッド距離)",
    )
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    print(args)

    # トラッカーの初期化
    tracker = BYTETracker(args)
    img_size = (args.img_h_size, args.img_w_size)

    # 距離計算メトリックの設定
    tracker.distance_metric = args.distance_metric
    print(f"Using distance metric: {tracker.distance_metric}")

    print(f"Loaded detections from {args.source_dir}")
    path2det_list = sorted(glob.glob(f"{args.source_dir}/*.txt"))
    all_results = []
    all_data = []
    for frame_id, path2det in enumerate(tqdm(path2det_list)):
        # print(f"\nProcessing frame {frame_id+1}/{len(path2det_list)}")
        detections = np.loadtxt(path2det)

        online_targets = tracker.update(detections, img_size, img_size)

        frame_results = []
        save_data = []
        for t in online_targets:
            track_id = t.track_id
            kalman_point = t.point.tolist()
            detection_point = t.detection_point.tolist()
            score = t.score
            frame_results.append(
                {
                    "frame": frame_id + 1,
                    "id": track_id,
                    "det_x": detection_point[0],
                    "det_y": detection_point[1],
                    "kalman_x": kalman_point[0],
                    "kalman_y": kalman_point[1],
                    "score": score,
                }
            )

            save_data.append([track_id, detection_point[0], detection_point[1]])
        save_dir = f"{args.save_dir}/frame_results"
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(save_data, columns=["id", "x", "y"])
        df.sort_values(by="id", inplace=True)
        df.to_csv(f"{save_dir}/{frame_id+1:04d}.csv", index=False)

        all_results.extend(frame_results)
        all_data.append(save_data)

    tracklets_by_id = {}
    for result in all_results:
        track_id = result["id"]
        if track_id not in tracklets_by_id:
            tracklets_by_id[track_id] = []
        tracklets_by_id[track_id].append(result)

    with open(f"{args.save_dir}/tracklets_by_id.json", "w") as f:
        json.dump({str(k): v for k, v in tracklets_by_id.items()}, f, indent=2)

    tracklets = {}
    for i, frame_data in enumerate(all_data):
        frame = i + 1
        for data in frame_data:
            track_id = data[0]
            x = data[1]
            y = data[2]
            if track_id not in tracklets:
                tracklets[track_id] = {}
                tracklets[track_id]["frames"] = []
                tracklets[track_id]["points"] = []
            tracklets[track_id]["frames"].append(frame)
            tracklets[track_id]["points"].append((x, y))
    with open(f"{args.save_dir}/tracklets.json", "w") as f:
        json.dump(tracklets, f, indent=2)

    interpolated_tracklets = interpolate_tracklets(tracklets)
    with open(f"{args.save_dir}/interpolated_tracklets.json", "w") as f:
        json.dump(interpolated_tracklets, f, indent=2)


def interpolate_tracklets(tracklets):
    for track_id, tracklet in tracklets.items():
        frames = tracklet["frames"]
        points = tracklet["points"]
        i = 0
        while i < len(frames) - 1:
            frame1 = frames[i]
            frame2 = frames[i + 1]
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if frame1 + 1 == frame2:
                i += 1
                continue
            if frame1 + 1 < frame2:
                num_missing_frames = frame2 - frame1 - 1
                for j in range(1, num_missing_frames + 1):
                    x = (x1 * (num_missing_frames + 1 - j) + x2 * j) / (num_missing_frames + 1)
                    y = (y1 * (num_missing_frames + 1 - j) + y2 * j) / (num_missing_frames + 1)
                    points.insert(i + j, (x, y))
                    frames.insert(i + j, frame1 + j)
                i += num_missing_frames
            i += 1
    return tracklets


if __name__ == "__main__":
    main()
