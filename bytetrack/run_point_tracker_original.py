import argparse
import numpy as np
from track_utils.byte_tracker import BYTETracker
import json
import pandas as pd
import os
import glob
from tqdm import tqdm
import logging
import time

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--img_w_size", type=int, default=7680, help="image width size")
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベルを設定",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # ログレベルの設定
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    logger.info("=== ByteTrack Point Tracker 開始 ===")

    save_detail = False

    # Initialize tracker
    tracker = BYTETracker(args)
    img_size = (args.img_h_size, args.img_w_size)

    # 距離計算メトリックの設定
    tracker.distance_metric = args.distance_metric

    logger.info("Generating tracking data...")

    path2det_list = sorted(glob.glob(f"{args.source_dir}/*.txt"))
    if len(path2det_list) == 0:
        raise FileNotFoundError(args.source_dir)

    path2det_list = path2det_list[:1000]

    time_A = time.time()

    all_results = []
    all_data = []
    for frame_id, path2det in enumerate(tqdm(path2det_list)):
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

        if save_detail:
            save_dir = f"{args.save_dir}/frame_results"
            os.makedirs(save_dir, exist_ok=True)
            df = pd.DataFrame(save_data, columns=["id", "x", "y"])
            df.sort_values(by="id", inplace=True)
            df.to_csv(f"{save_dir}/{frame_id+1:04d}.csv", index=False)
            all_results.extend(frame_results)

        all_data.append(save_data)

    if save_detail:
        tracklets_by_id = {}
        for result in all_results:
            track_id = result["id"]
            if track_id not in tracklets_by_id:
                tracklets_by_id[track_id] = []
            tracklets_by_id[track_id].append(result)

        with open(f"{args.save_dir}/tracklets_by_id.json", "w") as f:
            json.dump({str(k): v for k, v in tracklets_by_id.items()}, f, indent=2)

    time_B = time.time()
    logger.info(f"Tracking完了時間: {time_B - time_A:.2f}秒")

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
    if save_detail:
        with open(f"{args.save_dir}/tracklets.json", "w") as f:
            json.dump(tracklets, f, indent=2)
    time_C = time.time()
    logger.info(f"Tracklets生成完了時間: {time_C - time_B:.2f}秒")

    # 補間処理と統合保存
    interpolated_tracklets = interpolate_tracklets(tracklets)
    if save_detail:
        with open(f"{args.save_dir}/interpolated_tracklets.json", "w") as f:
            json.dump(interpolated_tracklets, f, indent=2)

    time_D = time.time()
    logger.info(f"補間完了時間: {time_D - time_C:.2f}秒")

    # 各フレームのtxtファイル作成（統合処理）
    logger.info("保存中...")
    create_frame_txt_files(interpolated_tracklets, args.save_dir, len(path2det_list))

    time_E = time.time()
    logger.info(f"保存完了時間: {time_E - time_D:.2f}秒")


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
                    x = (x1 * (num_missing_frames + 1 - j) + x2 * j) / (
                        num_missing_frames + 1
                    )
                    y = (y1 * (num_missing_frames + 1 - j) + y2 * j) / (
                        num_missing_frames + 1
                    )
                    points.insert(i + j, (x, y))
                    frames.insert(i + j, frame1 + j)
                i += num_missing_frames
            i += 1
    return tracklets


def create_frame_txt_files(interpolated_tracklets, save_dir, total_frames):
    os.makedirs(save_dir, exist_ok=True)

    frame_data = {}

    for track_id, tracklet in interpolated_tracklets.items():
        frames = tracklet["frames"]
        points = tracklet["points"]

        for i, frame in enumerate(frames):
            if frame not in frame_data:
                frame_data[frame] = []

            x, y = points[i]
            frame_data[frame].append([int(track_id), float(x), float(y)])

    for frame_num in tqdm(range(1, total_frames + 1)):
        filename = os.path.join(save_dir, f"{frame_num:04d}.txt")

        if frame_num in frame_data:
            data = frame_data[frame_num]
            data.sort(key=lambda x: x[0])
            np.savetxt(filename, data, fmt=["%d", "%.6f", "%.6f"], delimiter=",")
        else:
            open(filename, "w").close()


if __name__ == "__main__":
    main()
