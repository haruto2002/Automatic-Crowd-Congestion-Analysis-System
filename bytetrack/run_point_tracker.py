import argparse
import numpy as np
from track_utils.byte_tracker import BYTETracker
import pandas as pd
import os
import glob
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Point Tracker")
    parser.add_argument("--save_dir", type=str, default="byte_track/track_data")
    parser.add_argument(
        "--source_dir", type=str, default="byte_track/full_detection/WorldPorters_noon"
    )
    parser.add_argument("--img_h_size", type=int, default=4320)
    parser.add_argument("--img_w_size", type=int, default=7680)
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=10.0)
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["maha", "euclidean"],
    )
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--max_frames", type=int, default=0, help="0: all, >0: limit")
    return parser


# ---- 高速ローダ（並列 I/O） -----------------------------------------
def _load_txt_to_np(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None, sep=r"\s+", dtype=np.float32).to_numpy()


def load_all_detections(paths, workers: int, disable_tqdm: bool) -> list[np.ndarray]:
    if workers <= 1:
        return [
            _load_txt_to_np(p)
            for p in tqdm(paths, desc="Loading detections", disable=disable_tqdm)
        ]
    dets = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_txt_to_np, p): i for i, p in enumerate(paths)}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading detections (parallel)",
            disable=disable_tqdm,
        ):
            i = futures[fut]
            dets[i] = fut.result()
    return dets


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


# ---- Frame-by-frame output (parallel & fast with string buffer) -------------
def _write_frame_txt(path: str, rows: list[list[float]]) -> None:
    # np.savetxt tends to be slow with many small files. Use string concatenation for batch output.
    # Format: id(int), x, y separated by commas
    if not rows:
        open(path, "w").close()
        return
    # Sort & stringify
    rows.sort(key=lambda r: r[0])
    lines = [
        "{},{:.6f},{:.6f}\n".format(int(r[0]), float(r[1]), float(r[2])) for r in rows
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def create_frame_txt_files(
    interpolated_tracklets,
    save_dir: str,
    total_frames: int,
    workers: int,
    disable_tqdm: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    frame_data = defaultdict(list)

    # 各トラックの (frame, point) をフレーム側に寄せ集め
    for track_id, tr in interpolated_tracklets.items():
        tid = int(track_id)
        frames = tr["frames"]
        points = tr["points"]
        append_local = frame_data.__getitem__  # ルックアップ短縮
        for f, (x, y) in zip(frames, points):
            append_local(f).append([tid, x, y])

    # 並列書き出し
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for frame_num in range(1, total_frames + 1):
            filename = os.path.join(save_dir, f"{frame_num:04d}.txt")
            rows = frame_data.get(frame_num, [])
            tasks.append(ex.submit(_write_frame_txt, filename, rows))
        # 進捗バー
        for _ in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc="Writing frames",
            disable=disable_tqdm,
        ):
            pass


def main():
    parser = make_parser()
    args = parser.parse_args()

    # Set log level
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    disable_tqdm = logger.level >= logging.ERROR

    logger.info("=== ByteTrack Point Tracker started ===")

    # Initialize tracker
    tracker = BYTETracker(args)
    tracker.distance_metric = args.distance_metric
    img_size = (int(args.img_h_size), int(args.img_w_size))

    # Detection file list
    path2det_list = sorted(glob.glob(f"{args.source_dir}/*.txt"))
    if not path2det_list:
        raise FileNotFoundError(args.source_dir)
    if args.max_frames > 0:
        path2det_list = path2det_list[: args.max_frames]

    time_A = time.time()

    # 先読み I/O（並列）
    io_workers = os.cpu_count()
    dets_list = load_all_detections(
        path2det_list, workers=io_workers, disable_tqdm=disable_tqdm
    )

    all_data = []
    append_all = all_data.append

    # 逐次追跡（ここは依存あり）
    for detections in tqdm(dets_list, desc="Tracking", disable=disable_tqdm):
        online_targets = tracker.update(detections, img_size, img_size)

        n = len(online_targets)
        if n == 0:
            append_all(np.empty((0, 3), dtype=np.float32))
            continue

        out = np.empty((n, 3), dtype=np.float32)
        for i, t in enumerate(online_targets):
            dp = t.detection_point  # [x, y] (list/ndarray)
            out[i, 0] = t.track_id
            out[i, 1] = dp[0]
            out[i, 2] = dp[1]
        append_all(out)

    time_B = time.time()
    logger.info(f"Tracking completed in: {time_B - time_A:.2f} seconds")

    # tracklets 構築（辞書→配列）
    tracklets = {}
    for frame_idx, frame_arr in enumerate(all_data, start=1):
        if frame_arr.size == 0:
            continue
        # frame_arr: (N,3) = [id, x, y]
        tids = frame_arr[:, 0].astype(np.int64)
        xs = frame_arr[:, 1]
        ys = frame_arr[:, 2]
        for tid, x, y in zip(tids, xs, ys):
            tr = tracklets.get(tid)
            if tr is None:
                tr = {"frames": [], "points": []}
                tracklets[tid] = tr
            tr["frames"].append(frame_idx)
            tr["points"].append((float(x), float(y)))

    time_C = time.time()
    logger.info(f"Tracklets generation completed in: {time_C - time_B:.2f} seconds")

    interpolated_tracklets = interpolate_tracklets(tracklets)

    time_D = time.time()
    logger.info(f"Interpolation completed in: {time_D - time_C:.2f} seconds")

    # Save txt files by frame (parallel)
    logger.info("Saving...")
    save_workers = os.cpu_count()
    create_frame_txt_files(
        interpolated_tracklets,
        args.save_dir,
        total_frames=len(path2det_list),
        workers=save_workers,
        disable_tqdm=disable_tqdm,
    )

    time_E = time.time()
    logger.info(f"Save completed in: {time_E - time_D:.2f} seconds")
    logger.info("=== Completed ===")


if __name__ == "__main__":
    main()
