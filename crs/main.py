import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import logging
from utils.get_track_data import get_all_track, get_all_vec
from utils.clac_danger_score import (
    get_map_data,
    calc_Cd_map,
)
from utils.util_config import setup_config


def run_parallel(inputs):
    (
        res_save_dir,
        track_dir,
        bev_file,
        size_file,
        start_frame,
        grid_size,
        func_para,
        span,
        dan_ver,
        distance_decay_method,
        vec_decay_method,
        crop_area,
    ) = inputs
    press_map, dense_map, vec_list = main_single(
        res_save_dir,
        track_dir,
        bev_file,
        size_file,
        start_frame,
        grid_size,
        func_para,
        span=span,
        dan_ver=dan_ver,
        distance_decay_method=distance_decay_method,
        vec_decay_method=vec_decay_method,
        crop_area=crop_area,
    )
    return press_map, dense_map, vec_list


def main_single(
    save_dir,
    track_dir,
    bev_file,
    size_file,
    start_frame,
    grid_size,
    func_para,
    span,
    dan_ver="div",
    distance_decay_method="gaussian",
    vec_decay_method=None,
    crop_area=None,
):
    end_frame = start_frame + span
    all_track = get_all_track(track_dir, bev_file, start_frame, end_frame)
    width, height = np.loadtxt(size_file).astype(int)
    size = (width, height)
    vec_list = get_all_vec(all_track, end_frame)
    vec_data = np.array(vec_list)
    vec_data_copy = vec_data.copy()

    if dan_ver == "div" or dan_ver == "curl" or dan_ver == "crowd_pressure":
        map_data, dense_map = get_map_data(
            size,
            vec_data_copy,
            grid_size,
            func_para,
            dan_ver=dan_ver,
            distance_decay_method=distance_decay_method,
            vec_decay_method=vec_decay_method,
        )
    elif dan_ver == "Cd":
        decay, max_x, clip_value = func_para
        vec_data_stack_list = []
        stack_frame_start = start_frame
        stack_frame_end = end_frame
        for i in range(int(2.5 * 30)):
            stack_frame_start -= i
            stack_frame_end -= i
            if stack_frame_start <= 0:
                break
            all_track = get_all_track(
                track_dir, bev_file, stack_frame_start, stack_frame_end
            )
            frame_vec_list = get_all_vec(all_track, stack_frame_end)
            frame_vec_data = np.array(frame_vec_list)
            frame_vec_data_copy = frame_vec_data.copy()
            vec_data_stack_list.append(frame_vec_data_copy)
        map_data, dense_map, grid_vec_data = calc_Cd_map(
            size,
            vec_data_stack_list,
            grid_size,
            roi_size=3,
            distance_decay_method=distance_decay_method,
            max_x=max_x,
        )
        vec_data = grid_vec_data.reshape(-1, 2, 2)
        vec_data = np.nan_to_num(vec_data, nan=0)
    else:
        raise ValueError(f"Invalid dan_ver: {dan_ver}")

    save_map_data = True
    save_vec_data = True
    if crop_area is not None:
        crop_map_data = map_data[
            crop_area[1] // grid_size : crop_area[3] // grid_size,
            crop_area[0] // grid_size : crop_area[2] // grid_size,
        ]
        crop_dense_map = dense_map[
            crop_area[1] // grid_size : crop_area[3] // grid_size,
            crop_area[0] // grid_size : crop_area[2] // grid_size,
        ]
        crop_vec_data = vec_data[
            (vec_data[:, 0, 0] > crop_area[0])
            & (vec_data[:, 0, 0] < crop_area[2])
            & (vec_data[:, 0, 1] > crop_area[1])
            & (vec_data[:, 0, 1] < crop_area[3])
        ]
        crop_vec_data[:, 0, 0] = crop_vec_data[:, 0, 0] - crop_area[0]
        crop_vec_data[:, 0, 1] = crop_vec_data[:, 0, 1] - crop_area[1]

        if save_map_data:
            danger_save_dir = save_dir + "/danger_score"
            os.makedirs(danger_save_dir, exist_ok=True)
            np.savetxt(
                danger_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt",
                crop_map_data,
            )
        if save_vec_data:
            vec_save_dir = save_dir + "/vec_data"
            os.makedirs(vec_save_dir, exist_ok=True)
            save_vec_data = crop_vec_data.reshape(-1, 4)
            np.savetxt(
                vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", save_vec_data
            )
        return crop_map_data, crop_dense_map, crop_vec_data
    else:
        if save_map_data:
            danger_save_dir = save_dir + "/danger_score"
            os.makedirs(danger_save_dir, exist_ok=True)
            np.savetxt(
                danger_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", map_data
            )
        if save_vec_data:
            vec_save_dir = save_dir + "/vec_data"
            os.makedirs(vec_save_dir, exist_ok=True)
            save_vec_data = vec_data.reshape(-1, 4)
            np.savetxt(
                vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", save_vec_data
            )
        return map_data, dense_map, vec_data


def main(
    output_dir,
    track_dir,
    bev_file,
    size_file,
    crop_area,
    grid_size,
    vec_span,
    freq,
    func_para,
    frame_range,
    dan_ver,
    distance_decay_method,
    vec_decay_method,
):

    start_frame, end_frame = frame_range
    if start_frame is None:
        start_frame = 1
    if end_frame is None:
        end_frame = len(glob.glob(os.path.join(track_dir, "*.txt")))

    res_save_dir = f"{output_dir}/each_result"
    os.makedirs(res_save_dir, exist_ok=True)

    pool_list = []
    for frame in range(start_frame, end_frame, freq):
        input = (
            res_save_dir,
            track_dir,
            bev_file,
            size_file,
            frame,
            grid_size,
            func_para,
            vec_span,
            dan_ver,
            distance_decay_method,
            vec_decay_method,
            crop_area,
        )
        pool_list.append(input)

    # run parallel
    pool_size = min(os.cpu_count(), len(pool_list))
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(run_parallel, pool_list), total=len(pool_list)))


def run_with_yaml():
    cfg = setup_config()
    main(
        output_dir=cfg.output_dir,
        track_dir=cfg.track_dir,
        bev_file=cfg.bev_file,
        size_file=cfg.size_file,
        crop_area=cfg.crop_area,
        grid_size=cfg.grid_size,
        vec_span=cfg.vec_span,
        freq=cfg.freq,
        func_para=cfg.func_para,
        frame_range=cfg.frame_range,
        dan_ver=cfg.dan_ver,
        distance_decay_method=cfg.distance_decay_method,
        vec_decay_method=cfg.vec_decay_method,
    )


def run_with_argparse():
    parser = get_parser()
    args = parser.parse_args()

    func_para = [args.decay, args.max_x, args.clip_value]
    frame_range = (args.frame_start, args.frame_end)

    main(
        output_dir=args.output_dir,
        track_dir=args.track_dir,
        bev_file=args.bev_file,
        size_file=args.size_file,
        crop_area=args.crop_area,
        grid_size=args.grid_size,
        vec_span=args.vec_span,
        freq=args.freq,
        func_para=func_para,
        frame_range=frame_range,
        dan_ver=args.dan_ver,
        distance_decay_method=args.distance_decay_method,
        vec_decay_method=args.vec_decay_method,
    )


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track_dir",
        type=str,
        default="demo/track",
        help="Tracking data directory",
    )
    parser.add_argument(
        "--bev_file",
        type=str,
        default="crs/bev/WorldPorters_8K_matrix.txt",
        help="BEV data directory",
    )
    parser.add_argument(
        "--size_file",
        type=str,
        default="crs/size/WorldPorters_8K_size.txt",
        help="Size data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo/crowd_risk_score/debug",
        help="Output directory name",
    )
    parser.add_argument("--grid_size", type=int, default=5, help="Grid size")
    parser.add_argument(
        "--vec_span", type=int, default=10, help="Vector calculation span"
    )
    parser.add_argument(
        "--freq", type=int, default=10, help="Risk calculation frame interval"
    )
    parser.add_argument("--frame_start", type=int, default=1, help="Start frame")
    parser.add_argument("--frame_end", type=int, default=8990, help="End frame")
    parser.add_argument(
        "--dan_ver", type=str, default="div", help="Risk calculation version"
    )
    parser.add_argument(
        "--distance_decay_method",
        type=str,
        default=None,
        choices=[None, "gaussian", "liner"],
        help="Distance decay method",
    )
    parser.add_argument(
        "--vec_decay_method",
        type=str,
        default=None,
        choices=[None, "exp_func", "clip"],
        help="Vector decay method",
    )

    # func_para setting
    parser.add_argument(
        "--decay", type=float, default=1.0, help="Decay parameter: not used"
    )
    parser.add_argument("--max_x", type=float, default=50.0, help="Maximum distance")
    parser.add_argument(
        "--clip_value", type=float, default=0.1, help="Speed clip value"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== Crowd risk score analysis started ===")

    run_with_yaml()
