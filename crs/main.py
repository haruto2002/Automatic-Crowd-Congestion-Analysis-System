import numpy as np
import pandas as pd
import glob
import cv2
import os
import yaml
import subprocess
from multiprocessing import Pool
import argparse
from scipy.ndimage import uniform_filter1d
from utils.get_track_data import get_all_track, get_all_vec, get_all_track_byte_track
from utils.clac_danger_score import (
    get_map_data,
    calc_Cd_map,
)

from utils.vis_utils import (
    display_map_with_img_cv2,
    vis_hist,
    display_norm_parallel,
    create_movie,
)
from utils.graph import create_graph


def run_parallel(inputs):
    (
        place,
        start_frame,
        grid_size,
        func_para,
        span,
        display,
        res_save_dir,
        dan_ver,
        distance_decay_method,
        div_avg,
        vec_decay_method,
        track_method,
        crop_area,
        search,
    ) = inputs
    press_map, dense_map, vec_list, img = main_single(
        place,
        start_frame,
        grid_size,
        func_para,
        span=span,
        display=display,
        save_dir=res_save_dir,
        dan_ver=dan_ver,
        distance_decay_method=distance_decay_method,
        div_avg=div_avg,
        vec_decay_method=vec_decay_method,
        track_method=track_method,
        crop_area=crop_area,
        search=search,
    )
    # print("img:",img.shape)
    # print("map:",press_map.shape)
    return press_map, dense_map, vec_list, img


def main_single(
    place,
    start_frame,
    grid_size,
    func_para,
    span=90,
    display=True,
    save_dir=None,
    dir_name="press_map_crop",
    dan_ver=None,
    distance_decay_method=None,
    div_avg=False,
    vec_decay_method=None,
    track_method=None,
    crop_area=None,
    search=False,
):
    end_frame = start_frame + span
    if save_dir is None:
        save_dir = (
            f"{dir_name}/{place}_{grid_size}/each_result/{start_frame}_{end_frame}"
        )
    if track_method == "byte_track":
        all_track, img = get_all_track_byte_track(place, start_frame, end_frame, crop_area)
    elif track_method == "simple":
        all_track, img = get_all_track(place, start_frame, end_frame, crop_area)
    else:
        raise ValueError(f"Invalid track method: {track_method}")
    size = (img.shape[1], img.shape[0])
    vec_list = get_all_vec(all_track, end_frame)
    vec_data = np.array(vec_list)
    vec_data_copy=vec_data.copy()
    vec_data_copy[:,1] = vec_data_copy[:, 1] * 29.7 / 13.5
    if dan_ver == "div" or dan_ver == "curl" or dan_ver == "crowd_pressure":
        map_data, dense_map = get_map_data(
            size, vec_data_copy, grid_size, func_para, dan_ver=dan_ver, distance_decay_method=distance_decay_method, div_avg=div_avg, vec_decay_method=vec_decay_method
        )
    elif dan_ver == "Cd":
        decay, max_x, clip_value = func_para
        vec_data_stack_list=[]
        stack_frame_start=start_frame
        stack_frame_end=end_frame
        for i in range(int(2.5*30)):
            stack_frame_start-=i
            stack_frame_end-=i
            if stack_frame_start <= 0:
                break
            if track_method == "byte_track":
                all_track, _ = get_all_track_byte_track(place, stack_frame_start, stack_frame_end, crop_area)
            elif track_method == "simple":
                all_track, _ = get_all_track(place, stack_frame_start, stack_frame_end, crop_area)
            else:
                raise ValueError(f"Invalid track method: {track_method}")
            frame_vec_list = get_all_vec(all_track, stack_frame_end)
            frame_vec_data = np.array(frame_vec_list)
            frame_vec_data_copy=frame_vec_data.copy()
            frame_vec_data_copy[:,1] = frame_vec_data_copy[:, 1] * 29.7 / 13.5
            vec_data_stack_list.append(frame_vec_data_copy)
        map_data, dense_map,grid_vec_data = calc_Cd_map(
            size, vec_data_stack_list, grid_size, roi_size=3, distance_decay_method=distance_decay_method, max_x=max_x
        )
        vec_data = grid_vec_data.reshape(-1,2,2)
        vec_data[:,1] = vec_data[:,1] * 13.5 / 29.7
        vec_data = np.nan_to_num(vec_data, nan=0)
    else:
        raise ValueError(f"Invalid dan_ver: {dan_ver}")

    if not search and display:
        save_name = f"{start_frame:04d}_{end_frame:04d}_danger"
        vis_hist(map_data, save_dir, save_name)
        # display_map_with_img(
        #     map_data,
        #     vec_data,
        #     img,
        #     save_dir,
        #     save_name,
        #     grid_size,
        #     crop_area=crop_area,
        #     exp=False,
        # )
        decay, max_x, clip_value = func_para
        display_map_with_img_cv2(
            map_data,
            vec_data,
            img,
            save_dir,
            save_name,
            clip_value,
            crop_area=crop_area,
            exp=False,
        )

    if search:
        div_map, avg_div_map, dence_map, decay_dence_map = search_map_data(
            vec_data, grid_size, crop_area, func_para, exp_name=None
        )
        div_map *= -1
        avg_div_map *= -1

        map_list = [map_data, div_map, avg_div_map, dence_map, decay_dence_map]
        name_list = ["danger", "div", "avg_div", "dence", "decay_dence"]

        if display:
            for map, name in zip(map_list, name_list):
                save_name = f"{start_frame:04d}_{end_frame:04d}_{name}"
                vis_hist(map, save_dir, save_name)
                display_map_with_img_cv2(
                    map_data,
                    vec_data,
                    img,
                    save_dir,
                    save_name,
                    grid_size,
                    crop_area=crop_area,
                    exp=False,
                )
                # break

    save_map_data = True
    save_vec_data=True
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
        crop_img = img[
            crop_area[1]: crop_area[3],
            crop_area[0]: crop_area[2],
        ]
        
        if save_map_data:
            danger_save_dir = save_dir + "/danger_score"
            os.makedirs(danger_save_dir, exist_ok=True)
            np.savetxt(
                danger_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", crop_map_data
            )
        if save_vec_data:
            vec_save_dir = save_dir + "/vec_data"
            os.makedirs(vec_save_dir, exist_ok=True)
            save_vec_data=crop_vec_data.reshape(-1, 4)
            np.savetxt(
                vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", save_vec_data
            )
        return crop_map_data, crop_dense_map, crop_vec_data, crop_img
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
            save_vec_data=vec_data.reshape(-1, 4)
            np.savetxt(
                vec_save_dir + f"/{start_frame:04d}_{end_frame:04d}.txt", save_vec_data
            )
        return map_data, dense_map, vec_data, img


def main(
    place,
    results_base_dir_name,
    dir_name,
    crop_area,
    grid_size,
    vec_span,
    freq,
    func_para,
    frame_range,
    dan_ver,
    distance_decay_method,
    div_avg,
    vec_decay_method,
    track_method,
    space_ma=False,
    time_ma=False,
    space_window_size=None,
    time_window_size=None,
    danger_thred_list=[5, 10, 15],
    dense_thred_list=[75, 100, 125],
):
    if "debug" in dir_name:
        exist_ok = True
    else:
        exist_ok = False

    start_frame, end_frame = frame_range
    save_dir = f"danger/{results_base_dir_name}/{dir_name}/{place}/{start_frame}_{end_frame}_{vec_span}_{grid_size}_{func_para[0]}_{func_para[1]}_{func_para[2]}"
    print("SAVE_DIR >> ", save_dir)
    res_save_dir = f"{save_dir}/each_result"
    os.makedirs(res_save_dir, exist_ok=exist_ok)

    save_config(
        save_dir,
        results_base_dir_name,
        dir_name,
        place,
        crop_area,
        frame_range,
        grid_size,
        vec_span,
        freq,
        func_para,
        space_ma,
        time_ma,
        space_window_size,
        time_window_size,
        dan_ver,
        distance_decay_method,
        div_avg,
        vec_decay_method,
        track_method,
        danger_thred_list,
        dense_thred_list,
    )

    pool_list = []
    save_name_list = []
    display = False
    search = False
    for frame in range(start_frame, end_frame, freq):
        input = (
            place,
            frame,
            grid_size,
            func_para,
            vec_span,
            display,
            res_save_dir,
            dan_ver,
            distance_decay_method,
            div_avg,
            vec_decay_method,
            track_method,
            crop_area,
            search,
        )
        pool_list.append(input)

        save_name = f"{frame:04d}_{frame+vec_span:04d}"
        save_name_list.append(save_name)

    # run parallel
    print("calcurating")
    pool_size = min(os.cpu_count(),len(pool_list))
    with Pool(pool_size) as p:
        results = p.map(run_parallel, pool_list)

    # moving average
    if space_ma or time_ma:
        print("calcurating moving average")
        if space_ma and time_ma:
            ma_hm_data = space_moving_average(results, window_size=space_window_size)
            ma_hm_data = time_moving_average(
                ma_hm_data, window_size=time_window_size, input_map=True
            )
        elif time_ma and not space_ma:
            ma_hm_data = time_moving_average(results, window_size=time_window_size)
        elif space_ma and not time_ma:
            ma_hm_data = space_moving_average(results, window_size=space_window_size)

    # reshape results
    print("saving data")
    frame_list = list(range(start_frame, end_frame, freq))
    total_results = []
    data_list = []
    percentiles = [90, 95, 99]
    all_danger_values = np.concatenate([res[0].flatten() for res in results])
    danger_thresholds = np.percentile(all_danger_values, percentiles)
    all_dense_values = np.concatenate([res[1].flatten() for res in results])
    dense_thresholds = np.percentile(all_dense_values, percentiles)

    for i, res in enumerate(results):
        danger_map, dense_map, vec_data, img = res
        if space_ma or time_ma:
            danger_map = ma_hm_data[i]
        save_name = save_name_list[i]
        total_results.append((danger_map, dense_map, vec_data, img, save_name))
        num_people = len(vec_data)
        num_danger_list = [np.count_nonzero(danger_map > thred) for thred in danger_thresholds]
        num_dense_list = [np.count_nonzero(dense_map > thred) for thred in dense_thresholds]
        data = [frame_list[i], num_people] + num_danger_list + num_dense_list
        data_list.append(data)
    columns = (
        ["frame", "num_people"] +
        [f"num_danger_{i+1:02d}" for i in range(len(num_danger_list))] +
        [f"num_dense_{i+1:02d}" for i in range(len(num_dense_list))]
    )
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(save_dir + f"/data_{start_frame}_{end_frame}.csv", index=False)
    create_graph(df, None, save_dir)
    print("visualizing")
    decay, max_x, clip_value = func_para
    display_norm_parallel(
        total_results,
        res_save_dir,
        clip_value,
        grid_size,
        exp=False,
        crop_area=crop_area,
        set_max_score=danger_thred_list[-1]*2,
    )
    # save_name = f"{place}_{start_frame}_{end_frame}_{grid_size}_{vec_span}_norm"
    # create_movie(res_save_dir + "/danger_vis", save_dir, save_name)


def time_moving_average(results, window_size=10, input_map=False):
    if input_map:
        hm_data = results
    else:
        hm_data = np.array([hm for hm, _, _, _ in results])
    # hm_data.shape >> (num_frame, y_size, x_size)
    ma_hm_data = uniform_filter1d(hm_data, size=window_size, axis=0, mode="nearest")
    return ma_hm_data


def space_moving_average(results, window_size=3, input_map=False):
    # hm_data.shape >> (num_frame, y_size, x_size)
    if input_map:
        hm_data = results
    else:
        hm_data = np.array([hm for hm, _, _, _ in results])
    kernel = np.ones((1, window_size, window_size)) / (window_size * window_size)
    ma_hm_data = np.zeros_like(hm_data)

    for i in range(len(hm_data)):
        ma_hm_data[i] = cv2.filter2D(hm_data[i], -1, kernel[0])

    return ma_hm_data


def save_config(
    save_dir,
    results_base_dir_name,
    dir_name,
    place,
    crop_area,
    frame_range,    
    grid_size,
    vec_span,
    freq,
    func_para,
    space_ma,
    time_ma,
    space_window_size,
    time_window_size,
    dan_ver,
    distance_decay_method,
    div_avg,
    vec_decay_method,
    track_method,
    danger_thred_list,
    dense_thred_list,
):
    config = {
        "results_base_dir_name": results_base_dir_name,
        "dir_name": dir_name,
        "place": place,
        "crop_area": crop_area,
        "frame_range": list(frame_range),
        "grid_size": grid_size,
        "vec_span": vec_span,
        "freq": freq,
        "func_para": func_para,
        "space_ma": space_ma,
        "time_ma": time_ma,
        "space_window_size": space_window_size,
        "time_window_size": time_window_size,
        "dan_ver": dan_ver,
        "track_method": track_method,
        "distance_decay_method": distance_decay_method,
        "div_avg": div_avg,
        "vec_decay_method": vec_decay_method,
    }

    file_name = save_dir + "/config.yaml"
    with open(file_name, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print("Configration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Config saved to {file_name}\n")


def load_config(path2yaml):
    with open(path2yaml, "r") as file:
        config = yaml.safe_load(file)
    print(f"Config loaded from {path2yaml}")
    return config


def run_exp():
    parser = get_parser()
    args = parser.parse_args()

    if args.use_yaml:
        config = load_config(args.yaml_path)
        results_base_dir_name = config["results_base_dir_name"]
        dir_name = config["dir_name"]
        place = config["place"]
        crop_area = config["crop_area"]
        grid_size = config["grid_size"]
        vec_span = config["vec_span"]
        freq = config["freq"]
        func_para = config["func_para"]
        frame_range = config["frame_range"]
        dan_ver = config["dan_ver"]
        track_method = config["track_method"]
        vec_decay_method = config["vec_decay_method"]
        space_ma = config["space_ma"]
        time_ma = config["time_ma"]
        distance_decay_method = config["distance_decay_method"]
        div_avg = config["div_avg"]
        space_window_size = config["space_window_size"]
        time_window_size = config["time_window_size"]
        danger_thred_list = config["danger_thred_list"]
        dense_thred_list = config["dense_thred_list"]
        main(
            place=place,
            results_base_dir_name=results_base_dir_name,
            dir_name=dir_name,
            crop_area=crop_area,
            grid_size=grid_size,
            vec_span=vec_span,
            freq=freq,
            func_para=func_para,
            frame_range=frame_range,
            dan_ver=dan_ver,
            distance_decay_method=distance_decay_method,
            div_avg=div_avg,
            vec_decay_method=vec_decay_method,
            track_method=track_method,
            space_ma=space_ma,
            time_ma=time_ma,
            space_window_size=space_window_size,
            time_window_size=time_window_size,
            danger_thred_list=danger_thred_list,
            dense_thred_list=dense_thred_list,
        )

    else:
        func_para = [args.decay, args.max_x, args.clip_value]
        frame_range = (args.frame_start, args.frame_end)

        main(
            place=args.place,
            results_base_dir_name=args.results_base_dir_name,
            dir_name=args.dir_name,
            crop_area=args.crop_area,
            grid_size=args.grid_size,
            vec_span=args.vec_span,
            freq=args.freq,
            func_para=func_para,
            frame_range=frame_range,
            dan_ver=args.dan_ver,
            distance_decay_method=args.distance_decay_method,
            div_avg=args.div_avg,
            vec_decay_method=args.vec_decay_method,
            space_ma=args.space_ma,
            time_ma=args.time_ma,
            space_window_size=args.space_window_size,
            time_window_size=args.time_window_size,
            danger_thred_list=args.danger_thred_list,
            dense_thred_list=args.dense_thred_list,
        )



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_yaml", action="store_true", help="yamlファイルを使用するかどうか"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="danger/config/config.yaml",
        help="yamlファイルのパス",
    )

    parser.add_argument(
        "--place", type=str, default="WorldPorters_noon", help="場所の指定"
    )
    parser.add_argument(
        "--results_base_dir_name", type=str, default="results", help="結果保存ディレクトリ名"
    )
    parser.add_argument(
        "--dir_name", type=str, default="0228_exp", help="出力ディレクトリ名"
    )
    parser.add_argument("--grid_size", type=int, default=5, help="グリッドサイズ")
    parser.add_argument("--vec_span", type=int, default=10, help="ベクトル計算のスパン")
    parser.add_argument("--freq", type=int, default=10, help="危険度計算のフレーム間隔")
    parser.add_argument("--frame_start", type=int, default=1, help="開始フレーム")
    parser.add_argument("--frame_end", type=int, default=8990, help="終了フレーム")
    parser.add_argument(
        "--dan_ver", type=str, default="v2", help="危険度計算バージョン"
    )
    parser.add_argument("--distance_decay_method", type=str, default=None,choices=[None,"gaussian","liner"], help="距離減衰方法")
    parser.add_argument("--div_avg", action="store_true", help="発散の平均化")
    parser.add_argument("--vec_decay_method", type=str, default=None,choices=[None,"exp_func","clip"], help="ベクトル減衰方法")
    parser.add_argument("--space_ma", action="store_true", help="空間移動平均の適用")
    parser.add_argument("--time_ma", action="store_true", help="時間移動平均の適用")
    parser.add_argument(
        "--space_window_size",
        type=int,
        default=75,
        help="空間移動平均のウィンドウサイズ（※grid_sizeとの兼ね合いで決める）",
    )
    parser.add_argument(
        "--time_window_size",
        type=int,
        default=60,
        help="時間移動平均のウィンドウサイズ",
    )
    parser.add_argument(
        "--danger_thred_list",
        type=list,
        default=[5, 10, 15],
        help="危険度閾値リスト",
    )
    parser.add_argument(
        "--dense_thred_list",
        type=list,
        default=[75, 100, 125],
        help="密度閾値リスト",
    )

    # func_paraの設定
    parser.add_argument(
        "--decay", type=float, default=1.0, help="減衰パラメータ:使用しない"
    )
    parser.add_argument("--max_x", type=float, default=50.0, help="最大距離")
    parser.add_argument(
        "--clip_value", type=float, default=0.1, help="速度のクリップ値"
    )

    return parser


if __name__ == "__main__":
    run_exp()
