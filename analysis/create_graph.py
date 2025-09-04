import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from multiprocessing import Pool
from tqdm import tqdm


def extract_data_from_polygon_regions(source_dir, all_regions, smoothing):
    # 設定ファイルの読み込み
    path2cfg = os.path.join(source_dir, "config.yaml")
    with open(path2cfg, "r") as f:
        cfg = yaml.safe_load(f)
    grid_size = cfg["grid_size"]

    # Data directory configuration
    if smoothing:
        danger_source_dir = os.path.join(
            source_dir, "each_result", "danger_score_smoothed"
        )
        save_name = "polygon_regions_timeseries_smoothed.png"
    else:
        danger_source_dir = os.path.join(source_dir, "each_result", "danger_score")
        save_name = "polygon_regions_timeseries.png"

    vec_source_dir = os.path.join(source_dir, "each_result", "vec_data")
    danger_score_path_list = sorted(glob.glob(os.path.join(danger_source_dir, "*.txt")))
    vec_data_path_list = sorted(glob.glob(os.path.join(vec_source_dir, "*.txt")))

    assert len(danger_score_path_list) == len(vec_data_path_list)

    # Process data for each polygon region
    pool_list = []
    region_dict = {}

    for i, region in enumerate(all_regions):
        region_name = f"region_{i+1}"
        region_dict[region_name] = region

        for k in range(len(danger_score_path_list)):
            danger_score_path = danger_score_path_list[k]
            vec_data_path = vec_data_path_list[k]
            pool_list.append(
                [region, grid_size, danger_score_path, vec_data_path, region_name]
            )

    # Get data with parallel processing
    pool_size = os.cpu_count()
    with Pool(processes=pool_size) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(get_data_parallel_polygon, pool_list),
                total=len(pool_list),
            )
        )

    return results, save_name


def setting_data_for_graph(results):
    results_grouped_by_region = {}
    for result in results:
        region_name = result[0]
        if region_name not in results_grouped_by_region:
            results_grouped_by_region[region_name] = []
        results_grouped_by_region[region_name].append(result)

    # Sort data for each region
    results_for_graph = {}
    for region_name, result_list in results_grouped_by_region.items():
        sorted_result_list = sorted(result_list, key=lambda x: x[1])
        results_for_graph[region_name] = {
            "frame_list": [result[1] for result in sorted_result_list],
            "danger_score_list": [result[2] for result in sorted_result_list],
            "velocity_norm_list": [result[3] for result in sorted_result_list],
            "num_people_list": [result[4] for result in sorted_result_list],
        }

    return results_for_graph


def get_data_parallel_polygon(input_list):
    region, grid_size, danger_score_path, vec_data_path, region_name = input_list
    frame, danger_score, velocity_norm, num_people = get_single_data_polygon(
        region, grid_size, danger_score_path, vec_data_path
    )
    return region_name, frame, danger_score, velocity_norm, num_people


def get_single_data_polygon(region, grid_size, danger_score_path, vec_data_path):
    # 危険度スコアの取得
    danger_score = get_polygon_map_data(
        np.loadtxt(danger_score_path), region, grid_size
    )

    # Get vector data
    velocity_norm, num_people = get_polygon_vec_data(np.loadtxt(vec_data_path), region)

    # フレーム番号の取得
    frame = int(os.path.basename(danger_score_path).split(".")[0].split("_")[-1])

    return frame, danger_score, velocity_norm, num_people


def get_polygon_map_data(map_data, region, grid_size):
    """
    ポリゴン領域内の危険度スコアの平均を計算

    処理手順:
    1. ポリゴンの境界ボックスを取得
    2. grid_sizeで座標を正規化（ピクセル座標に変換）
    3. 画像座標系に変換（Y軸反転）
    4. Limit to boundaries to ensure valid range
    5. ポリゴン内の各ピクセルをチェック
    6. Calculate average of data within polygon
    """
    # 1. ポリゴンの境界ボックスを取得
    minx, miny, maxx, maxy = region.bounds

    # 2. grid_sizeで座標を正規化（ピクセル座標に変換）
    minx = minx / grid_size
    maxx = maxx / grid_size
    miny = miny / grid_size
    maxy = maxy / grid_size

    # 3. 画像座標系に変換（Y軸反転）
    img_height = map_data.shape[0]
    minx_img = int(minx)
    maxx_img = int(maxx)
    miny_img = int(img_height - maxy)  # Y軸反転
    maxy_img = int(img_height - miny)  # Y軸反転

    # 4. Limit to boundaries to ensure valid range
    minx_img = max(0, minx_img)
    maxx_img = min(map_data.shape[1], maxx_img)
    miny_img = max(0, miny_img)
    maxy_img = min(map_data.shape[0], maxy_img)

    # 5. ポリゴン内の各ピクセルをチェック
    points_in_polygon = []
    if minx_img < maxx_img and miny_img < maxy_img:
        for y in range(miny_img, maxy_img):
            for x in range(minx_img, maxx_img):
                # Convert from image coordinate system to polygon coordinate system
                poly_x = x * grid_size
                poly_y = (img_height - y) * grid_size

                # ポリゴン内かどうかをチェック
                if region.contains(Point(poly_x, poly_y)):
                    points_in_polygon.append(map_data[y, x])

    # 6. Calculate average of data within polygon
    if len(points_in_polygon) > 0:
        average_danger_score = np.mean(np.clip(points_in_polygon, 0, None))
    else:
        average_danger_score = 0.0

    return average_danger_score


def get_polygon_vec_data(vec_data, region):
    """
    Process vector data within polygon region
    """
    # Filter points within polygon
    points_in_region = []

    for point in vec_data:
        x, y = point[0], point[1]
        # 画像座標系に変換（Y軸反転）
        img_height = 1100  # 画像の高さ
        y_img = img_height - y

        # ポリゴン内かどうかをチェック
        if region.contains(Point(x, y_img)):
            points_in_region.append(point)

    if len(points_in_region) > 0:
        points_in_region = np.array(points_in_region)
        # 速度のノルムを計算
        velocity_norm_list = np.linalg.norm(points_in_region[:, 2:4], axis=1)
        average_velocity_norm = np.mean(velocity_norm_list)
        num_people = len(points_in_region)
    else:
        average_velocity_norm = 0.0
        num_people = 0

    return average_velocity_norm, num_people


def plot_polygon_regions_graph(data, save_dir, save_name):
    """
    Create time series graphs for each polygon region
    """
    num_regions = len(data)
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    axs = axs.flatten()

    # Get overall maximum and minimum values
    max_danger_score = max(
        max(each_region_data["danger_score_list"]) for each_region_data in data.values()
    )
    min_danger_score = min(
        min(each_region_data["danger_score_list"]) for each_region_data in data.values()
    )
    max_velocity_norm = max(
        max(each_region_data["velocity_norm_list"])
        for each_region_data in data.values()
    )
    min_velocity_norm = min(
        min(each_region_data["velocity_norm_list"])
        for each_region_data in data.values()
    )
    max_num_people = max(
        max(each_region_data["num_people_list"]) for each_region_data in data.values()
    )
    min_num_people = min(
        min(each_region_data["num_people_list"]) for each_region_data in data.values()
    )

    labels = [
        "Region 1",
        "Region 2",
        "Region 3",
        "Region 4",
        "Region 5",
    ]

    for i, (region_name, data_dict) in enumerate(data.items()):
        if i >= len(axs):
            break

        ax = axs[i]

        # Create three y-axes
        ax.set_xlabel("Frame")
        ax.set_ylabel("Danger Score", color="red")
        ax.tick_params(axis="y", labelcolor="red")

        # Second y-axis (for velocity)
        ax2 = ax.twinx()
        ax2.set_ylabel("Average Velocity", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        # Third y-axis (for number of people)
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.set_ylabel("Number of People", color="green")
        ax3.tick_params(axis="y", labelcolor="green")

        # Calculate and plot moving average
        window_size = 30
        if len(data_dict["frame_list"]) >= window_size:
            # Plot original data with alpha=0.3
            ax.plot(
                data_dict["frame_list"],
                data_dict["danger_score_list"],
                color="red",
                alpha=0.3,
                linewidth=1,
            )
            ax2.plot(
                data_dict["frame_list"],
                data_dict["velocity_norm_list"],
                color="blue",
                alpha=0.3,
                linewidth=1,
            )
            ax3.plot(
                data_dict["frame_list"],
                data_dict["num_people_list"],
                color="green",
                alpha=0.3,
                linewidth=1,
            )

            # Moving average for Danger Score
            danger_ma = np.convolve(
                data_dict["danger_score_list"],
                np.ones(window_size) / window_size,
                mode="same",
            )
            danger_line = ax.plot(
                data_dict["frame_list"][window_size // 2 : -window_size // 2],
                danger_ma[window_size // 2 : -window_size // 2],
                label="Danger Score (MA)",
                color="red",
                linewidth=2,
            )

            # Moving average for Velocity
            velocity_ma = np.convolve(
                data_dict["velocity_norm_list"],
                np.ones(window_size) / window_size,
                mode="same",
            )
            velocity_line = ax2.plot(
                data_dict["frame_list"][window_size // 2 : -window_size // 2],
                velocity_ma[window_size // 2 : -window_size // 2],
                label="Average Velocity (MA)",
                color="blue",
                linewidth=2,
            )

            # Moving average for Number of People
            people_ma = np.convolve(
                data_dict["num_people_list"],
                np.ones(window_size) / window_size,
                mode="same",
            )
            people_line = ax3.plot(
                data_dict["frame_list"][window_size // 2 : -window_size // 2],
                people_ma[window_size // 2 : -window_size // 2],
                label="Number of People (MA)",
                color="green",
                linewidth=2,
            )
        else:
            # Display original data if insufficient data
            danger_line = ax.plot(
                data_dict["frame_list"],
                data_dict["danger_score_list"],
                label="Danger Score",
                color="red",
                linewidth=2,
            )
            velocity_line = ax2.plot(
                data_dict["frame_list"],
                data_dict["velocity_norm_list"],
                label="Average Velocity",
                color="blue",
                linewidth=2,
            )
            people_line = ax3.plot(
                data_dict["frame_list"],
                data_dict["num_people_list"],
                label="Number of People",
                color="green",
                linewidth=2,
            )

        ax.set_title(f"{labels[i] if i < len(labels) else region_name}")
        ax.grid(True, alpha=0.3)

        # Add legend
        lines = danger_line + velocity_line + people_line
        labels_legend = ["Danger Score", "Average Velocity", "Number of People"]
        ax.legend(lines, labels_legend, loc="upper right")

        ax.set_ylim(min_danger_score, max_danger_score)
        ax2.set_ylim(min_velocity_norm, max_velocity_norm)
        ax3.set_ylim(min_num_people, max_num_people)

    # Hide unused subplots
    for i in range(num_regions, len(axs)):
        axs[i].set_visible(False)

    # Save the graph
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"Graph saved as '{save_path}'.")
