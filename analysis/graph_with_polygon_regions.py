import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from shapely.geometry import Polygon
from devide_region import devide_polygon_regions, visualize_regions_on_bev_image
from create_graph import (
    extract_data_from_polygon_regions,
    setting_data_for_graph,
    plot_polygon_regions_graph,
)


def main():
    save_dir = "demo_graph_polygon"
    os.makedirs(save_dir, exist_ok=True)
    source_dir = "demo_5min_8K/crowd_risk_score/debug"
    smoothing = True

    # BEV変換の設定
    bev_file = "crs/bev/WorldPorters_8K_matrix.txt"
    size_file = "crs/size/WorldPorters_8K_size.txt"
    # path2img = "demo/img/0001.png"
    path2img = "demo_5min_8K/img/0001.png"
    original_square = [0, 0, 7680, 4320]

    # ポリゴン領域の分割
    print("Deviding polygon regions...")
    all_regions, bev_img = devide_polygon_regions(
        bev_file, original_square, size_file, path2img
    )

    # BEV画像上に領域を可視化
    print("Visualizing regions on BEV image...")
    visualize_regions_on_bev_image(bev_img, all_regions, save_dir)

    # Data extraction
    print("Extracting data from polygon regions...")
    results, save_name = extract_data_from_polygon_regions(
        source_dir, all_regions, smoothing
    )
    # 結果を整理
    print("Setting data for graph...")
    results_for_graph = setting_data_for_graph(results)
    # グラフを作成
    print("Plotting polygon regions graph...")
    plot_polygon_regions_graph(results_for_graph, save_dir, save_name)


if __name__ == "__main__":
    main()
