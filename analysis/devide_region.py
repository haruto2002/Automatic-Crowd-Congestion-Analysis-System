import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from matplotlib.patches import Polygon as mplPolygon


def devide_polygon_regions(bev_file, original_square, size_file, path2img):
    # ポリゴン領域を取得
    transformed_square, bev_img = bev_trans(
        bev_file, original_square, size_file, path2img
    )
    corners = transformed_square.reshape(-1, 2)
    poly = Polygon(corners)

    # 上下2:3に分割（下側が3/5、上側が2/5）
    bottom_part, top_part = split_polygon_horizontal_equal(poly, ratio=0.6)

    # 上側の領域を2等分、下側の領域を3等分
    top_regions = split_polygon_vertical_equal(top_part, 2)
    bottom_regions = split_polygon_vertical_equal(bottom_part, 3)

    # 全5つの領域を結合
    all_regions = top_regions + bottom_regions

    print(f"元ポリゴン面積: {poly.area:.2f}")
    for i, region in enumerate(all_regions):
        print(f"領域 {i+1}: 面積 {region.area:.2f} (比率: {region.area/poly.area:.3f})")

    return all_regions, bev_img


def bev_trans(path2bev_matrix, original_square, path2bev_size, path2img):
    matrix = np.loadtxt(path2bev_matrix)
    # original_squareの4つの角の座標を取得
    x1, y1, x2, y2 = original_square
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  # 左上  # 右上  # 右下  # 左下
        dtype=np.float32,
    )

    transformed_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), matrix
    ).reshape(-1, 2)

    # 変換後の座標を整数に変換
    transformed_square = transformed_corners.astype(int)

    # 画像を射影変換
    img = cv2.imread(path2img)
    img_w, img_h = np.loadtxt(path2bev_size).astype(int)
    bev_img = cv2.warpPerspective(
        img,
        matrix,
        (img_w, img_h),
        borderValue=(255, 255, 255),
    )

    return transformed_square, bev_img


def split_polygon_horizontal_equal(
    poly: Polygon, ratio: float = 0.5, tol: float = 1e-6, max_iter: int = 50
):
    """
    ポリゴン poly を水平線 y = cut で分割し、下側部分（y <= cut）が poly 面積 * ratio になるような cut を二分探索で探す。
    """
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("無効なポリゴンまたは面積 <= 0")
    total_area = poly.area
    target = total_area * ratio

    # bounding box 取得
    minx, miny, maxx, maxy = poly.bounds
    lo, hi = miny, maxy
    found_cut = None

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        half_poly = box(
            minx - (maxx - minx) * 10,
            miny - (maxy - miny) * 10,
            maxx + (maxx - minx) * 10,
            mid,
        )
        bottom = poly.intersection(half_poly)
        area_bot = bottom.area
        if area_bot < target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tol:
            found_cut = (lo + hi) / 2
            break
    else:
        found_cut = (lo + hi) / 2

    # 最終的な切断
    half_poly = box(
        minx - (maxx - minx) * 10,
        miny - (maxy - miny) * 10,
        maxx + (maxx - minx) * 10,
        found_cut,
    )
    bottom = poly.intersection(half_poly)
    top = poly.difference(bottom)
    return bottom, top


def split_polygon_vertical_equal(
    poly: Polygon, n: int, tol: float = 1e-6, max_iter: int = 50
):
    """
    ポリゴン poly を垂直線 x = cut で分割し、左側部分（x <= cut）が poly 面積 * 1/n になるような cut を二分探索で探す。
    """
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("無効なポリゴンまたは面積 <= 0")

    total_area = poly.area
    target = total_area / n
    regions = []
    remaining_poly = poly

    # bounding box 取得
    minx, miny, maxx, maxy = remaining_poly.bounds

    for i in range(n - 1):
        if remaining_poly.area <= 0:
            break

        lo, hi = minx, maxx
        found_cut = None

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            half_poly = box(
                minx - (maxx - minx) * 10,
                miny - (maxy - miny) * 10,
                mid,
                maxy + (maxy - miny) * 10,
            )
            left = remaining_poly.intersection(half_poly)
            area_left = left.area

            if area_left < target:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                found_cut = (lo + hi) / 2
                break
        else:
            found_cut = (lo + hi) / 2

        # 最終的な切断
        half_poly = box(
            minx - (maxx - minx) * 10,
            miny - (maxy - miny) * 10,
            found_cut,
            maxy + (maxy - miny) * 10,
        )
        left = remaining_poly.intersection(half_poly)
        right = remaining_poly.difference(left)

        if left.is_valid and left.area > 0:
            if isinstance(left, Polygon):
                regions.append(left)
            elif hasattr(left, "geoms"):
                for geom in left.geoms:
                    if geom.is_valid and geom.area > 0:
                        regions.append(geom)

        remaining_poly = right
        if remaining_poly.is_valid and remaining_poly.area > 0:
            minx, miny, maxx, maxy = remaining_poly.bounds

    # 最後の部分
    if remaining_poly.is_valid and remaining_poly.area > 0:
        if isinstance(remaining_poly, Polygon):
            regions.append(remaining_poly)
        elif hasattr(remaining_poly, "geoms"):
            for geom in remaining_poly.geoms:
                if geom.is_valid and geom.area > 0:
                    regions.append(geom)

    return regions


def visualize_regions_on_bev_image(bev_img, regions, save_dir):
    """
    BEV画像上に分割された領域を描画
    """
    # BGRからRGBに変換
    bev_img_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(15, 12))

    # BEV画像を表示
    ax.imshow(bev_img_rgb)

    # 全5つの領域を描画
    colors = ["red", "blue", "green", "orange", "purple"]

    for idx, region in enumerate(regions):
        if not region.is_valid or region.area <= 0:
            continue

        if isinstance(region, Polygon):
            xs, ys = region.exterior.xy
            patch = mplPolygon(
                list(zip(xs, ys)),
                facecolor=colors[idx % len(colors)],
                alpha=0.3,
                edgecolor=colors[idx % len(colors)],
                linewidth=2,
            )
            ax.add_patch(patch)

            # 領域番号を重心に表示
            centroid = region.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(idx + 1),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colors[idx % len(colors)],
                    alpha=0.8,
                ),
            )

    # 軸の設定
    ax.set_xlim(0, bev_img.shape[1])
    ax.set_ylim(bev_img.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")  # 軸を非表示

    plt.tight_layout()
    save_path = os.path.join(save_dir, "bev_image_with_regions.png")
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    print(f"BEV image with regions saved as {save_path}")
