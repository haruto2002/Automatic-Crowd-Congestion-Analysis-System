import numpy as np
import cv2
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted
import glob


def display_map_with_img(
    press_map,
    vec_data,
    img,
    save_dir,
    name,
    grid_size,
    crop_area=None,
    exp=True,
):
    if exp:
        press_map[~np.isnan(press_map)] = np.exp2(press_map[~np.isnan(press_map)])
    # press_map[~np.isnan(press_map)] = np.exp(press_map[~np.isnan(press_map)])

    if crop_area is None:
        aspect_ratio = img.shape[0] / img.shape[1]  # h/w
        plt.figure(figsize=(20, int(20 * aspect_ratio)))
    else:
        aspect_ratio = (crop_area[3] - crop_area[1]) / (crop_area[2] - crop_area[0])
        plt.figure(figsize=(int(20 * aspect_ratio), 20))

    # RGB変換
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = rgb_img.shape

    # ヒートマップを画像サイズにリサイズ
    # 補完で滑らかに
    press_map_resized = cv2.resize(
        press_map,
        (img_width, img_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # 補完せずにグリッドサイズを順守
    # press_map_resized = np.zeros((img_height, img_width))
    # for i in range(press_map.shape[0]):
    #     for j in range(press_map.shape[1]):
    #         press_map_resized[
    #             i * grid_size : (i + 1) * grid_size,
    #             j * grid_size : (j + 1) * grid_size,
    #         ] = press_map[i, j]

    if crop_area is not None:
        vec_data = vec_data[
            (vec_data[:, 0, 0] > crop_area[0])
            & (vec_data[:, 0, 0] < crop_area[2])
            & (vec_data[:, 0, 1] > crop_area[1])
            & (vec_data[:, 0, 1] < crop_area[3])
        ]
        press_map_resized = press_map_resized[
            crop_area[1] : crop_area[3],
            crop_area[0] : crop_area[2],
        ]
        rgb_img = rgb_img[
            crop_area[1] : crop_area[3],
            crop_area[0] : crop_area[2],
        ]

    mask = np.isnan(press_map_resized)

    # 画像表示
    plt.imshow(rgb_img)

    # ヒートマップ表示
    min_score = 5
    sns.heatmap(
        press_map_resized,
        cmap="OrRd",
        cbar=False,
        annot=False,
        mask=mask,
        alpha=0.7,
        vmin=min_score,
    )

    # 矢印の描画
    for pos, vec in vec_data:
        vec = vec * 50
        if crop_area is None:
            start_point = (pos[0], pos[1])
        else:
            start_point = (pos[0] - crop_area[0], pos[1] - crop_area[1])
        if name[-2:] == "_x":
            vx = vec[0]
            vy = 0
        elif name[-2:] == "_y":
            vx = 0
            vy = vec[1]
        else:
            vx = vec[0]
            vy = vec[1]

        plt.arrow(
            start_point[0],
            start_point[1],
            vx,
            vy,
            head_width=3,
            head_length=3,
            fc="green",
            ec="green",
            linewidth=3,
        )

    # 保存
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_with_img.png")  # , dpi=300)
    plt.clf()
    plt.close()


def display_map_with_img_cv2(
    press_map,
    vec_data,
    img,
    save_dir,
    name,
    clip_value,
    crop_area=None,
    exp=True,
):
    if exp:
        press_map[~np.isnan(press_map)] = np.exp2(press_map[~np.isnan(press_map)])

    if clip_value is not None:
        vec_data[:, 1] = np.clip(vec_data[:, 1], -clip_value, clip_value)

    scale = 3
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img_height, img_width, _ = img.shape
    vec_data *= scale
    if crop_area:
        crop_area = [coor * scale for coor in crop_area]

    # ヒートマップをリサイズ
    press_map_resized = cv2.resize(
        press_map,
        (img_width, img_height),
        interpolation=cv2.INTER_LINEAR,
    )

    if crop_area:
        x1, y1, x2, y2 = crop_area
        vec_data = vec_data[
            (vec_data[:, 0, 0] > x1)
            & (vec_data[:, 0, 0] < x2)
            & (vec_data[:, 0, 1] > y1)
            & (vec_data[:, 0, 1] < y2)
        ]
        img = img[y1:y2, x1:x2]
        press_map_resized = press_map_resized[y1:y2, x1:x2]
        # print(press_map_resized.shape)
        # print(img.shape)

    # NaN を最小値に置き換え
    min_score = 5
    press_map_resized[press_map_resized < min_score] = np.nan
    press_map_resized[np.isnan(press_map_resized)] = min_score

    # ヒートマップを適用
    norm_map = cv2.normalize(press_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # 画像と合成
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 矢印の描画
    for pos, vec in vec_data:
        vec = vec * 40
        start_point = (int(pos[0]), int(pos[1]))
        if crop_area:
            start_point = (int(pos[0] - crop_area[0]), int(pos[1] - crop_area[1]))

        if name[-2:] == "_x":
            vx, vy = vec[0], 0
        elif name[-2:] == "_y":
            vx, vy = 0, vec[1]
        else:
            vx, vy = vec[0], vec[1]

        end_point = (int(start_point[0] + vx), int(start_point[1] + vy))
        cv2.arrowedLine(overlay, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_with_img.png")
    cv2.imwrite(save_path, overlay)


def display_map(press_map, save_dir, name):
    plt.figure(figsize=(10, 10))
    mask = np.isnan(press_map)
    sns.heatmap(
        press_map,
        cmap="OrRd",
        cbar=False,
        annot=True,
        mask=mask,
        alpha=0.7,
        annot_kws={"fontsize": 5},
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png")  # , dpi=300)
    plt.clf()
    plt.close()


def vis_hist(press_map, save_dir, name):
    # print(np.sum(press_map[~np.isnan(press_map)]))
    plt.figure()
    plt.hist(press_map.reshape(-1), color="skyblue", bins=100)
    # plt.xlim(-0.01, 0.01)
    # plt.xlabel("Pressure Values")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Pressure")
    plt.savefig(f"{save_dir}/{name}_hist.png")


def display_norm_parallel(
    results, save_dir, clip_value, grid_size, exp=False, crop_area=None, set_max_score=None,
):
    # 全体の最小値と最大値を取得
    if exp:
        danger_global_max, danger_global_min = get_global_min_max_with_exp(results)
    else:
        danger_global_max, danger_global_min = get_global_min_max(results)

    dense_global_max, dense_global_min = get_global_min_max(results, dense=True)

    pool_list = []
    for danger_map, dense_map, vec_list, img, name in results:
        inputs = (
            danger_map,
            dense_map,
            vec_list,
            img,
            name,
            grid_size,
            danger_global_min,
            danger_global_max,
            dense_global_max,
            dense_global_min,
            save_dir,
            clip_value,
            exp,
            crop_area,
            set_max_score,
        )
        pool_list.append(inputs)

    pool_size = os.cpu_count()
    with Pool(pool_size) as p:
        results = p.map(display_norm_cv2, pool_list)


def get_global_min_max(results, dense=False):
    max_values = [hm.max() for hm, _, _, _, _ in results]
    min_values = [hm.min() for hm, _, _, _, _ in results]

    if dense:
        max_values = [hm.max() for _, hm, _, _, _ in results]
        min_values = [hm.min() for _, hm, _, _, _ in results]

    # 降順ソートして上位 100 個を取得
    top = sorted(max_values, reverse=True)[:50]
    bottom = sorted(min_values, reverse=False)[:50]

    # 上位 100 個の平均を計算
    max_mean = np.mean(top)
    min_mean = np.mean(bottom)

    max_value = max_mean
    min_value = 0

    return max_value, min_value


def get_global_min_max_with_exp(results):
    max_values = [np.exp2(hm[~np.isnan(hm)]).max() for hm, _, _, _, _ in results]
    min_values = [np.exp2(hm[~np.isnan(hm)]).min() for hm, _, _, _, _ in results]
    # max_values = [np.exp(hm[~np.isnan(hm)]).max() for hm, _, _, _ in results]
    # min_values = [np.exp(hm[~np.isnan(hm)]).min() for hm, _, _, _ in results]

    # 降順ソートして上位 100 個を取得
    top = sorted(max_values, reverse=True)[:50]
    bottom = sorted(min_values, reverse=False)[:50]

    # 上位 100 個の平均を計算
    max_mean = np.mean(top)
    min_mean = np.mean(bottom)

    return max_mean, min_mean


def display_norm(inputs):
    (
        press_map,
        vec_list,
        img,
        name,
        global_min,
        global_max,
        save_dir,
        exp,
        crop_area,
    ) = inputs
    if exp:
        press_map[~np.isnan(press_map)] = np.exp2(press_map[~np.isnan(press_map)])

    if crop_area is None:
        aspect_ratio = img.shape[0] / img.shape[1]  # h/w
        plt.figure(figsize=(20, int(20 * aspect_ratio)))
    else:
        aspect_ratio = (crop_area[3] - crop_area[1]) / (crop_area[2] - crop_area[0])
        plt.figure(
            figsize=(
                20,
                int(20 * aspect_ratio),
            )
        )

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = rgb_img.shape

    # 計算していない地点を補完
    # press_map_resized = cv2.resize(
    #     press_map,
    #     (img_width, img_height),
    #     interpolation=cv2.INTER_LINEAR,
    # )

    # 計算地点をを遵守
    press_map_resized = np.zeros((img_height, img_width))
    for i in range(press_map.shape[0]):
        for j in range(press_map.shape[1]):
            press_map_resized[
                i * grid_size : (i + 1) * grid_size,
                j * grid_size : (j + 1) * grid_size,
            ] = press_map[i, j]

    if crop_area is not None:
        vec_data = vec_data[
            (vec_data[:, 0, 0] > crop_area[0])
            & (vec_data[:, 0, 0] < crop_area[2])
            & (vec_data[:, 0, 1] > crop_area[1])
            & (vec_data[:, 0, 1] < crop_area[3])
        ]
        press_map_resized = press_map_resized[
            crop_area[1] : crop_area[3],
            crop_area[0] : crop_area[2],
        ]
        rgb_img = rgb_img[
            crop_area[1] : crop_area[3],
            crop_area[0] : crop_area[2],
        ]

    plt.imshow(rgb_img)
    sns.heatmap(
        press_map_resized,
        cmap="OrRd",
        vmin=global_min,
        vmax=global_max,
        cbar=False,
        annot=False,
        alpha=0.7,
    )
    for pos, vec in vec_data:
        vec = vec * 40
        start_point = (pos[0] - crop_area[0], pos[1] - crop_area[1])
        plt.arrow(
            start_point[0],
            start_point[1],
            vec[0],
            vec[1],
            head_width=3,
            head_length=3,
            fc="green",
            ec="green",
            linewidth=3,
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png")
    plt.clf()
    plt.close()


def display_norm_cv2(inputs):
    (
        danger_map,
        dense_map,
        vec_data,
        img,
        name,
        grid_size,
        danger_global_min,
        danger_global_max,
        dense_global_max,
        dense_global_min,
        save_dir,
        clip_value,
        exp,
        crop_area,
        set_max_score,
    ) = inputs
    if exp:
        danger_map[~np.isnan(danger_map)] = np.exp2(danger_map[~np.isnan(danger_map)])

    if clip_value is not None:
        vec_data[:, 1] = np.clip(vec_data[:, 1], -clip_value, clip_value)

    scale = 3
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img_height, img_width, _ = img.shape
    vec_data *= scale
    if crop_area:
        crop_area = [coor * scale for coor in crop_area]

    # ヒートマップをリサイズ
    # danger_map_resized = cv2.resize(
    #     danger_map,
    #     (img_width, img_height),
    #     interpolation=cv2.INTER_LINEAR,
    # )
    danger_map_resized = np.zeros((img_height, img_width))
    for i in range(danger_map.shape[0]):
        for j in range(danger_map.shape[1]):
            danger_map_resized[
                i * grid_size * scale : (i + 1) * grid_size * scale,
                j * grid_size * scale : (j + 1) * grid_size * scale,
            ] = danger_map[i, j]
    dense_map_resized = np.zeros((img_height, img_width))
    for i in range(dense_map.shape[0]):
        for j in range(dense_map.shape[1]):
            dense_map_resized[
                i * grid_size * scale : (i + 1) * grid_size * scale,
                j * grid_size * scale : (j + 1) * grid_size * scale,
            ] = dense_map[i, j]

    # if crop_area:
    #     x1, y1, x2, y2 = crop_area
    #     vec_data = vec_data[
    #         (vec_data[:, 0, 0] > x1)
    #         & (vec_data[:, 0, 0] < x2)
    #         & (vec_data[:, 0, 1] > y1)
    #         & (vec_data[:, 0, 1] < y2)
    #     ]
    #     img = img[y1:y2, x1:x2]
    #     danger_map_resized = danger_map_resized[y1:y2, x1:x2]
    #     dense_map_resized = dense_map_resized[y1:y2, x1:x2]

    # 矢印の描画
    display_vec = True
    if display_vec:
        arrow_len = 3
        tipLength = 0.3
        arrow_scale = 40
        for pos, vec in vec_data:
            vec = vec * arrow_scale
            start_point = (int(pos[0]), int(pos[1]))
            # if crop_area:
            #     start_point = (int(pos[0] - x1), int(pos[1] - y1))
            end_point = (int(start_point[0] + vec[0]), int(start_point[1] + vec[1]))
            cv2.arrowedLine(
                img, start_point, end_point, (0, 255, 0), arrow_len, tipLength=tipLength
            )

    # set_score = True
    set_score = False
    if not set_score:
        # NaN を最小値に置き換え
        danger_map_resized[np.isnan(danger_map_resized)] = danger_global_min

        # global_min, global_max を用いた正規化
        danger_map_resized = np.clip(
            danger_map_resized, danger_global_min, danger_global_max
        )
        danger_norm_map = (
            (danger_map_resized - danger_global_min)
            / (danger_global_max - danger_global_min)
            * 255
        ).astype(np.uint8)
        danger_heatmap = cv2.applyColorMap(danger_norm_map, cv2.COLORMAP_JET)
    else:
        min_score = 0
        max_score = set_max_score
        # max_score = 30
        # NaN を最小値に置き換え
        danger_map_resized[np.isnan(danger_map_resized)] = min_score
        # min_score, max_score を用いた正規化
        danger_map_resized = np.clip(danger_map_resized, min_score, max_score)
        danger_norm_map = (
            (danger_map_resized - min_score) / (max_score - min_score) * 255
        ).astype(np.uint8)
        # heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_AUTUMN)
        danger_heatmap = cv2.applyColorMap(danger_norm_map, cv2.COLORMAP_JET)
        # vis_hist(heatmap, save_dir, name)
        # print(heatmap[0, 0])

    # 画像と合成
    cut_low = False
    mix = False
    if cut_low and mix:
        overlay_high = cv2.addWeighted(img, 0.3, danger_heatmap, 0.7, 0)
        overlay_low = cv2.addWeighted(img, 0.7, danger_heatmap, 0.3, 0)

        mask_low = np.all(danger_heatmap == np.array([0, 0, 255]), axis=-1)

        output = np.zeros_like(img)
        output[mask_low] = overlay_low[mask_low]
        output[~mask_low] = overlay_high[~mask_low]
    elif cut_low and not mix:
        overlay = cv2.addWeighted(img, 0.3, danger_heatmap, 0.7, 0)
        mask = np.any(danger_heatmap != np.array([0, 0, 255]), axis=-1)
        output = img.copy()
        output[mask] = overlay[mask]

    else:
        output = cv2.addWeighted(img, 0.6, danger_heatmap, 0.4, 0)
        # print("simple")

    # 保存
    danger_vis_save_dir = save_dir + "/danger_vis"
    danger_hm_save_dir = save_dir + "/danger_heatmap"
    os.makedirs(danger_vis_save_dir, exist_ok=True)
    os.makedirs(danger_hm_save_dir, exist_ok=True)
    danger_vis_save_path = os.path.join(danger_vis_save_dir, f"{name}.png")
    danger_hm_save_path = os.path.join(danger_hm_save_dir, f"{name}.png")
    cv2.imwrite(danger_vis_save_path, output)
    cv2.imwrite(danger_hm_save_path, danger_heatmap)

    dense_vis = True
    if dense_vis:
        # dence_map
        dense_map_resized[np.isnan(dense_map_resized)] = dense_global_min
        # global_min, global_max を用いた正規化
        dense_map_resized = np.clip(
            dense_map_resized, dense_global_min, dense_global_max
        )
        dense_norm_map = (
            (dense_map_resized - dense_global_min)
            / (dense_global_max - dense_global_min)
            * 255
        ).astype(np.uint8)
        dense_heatmap = cv2.applyColorMap(dense_norm_map, cv2.COLORMAP_JET)
        dense_output = cv2.addWeighted(img, 0.6, dense_heatmap, 0.4, 0)
        dense_vis_save_dir = save_dir + "/dense_vis"
        dense_hm_save_dir = save_dir + "/dense_heatmap"
        os.makedirs(dense_vis_save_dir, exist_ok=True)
        os.makedirs(dense_hm_save_dir, exist_ok=True)
        dense_vis_save_path = os.path.join(dense_vis_save_dir, f"{name}.png")
        dense_hm_save_path = os.path.join(dense_hm_save_dir, f"{name}.png")
        cv2.imwrite(dense_hm_save_path, dense_heatmap)
        cv2.imwrite(dense_vis_save_path, dense_output)


def create_movie(source_dir, save_dir, save_name):
    path2img_list = natsorted(glob.glob(f"{source_dir}/*.png"))
    # for path2img in path2img_list:
    #     print(path2img)
    img = cv2.imread(path2img_list[0])
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 5
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for i, path2img in enumerate(tqdm(path2img_list)):
        img = cv2.imread(path2img)
        frame = path2img_list[i].split("_")[-1][:-4]
        add_frame(img, frame)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()


def create_pick_movie(source_dir, start, end, save_dir, save_name):
    path2img_list = natsorted(glob.glob(f"{source_dir}/*.png"))
    path2img_list = path2img_list[start:end]
    img = cv2.imread(path2img_list[0])
    # frame = path2img_list[0].split("_")[-1][:-4]
    # add_frame(img, frame)
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 5
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for i, path2img in enumerate(tqdm(path2img_list)):
        img = cv2.imread(path2img)
        # frame = path2img_list[i].split("_")[-1][:-4]
        # add_frame(img, frame)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def add_frame(img, frame):
    h, w, c = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    color = (0, 0, 0)
    thickness = 2
    text_size, _ = cv2.getTextSize(frame, font, font_scale, thickness)
    text_x = w - text_size[0] - 10
    text_y = 20 + text_size[1]

    cv2.putText(img, frame, (text_x, text_y), font, font_scale, color, thickness)


if __name__ == "__main__":
    source_dir = "/homes/hnakayama/P2P/P2PNet/congestion/divergence/evaluations/12_19_fin/WorldPorters_noon_5_1_50_None/heat_map"
    save_dir = "/homes/hnakayama/P2P/P2PNet/congestion/divergence/evaluations/12_19_fin/WorldPorters_noon_5_1_50_None"
    save_name = "demo"
    create_movie(source_dir, save_dir, save_name)
