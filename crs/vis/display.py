import numpy as np
import cv2
import os
from multiprocessing import Pool
import glob
from tqdm import tqdm
from natsort import natsorted
import argparse


def display_parallel(back_img_dir, heatmap_dir, save_dir):
    path2back_list = natsorted(glob.glob(back_img_dir + "/*.png"))
    path2hm_list = sorted(glob.glob(heatmap_dir + "/*.png"))

    pool_list = []
    for path2back in path2back_list:
        back_frame = path2back.split("/")[-1][-8:-4]
        path2hm = next(
            (
                path2hm
                for path2hm in path2hm_list
                if int(path2hm.split("/")[-1][-8:-4]) == int(back_frame)
            ),
            None,
        )
        if path2hm is None:
            continue
        inputs = (
            path2back,
            path2hm,
            save_dir,
            f"{int(back_frame):04d}",
        )
        pool_list.append(inputs)

    pool_size = os.cpu_count()
    with Pool(pool_size) as p:
        results = p.map(display_frame, pool_list)
    return results


def display_frame(inputs):
    (path2back, path2hm, save_dir, name) = inputs
    back_img = cv2.imread(path2back)
    heatmap = cv2.imread(path2hm)

    # 画像と合成
    cut_low = False
    mix = False
    if cut_low and mix:
        overlay_high = cv2.addWeighted(back_img, 0.3, heatmap, 0.7, 0)
        overlay_low = cv2.addWeighted(back_img, 0.7, heatmap, 0.3, 0)

        mask_low = np.all(heatmap == np.array([0, 0, 255]), axis=-1)

        output = np.zeros_like(back_img)
        output[mask_low] = overlay_low[mask_low]
        output[~mask_low] = overlay_high[~mask_low]
    elif cut_low and not mix:
        overlay = cv2.addWeighted(back_img, 0.3, heatmap, 0.7, 0)
        mask = np.any(heatmap != np.array([0, 0, 255]), axis=-1)
        output = back_img.copy()
        output[mask] = overlay[mask]

    else:
        output = cv2.addWeighted(back_img, 0.6, heatmap, 0.4, 0)
        # print("simple")

    # 保存
    vis_save_path = os.path.join(save_dir, f"{name}.png")
    cv2.imwrite(vis_save_path, output)
    return output


def create_movie(img_list, save_dir, save_name, fps=10):
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img in tqdm(img_list):
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def main_on_map():
    back_img_dir = "danger/track_vis/WorldPorters_noon/frames_with_map"
    heatmap_dir = "danger/results/0219/WorldPorters_noon/0_3000/10_5_1_50_0.1/v2_2/each_result/heatmap"
    save_dir = "danger/custom_vis/WorldPorters_noon_crop_0_3000_div_v2_2"
    frame_save_dir = save_dir + "/frames"
    os.makedirs(frame_save_dir)
    output_list = display_parallel(back_img_dir, heatmap_dir, frame_save_dir)
    create_movie(output_list, save_dir, "danger_on_map")


def main_original_view():
    back_img_dir = "tracking_results/WorldPorters_noon/img"
    heatmap_dir = "danger/original_vis/WorldPorters_noon/heatmap"
    save_dir = "danger/original_vis/WorldPorters_noon"
    frame_save_dir = save_dir + "/vis"
    os.makedirs(frame_save_dir)
    output_list = display_parallel(back_img_dir, heatmap_dir, frame_save_dir)
    create_movie(output_list, save_dir, "danger_original_view")


def main_on_vec():
    back_img_dir = "danger/vec_vis/WorldPorters_noon/frames_with_map"
    heatmap_dir = "danger/results/0219/WorldPorters_noon/0_3000/10_5_1_50_0.1/v2/each_result/heatmap"
    save_dir = "danger/vec_vis/WorldPorters_noon"
    frame_save_dir = save_dir + "/frames_with_danger"
    os.makedirs(frame_save_dir)
    output_list = display_parallel(back_img_dir, heatmap_dir, frame_save_dir)
    create_movie(output_list, save_dir, "danger_on_map")


def only_movie():
    img_dir = "danger/original_vis/WorldPorters_noon/vis"
    save_dir = "danger/original_vis/WorldPorters_noon"
    pool_size = int(os.cpu_count())
    freq = 1
    path2img_list = sorted(glob.glob(img_dir + "/*.png"))[::freq]
    print("Reading images")
    with Pool(pool_size) as p:
        img_list = p.map(parallel_read, path2img_list)
    print("Creating movie")
    create_movie(img_list, save_dir, "danger_original_view_faster", fps=15)


def parallel_read(path2img):
    img = cv2.imread(path2img)
    # frame = path2img.split("/")[-1][-8:-4]
    return img


def display_hm_on_back_img(save_dir, back_img_dir, heatmap_dir, save_mov_name, fps=15):
    frame_save_dir = save_dir + "/frames"
    os.makedirs(frame_save_dir)
    output_list = display_parallel(back_img_dir, heatmap_dir, frame_save_dir)
    create_movie(output_list, save_dir, save_mov_name, fps)


if __name__ == "__main__":
    # main_original_view()
    main_on_vec()
    # only_movie()
