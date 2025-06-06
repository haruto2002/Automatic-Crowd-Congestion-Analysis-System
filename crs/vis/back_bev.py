import cv2
import numpy as np
import glob
import os
from multiprocessing import Pool


def main(save_dir, hm_source_dir, path2matrix="danger/bev/WorldPorters_8K/matrix.txt"):
    # save_dir = "danger/original_vis/WorldPorters_noon/heatmap"
    # hm_source_dir = "danger/results/0226_exp_all/WorldPorters_noon/1_8990/10_1_1_50_0.1_True_True_75_60/v2/each_result/heatmap"
    # path2mtrix = "danger/bev/WorldPorters_8K/matrix.txt"
    os.makedirs(save_dir)
    bev_matrix = np.loadtxt(path2matrix)
    matrix = np.linalg.inv(bev_matrix)
    path2hm_list = glob.glob(f"{hm_source_dir}/*.png")
    pool_list = []
    for path2hm in path2hm_list:
        input = [path2hm, matrix, save_dir]
        pool_list.append(input)

    pool_size = os.cpu_count()
    with Pool(pool_size) as p:
        p.map(parallel_run, pool_list)


def parallel_run(input):
    path2hm, matrix, save_dir = input
    back_trans(path2hm, matrix, save_dir)


def back_trans(path2hm, matrix, save_dir):
    hm = cv2.imread(path2hm)
    hm = cv2.resize(
        hm,
        (1500, 1100),
        interpolation=cv2.INTER_LINEAR,
    )

    back_hm = cv2.warpPerspective(hm, matrix, (7680, 4320), borderValue=(255, 255, 255))

    name = path2hm.split("/")[-1][-8:]

    cv2.imwrite(save_dir + "/" + name, back_hm)


if __name__ == "__main__":
    main()
    # path2hm_list = glob.glob(
    #     "danger/results/0219_all/WorldPorters_noon/1_8990/10_5_1_50_0.1/v2/each_result/heatmap/*.png"
    # )
    # for path2img in path2hm_list:
    #     end_frame = path2img.split("/")[-1][-8:-4]
    #     new_end_frame = int(end_frame) - 1
    #     new_file_path = path2img.replace(end_frame, f"{new_end_frame:04d}")
    #     command = f"mv {path2img} {new_file_path}"
    #     subprocess.run(command, shell=True)
