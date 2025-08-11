import cv2
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.util_dnn import suggest_network
import click
from omegaconf import OmegaConf
from dataset.dataset_utils import copy_datasets, suggest_dataset_root_dir
from tqdm import tqdm
import glob
from prediction.predict_utils import (
    transform,
    reconstruction,
    separate,
)
from matplotlib.ticker import MaxNLocator


def main_conf_map():
    save_dir = "demo"
    os.makedirs(save_dir, exist_ok=True)
    # data_dir = "debug_conf/DSC_7136_2024-08-05_19:45:33_3/2-2"
    data_dir = "debug_conf/DSC_7130_2024-08-05_19:13:14_1/3-3"
    path2img = glob.glob(data_dir + "/*.jpg")[0]
    path2csv = glob.glob(data_dir + "/*.csv")[0]
    img = cv2.imread(path2img)
    df = pd.read_csv(path2csv)
    crop_pos = (850, 350)
    crop_size = 256
    cropped_img = img[
        crop_pos[1] : crop_pos[1] + crop_size, crop_pos[0] : crop_pos[0] + crop_size, :
    ]
    print(cropped_img.shape)
    print(
        f"{crop_pos[1]} : {crop_pos[1] + crop_size}, {crop_pos[0]} : {crop_pos[0] + crop_size}"
    )
    cropped_df = df[
        (df["x"] > crop_pos[0])
        & (df["x"] < crop_pos[0] + crop_size)
        & (df["y"] > crop_pos[1])
        & (df["y"] < crop_pos[1] + crop_size)
    ]
    cropped_df.loc[:, "x"] -= crop_pos[0]
    cropped_df.loc[:, "y"] -= crop_pos[1]

    # 3D
    # x = cropped_df["x"].to_list()
    # y = cropped_df["y"].to_list()
    # z = cropped_df["conf"].to_list()
    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    # ax.scatter(x, y, z, s=1)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # z_max = 0.1
    # ax.set_zlim(0, z_max)
    # plt.savefig(save_dir + f"/conf_max{z_max}.png")
    # plt.close()

    # Color map
    z_max = 1
    map_df = cropped_df[cropped_df["conf"] < z_max]
    x = map_df["x"].to_list()
    y = map_df["y"].to_list()
    z = map_df["conf"].to_list()
    fig, ax = plt.subplots(figsize=(12, 12))
    # GT
    df_gt = pd.read_csv("demo/raw_img.csv")
    x_gt = df_gt["x"].to_list()
    y_gt = df_gt["y"].to_list()
    ax.scatter(x_gt, y_gt, color="g", alpha=0.5, s=7)
    ax.scatter(x, y, c=z, cmap="Reds", s=1)
    # sc = ax.scatter(x, y, c=z, cmap="Reds", s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, crop_size)
    ax.set_ylim(crop_size, 0)
    # cbar = plt.colorbar(sc)
    # cbar.set_label("Confidence")
    plt.savefig(save_dir + f"/confmap_max{z_max}_2d.png")
    plt.close()

    cv2.imwrite(save_dir + f"/raw_img.png", cropped_img)

    img_copy = cropped_img.copy()
    p_size = 5
    # p_size = 3
    df_positive = cropped_df[cropped_df["conf"] > 0.5]
    for i in range(len(df_positive)):
        cv2.circle(
            img_copy,
            (int(df_positive.iloc[i, 0]), int(df_positive.iloc[i, 1])),
            p_size,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        save_dir + f"/pred{len(df_positive)}.png",
        img_copy,
    )


def get_data(save_dir, data_dir):
    path2img = glob.glob(data_dir + "/*.jpg")[0]
    path2csv = glob.glob(data_dir + "/*.csv")[0]
    img = cv2.imread(path2img)
    df = pd.read_csv(path2csv)
    crop_pos = (850, 350)
    crop_size = 256
    cropped_img = img[
        crop_pos[1] : crop_pos[1] + crop_size, crop_pos[0] : crop_pos[0] + crop_size, :
    ]
    print(cropped_img.shape)
    print(
        f"{crop_pos[1]} : {crop_pos[1] + crop_size}, {crop_pos[0]} : {crop_pos[0] + crop_size}"
    )
    cv2.imwrite(save_dir + "/raw_img.png", cropped_img)
    cropped_df = df[
        (df["x"] > crop_pos[0])
        & (df["x"] < crop_pos[0] + crop_size)
        & (df["y"] > crop_pos[1])
        & (df["y"] < crop_pos[1] + crop_size)
    ]
    cropped_df.loc[:, "x"] -= crop_pos[0]
    cropped_df.loc[:, "y"] -= crop_pos[1]
    cropped_df.to_csv(save_dir + "/data.csv")

    img_copy = cropped_img.copy()
    p_size = 5
    df_positive = cropped_df[cropped_df["conf"] > 0.5]
    for i in range(len(df_positive)):
        cv2.circle(
            img_copy,
            (int(df_positive.iloc[i, 0]), int(df_positive.iloc[i, 1])),
            p_size,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        save_dir + f"/pred{len(df_positive)}.png",
        img_copy,
    )


def create_distrubution_hist(save_dir, data_dir):
    # load data
    path2det = data_dir + "/data.csv"
    path2gt = data_dir + "/GT.csv"
    path2img = data_dir + "/raw_img.png"
    df_det = pd.read_csv(path2det)
    df_gt = pd.read_csv(path2gt)
    img = cv2.imread(path2img)
    h, w, c = img.shape

    # set_data
    radius = 10
    for i in range(len(df_gt)):
        gt_coor = (df_gt.iloc[i, 0], df_gt.iloc[i, 1])
        picked_df = pick_around_data(gt_coor, df_det, radius, h, w)
        conf = picked_df["conf"].to_list()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(conf)
        ax.set_title(f"GT ID:{i}")
        ax.set_xlabel("confidence", fontsize=20)
        ax.set_ylabel("freq", fontsize=20)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tick_params(labelsize=15)
        # ax.set_xlim(0, 1)
        plt.savefig(save_dir + f"/hist_gt{i}.png")
        plt.close()


def create_distrubution_map(save_dir, data_dir):
    # load data
    path2det = data_dir + "/data.csv"
    path2gt = data_dir + "/GT.csv"
    path2img = data_dir + "/raw_img.png"
    df_det = pd.read_csv(path2det)
    df_gt = pd.read_csv(path2gt)
    img = cv2.imread(path2img)
    h, w, c = img.shape

    # set_data
    radius = 10
    for i in range(len(df_gt)):
        gt_coor = (df_gt.iloc[i, 0].item(), df_gt.iloc[i, 1].item())
        picked_df = pick_around_data(gt_coor, df_det, radius)
        x = picked_df["x"].to_list()
        y = picked_df["y"].to_list()
        z = picked_df["conf"].to_list()
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(gt_coor[0], gt_coor[1], color="g", alpha=0.5, s=20)
        ax.scatter(x, y, c=z, cmap="Reds", s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.set_xlim(gt_coor[0] - radius, gt_coor[0] + radius)
        # ax.set_ylim(gt_coor[1] - radius, gt_coor[1] + radius)
        # cbar = plt.colorbar(sc)
        # cbar.set_label("Confidence")
        plt.savefig(save_dir + f"/confmap_gt{i}.png")
        plt.close()


def create_distrubution_map_with_img(save_dir, data_dir):
    # load data
    path2det = data_dir + "/data.csv"
    path2gt = data_dir + "/GT.csv"
    path2img = data_dir + "/raw_img.png"
    df_det = pd.read_csv(path2det)
    df_gt = pd.read_csv(path2gt)
    radius = 10

    fig, ax = plt.subplots(figsize=(12, 12))
    img = cv2.imread(path2img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    ax.imshow(img)

    for i in range(len(df_gt)):
        gt_coor = (df_gt.iloc[i, 0].item(), df_gt.iloc[i, 1].item())
        picked_df = pick_around_data(
            gt_coor,
            df_det,
            radius,
            h,
            w,
        )
        sorted_picked_df = picked_df.sort_values("conf")
        x = sorted_picked_df["x"].to_list()
        y = sorted_picked_df["y"].to_list()
        z = sorted_picked_df["conf"].to_list()
        ax.scatter(x, y, c=z, cmap="Reds", s=70)
        ax.scatter(gt_coor[0], gt_coor[1], color="g", alpha=0.9, s=100)
        ax.axis("off")
    plt.savefig(save_dir + "/confmap_with_gt.png")
    plt.close()


def pick_around_data(
    gt_coor,
    df_det,
    radius,
    h,
    w,
):
    picked_df = df_det[
        (df_det["x"] > max(0, gt_coor[0] - radius))
        & (df_det["x"] < min(w, gt_coor[0] + radius))
        & (df_det["y"] > max(0, gt_coor[1] - radius))
        & (df_det["y"] < min(h, gt_coor[1] + radius))
    ].copy()

    return picked_df


def save_gt_ID(save_dir, data_dir):
    path2gt = data_dir + "/GT.csv"
    path2img = data_dir + "/raw_img.png"
    df_gt = pd.read_csv(path2gt)
    img = cv2.imread(path2img)
    img_with_id = img.copy()

    for i in range(len(df_gt)):
        gt_coor = (int(df_gt.iloc[i, 0].item()), int(df_gt.iloc[i, 1].item()))
        img_with_id = cv2.putText(
            img, str(i), gt_coor, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
        )
    cv2.imwrite(save_dir + "/GT_ID.png", img_with_id)


def set_data():
    data_dir = "debug_conf/DSC_7130_2024-08-05_19:13:14_1/3-3"
    save_dir = "vis_conf/" + data_dir.split("/")[-1]
    os.makedirs(save_dir, exist_ok=True)
    get_data(save_dir, data_dir)


def main_vis():
    data_dir = "vis_conf/3-3"
    save_dir = data_dir + "/vis"
    os.makedirs(save_dir, exist_ok=True)
    create_distrubution_hist(save_dir, data_dir)
    # create_distrubution_map(save_dir, data_dir)
    # create_distrubution_map_with_img(save_dir, data_dir)


def main_gt_id():
    data_dir = "vis_conf/3-3"
    save_dir = data_dir
    save_gt_ID(save_dir, data_dir)


if __name__ == "__main__":
    # set_data()
    main_vis()
    # main_gt_id()
