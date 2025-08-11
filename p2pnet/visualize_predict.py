import cv2
import numpy as np
import pandas as pd
import os
import torch
import click
from omegaconf import OmegaConf
from utils.util_dnn import (
    fixed_r_seed,
    setup_device,
    suggest_network,
)
import glob
import albumentations as A
from natsort import natsorted
from dataset.dataset_utils import copy_datasets, suggest_dataset_root_dir
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def predict(
    cfg, img_path, gt_path, save_dir, model, weight_name, device, vis_conf=False
):
    # print(img_path)
    # if device_id in [0, 1, 2, 3]:
    #     device = f"cuda:{device_id}"
    # else:
    #     device = "cuda"

    # Load the images
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load ground truth points
    h, w, _ = img.shape
    points = np.loadtxt(gt_path)
    points = points[0 <= points[:, 0]]
    points = points[points[:, 0] <= w]
    points = points[0 <= points[:, 1]]
    points = points[points[:, 1] <= h]

    # Apply letter box
    if cfg.dataset.name in ["UCF-QNRF", "NWPU-Crowd"]:
        if cfg.dataset.name == "UCF-QNRF":
            max_size = 1408
        elif cfg.dataset.name == "NWPU-Crowd":
            max_size = 1920
        if max_size < max(img.shape[:3]):
            trainform_letterbox = A.Compose(
                A.LongestMaxSize(max_size=max_size, p=1.0),
                keypoint_params=A.KeypointParams(format="xy"),
            )
            transfromed_letter_box = trainform_letterbox(image=img, keypoints=points)
            img = transfromed_letter_box["image"]
            points = transfromed_letter_box["keypoints"]

    # Apply resize
    height, width, _ = img.shape
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    # transform
    transform = A.Compose(
        [
            A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
            A.Normalize(p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transformed = transform(image=img, keypoints=points)
    transformed_img = transformed["image"]
    gt_points = transformed["keypoints"]
    gt_cnt = len(gt_points)

    # save transformed img
    transform_for_save = A.Compose(
        [A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA)],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transformed_for_save = transform_for_save(image=img, keypoints=points)
    img_for_save = transformed_for_save["image"]
    img_for_save = cv2.cvtColor(img_for_save, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_dir + "/resized.png", img_for_save)

    sample = np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
    sample = torch.from_numpy(sample).clone()

    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    # run inference
    outputs = model(sample)
    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]

    outputs_points = outputs["pred_points"][0]

    if not vis_conf:

        threshold = 0.5
        # filter the predictions
        points = (
            outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        )
        confidence = (
            outputs_scores[outputs_scores > threshold].detach().cpu().numpy().tolist()
        )
        predict_cnt = int((outputs_scores > threshold).sum())

        # draw the predictions
        img_copy1 = img_for_save.copy()
        img_copy2 = img_for_save.copy()
        preedict_size = 3
        GT_size = 5
        for p in gt_points:
            cv2.circle(
                img_copy1,
                (int(p[0]), int(p[1])),
                GT_size,
                (0, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )

        for p in points:
            cv2.circle(
                img_copy1,
                (int(p[0]), int(p[1])),
                preedict_size,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                img_copy2,
                (int(p[0]), int(p[1])),
                preedict_size,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

        # save the visualized image
        cv2.imwrite(
            save_dir + f"/{weight_name}_pred{predict_cnt}_gt{gt_cnt}.png", img_copy1
        )
        cv2.imwrite(save_dir + f"/{weight_name}_pred{predict_cnt}.png", img_copy2)

    else:
        points = outputs_points.detach().cpu().numpy().tolist()
        confidence = outputs_scores.detach().cpu().numpy().tolist()

        points_and_confidence = [
            [points[i][0], points[i][1], confidence[i]] for i in range(len(points))
        ]
        df = pd.DataFrame(points_and_confidence, columns=["x", "y", "confidence"])
        plot_conf(
            df,
            img_for_save,
            min_threshold=0.0001,
            save_dir=save_dir,
            weight_name=weight_name,
        )


def plot_conf(df, img, min_threshold, save_dir, weight_name):
    df1 = df.query("confidence > 0.75")  # green
    df2 = df.query("0.75 >= confidence > 0.5")  # red
    df3 = df.query("0.5 >= confidence > 0.25")  # blue
    df4 = df.query(f"0.25 >= confidence > {min_threshold}")  # yellow
    size = 1
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    cnt_list = [0, 0, 0, 0]
    for i, df in enumerate([df1, df2, df3, df4]):
        df = df.reset_index()
        color = color_list[i]
        cnt_list[i] = len(df)
        for j in range(len(df)):
            cv2.circle(
                img,
                (int(df["x"][j]), int(df["y"][j])),
                size,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(
        save_dir
        + f"/{weight_name}_min_threshold_{min_threshold}_{cnt_list[0]}_{cnt_list[1]}_{cnt_list[2]}_{cnt_list[3]}.png",
        img,
    )


@click.command()
@click.argument("result_dir_path", type=str, default="vis_predicts")
@click.argument("weight_path", type=str, default="SHTechA.pth")
@click.argument("weight_name", type=str, default="SHTechA")
@click.argument("dataset_name", type=str, default="ShanghaiA")
@click.argument("resource", type=str, default="Tsukuba")
@click.argument("device_id", type=int, default=0)
def main(result_dir_path, weight_path, weight_name, dataset_name, resource, device_id):
    if torch.cuda.is_available():
        device = f"cuda:{device_id}"

    output_dir_path = result_dir_path + f"/{dataset_name}/"
    os.makedirs(output_dir_path, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dataset.name = dataset_name
    cfg.default.resource = resource
    # cfg.network.dnn_task = "counting_MIRU2024"
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    copy_datasets(cfg)
    root_dir = suggest_dataset_root_dir(cfg)
    test_dataset_dir_path = root_dir + "test/"

    for i, dir_path in enumerate(
        tqdm(natsorted(glob.glob(test_dataset_dir_path + "*")), leave=False)
    ):
        img_path = glob.glob(dir_path + "/*.jpg")[-1]
        anno_path = glob.glob(dir_path + "/*.txt")[-1]
        save_dir = dir_path.split("/")[-1]
        save_dir = output_dir_path + save_dir
        os.makedirs(save_dir, exist_ok=True)

        predict(
            cfg,
            img_path,
            anno_path,
            save_dir,
            model,
            weight_name,
            device,
            vis_conf=False,
        )


@click.command()
@click.argument("result_dir_path", type=str, default="vis_predicts")
@click.argument("weight_path", type=str, default="SHTechA.pth")
@click.argument("weight_name", type=str, default="SHTechA")
@click.argument("dataset_name", type=str, default="ShanghaiA")
@click.argument("resource", type=str, default="Tsukuba")
@click.argument("device_id", type=int, default=0)
def main_custom(
    result_dir_path, weight_path, weight_name, dataset_name, resource, device_id
):
    if torch.cuda.is_available():
        device = f"cuda:{device_id}"

    output_dir_path = result_dir_path + f"/{dataset_name}/"
    os.makedirs(output_dir_path, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dataset.name = dataset_name
    cfg.default.resource = resource
    # cfg.network.dnn_task = "counting_MIRU2024"
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    copy_datasets(cfg)
    root_dir = suggest_dataset_root_dir(cfg)
    test_dataset_dir_path = root_dir + "test/"

    for i, dir_path in enumerate(
        tqdm(natsorted(glob.glob(test_dataset_dir_path + "*")), leave=False)
    ):
        img_path = glob.glob(dir_path + "/*.jpg")[-1]
        anno_path = glob.glob(dir_path + "/*.txt")[-1]
        save_dir = dir_path.split("/")[-1]
        save_dir = output_dir_path + save_dir
        os.makedirs(save_dir, exist_ok=True)

        predict(
            cfg,
            img_path,
            anno_path,
            save_dir,
            model,
            weight_name,
            device,
            vis_conf=False,
        )


@click.command()
@click.argument("weight_path", type=str, default="weight_hanabi/yokohama2023_70.pth")
def test(weight_path):
    if torch.cuda.is_available():
        device = "cuda"

    output_dir_path = "test/"
    os.makedirs(output_dir_path, exist_ok=True)

    weight_name = weight_path.split("/")[-1][:-4]

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    test_set_dir_path = natsorted(glob.glob("test_set/*"))
    for i, dir_path in enumerate(tqdm(test_set_dir_path, leave=False)):
        img_path = glob.glob(dir_path + "/*.jpg")[-1]
        anno_path = glob.glob(dir_path + "/*.txt")[-1]
        save_dir = dir_path.split("/")[-1]
        save_dir = output_dir_path + save_dir
        os.makedirs(save_dir, exist_ok=True)

        predict(
            cfg,
            img_path,
            anno_path,
            save_dir,
            model,
            weight_name,
            device,
            vis_conf=False,
        )


if __name__ == "__main__":
    # main()
    # main_custom()
    test()
