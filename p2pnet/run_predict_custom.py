import cv2
import os
import torch
import pandas as pd
import numpy as np
from utils.util_dnn import suggest_network
import click
from omegaconf import OmegaConf
from tqdm import tqdm
import glob
from prediction.predict_utils import predict_points, predict_slide


def predict_main(save_dir, img_list, model, device):
    for path2img in tqdm(img_list):
        img_name = path2img.split("/")[-1][:-4]
        raw_img = cv2.imread(path2img)
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        if img.shape == (4320, 7680, 3) or img.shape == (4320, 3840, 3):
            # print("slide")
            h_size = 720
            w_size = 1280
            detections, scores = predict_slide(
                img, model, device, h_size=h_size, w_size=w_size
            )
        else:
            detections, scores = predict_points(img, model, device)

        save_plot(save_dir, raw_img, img_name, detections)
        save_csv(save_dir, img_name, detections, scores)


def save_plot(save_dir, img, img_name, detections):
    os.makedirs(save_dir + "/plot", exist_ok=True)
    img_copy = img.copy()
    p_size = 10
    # p_size = 3
    for p in detections:
        cv2.circle(
            img_copy,
            (int(p[0]), int(p[1])),
            p_size,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        save_dir + f"/plot/{img_name}_pred{len(detections)}.png",
        img_copy,
    )


def save_csv(save_dir, img_name, detections, scores):
    os.makedirs(save_dir + "/csv_data", exist_ok=True)
    if np.ndim(detections) == 2:
        scores = np.expand_dims(scores, 1)
    all_data = np.concatenate([detections, scores], 1)
    df = pd.DataFrame(all_data, columns=["x", "y", "score"])
    df.to_csv(save_dir + f"/csv_data/{img_name}.csv", index=False)


@click.command()
@click.argument("save_dir", type=str, default="vis/plot/WorldPorters_8K_sec_damage")
@click.argument(
    "weight_path",
    type=str,
    default="weights_hanabi/cutout.pth",
)
@click.argument(
    "img_dir",
    type=str,
    default="datasets/WorldPorters_8K_sec_damage",
)
def main(save_dir, weight_path, img_dir):
    if torch.cuda.is_available():
        device = "cuda"

    # os.makedirs(save_dir, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    img_list = glob.glob((f"{img_dir}/*.png"))
    # img_list = ["frames/image_0009.png"]

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    predict_main(save_dir, img_list, model, device)


if __name__ == "__main__":
    main()
