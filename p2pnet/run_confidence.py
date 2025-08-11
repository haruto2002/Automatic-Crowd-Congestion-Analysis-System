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


def predict_main(save_dir, img_list, model, device):
    for path2img in tqdm(img_list):
        img_name = path2img.split("/")[-1][:-4]
        raw_img = cv2.imread(path2img)
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        if img.shape == (4320, 7680, 3) or img.shape == (4320, 3840, 3):
            h_size = 720
            w_size = 1280
            img_raw_list = separate(raw_img, h_size, w_size)
            h_num = len(img_raw_list)
            w_num = len(img_raw_list[0])
            bar = tqdm(total=h_num * w_num, leave=False)
            for i in range(h_num):
                for j in range(w_num):
                    path_save_dir = save_dir + "/" + img_name + f"/{j}-{i}"
                    os.makedirs(path_save_dir, exist_ok=True)
                    pathch_img = img_raw_list[i][j]
                    detections, confidences = predict_points(pathch_img, model, device)
                    pathch_img_name = img_name + f"_{j}-{i}"
                    save_confidence(
                        path_save_dir, pathch_img_name, detections, confidences
                    )
                    save_detection(
                        path_save_dir,
                        pathch_img,
                        pathch_img_name,
                        detections,
                        confidences,
                    )
                    save_csv(path_save_dir, img_name, detections, confidences)
                    save_img(path_save_dir, img_name, pathch_img)
                    bar.update(1)
        else:
            detections, confidences = predict_points(img, model, device)
            save_confidence(save_dir, img_name, detections, confidences)


def predict_points(img, model, device):
    height, width, _ = img.shape
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    trans_data = height, width, new_height, new_width

    transformed_img = transform(img, trans_data)
    sample = np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
    sample = torch.from_numpy(sample).clone()
    sample = sample.unsqueeze(0)
    sample = sample.to(device)

    # run inference
    outputs = model(sample)
    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
    outputs_points = outputs["pred_points"][0]
    threshold = 0
    detect_scores = outputs_scores[outputs_scores > threshold].detach().cpu().numpy()
    detect_points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()

    detect_points = reconstruction(transformed_img, detect_points, trans_data)

    if not len(detect_scores) == len(detect_points):
        raise ValueError("The number of detect_score and detect_points are not same!")

    return detect_points, detect_scores


def save_detection(save_dir, img, img_name, detections, confidences):
    img_copy = img.copy()
    p_size = 5
    # p_size = 3
    for p in detections[confidences > 0.5]:
        cv2.circle(
            img_copy,
            (int(p[0]), int(p[1])),
            p_size,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        save_dir + f"/{img_name}_pred{len(detections)}.png",
        img_copy,
    )


def save_confidence(save_dir, img_name, detections, confidences):
    x = detections[:, 0]
    y = detections[:, 1]
    z = confidences
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    z_max = 1
    ax.set_zlim(0, z_max)
    plt.savefig(save_dir + f"/{img_name}_conf_max{z_max}.png")
    plt.close()


def save_csv(save_dir, img_name, detections, confidences):
    results = np.hstack((detections, confidences[:, np.newaxis]))
    df = pd.DataFrame(results, columns=["x", "y", "conf"])
    df.to_csv(save_dir + f"/{img_name}_data.csv", index=False)


def save_img(save_dir, img_name, img):
    cv2.imwrite(save_dir + f"/{img_name}_data.jpg", img)


@click.command()
@click.argument("save_dir", type=str, default="vis/plot")
@click.argument(
    "weight_path",
    type=str,
    default="weight_hanabi/SHTechB+2023.pth",
)
@click.argument("weight_name", type=str, default="hanabi")
@click.argument(
    "dataset_name",
    type=str,
    default="WorldPorters2023",
)
@click.argument("resource", type=str, default="local")
@click.argument("gpu_id", type=str, default=0)
def main(save_dir, weight_path, weight_name, dataset_name, resource, gpu_id):
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"

    save_dir = f"{save_dir}/{dataset_name}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)

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

    # 2023
    datasets_2023 = ["Kokusaibashi2023", "WorldPorters2023"]
    # 2024
    # datasets_2024 = [
    #     "WorldPorters2024_8K",
    #     "Kokusaibashi2024_4K-2",
    #     "WorldPorters2024_4K-2",
    #     "Chosha2024_4K-2",
    #     "Chosha2024_8K",
    #     "Chosha2024_4K-1",
    #     "Kishamichi2024_8K",
    #     "WorldPorters2024_4K-1",
    #     "Chosha2024_4K-3",
    #     "Kokusaibashi2024_4K-1",
    #     "Chuohiroba2024_4K",
    # ]

    if dataset_name in datasets_2023:
        copy_datasets(cfg)
        root_dir = suggest_dataset_root_dir(cfg)
        img_list = sorted(glob.glob(root_dir + "*.jpg"))
    # elif dataset_name in datasets_2024:
    #     copy_datasets_from_mac(cfg)
    #     root_dir = suggest_dataset_root_dir(cfg)
    #     img_list = sorted(glob.glob(root_dir + "*.jpg"))
    elif resource == "local":
        root_dir = suggest_dataset_root_dir(cfg)
        img_list = sorted(glob.glob(root_dir + "*.jpg"))
    # else:
    #     raise ValueError("add more code")

    predict_main(save_dir, img_list, model, device)


def main_custom(
    par_dir,
    weight_path,
    weight_name=None,
    dataset_name=None,
    resource=None,
    gpu_id=0,
):
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"

    # save_dir = f"{par_dir}/{dataset_name}/{weight_name}"
    save_dir = par_dir
    os.makedirs(save_dir, exist_ok=True)

    cfg_path = "p2pnet/conf/p2p.yaml"
    cfg = OmegaConf.load(cfg_path)
    # cfg.dataset.name = dataset_name
    # cfg.default.resource = resource
    cfg.default.finetune = True
    cfg.network.init_weight = weight_path

    model = suggest_network(cfg, device)
    model.to(device)
    model.eval()

    # img_list = glob.glob("tokyo_station/*.png")
    img_list = ["/homes/hnakayama/Fin_p2p/DSC_7130_2024-08-05_19:13:14_1.jpg"]

    predict_main(save_dir, img_list, model, device)


if __name__ == "__main__":
    # main()
    par_dir = "debug_conf"
    weight_path = "outputs_hanabi2023/500epochs/weights/check_points_epochs_0500.pth"
    # weight_path = "outputs_hanabi2023/new_loader/weights/check_points_epochs_0500.pth"
    main_custom(par_dir, weight_path)
