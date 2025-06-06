import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import cv2
import glob
from tqdm import tqdm
import os
from natsort import natsorted
import datetime
import click

# from multiprocessing import Pool

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)


def add_time(img, df, frame):
    text = df["time"][frame].strftime("%H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)

    if img.shape == (2160, 3840, 3):
        font_scale = 4
        font_thickness = 5
        position = (3200, 150)  # テキストの位置
        x, y = position
    elif img.shape == (4320, 7680, 3):
        font_scale = 7
        font_thickness = 10
        position = (6500, 300)  # テキストの位置
        x, y = position

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )

    # White background
    cv2.rectangle(
        img,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        thickness=cv2.FILLED,
    )

    cv2.putText(
        img, text, position, font, font_scale, text_color, thickness=font_thickness
    )

    return img


def save_frame(save_dir, frame, fig, ax1, ax2, path2img, df, dataset_name):
    # ax1 >> Images of inference results
    img = cv2.imread(path2img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_time(img, df, frame)

    # if img.shape != (2160, 3840, 3):
    #     print(img.shape)
    #     raise ValueError("img size is not correct")

    ax1.clear()
    ax1.imshow(img)
    ax1.axis("off")
    # ax1.set_title(df["time"][frame], fontsize=30)

    # ax2 >> Number of people graph
    ax2.clear()
    ax2.plot(
        df["time"][: frame + 1],
        df["num_people"][: frame + 1],
        marker="o",
        linestyle="-",
    )
    ax2.set_xlabel("Time", fontsize=22)
    ax2.set_ylabel("Number of People", fontsize=22)
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0, df["num_people"][: frame + 1].max() + 100)
    ax2.set_xlim(
        df["time"][0] - datetime.timedelta(minutes=2),
        df["time"][frame] + datetime.timedelta(minutes=5),
    )
    ax2.tick_params(labelsize=20)

    fig.suptitle(dataset_name, fontsize=25)
    # fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(f"{save_dir}/frames/{frame}.png")


def create_mov(save_dir, dataset_name, weight_name):
    path2img_list = natsorted(glob.glob(f"{save_dir}/frames/*.png"))
    img = cv2.imread(path2img_list[0])
    h, w, ch = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        f"{save_dir}/{dataset_name}_{weight_name}.mp4", fourcc, 10, (w, h)
    )
    for path2img in tqdm(path2img_list):
        img = cv2.imread(path2img)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


@click.command()
@click.argument("save_dir", type=str, default="vis")
@click.argument("source_dir", type=str, default="vis")
@click.argument("dataset_name", type=str)
@click.argument("weight_name", type=str)
def main(save_dir, source_dir, dataset_name, weight_name):
    # load data
    path2plot_list = sorted(
        glob.glob(f"{source_dir}/plot/{dataset_name}/{weight_name}/*.png")
    )
    path2csv = f"{source_dir}/graph/{dataset_name}/{weight_name}/num_people.csv"
    df = pd.read_csv(path2csv)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    # set graph
    fig = plt.figure(figsize=(16, 19))
    fig.subplots_adjust(left=0.03, bottom=0.0, right=0.97, top=1.0)
    ax1 = plt.subplot2grid((19, 16), (1, 0), rowspan=10, colspan=16)
    ax2 = plt.subplot2grid((19, 16), (11, 1), rowspan=7, colspan=14)

    # set save dir
    save_dir = f"{save_dir}/all_vis/{dataset_name}/{weight_name}"
    os.makedirs(f"vis/all_vis/{dataset_name}/{weight_name}/frames", exist_ok=True)

    # create frame
    for i, path2plot in enumerate(tqdm(path2plot_list)):
        save_frame(save_dir, i, fig, ax1, ax2, path2plot, df, dataset_name)

    # create movie
    create_mov(save_dir, dataset_name, weight_name)


def main_demo(dataset_name, weight_name):
    # load data
    path2plot_list = sorted(glob.glob(f"vis/plot/{dataset_name}/{weight_name}/*.png"))
    path2csv = f"vis/graph/{dataset_name}/{weight_name}/num_people.csv"
    df = pd.read_csv(path2csv)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    # set graph
    fig = plt.figure(figsize=(16, 19))
    fig.subplots_adjust(left=0.03, bottom=0.0, right=0.97, top=1.0)
    ax1 = plt.subplot2grid((19, 32), (1, 0), rowspan=10, colspan=32)
    ax2 = plt.subplot2grid((19, 32), (11, 2), rowspan=7, colspan=29)

    # set save dir
    save_dir = "vis_demo"
    os.makedirs("vis_demo/frames", exist_ok=True)

    # create frame
    for i, path2plot in enumerate(tqdm(path2plot_list[:3])):
        save_frame(save_dir, i, fig, ax1, ax2, path2plot, df, dataset_name)


if __name__ == "__main__":
    main()
    # main_demo("Akarenga_4K_2024_08_05", "2023_all")
    # create_mov("vis/all_vis/WorldPorters2023/2023", "WorldPorters2023")
