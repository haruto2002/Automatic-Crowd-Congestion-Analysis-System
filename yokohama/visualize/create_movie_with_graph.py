import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import cv2
import glob
from tqdm import tqdm
import os
import datetime
import argparse
import numpy as np
from multiprocessing import Pool
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


def parallel_save_frame(pool_list):
    save_dir, frame, fig, ax1, ax2, path2img, df, place = pool_list
    save_frame(save_dir, frame, fig, ax1, ax2, path2img, df, place)


def save_frame(save_dir, frame, fig, ax1, ax2, path2img, df, place):
    # ax1 >> Images of inference results
    img = cv2.imread(path2img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_time(img, df, frame)

    ax1.clear()
    ax1.imshow(img)
    ax1.axis("off")

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

    fig.suptitle(place, fontsize=25)
    # fig.autofmt_xdate()
    # fig.tight_layout()
    plt.savefig(f"{save_dir}/{frame:04d}.png")


def create_mov(save_dir, source_dir, place):
    path2img_list = sorted(glob.glob(f"{source_dir}/*.png"))
    img = cv2.imread(path2img_list[0])
    h, w, ch = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        f"{save_dir}/{place}_movie_with_graph.mp4", fourcc, 10, (w, h)
    )
    for path2img in tqdm(path2img_list):
        img = cv2.imread(path2img)
        out.write(img)

    out.release()


def load_num_people(save_dir, source_dir, threshold=0.5):
    cnt_data = []
    path2txt_list = sorted(glob.glob(f"{source_dir}/*.txt"))
    for path2txt in path2txt_list:
        result = np.loadtxt(path2txt)
        det_result = result[result[:, 2] > threshold]
        num_people = len(det_result)
        time = path2txt[:-4].split("_")[-1]
        cnt_data.append([time, num_people])
    df = pd.DataFrame(cnt_data, columns=["time", "num_people"])
    df.to_csv(f"{save_dir}/num_people.csv", index=False)
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="yokohama/WorldPorter/vis")
    parser.add_argument("--detection_img_dir", type=str, default="yokohama/WorldPorter/detection_plot")
    parser.add_argument("--detection_data_dir", type=str, default="yokohama/WorldPorter/full_detection")
    parser.add_argument("--place", type=str, default="WorldPorter")
    parser.add_argument("--path2num_people_data", type=str, default=None)
    return parser.parse_args()


def main(save_dir, detection_img_dir, detection_data_dir, place, path2num_people_data=None):
    # load data
    path2plot_list = sorted(
        glob.glob(f"{detection_img_dir}/*.png")
    )
    if path2num_people_data is None:
        df = load_num_people(save_dir, detection_data_dir)
    else:
        df = pd.read_csv(path2num_people_data)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    # set graph
    fig = plt.figure(figsize=(16, 19))
    fig.subplots_adjust(left=0.03, bottom=0.0, right=0.97, top=1.0)
    ax1 = plt.subplot2grid((19, 16), (1, 0), rowspan=10, colspan=16)
    ax2 = plt.subplot2grid((19, 16), (11, 1), rowspan=7, colspan=14)

    # set save dir
    frame_save_dir = f"{save_dir}/frames_movie_with_graph"
    os.makedirs(frame_save_dir, exist_ok=True)

    # create frame
    pool_list = []
    for i, path2plot in enumerate(path2plot_list):
        pool_list.append([frame_save_dir, i, fig, ax1, ax2, path2plot, df, place])
    pool_size = os.cpu_count()//2
    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(parallel_save_frame, pool_list), total=len(pool_list)))

    # create movie
    create_mov(save_dir, frame_save_dir, place)

def run_main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args.save_dir, args.detection_img_dir, args.detection_data_dir, args.place, args.path2num_people_data)


if __name__ == "__main__":
    run_main()
