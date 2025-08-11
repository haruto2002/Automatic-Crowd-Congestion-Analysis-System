import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import glob
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Palatino"],
#     }
# )


def plot_detection(path2img, path2detection, threshold=0.5, resize_ratio=1, plot_size=5):
    img = cv2.imread(path2img)
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    det = np.loadtxt(path2detection)
    det = det[det[:, 2] > threshold]
    for d in det:
        x, y, _ = d
        x = int(x * resize_ratio)
        y = int(y * resize_ratio)
        cv2.circle(img, (x, y), plot_size, (0, 0, 255), -1)
    return img


def plot_graph(df, frame, fig_size=(16, 9), img_size=(768, 432)):
    dpi = img_size[0] / fig_size[0]
    assert img_size[1] / fig_size[1] == dpi
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    canvas = FigureCanvas(fig)
    
    total_time_range = df["time"].max() - df["time"].min()
    x_margin = total_time_range * 0.05
    
    ax.plot(
        df["time"][: frame + 1],
        df["num_people"][: frame + 1],
        marker="o",
        linestyle="-",
    )
    ax.set_xlabel("Time", fontsize=22)
    ax.set_ylabel("Number of People", fontsize=22)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_ylim(0, df["num_people"][: frame + 1].max() + 100)
    
    ax.set_xlim(
        df["time"].min() - x_margin,
        df["time"].max() + x_margin,
    )
    ax.set_ylim(0, df["num_people"].max() + 100)
    ax.tick_params(labelsize=20)
    fig.tight_layout()

    # figをimgに変換
    canvas.draw()
    img = np.array(canvas.buffer_rgba())
    img = img[:, :, :3]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return img_bgr


def parallel_save_frame(inputs):
    path2img, path2detection, frame, df, resize_ratio, save_dir = inputs
    plot_size = int(15 * resize_ratio)
    fig_size = (16, 9)
    detection_vis = plot_detection(
        path2img, path2detection, threshold=0.5, 
        resize_ratio=resize_ratio, plot_size=plot_size
    )
    graph = plot_graph(
        df, frame, fig_size=fig_size, 
        img_size=(detection_vis.shape[1], detection_vis.shape[0])
    )
    all_vis = np.concatenate([detection_vis, graph], axis=0)
    cv2.imwrite(f"{save_dir}/{frame:04d}.png", all_vis)
    return all_vis


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


def create_video(save_dir, source_dir):
    path2img_list = sorted(glob.glob(f"{source_dir}/*.png"))
    img = cv2.imread(path2img_list[0])
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        f"{save_dir}/video.mp4", fourcc, 10, (w, h)
    )
    for path2img in tqdm(path2img_list):
        img = cv2.imread(path2img)
        out.write(img)
    out.release()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="yokohama/WorldPorter/vis")
    parser.add_argument("--img_dir", type=str, default="yokohama/WorldPorter/img")
    parser.add_argument("--detection_data_dir", type=str, default="yokohama/WorldPorter/full_detection")
    parser.add_argument("--resize_ratio", type=float, default=0.25)
    return parser.parse_args()


def main(save_dir, img_dir, detection_data_dir, resize_ratio):
    frame_save_dir = f"{save_dir}/frames"
    os.makedirs(frame_save_dir, exist_ok=True)

    path2img_list = sorted(glob.glob(f"{img_dir}/*.png"))
    path2detection_list = sorted(glob.glob(f"{detection_data_dir}/*.txt"))
    assert len(path2img_list) == len(path2detection_list)

    # debug = True
    # if debug:
    #     path2img_list = path2img_list[:10]
    #     path2detection_list = path2detection_list[:10]

    df = load_num_people(save_dir, detection_data_dir)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    pool_list = []
    for i, (path2img, path2detection) in enumerate(zip(path2img_list, path2detection_list)):
        frame = i
        pool_list.append([path2img, path2detection, frame, df, resize_ratio, frame_save_dir])
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap_unordered(parallel_save_frame, pool_list), total=len(pool_list)))

    create_video(save_dir, frame_save_dir)


def run_main():
    args = get_args()
    main(args.save_dir, args.img_dir, args.detection_data_dir, args.resize_ratio)


if __name__ == "__main__":
    run_main()