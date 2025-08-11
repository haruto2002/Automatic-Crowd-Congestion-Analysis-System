import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


def graph(save_dir, df, place):
    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df["time"], df["num_people"], linestyle="-")
    # ax.fill_between(df["time"], 0, df["num_people"], alpha=0.8, color="g", linewidth=0)
    ax.tick_params(labelsize=30)
    ax.set_xlabel("Date time", fontsize=30)
    ax.set_xlim(df["time"].min(), df["time"].max())
    ax.set_ylabel("Number of People", fontsize=30)
    ax.set_ylim(0, (df["num_people"].max() // 50 + 1) * 50)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.title(place, fontsize=40)
    plt.savefig(f"{save_dir}/{place}_{start_time}_{end_time}.png")


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
    parser.add_argument("--save_dir", type=str, default="vis")
    parser.add_argument("--source_dir", type=str, default="vis")
    return parser.parse_args()


def main(save_dir, source_dir, place):
    os.makedirs(save_dir, exist_ok=True)
    df = load_num_people(save_dir, source_dir)
    graph(save_dir, df, place)

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="yokohama/WorldPorter/full_detection")
    parser.add_argument("--source_dir", type=str, default="yokohama/WorldPorter/vis")
    parser.add_argument("--place", type=str, default="WorldPorter")
    args = parser.parse_args()
    os.makedirs(args.save_dir)
    main(args.save_dir, args.source_dir, args.place)


if __name__ == "__main__":
    run_main()