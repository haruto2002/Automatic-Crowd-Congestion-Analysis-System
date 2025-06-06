import click
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess
from tqdm import tqdm

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


def graph(save_dir, place, df):
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df["time"], df["num_people"], linestyle="-")
    ax.tick_params(labelsize=30)
    ax.set_xlabel("Date time", fontsize=30)
    ax.set_xlim(df["time"].min(), df["time"].max())
    ax.set_ylabel("Number of People", fontsize=30)
    ax.set_ylim(0, (df["num_people"].max() // 50 + 1) * 50)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.title(place, fontsize=40)
    # plt.show()
    plt.savefig(f"{save_dir}/{place}.png")


def load_num_people(save_dir, place):
    cnt_data = []
    path2csv_list = sorted(glob.glob(f"2023_data/{place}3596_anno/*.csv"))
    for path2csv in path2csv_list:
        df = pd.read_csv(path2csv)
        num_people = len(df)
        time = path2csv.split("_")[-4]
        cnt_data.append([time, num_people])
    df = pd.DataFrame(cnt_data, columns=["time", "num_people"])
    df.to_csv(f"{save_dir}/{place}_num_people.csv", index=False)
    return df


def main():
    save_dir = "2023_data"
    place = "Kokusaibashi"
    # place = "WorldPorters"
    df = load_num_people(save_dir, place)
    graph(save_dir, place, df)


if __name__ == "__main__":
    main()
