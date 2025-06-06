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


def graph_2comp(save_dir, place, df1, df2):
    color_list = ["tab:blue", "tab:orange"]
    label_list = ["Predict", "GT"]
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, df in enumerate([df1, df2]):
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        ax.plot(
            df["time"],
            df["num_people"],
            linestyle="-",
            color=color_list[i],
            label=label_list[i],
        )
        ax.tick_params(labelsize=30)
        ax.set_xlabel("Date time", fontsize=30)
        # ax.set_xlim(df["time"].min(), df["time"].max())
        ax.set_ylabel("Number of People", fontsize=30)
        # ax.set_ylim(0, (df["num_people"].max() // 50 + 1) * 50)
        ax.grid(True, axis="y", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.legend(fontsize=25)
    plt.title(place, fontsize=40)
    # plt.show()
    plt.savefig(f"{save_dir}/{place}.png")


def main():
    save_dir = "vis/comparison"
    os.makedirs(save_dir, exist_ok=True)
    # set data
    place = "WorldPorters"
    csv1 = "vis/graph/WorldPorters2023/2023/num_people.csv"
    df1 = pd.read_csv(csv1)
    csv2 = "2023_data/WorldPorters_num_people.csv"
    df2 = pd.read_csv(csv2)

    # create graph
    graph_2comp(save_dir, place, df1, df2)


def graph_3comp():
    save_dir = "vis/comparison"
    os.makedirs(save_dir, exist_ok=True)

    # set data
    place = "Kokusaibashi"
    csv1 = "vis/graph/Kokusaibashi2023/yokohama2023_70/num_people.csv"
    df1 = pd.read_csv(csv1)
    csv2 = "vis/graph/Kokusaibashi2023/2023/num_people.csv"
    df2 = pd.read_csv(csv2)
    csv3 = "2023_data/Kokusaibashi_num_people.csv"
    df3 = pd.read_csv(csv3)
    label_list = ["Predict1", "Predict2", "GT"]

    # create graph
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, df in enumerate([df1, df2, df3]):
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        ax.plot(
            df["time"],
            df["num_people"],
            linestyle="-",
            label=label_list[i],
        )
        ax.tick_params(labelsize=30)
        ax.set_xlabel("Date time", fontsize=30)
        # ax.set_xlim(df["time"].min(), df["time"].max())
        ax.set_ylabel("Number of People", fontsize=30)
        # ax.set_ylim(0, (df["num_people"].max() // 50 + 1) * 50)
        ax.grid(True, axis="y", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.legend(fontsize=25)
    plt.title(place, fontsize=40)
    # plt.show()
    plt.savefig(f"{save_dir}/{place}_3.png")


if __name__ == "__main__":
    # main()
    graph_3comp()
