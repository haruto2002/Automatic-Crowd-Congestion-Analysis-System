import click
import glob
import os
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


def graph(save_dir, dataset_name, weight_name, df):
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
    plt.title(dataset_name, fontsize=40)
    # plt.show()
    plt.savefig(f"{save_dir}/{dataset_name}_{weight_name}.png")


def load_num_people(save_dir, source_dir, dataset_name, weight_name):
    cnt_data = []
    path2img_list = sorted(
        glob.glob(f"{source_dir}/plot/{dataset_name}/{weight_name}/*.png")
    )
    for path2img in path2img_list:
        num_people = int(path2img.split("_")[-1][4:-4])
        time = path2img.split("_")[-2]
        cnt_data.append([time, num_people])
    df = pd.DataFrame(cnt_data, columns=["time", "num_people"])
    df.to_csv(f"{save_dir}/num_people.csv", index=False)
    return df


@click.command()
@click.argument("save_dir", type=str, default="vis")
@click.argument("source_dir", type=str, default="vis")
@click.argument("dataset_name", type=str)
@click.argument("weight_name", type=str)
def main(save_dir, source_dir, dataset_name, weight_name):
    save_dir = f"{save_dir}/graph/{dataset_name}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)
    df = load_num_people(save_dir, source_dir, dataset_name, weight_name)
    graph(save_dir, dataset_name, weight_name, df)


if __name__ == "__main__":
    main()
