import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)


def create_graph(df, path2df, save_dir):
    if df is None:
        df = pd.read_csv(path2df)
        # df = df[df["frame"] <= 3500]
    window_size = 10
    df["dan1_ma"] = df["num_danger_01"].rolling(window=window_size).mean()
    df["dan2_ma"] = df["num_danger_02"].rolling(window=window_size).mean()
    df["dan3_ma"] = df["num_danger_03"].rolling(window=window_size).mean()
    df["dense1_ma"] = df["num_dense_01"].rolling(window=window_size).mean()
    df["dense2_ma"] = df["num_dense_02"].rolling(window=window_size).mean()
    df["dense3_ma"] = df["num_dense_03"].rolling(window=window_size).mean()

    fig, ax1 = plt.subplots(figsize=(15, 10))
    # fig, ax1 = plt.subplots(figsize=(10, 10))

    danger_colors = ["#FEE090", "#FC8D59", "#D73027"]
    ax1.plot(
        df["frame"],
        df["num_danger_01"],
        label="Original (thred=-1.0)",
        alpha=0.3,
        color=danger_colors[0],  # lw=0.5
    )
    ax1.plot(
        df["frame"],
        df["dan1_ma"],
        label=f"{window_size}-Frame Moving Average (thred=-1.0)",
        color=danger_colors[0],
    )

    ax1.plot(
        df["frame"],
        df["num_danger_02"],
        label="Original (thred=-0.5)",
        alpha=0.3,
        color=danger_colors[1],
    )
    ax1.plot(
        df["frame"],
        df["dan2_ma"],
        label=f"{window_size}-Frame Moving Average (thred=-0.5)",
        color=danger_colors[1],
    )
    ax1.plot(
        df["frame"],
        df["num_danger_03"],
        label="Original (thred=0.0)",
        alpha=0.3,
        color=danger_colors[2],
    )
    ax1.plot(
        df["frame"],
        df["dan3_ma"],
        label=f"{window_size}-Frame Moving Average (thred=0.0)",
        color=danger_colors[2],
    )

    dense_colors = ["#91BFDB", "#74ADD1", "#4575B4"]
    ax1.plot(
        df["frame"],
        df["num_dense_01"],
        label="Original (thred=-1.0)",
        alpha=0.3,
        color=dense_colors[0],  # lw=0.5
    )
    ax1.plot(
        df["frame"],
        df["dense1_ma"],
        label=f"{window_size}-Frame Moving Average (thred=-1.0)",
        color=dense_colors[0],
    )

    ax1.plot(
        df["frame"],
        df["num_dense_02"],
        label="Original (thred=-0.5)",
        alpha=0.3,
        color=dense_colors[1],
    )
    ax1.plot(
        df["frame"],
        df["dense2_ma"],
        label=f"{window_size}-Frame Moving Average (thred=-0.5)",
        color=dense_colors[1],
    )
    ax1.plot(
        df["frame"],
        df["num_dense_03"],
        label="Original (thred=0.0)",
        alpha=0.3,
        color=dense_colors[2],
    )
    ax1.plot(
        df["frame"],
        df["dense3_ma"],
        label=f"{window_size}-Frame Moving Average (thred=0.0)",
        color=dense_colors[2],
    )

    # ax1.set_xlabel("Frame",fontsize=30)
    # ax1.set_ylabel("Number of Danger", fontsize=30)
    # ax1.tick_params(labelsize=25)
    ax1.set_xlabel("Frame", fontsize=25)
    ax1.set_ylabel("Number of Danger and Dense", fontsize=25)
    ax1.tick_params(labelsize=20)
    ax1.grid()
    # ax1.legend(loc="upper left", fontsize=15)

    ax2 = ax1.twinx()
    ax2.plot(
        df["frame"],
        df["num_people"],
        label="Number of People",
        color="green",
    )
    ax2.set_ylabel("Number of People", fontsize=25)
    ax2.tick_params(labelsize=20)
    # ax2.legend(loc="upper right",fontsize=15)

    # ax1.set_xlim(0, 4000)

    plt.savefig(f"{save_dir}/graph.pdf")
    # plt.savefig(f"{save_dir}/graph_crop.pdf")
    # plt.savefig(f"{save_dir}/graph_crop_3500.pdf")
    # plt.xlim(0, 4000)
    # plt.savefig("fig2_graph_v2.pdf")
    # plt.xlim(1800, 2300)
    # plt.savefig("fig3_graph.pdf")
    # plt.show()


if __name__ == "__main__":
    create_graph(
        None,
        "danger/results/0228_exp/WorldPorters_noon/1_8990/10_5_1_50_0.1_False_True_75_3/v2/data_1_8990_crop_130_250_480_800.csv",
        "danger/results/0228_exp/WorldPorters_noon/1_8990/10_5_1_50_0.1_False_True_75_3/v2/",
    )
