import numpy as np
import cv2
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


def get_all_track(place, start_frame, end_frame, crop_area=None, bev=True, remove=True):
    # crop_area >> [xmin,ymin,xmax,ymax]
    if os.path.isdir(f"tracking_results/{place}/tracking3/results"):
        # print("source data >> tracking3")
        csv_files = sorted(
            glob.glob(f"tracking_results/{place}/tracking3/results/*.csv")
        )
    else:
        if not os.path.isdir(f"tracking_results/{place}/tracking/results"):
            raise ValueError("no tracking data")
        # print("source data >> tracking")
        csv_files = sorted(
            glob.glob(f"tracking_results/{place}/tracking/results/*.csv")
        )
    df_list = [
        pd.read_csv(path2csv) for path2csv in csv_files[start_frame - 1 : end_frame]
    ]
    if start_frame > 1:
        pre_num = 90
        pre_start_frame = max(1, start_frame - 1 - pre_num)
        pre_df_list = [
            pd.read_csv(path2csv)
            for path2csv in csv_files[pre_start_frame - 1 : start_frame - 1]
        ]
        df_list = pre_df_list + df_list
        start_frame = pre_start_frame
    all_track = []
    for i, df in enumerate(df_list):
        frame = start_frame + i
        df["frame"] = frame
        track = df[["frame", "id", "x", "y"]].to_numpy()
        # track = track[
        #     (crop_area[0] < track[:, 2])
        #     & (crop_area[1] < track[:, 3])
        #     & (track[:, 2] < crop_area[2])
        #     & (track[:, 3] < crop_area[3])
        # ]
        all_track += list(track)
    all_track = np.array(all_track)

    path2img = sorted(glob.glob(f"tracking_results/{place}/img/*.png"))[end_frame - 1]
    img = cv2.imread(path2img)
    # img = img[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2], :]

    # BEV変換
    if bev:
        all_track, bev_img = bev_trans(place, all_track, img)

    # あり得ない検出結果を除外
    if remove:
        all_track = remove_error(place, all_track)

    # all_track=[[frame,id,x,y],...]
    return all_track, bev_img

def get_all_track_byte_track(place, start_frame, end_frame, crop_area=None, bev=True, remove=True):
    # crop_area >> [xmin,ymin,xmax,ymax]
    if place == "WorldPorters_noon":
        txt_files = sorted(
                glob.glob(f"byte_track/track_data/{place}/interpolated_frame_data/*.txt")
            )
        track_list = [
            np.loadtxt(path2txt,delimiter=",") for path2txt in txt_files[start_frame - 1 : end_frame]
        ]
    else:
        raise ValueError("The place is not found!")

    if start_frame > 1:
        pre_num = 90
        pre_start_frame = max(1, start_frame - pre_num)
        pre_track_list = [
            np.loadtxt(path2txt,delimiter=",")
            for path2txt in txt_files[pre_start_frame - 1 : start_frame - 1]
        ]
        track_list = pre_track_list + track_list
        start_frame = pre_start_frame

    all_track = []
    for i, track in enumerate(track_list):
        frame = start_frame + i
        track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
        all_track += list(track)
    all_track = np.array(all_track)

    path2img = sorted(glob.glob(f"tracking_results/{place}/img/*.png"))[end_frame - 1]
    img = cv2.imread(path2img)
    # img = img[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2], :]

    # BEV変換
    if bev:
        all_track, bev_img = bev_trans(place, all_track, img)

    # あり得ない検出結果を除外
    if remove:
        all_track = remove_error(place, all_track)

    # all_track=[[frame,id,x,y],...]
    return all_track, bev_img

def bev_trans(place, all_track, img):
    if place in ["WorldPorters_noon", "WorldPorters_night", "WorldPorters_night2"]:
        place_dir = "WorldPorters_8K"
    elif place in ["Chosha_noon", "Chosha_night"]:
        place_dir = "Chosha_8K"
    elif place in ["Kokusaibashi_noon", "Kokusaibashi_night"]:
        place_dir = "Kokusaibashi_8K"
    else:
        raise ValueError("The place is not found!")

    matrix = np.loadtxt(f"danger/bev/{place_dir}/matrix.txt")
    img_w, img_h = np.loadtxt(f"danger/bev/{place_dir}/size.txt").astype(int)
    all_track[:, 2:4] = cv2.perspectiveTransform(
        all_track[:, 2:4].reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    bev_img = cv2.warpPerspective(
        img,
        matrix,
        (img_w, img_h),
        borderValue=(255, 255, 255),
    )
    return all_track, bev_img


def remove_error(place, all_track):
    if "Chosha" in place:
        polygon_points = np.loadtxt("danger/error_region/Chosha_error.txt")
    # elif "WorldPorters" in place:
    #     polygon_points = np.loadtxt("danger/error_region/WorldPorters_error.txt")
    else:
        return all_track

    polygon = Polygon(polygon_points)

    # all_track=[[frame, id, x, y], ...] から、エラー領域内にいないトラッキングデータを選択
    filtered_tracks = [
        track for track in all_track if not polygon.contains(Point(track[2], track[3]))
    ]

    return np.array(filtered_tracks)


def get_all_vec(all_track, end_frame):
    id_list = np.unique(all_track[all_track[:, 0] == end_frame][:, 1])
    pool_list = [(id, all_track) for id in id_list]

    # pool_size = os.cpu_count()
    # with Pool(pool_size) as p:
    #     vec_list = p.map(get_vec, pool_list)

    vec_list = []
    for input in pool_list:
        vec_list.append(get_vec(input))

    if len(vec_list) == 0:
        print(all_track.shape)
        print("no vec! Check the track data or coding")
        print(end_frame)
        print(pool_list)
        # raise ValueError("no vec! Check the track data or coding")
    return vec_list


def get_vec(pool):
    id, all_track = pool
    track = all_track[all_track[:, 1] == id]
    start_point = track[0, 2:]
    end_point = track[-1, 2:]
    vec = (end_point - start_point) / len(track)
    pos = end_point
    return (pos, vec)


def display_vec(vec_list, save_path, img=None):
    plt.figure(figsize=(15, 11))

    if img is not None:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)

    for pos, vec in vec_list:
        vec = vec * 50
        start_point = (pos[0], pos[1])
        vx = vec[0]
        vy = vec[1]

        plt.arrow(
            start_point[0],
            start_point[1],
            vx,
            vy,
            head_width=3,
            head_length=3,
            fc="lime",
            ec="lime",
            linewidth=0.5,
        )
    plt.xlim(0, 1500)
    plt.ylim(1100, 0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    place = "WorldPorters_noon"
    start_frame = 0
    end_frame = start_frame + 90
    track, img = get_all_track(place, start_frame, end_frame)
    vec_list = get_all_vec(track, end_frame)
    # vec_data = np.array(vec_list)
    # save_dir = "danger/vec_vis"
    # save_path = f"{save_dir}/{place}_{start_frame}_{end_frame}.png"
    # os.makedirs(save_dir, exist_ok=True)
    # display_vec(vec_data, save_path, img=img)
