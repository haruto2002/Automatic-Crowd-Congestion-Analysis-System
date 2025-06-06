import numpy as np
import pandas as pd
import cv2
import os
from multiprocessing import Pool
from utils.get_track_data import get_all_track, get_all_vec
from tqdm import tqdm
import pickle
import time


def display_tracking_result(track, save_dir, img_name, frame, color_list):
    multi_color = True
    scale = 3
    back_img = (np.zeros((550 * scale, 350 * scale, 3)) + 255).astype(np.uint8)
    img_copy = back_img.copy()
    existing_id = track[track[:, 0] == np.max(track[:, 0])][:, 1]
    for id in existing_id:
        one_track = track[track[:, 1] == id]
        use_length = min(30, len(one_track))
        for i, data in enumerate(one_track[-use_length:]):
            p = data[2:]  # coordinate
            if i == use_length - 1:
                p_size = 7
            else:
                p_size = 1
            if multi_color:
                cv2.circle(
                    img_copy,
                    (int(p[0]) * 3, int(p[1]) * 3),
                    p_size,
                    color_list[(int(id) - 1) % len(color_list)],
                    -1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    img_copy,
                    (int(p[0]) * 3, int(p[1]) * 3),
                    p_size,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )
    # if frame is not None:
    #     # print(frame)
    #     add_frame(img_copy, str(frame))
    cv2.imwrite(save_dir + f"/{img_name}.png", img_copy)
    return img_copy


def add_frame(img, frame):
    h, w, c = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    color = (0, 0, 0)
    thickness = 2
    text_size, _ = cv2.getTextSize(frame, font, font_scale, thickness)
    text_x = w - text_size[0] - 10
    text_y = 20 + text_size[1]

    cv2.putText(img, frame, (text_x, text_y), font, font_scale, color, thickness)


def create_mov(save_dir, img_list, save_name):
    # path2img_list = sorted(glob.glob(f"{source_dir}/*.png"))
    # img = cv2.imread(img_list[0])
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 30
    # fps = 10
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img in tqdm(img_list[1:]):
        # img = cv2.imread(path2img)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def parallel_display(inputs):
    track, save_dir, img_name, frame, color_list = inputs
    img = display_tracking_result(track, save_dir, img_name, frame, color_list)
    return img


def main(
    place="WorldPorters_noon",
    crop_area=[130, 250, 130 + 350, 250 + 550],
    frame_range=(1, 8990),
    frame_save_dir="danger/track_vis/WorldPorters_noon/frames",
    freq=1,
):
    os.makedirs(frame_save_dir, exist_ok=True)
    start_frame, end_frame = frame_range
    start_frame, end_frame = frame_range
    np.random.seed(seed=32)
    color_list = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(10000)]
    pool_size = int(os.cpu_count())

    print("Setting Data")
    # all_track=[[frame,id,x,y],...]
    all_track, _ = get_all_track(place, start_frame, end_frame, crop_area)

    all_track = all_track[
        (all_track[:, 2] > crop_area[0])
        & (all_track[:, 2] < crop_area[2])
        & (all_track[:, 3] > crop_area[1])
        & (all_track[:, 3] < crop_area[3])
    ]
    all_track[:, 2] += -crop_area[0]
    all_track[:, 3] += -crop_area[1]
    # print(all_track[0])

    pool_list = []
    for i in tqdm(range(start_frame, end_frame + 1, freq)):
        track = all_track[all_track[:, 0] <= i]
        frame = i
        img_name = f"{frame:04d}"
        pool_list.append(
            [
                track,
                frame_save_dir,
                img_name,
                frame,
                color_list,
            ]
        )

    print("Display Frames")
    with Pool(pool_size) as p:
        tracking_vis = p.map(parallel_display, pool_list)

    # print("Create Movie")
    # movie_save_dir = f"{save_par_dir}/{place}"
    # create_mov(save_dir, tracking_vis, save_name)


def many_frame_main():
    place = ("WorldPorters_noon",)
    crop_area = ([130, 250, 130 + 350, 250 + 550],)
    frame_range = ((1, 8990),)
    frame_save_dir = "danger/track_vis/WorldPorters_noon/frames"
    all_frame_range = (1, 8990)
    part_num = 1000
    for start_frame in range(all_frame_range[0], all_frame_range[1], part_num):
        end_frame = start_frame + part_num - 1
        if end_frame <= all_frame_range[1]:
            frame_range = (start_frame, end_frame)
        else:
            frame_range = (start_frame, all_frame_range[1])
        print(frame_range)
        main(place, crop_area, frame_range, frame_save_dir)


def create_track_img(
    place,
    crop_area,
    all_frame_range,
    frame_save_dir,
    saparate_run=False,
    part_num=None,
    freq=1,
):
    if saparate_run:
        for start_frame in range(all_frame_range[0], all_frame_range[1], part_num):
            end_frame = start_frame + part_num - 1
            if end_frame <= all_frame_range[1]:
                frame_range = (start_frame, end_frame)
            else:
                frame_range = (start_frame, all_frame_range[1])
            print(frame_range)
            main(place, crop_area, frame_range, frame_save_dir, freq=freq)
    else:
        main(place, crop_area, all_frame_range, frame_save_dir, freq=freq)


if __name__ == "__main__":
    # s = time.time()
    # main()
    # e = time.time()
    # print(e - s)
    # 586.6551077365875

    many_frame_main()
