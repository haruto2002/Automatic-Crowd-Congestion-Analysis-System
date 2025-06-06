import numpy as np
import cv2
import os
from multiprocessing import Pool
from utils.get_track_data import get_all_track, get_all_vec
from tqdm import tqdm


def display_vec(track, end_frame, save_dir, img_name):
    vec_list = get_all_vec(track, end_frame)
    vec_data = np.array(vec_list)
    scale = 3
    img = (np.zeros((550 * scale, 350 * scale, 3)) + 255).astype(np.uint8)
    img_height, img_width, _ = img.shape
    vec_data *= scale

    arrow_len = 3
    tipLength = 0.3
    arrow_scale = 30
    for pos, vec in vec_data:
        vec = vec * arrow_scale
        start_point = (int(pos[0]), int(pos[1]))
        end_point = (int(start_point[0] + vec[0]), int(start_point[1] + vec[1]))
        cv2.circle(img, start_point, radius=5, color=(128, 128, 128), thickness=-1)
        cv2.arrowedLine(
            img,
            start_point,
            end_point,
            (0, 200, 0),
            arrow_len,
            tipLength=tipLength,
        )

    cv2.imwrite(save_dir + f"/{img_name}.png", img)
    return img


def create_mov(save_dir, img_list, save_name):
    # path2img_list = sorted(glob.glob(f"{source_dir}/*.png"))
    # img = cv2.imread(img_list[0])
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 10
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img in tqdm(img_list[1:]):
        # img = cv2.imread(path2img)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def parallel_display(inputs):
    track, frame, frame_save_dir, img_name = inputs
    img = display_vec(track, frame, frame_save_dir, img_name)
    return img


def main(
    place="WorldPorters_noon",
    crop_area=[130, 250, 130 + 350, 250 + 550],
    frame_range=(1, 8990),
    frame_save_dir="danger/vec_vis/WorldPorters_noon/frames",
    freq=10,
):
    start_frame, end_frame = frame_range
    pool_size = int(os.cpu_count())
    os.makedirs(frame_save_dir, exist_ok=True)

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
        pool_list.append([track, frame, frame_save_dir, img_name])

    print("Display Frames")
    with Pool(pool_size) as p:
        vec_vis = p.map(parallel_display, pool_list)

    # print("Create Movie")
    # save_name = f"{start_frame}_{end_frame}_vec"
    # create_mov(save_dir, vec_vis, save_name)


def many_frame_main():
    # save_par_dir = "danger/track_vis_demo"
    # all_frame_range = (1000, 1200)
    # part_num = 50
    save_par_dir = "danger/vec_vis"
    all_frame_range = (1, 8990)
    part_num = 1000
    for start_frame in range(all_frame_range[0], all_frame_range[1], part_num):
        end_frame = start_frame + part_num - 1
        if end_frame <= all_frame_range[1]:
            frame_range = (start_frame, end_frame)
        else:
            frame_range = (start_frame, all_frame_range[1])
        print(frame_range)
        main(frame_range, save_par_dir)
        # break


def create_vec_img(
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
    many_frame_main()
