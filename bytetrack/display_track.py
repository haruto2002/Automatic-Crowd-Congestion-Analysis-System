import numpy as np
import cv2
import os
from multiprocessing import Pool
from get_track_data import get_all_track
from tqdm import tqdm
import argparse

def display_tracking_result(track, img, save_path, color_list):
    img_copy = img.copy()
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

            cv2.circle(
                img_copy,
                (int(p[0]), int(p[1])),
                p_size,
                color_list[(int(id) - 1) % len(color_list)],
                -1,lineType=cv2.LINE_AA,
            )
    cv2.imwrite(save_path, img_copy)
    return img_copy


def create_mov(img_list, save_path):
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for img in tqdm(img_list):
        out.write(img)
    out.release()


def parallel_display(inputs):
    track, img, save_path, color_list, order = inputs
    img = display_tracking_result(track,img, save_path, color_list)
    return img, order

def parallel_set_data(inputs):
    place, s, e, crop_area, save_path, color_list, order = inputs
    track, img = get_all_track(place, s, e, crop_area)
    return track, img, save_path, color_list, order


def main(
    place,
    crop_area,
    frame_range,
    save_base_dir,
    freq,
    vis_span
):
    save_dir=f"{save_base_dir}/{place}/{crop_area[0]}_{crop_area[1]}_{crop_area[2]}_{crop_area[3]}"
    print("SAVE_DIR:",save_dir)
    frame_save_dir=f"{save_dir}/frames"
    os.makedirs(frame_save_dir, exist_ok=True)
    start_frame, end_frame = frame_range
    np.random.seed(seed=32)
    color_list = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(10000)]
    pool_size = int(os.cpu_count())

    print("Setting Data")
    display_inputs = []
    set_data_inputs = []
    save_path_list = []
    order=0
    for e in range(start_frame, end_frame + 1, freq):
        order+=1
        if e - vis_span <=0:
            s = 1
        else:
            s = e - vis_span
        save_path = f"{frame_save_dir}/{e:04d}.png"
        save_path_list.append(save_path)
        set_data_inputs.append((place, s, e, crop_area,save_path,color_list,order))

    with Pool(pool_size) as p:
        display_inputs = list(tqdm(p.imap_unordered(parallel_set_data, set_data_inputs), total=len(set_data_inputs)))

    print("Displaying Frames")
    with Pool(pool_size) as p:
        tracking_vis = list(tqdm(p.imap_unordered(parallel_display, display_inputs), total=len(display_inputs)))

    print("Creating Movie")
    img_list = [img for img, order in sorted(tracking_vis, key=lambda x: x[-1])]
    movie_save_path=f"{save_dir}/{frame_range[0]}_{frame_range[1]}.mp4"
    create_mov(img_list, movie_save_path)


def run_mian():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", type=str, default="WorldPorters_noon")
    parser.add_argument("--crop_area", type=list, default=[0, 0, 2000, 2000])
    parser.add_argument("--frame_range", type=list, default=(1, 1500))
    parser.add_argument("--save_base_dir", type=str, default="byte_track/track_vis")
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--vis_span", type=int, default=30)
    args = parser.parse_args()
    main(args.place, args.crop_area, args.frame_range, args.save_base_dir, args.freq, args.vis_span)

if __name__ == "__main__":
    run_mian()
