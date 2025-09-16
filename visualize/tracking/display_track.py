import numpy as np
import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import glob
import logging


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
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(save_path, img_copy)
    return img_copy


def create_mov(img_list, save_path, fps, disable_tqdm=False, tqdm_pos=0):
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for img in tqdm(
        img_list,
        desc="Creating Movie",
        leave=False,
        disable=disable_tqdm,
        position=tqdm_pos,
    ):
        out.write(img)
    out.release()


def parallel_display(inputs):
    track, img, save_path, color_list, order = inputs
    img = display_tracking_result(track, img, save_path, color_list)
    return img, order


def parallel_set_data(inputs):
    track_dir, img_dir, s, e, crop_area, save_path, color_list, order = inputs
    track, img = get_all_track(track_dir, img_dir, s, e, crop_area)
    return track, img, save_path, color_list, order


def get_all_track(track_dir, img_dir, start_frame, end_frame, crop_area=None):
    # crop_area >> [xmin,ymin,xmax,ymax]
    txt_files = sorted(glob.glob(f"{track_dir}/*.txt"))
    track_list = [
        np.loadtxt(path2txt, delimiter=",")
        for path2txt in txt_files[start_frame - 1 : end_frame]
    ]
    all_track = []
    for i, track in enumerate(track_list):
        frame = start_frame + i
        track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
        all_track += list(track)
    all_track = np.array(all_track)

    path2img = sorted(glob.glob(f"{img_dir}/*.jpg"))[end_frame - 1]
    img = cv2.imread(path2img)

    if crop_area is not None:
        all_track = all_track[
            (all_track[:, 2] > crop_area[0])
            & (all_track[:, 2] < crop_area[2])
            & (all_track[:, 3] > crop_area[1])
            & (all_track[:, 3] < crop_area[3])
        ]
        all_track[:, 2] += -crop_area[0]
        all_track[:, 3] += -crop_area[1]

        img = img[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]

    # all_track=[[frame,id,x,y],...]
    return all_track, img


def main(
    track_dir,
    img_dir,
    save_base_dir,
    freq,
    vis_track_length,
    start_frame=None,
    end_frame=None,
    crop_area=None,
    node_type=None,
    disable_tqdm=False,
    tqdm_pos=0,
):
    frame_save_dir = f"{save_base_dir}/frames"
    os.makedirs(frame_save_dir, exist_ok=True)
    if start_frame is None:
        start_frame = 1
    if end_frame is None:
        end_frame = len(glob.glob(f"{track_dir}/*.txt"))
    np.random.seed(seed=32)
    color_list = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(10000)]
    if node_type == "rt_HF":
        pool_size = 192
    elif node_type == "rt_HG":
        pool_size = 16
    elif node_type == "rt_HC":
        pool_size = 32
    else:
        pool_size = int(os.cpu_count())

    display_inputs = []
    set_data_inputs = []
    save_path_list = []
    order = 0
    for e in range(start_frame, end_frame + 1, freq):
        order += 1
        if e - vis_track_length <= 0:
            s = 1
        else:
            s = e - vis_track_length
        save_path = f"{frame_save_dir}/{e:04d}.jpg"
        save_path_list.append(save_path)
        set_data_inputs.append(
            (track_dir, img_dir, s, e, crop_area, save_path, color_list, order)
        )

    with Pool(pool_size) as p:
        display_inputs = list(
            tqdm(
                p.imap_unordered(parallel_set_data, set_data_inputs),
                total=len(set_data_inputs),
                desc="Setting Data",
                leave=False,
                position=tqdm_pos,
                disable=disable_tqdm,
            )
        )

    with Pool(pool_size) as p:
        tracking_vis = list(
            tqdm(
                p.imap_unordered(parallel_display, display_inputs),
                total=len(display_inputs),
                desc="Displaying Frames",
                leave=False,
                position=tqdm_pos,
                disable=disable_tqdm,
            )
        )

    img_list = [img for img, order in sorted(tracking_vis, key=lambda x: x[-1])]
    movie_save_path = f"{save_base_dir}/{start_frame}_{end_frame}.mp4"
    fps = 30 / freq
    create_mov(img_list, movie_save_path, fps, disable_tqdm, tqdm_pos)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_dir", type=str, default="demo/track")
    parser.add_argument("--img_dir", type=str, default="demo/img")
    parser.add_argument("--save_base_dir", type=str, default="demo/track_vis")
    parser.add_argument("--freq", type=int, default=1, help="display frame interval")
    parser.add_argument(
        "--vis_track_length", type=int, default=30, help="length of visualised track"
    )
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--crop_area",
        type=int,
        nargs=4,
        default=None,
        help="crop area as four integers: x1 y1 x2 y2",
    )
    parser.add_argument("--node_type", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--multi_video_mode", type=bool, default=False)
    args = parser.parse_args()

    node_type = args.node_type
    log_level = args.log_level
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    disable_tqdm = logger.level >= logging.ERROR

    if args.multi_video_mode:
        tqdm_pos = 1
    else:
        tqdm_pos = 0

    main(
        args.track_dir,
        args.img_dir,
        args.save_base_dir,
        args.freq,
        args.vis_track_length,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        crop_area=args.crop_area,
        node_type=node_type,
        disable_tqdm=disable_tqdm,
        tqdm_pos=tqdm_pos,
    )


if __name__ == "__main__":
    run_main()
