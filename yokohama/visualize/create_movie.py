import glob
from tqdm import tqdm
import cv2
import os
import argparse

def create_crop_mov(save_dir, source_dir, place, v_devide_num, h_devide_num):
    path2img_list = sorted(
        glob.glob(f"{source_dir}/*.png")
    )
    img = cv2.imread(path2img_list[0])
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    h, w, c = img.shape
    crop_h = h // v_devide_num
    crop_w = w // h_devide_num
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_list = []
    for i in range(v_devide_num):
        for j in range(h_devide_num):
            out = cv2.VideoWriter(
                f"{save_dir}/{place}_{i}_{j}.mp4", fourcc, 5, (crop_w, crop_h)
            )
            out_list.append(out)
    for path2img in tqdm(path2img_list):
        img = cv2.imread(path2img)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        for i in range(v_devide_num):
            for j in range(h_devide_num):
                crop_img = img[i * crop_h:(i + 1) * crop_h, j * crop_w:(j + 1) * crop_w]
                out_list[i * h_devide_num + j].write(crop_img)
    for out in out_list:
        out.release()

def create_full_mov(save_dir, source_dir, place):
    path2img_list = sorted(
        glob.glob(f"{source_dir}/*.png")
    )
    img = cv2.imread(path2img_list[0])
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        f"{save_dir}/{place}.mp4", fourcc, 5, (w, h)
    )
    for path2img in tqdm(path2img_list):
        img = cv2.imread(path2img)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)
    out.release()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="yokohama/WorldPorter/vis")
    parser.add_argument("--source_dir", type=str, default="yokohama/WorldPorter/detection_plot")
    parser.add_argument("--place", type=str, default="WorldPorter")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--v_devide_num", type=int, default=1)
    parser.add_argument("--h_devide_num", type=int, default=1)
    return parser.parse_args()


def main(save_dir, source_dir, place, v_devide_num, h_devide_num, full):
    os.makedirs(save_dir, exist_ok=True)
    if full:
        create_full_mov(save_dir, source_dir, place)
    else:
        create_crop_mov(save_dir, source_dir, place, v_devide_num, h_devide_num)


def run_main():
    args = get_args()
    main(args.save_dir, args.source_dir, args.place, args.v_devide_num, args.h_devide_num, args.full)


if __name__ == "__main__":
    run_main()
