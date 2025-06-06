import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool


def add_map(path2img, map_data, save_path):
    img = cv2.imread(path2img)
    start = 0
    for num in [7, 4, 4, 4, 8]:
        points = map_data[start : start + num]
        points = points.astype(int)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], (0, 0, 0))
        start = start + num

    cv2.imwrite(save_path, img)

    return img


def parallel_add_map(input):
    path2img, map_data, save_path = input
    img_with_map = add_map(path2img, map_data, save_path)
    return img_with_map


def create_simple_mov(save_dir, img_list, save_name, fps=10):
    img = img_list[0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img in tqdm(img_list):
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def create_mov(save_dir, data, save_name, fps=10):
    # data=[(img,frame),...]
    img = data[0][0]
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", fourcc, fps, (w, h))
    for img, frame_num in tqdm(data):
        # add_frame(img, frame_num)
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


def add_frame(img, frame):
    h, w, c = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    color = (255, 255, 255)
    thickness = 2
    text_size, _ = cv2.getTextSize(frame, font, font_scale, thickness)
    text_x = w - text_size[0] - 10
    text_y = 20 + text_size[1]

    cv2.putText(img, frame, (text_x, text_y), font, font_scale, color, thickness)


def main_single():
    path2img = "danger/track_vis/WorldPorters_noon/frames/2866.png"
    scale = 3
    map_data = np.loadtxt("danger/map_data/WorldPorters_crop.txt", delimiter=",")
    map_data *= scale
    save_path = "demo.png"
    add_map(path2img, map_data, save_path)


def main_movie(place="WorldPorters_noon"):
    # par_dir_name = "track_vis"
    par_dir_name = "vec_vis"
    raw_img_dir = f"danger/{par_dir_name}/{place}/frames"
    save_dir = f"danger/{par_dir_name}/{place}/frames_with_map"
    os.makedirs(save_dir)
    pool_size = int(os.cpu_count())
    path2img_list = sorted(glob.glob(raw_img_dir + "/*.png"))
    # print(path2img_list)
    scale = 3
    map_data = np.loadtxt("danger/map_data/WorldPorters_crop.txt", delimiter=",")
    map_data *= scale
    pool_list = []
    print("Setting data")
    for path2img in tqdm(path2img_list):
        save_name = path2img.split("/")[-1]
        save_path = os.path.join(save_dir, save_name)
        input = [path2img, map_data, save_path]
        pool_list.append(input)
    print("Adding data")
    with Pool(pool_size) as p:
        img_list = p.map(parallel_add_map, pool_list)
    print("Creating movie")
    create_simple_mov(f"danger/{par_dir_name}/{place}", img_list, "with_map")


def only_movie(place="WorldPorters_noon"):
    par_dir_name = "track_vis"
    img_dir = f"danger/{par_dir_name}/{place}/frames_with_map"
    pool_size = int(os.cpu_count())
    freq = 10
    path2img_list = sorted(glob.glob(img_dir + "/*.png"))[::freq]
    print("Reading images")
    with Pool(pool_size) as p:
        results = p.map(parallel_read, path2img_list)
    print("Creating movie")
    # create_mov(f"danger/{par_dir_name}/{place}", results, "with_map_faster", fps=5)
    create_mov(
        f"danger/{par_dir_name}/{place}", results, "with_map_faster_no_frame", fps=10
    )


def parallel_read(path2img):
    img = cv2.imread(path2img)
    frame = path2img.split("/")[-1][-8:-4]
    return img, frame


def main_add_map(
    raw_img_dir,
    save_dir,
    save_mov_path,
    path2map_data="danger/map_data/WorldPorters_crop.txt",
    scale=3,
):
    os.makedirs(save_dir)
    pool_size = int(os.cpu_count())
    path2img_list = sorted(glob.glob(raw_img_dir + "/*.png"))
    map_data = np.loadtxt(path2map_data, delimiter=",")
    map_data *= scale
    pool_list = []
    print("Setting data")
    for path2img in tqdm(path2img_list):
        save_name = path2img.split("/")[-1]
        save_path = os.path.join(save_dir, save_name)
        input = [path2img, map_data, save_path]
        pool_list.append(input)
    print("Adding data")
    with Pool(pool_size) as p:
        img_list = p.map(parallel_add_map, pool_list)
    print("Creating movie")
    create_simple_mov(save_mov_path, img_list, "with_map")
    return img_list


if __name__ == "__main__":
    main_movie()
    # main_single()
    # only_movie()
