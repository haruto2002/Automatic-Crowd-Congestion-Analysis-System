import numpy as np
import os
import glob
from tqdm import tqdm
import yaml
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def create_full_detection(source_dir, place, save_dir):
    full_img_size, patch_size = load_info(f"{source_dir}/{place}")
    patch_size = patch_size[::-1]
    frame_list = sorted(glob.glob(f"{source_dir}/{place}/detection/*"))
    pool_list = []
    poool_size = os.cpu_count()
    for frame_path in tqdm(frame_list):
        frame_name = frame_path.split("/")[-1]
        patch_list = glob.glob(f"{frame_path}/data/*.txt")
        frame_save_path = f"{save_dir}/{frame_name}.txt"
        pool_list.append((patch_list, patch_size, frame_save_path))

    with Pool(poool_size) as p:
        p.map(parallel_merge_patch, pool_list)


def load_info(source_dir):
    with open(f"{source_dir}/input_info.yaml", "r") as f:
        info = yaml.safe_load(f)
    full_img_size = info["img_size"]
    img_size = info["separate_size"]
    return full_img_size, img_size


def parallel_merge_patch(pool_list):
    patch_list, patch_size, frame_save_path = pool_list
    merge_patch(patch_list, patch_size, frame_save_path)


def merge_patch(patch_list, patch_size, frame_save_path):
    full_det = []
    for patch_path in patch_list:
        patch_pos = (
            int(os.path.basename(patch_path).split(".")[0].split("_")[1]),
            int(os.path.basename(patch_path).split(".")[0].split("_")[0]),
        )
        det = np.loadtxt(patch_path, delimiter=",")
        if len(det) == 0:
            continue
        if det.ndim == 1:
            det = det.reshape(1, det.shape[0])
        det[:, 0] = det[:, 0] + patch_pos[0] * patch_size[0]
        det[:, 1] = det[:, 1] + patch_pos[1] * patch_size[1]
        full_det.extend(det)
    full_det = np.array(full_det)
    np.savetxt(frame_save_path, full_det)


def main():
    source_dir = "inference_0328"
    place = "WorldPorters_noon"
    save_dir = f"byte_track/full_detection/{place}"
    os.makedirs(save_dir, exist_ok=True)
    create_full_detection(source_dir, place, save_dir)

def run_multi_place():
    source_dir = "inference_0328"
    place_list=["WorldPorters_noon","WorldPorters_night","WorldPorters_night2"]
    for place in place_list[2:]:
        save_dir=f"byte_track/full_detection/{place}"
        os.makedirs(save_dir)
        create_full_detection(source_dir,place,save_dir)


if __name__ == "__main__":
    run_multi_place()
