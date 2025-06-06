import glob
from tqdm import tqdm
import cv2
import click
import os


def create_mov(save_dir, source_dir, dataset_name, weight_name):
    path2img_list = sorted(
        glob.glob(f"{source_dir}/plot/{dataset_name}/{weight_name}/*.png")
    )

    img = cv2.imread(path2img_list[0])
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    h, w, c = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        f"{save_dir}/{dataset_name}_{weight_name}.mp4", fourcc, 5, (w, h)
    )
    for path2img in tqdm(path2img_list[1:]):
        img = cv2.imread(path2img)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


def add_time(img, df, frame):
    text = df["time"][frame].strftime("%H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 7
    font_thickness = 10
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    position = (6500, 300)  # テキストの位置
    x, y = position

    # White background
    cv2.rectangle(
        img,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        thickness=cv2.FILLED,
    )

    cv2.putText(
        img, text, position, font, font_scale, text_color, thickness=font_thickness
    )

    return img


@click.command()
@click.argument("save_dir", type=str, default="vis")
@click.argument("source_dir", type=str, default="vis")
@click.argument("dataset_name", type=str)
@click.argument("weight_name", type=str)
def main(save_dir, source_dir, dataset_name, weight_name):
    save_dir = f"{save_dir}/movie/{dataset_name}/{weight_name}"
    os.makedirs(save_dir, exist_ok=True)
    create_mov(save_dir, source_dir, dataset_name, weight_name)


if __name__ == "__main__":
    main()
    # python vis_codes/create_mov.py WorldPorters2023 yokohama2023_70
