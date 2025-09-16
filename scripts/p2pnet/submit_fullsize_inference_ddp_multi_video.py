import json
import argparse
import subprocess
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--io_info_file", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--img_sub_dir", type=str, required=True)
    parser.add_argument("--out_sub_dir", type=str, required=True)
    parser.add_argument("--full_det_sub_dir", type=str, required=True)
    parser.add_argument("--node_type", type=str, required=True)
    parser.add_argument("--log_level", type=str, required=True)
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()

    script = "scripts/p2pnet/run_fullsize_inference_ddp.sh"
    if not os.path.exists(script):
        raise FileNotFoundError(script)

    with open(args.io_info_file, "r") as f:
        io_info = json.load(f)
    for video_path, save_dir in io_info.items():
        img_dir = os.path.join(save_dir, args.img_sub_dir)
        out_dir = os.path.join(save_dir, args.out_sub_dir)
        full_det_dir = os.path.join(save_dir, args.full_det_sub_dir)
        run_cmd = f"bash {script} {args.weight_path} {img_dir} {out_dir} {full_det_dir} {args.node_type} {args.log_level}"
        subprocess.run(run_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
