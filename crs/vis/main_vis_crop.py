import argparse
from display import display_hm_on_back_img
from display_track import create_track_img
from display_vec import create_vec_img

"""1. create back img (display_vec.py or display_track.py)"""
"""2. add map (add_map.py)"""
"""3. add heatmap (this file)"""


def get_parser_for_display():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap_dir")
    parser.add_argument("--back_img_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--save_mov_name")
    parser.add_argument("--movie_fps", type=int, default=15)
    return parser


def get_parser_for_backimg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_vec_frames", type="store_true")
    parser.add_argument("--create_track_frames", type="store_true")
    parser.add_argument("--place", default="WorldPorters_noon")
    parser.add_argument("--crop_area", default=[130, 250, 130 + 350, 250 + 550])
    parser.add_argument("--frame_range", default=(1, 8990))
    parser.add_argument(
        "--frame_save_dir", default="danger/track_vis/WorldPorters_noon/frames"
    )
    parser.add_argument("--saparate_run", default=True)
    parser.add_argument("--part_num", default=1000)
    parser.add_argument("--freq", default=10)
    return parser


def get_parser():
    parser = argparse.ArgumentParser()

    # for back img
    parser.add_argument("--create_back_img", action="store_true")
    parser.add_argument("--create_vec_frames", action="store_true")
    parser.add_argument("--create_track_frames", action="store_true")
    parser.add_argument("--place", default="WorldPorters_noon")
    parser.add_argument("--crop_area", default=[130, 250, 130 + 350, 250 + 550])
    parser.add_argument("--frame_range", default=(1, 8990))
    parser.add_argument(
        "--back_img_save_dir", default="danger/track_vis/WorldPorters_noon/frames"
    )
    parser.add_argument("--saparate_run", default=True)
    parser.add_argument("--part_num", default=1000)
    parser.add_argument("--freq", default=10)

    # for display heatmap on back img
    parser.add_argument("--display_on_map", action="store_true")
    parser.add_argument("--heatmap_dir")
    parser.add_argument("--back_img_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--save_mov_name")
    parser.add_argument("--movie_fps", type=int, default=15)

    return parser


def main_create_back_track_img(args):
    # parser = get_parser_for_backimg()
    # args = parser.parse_args()
    if args.create_vec_frames and args.create_track_frames:
        raise ValueError(
            "create_vec_frames and create_track_frames cannot be True at the same time"
        )
    if args.create_vec_frames:
        print("Creating vec frames...")
        print("place: ", args.place)
        print("crop_area: ", args.crop_area)
        print("frame_range: ", args.frame_range)
        print("frame_save_dir: ", args.frame_save_dir)
        print("saparate_run: ", args.saparate_run)
        print("part_num: ", args.part_num)
        print("freq: ", args.freq)
        create_vec_img(
            args.place,
            args.crop_area,
            args.frame_range,
            args.frame_save_dir,
            saparate_run=args.saparate_run,
            part_num=args.part_num,
            freq=args.freq,
        )
    if args.create_track_frames:
        print("Creating track frames...")
        print("place: ", args.place)
        print("crop_area: ", args.crop_area)
        print("frame_range: ", args.frame_range)
        print("frame_save_dir: ", args.frame_save_dir)
        print("saparate_run: ", args.saparate_run)
        print("part_num: ", args.part_num)
        print("freq: ", args.freq)
        create_track_img(
            args.place,
            args.crop_area,
            args.frame_range,
            args.frame_save_dir,
            saparate_run=args.saparate_run,
            part_num=args.part_num,
            freq=args.freq,
        )


def main_display_on_map(args):
    # parser = get_parser_for_display()
    # args = parser.parse_args()
    print("Displaying heatmap on back img...")
    print("save_dir: ", args.save_dir)
    print("back_img_dir: ", args.back_img_dir)
    print("heatmap_dir: ", args.heatmap_dir)
    print("save_mov_name: ", args.save_mov_name)
    print("movie_fps: ", args.movie_fps)
    display_hm_on_back_img(
        args.save_dir,
        args.back_img_dir,
        args.heatmap_dir,
        args.save_mov_name,
        fps=args.movie_fps,
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.create_back_img:
        main_create_back_track_img(args)
    if args.display_on_map:
        main_display_on_map(args)


if __name__ == "__main__":
    main()
