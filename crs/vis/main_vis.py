import argparse
from display import display_hm_on_back_img
from back_bev import main as back_bev


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bev_heatmap_dir")
    parser.add_argument("--path2bev_back_matrix")
    parser.add_argument("--reprojected_heatmap_dir")
    parser.add_argument("--back_img_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--save_mov_name")
    parser.add_argument("--movie_fps", type=int, default=15)
    return parser


def main_back_bev():
    parser = get_parser()
    args = parser.parse_args()

    print("Creating reprojected heatmap...")
    print("bev_heatmap_dir: ", args.bev_heatmap_dir)
    print("path2bev_back_matrix: ", args.path2bev_back_matrix)
    print("reprojected_heatmap_dir: ", args.reprojected_heatmap_dir)
    back_bev(
        args.reprojected_heatmap_dir,
        args.bev_heatmap_dir,
        path2matrix=args.path2bev_back_matrix,
    )

    print("Displaying reprojected heatmap on back img...")
    print("back_img_dir: ", args.back_img_dir)
    print("reprojected_heatmap_dir: ", args.reprojected_heatmap_dir)
    print("save_dir: ", args.save_dir)
    print("save_mov_name: ", args.save_mov_name)
    print("movie_fps: ", args.movie_fps)
    display_hm_on_back_img(
        args.save_dir,
        args.back_img_dir,
        args.reprojected_heatmap_dir,
        args.save_mov_name,
        fps=args.movie_fps,
    )


if __name__ == "__main__":
    main_back_bev()
