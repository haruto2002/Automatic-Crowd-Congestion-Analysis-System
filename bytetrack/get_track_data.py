import numpy as np
import cv2
import glob


def get_all_track(place, start_frame, end_frame, crop_area=None):
    # crop_area >> [xmin,ymin,xmax,ymax]
    txt_files = sorted(
            glob.glob(f"byte_track/track_data/{place}/interpolated_frame_data/*.txt")
        )
    track_list = [
        np.loadtxt(path2txt,delimiter=",") for path2txt in txt_files[start_frame - 1 : end_frame]
    ]
    all_track = []
    for i, track in enumerate(track_list):
        frame = start_frame + i
        track = np.concatenate([np.full((len(track), 1), frame), track], axis=1)
        all_track += list(track)
    all_track = np.array(all_track)

    path2img = sorted(glob.glob(f"tracking_results/{place}/img/*.png"))[end_frame - 1]
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

        img = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]

    # all_track=[[frame,id,x,y],...]
    return all_track, img


