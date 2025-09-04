import subprocess
import glob
import os


def main():
    samples_dir = "samples"
    parent_dir_list = sorted(glob.glob(f"{samples_dir}/*"))
    for parent_dir in parent_dir_list:
        parent_dir_name = os.path.basename(parent_dir)
        child_dir_list = sorted(glob.glob(f"{parent_dir}/*"))
        for child_dir in child_dir_list:
            child_dir_name = os.path.basename(child_dir)
            video_list = sorted(glob.glob(f"{child_dir}/*.MOV"))
            if len(video_list) == 0:
                video_list = sorted(glob.glob(f"{child_dir}/*.MP4"))
            for video_path in video_list:
                video_name = video_path.split("/")[-1].split(".")[0]
                exp_name = f"{parent_dir_name}_{child_dir_name}_{video_name}"
                print(video_path)
                subprocess.run(
                    f"qsub -v EXP_NAME={exp_name},VIDEO_PATH={video_path} run_pipeline.sh",
                    shell=True,
                )


def main_custom():
    samples_dir = "samples"
    parent_dir_name = "2024_moji"
    child_dir_name = "02"
    child_dir = f"{samples_dir}/{parent_dir_name}/{child_dir_name}"
    video_list = sorted(glob.glob(f"{child_dir}/*.MOV"))
    if len(video_list) == 0:
        video_list = sorted(glob.glob(f"{child_dir}/*.MP4"))
    for video_path in video_list:
        video_name = video_path.split("/")[-1].split(".")[0]
        exp_name = f"{parent_dir_name}_{child_dir_name}_{video_name}"
        print(video_path)
        subprocess.run(
            f"qsub -v EXP_NAME={exp_name},VIDEO_PATH={video_path} run_pipeline.sh",
            shell=True,
        )


def main_fail():
    with open("false_list.txt", "r") as f:
        for line in f:
            line = line.strip()
            info = line.split("/")[-3]
            parent_dir_name = "_".join(info.split("_")[:-4])
            child_dir_name = "_".join(info.split("_")[-4:-3])
            video_name = "_".join(info.split("_")[-3:])
            exp_name = f"{parent_dir_name}_{child_dir_name}_{video_name}"
            video_path = (
                f"./samples/{parent_dir_name}/{child_dir_name}/{video_name}.MOV"
            )
            if not os.path.exists(video_path):
                video_path = (
                    f"./samples/{parent_dir_name}/{child_dir_name}/{video_name}.MP4"
                )
            if not os.path.exists(video_path):
                raise ValueError(f"Video path not found: {video_path}")
            cmd = f"qsub -v EXP_NAME={exp_name},VIDEO_PATH={video_path} run_pipeline.sh"
            print(cmd)
            subprocess.run(cmd, shell=True)


def check_completed():
    path2exp_parent_dir_list = sorted(glob.glob("outputs/*"))
    path2exp_child_dir_list = [
        sorted(glob.glob(f"{parent_dir}/*"))[-1]
        for parent_dir in path2exp_parent_dir_list
    ]
    path2log_list = [
        f"{path2exp_child_dir}/run_job_hydra.log"
        for path2exp_child_dir in path2exp_child_dir_list
        # if "2024_moji_02" in path2exp_child_dir
    ]

    fail_list = []
    for path2log in path2log_list:
        fail_lag = check(path2log)
        if fail_lag:
            fail_list.append(path2log)
    with open("false_list.txt", "w") as f:
        for path2log in fail_list:
            f.write(path2log + "\n")

    if len(fail_list) == 0:
        print("All jobs are completed")
    else:
        print(f"{len(fail_list)} jobs failed")


def check(log_file_path):
    keyword = "Pipeline completed successfully"
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if keyword in line:
                return False
    return True


def write_cmd_for_local():
    save_cmd_dir = "local_cmd_script/samples"
    os.makedirs(save_cmd_dir, exist_ok=True)
    local_save_dir = "/Users/haruto/Desktop/video_setting/samples_results"
    dir_list = sorted(glob.glob("outputs/*"))
    for dir in dir_list:
        dir_name = os.path.basename(dir)
        if "2024_moji_02" not in dir_name:
            continue
        result_dir = sorted(glob.glob(f"{dir}/*"))[-1]
        det_vis_path = f"{result_dir}/results/detection_vis_30"
        track_vis_path = f"{result_dir}/results/track_vis_5"
        absolute_det_vis_path = os.path.abspath(det_vis_path)
        absolute_track_vis_path = os.path.abspath(track_vis_path)
        cmd_mkldir = f"mkdir -p {local_save_dir}/{dir_name}/"
        command_det_vis = (
            f"scp -r abci:{absolute_det_vis_path} {local_save_dir}/{dir_name}/"
        )
        command_track_vis = (
            f"scp -r abci:{absolute_track_vis_path} {local_save_dir}/{dir_name}/"
        )
        with open(f"{save_cmd_dir}/{dir_name}.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd_mkldir + "\n")
            f.write(command_det_vis + "\n")
            f.write(command_track_vis)


if __name__ == "__main__":
    main()
    # main_custom()
    # check_completed()
    # main_fail()
    # write_cmd_for_local()
