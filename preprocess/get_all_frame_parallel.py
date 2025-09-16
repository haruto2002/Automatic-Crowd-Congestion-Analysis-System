import subprocess
import os
import time
import argparse
import logging
import json
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_video_creation_time(path2video):
    """
    Function to get recording time of video file

    Args:
        path2video (str): Path to video file

    Returns:
        str: Recording time (ISO format string), None if not available
    """
    logger.info(f"Getting recording time for video file {path2video}...")

    # Get metadata using ffprobe
    command = (
        f'ffprobe -v quiet -print_format json -show_format -show_streams "{path2video}"'
    )
    logger.debug(f"Executing command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            metadata = json.loads(result.stdout)

            # Try multiple methods to get recording time
            creation_time = None

            # Method 1: creation_time from format tags
            if "format" in metadata and "tags" in metadata["format"]:
                tags = metadata["format"]["tags"]
                if "creation_time" in tags:
                    creation_time = tags["creation_time"]
                    logger.info(
                        f"Got recording time from format.tags.creation_time: {creation_time}"
                    )

            # Method 2: creation_time from first video stream
            if not creation_time and "streams" in metadata:
                for stream in metadata["streams"]:
                    if stream.get("codec_type") == "video":
                        if "tags" in stream and "creation_time" in stream["tags"]:
                            creation_time = stream["tags"]["creation_time"]
                            logger.info(
                                f"Got recording time from video stream.tags.creation_time: {creation_time}"
                            )
                            break

            # Method 3: File creation time (fallback)
            if not creation_time:
                file_stat = os.stat(path2video)
                creation_time = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                logger.info(
                    f"Using file creation time as recording time: {creation_time}"
                )

            return creation_time

        else:
            logger.error(f"Error occurred while getting metadata: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Exception occurred while getting recording time: {e}")
        return None


def get_frame(path2video, save_dir):
    logger.info(f"Extracting frames from video file {path2video}...")
    command = f"ffmpeg -i {path2video} -q:v 1 {save_dir}/%04d.jpg"
    # command = f"ffmpeg -i {path2video} -vcodec png {save_dir}/%04d.png"
    logger.debug(f"Executing command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Frame extraction completed successfully")
        else:
            logger.error(f"Error occurred during frame extraction: {result.stderr}")
            raise RuntimeError(result.stderr)
    except Exception as e:
        logger.error(f"Exception occurred during frame extraction: {e}")
        raise RuntimeError(e)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--io_info_file",
        type=str,
        default="parallel_hydra/parallel_time_test/2025-09-16_12-44-12/IO_info/IO_info.json",
    )
    parser.add_argument("--img_dir_name", type=str, default="debug_img")
    parser.add_argument(
        "--node_type", type=str, default="rt_HC", choices=["rt_HF", "rt_HG", "rt_HC"]
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def run_parallel(pool_list):
    path2video, save_dir = pool_list
    run_single(path2video, save_dir)


def run_single(path2video, save_dir):
    # Check input file existence
    if not os.path.exists(path2video):
        raise FileNotFoundError(path2video)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    # Get recording time
    creation_time = get_video_creation_time(path2video)
    if creation_time:
        time_info_path = os.path.join(save_dir, "video_creation_time.txt")
        with open(time_info_path, "w") as f:
            f.write(f"Recording time: {creation_time}\n")
    else:
        logger.warning("Failed to get recording time")

    # Execute frame extraction
    get_frame(path2video, save_dir)
    frame_end = time.time()
    processing_time = frame_end - start

    # Record processing time
    time_log_path = os.path.join(save_dir, "time_get_frame.txt")
    with open(time_log_path, "w") as f:
        text = f"get frame:{processing_time}\n"
        f.write(text)


def main():
    args = get_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    io_info_dict = json.load(open(args.io_info_file, "r"))

    pool_list = []
    for path2video, save_dir in io_info_dict.items():
        img_save_dir = os.path.join(save_dir, args.img_dir_name)
        pool_list.append((path2video, img_save_dir))

    if args.node_type == "rt_HF":
        pool_size = 192 // 2
    elif args.node_type == "rt_HG":
        pool_size = 16 // 2
    elif args.node_type == "rt_HC":
        pool_size = 32 // 2

    with Pool(pool_size) as p:
        list(tqdm(p.imap_unordered(run_parallel, pool_list), total=len(pool_list)))


if __name__ == "__main__":
    main()
