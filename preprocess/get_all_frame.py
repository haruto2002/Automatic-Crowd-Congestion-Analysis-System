import subprocess
import os
import time
import argparse
import logging
import json
from datetime import datetime

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
    command = f"ffmpeg -i {path2video} -vcodec png {save_dir}/%04d.png"
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
    parser.add_argument("--path2video", type=str, default="demo.mov")
    parser.add_argument("--save_dir", type=str, default="demo/img")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    path2video = args.path2video
    save_dir = args.save_dir

    logger.info("=== Frame extraction process started ===")
    logger.info(f"Video file: {path2video}")
    logger.info(f"Save directory: {save_dir}")

    # Check input file existence
    if not os.path.exists(path2video):
        raise FileNotFoundError(path2video)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    # Get recording time
    # creation_time = get_video_creation_time(path2video)
    # if creation_time:
    #     logger.info(f"Video recording time: {creation_time}")

    #     # Save recording time to file
    #     time_info_path = os.path.join(save_dir, "video_creation_time.txt")
    #     with open(time_info_path, "w") as f:
    #         f.write(f"Recording time: {creation_time}\n")
    # else:
    #     logger.warning("Failed to get recording time")

    # Execute frame extraction
    get_frame(path2video, save_dir)
    frame_end = time.time()
    processing_time = frame_end - start

    # Record processing time
    time_log_path = os.path.join(save_dir, "time_get_frame.txt")
    with open(time_log_path, "w") as f:
        text = f"get frame:{processing_time}\n"
        f.write(text)


if __name__ == "__main__":
    main()
