import numpy as np
from scipy import stats as st
import cv2 as cv
import general_functions as gf
from typing import Optional
from global_settings import *


def calculate_images(path: Path, use_mean: bool, use_std: bool,
                     use_median: bool, use_mode: bool):

    video, count = load_video(path)

    mean_image = None
    std_image = None
    median_image = None
    mode_image = None

    if use_mean:
        mean_image = calculate_mean_image(video)
    if use_std:
        std_image = calculate_std(video)
    if use_median:
        median_image = calculate_median_image(video)
    if use_mode:
        mode_image = calculate_mode_image(video)

    ret = {'mean': mean_image, 'std': std_image, 'median': median_image, 'mode': mode_image}

    return ret


def calculate_mode_image(video: np.ndarray):

    mode_image = st.mode(video, axis=-1, keepdims=False, dtype=np.dtype('float32'))

    return mode_image[0].astype(np.dtype('uint8'))


def calculate_std(video: np.ndarray):

    std_image = np.std(video, axis=-1, keepdims=False, dtype=np.dtype('float32'))

    return std_image.astype(np.dtype('uint8'))


def calculate_mean_image(video: np.ndarray):

    mean_image = np.mean(video, axis=-1, keepdims=False, dtype=np.dtype('float32'))

    return mean_image.astype(np.dtype('uint8'))


def calculate_median_image(video: np.ndarray):

    median_image = np.median(video, axis=-1, keepdims=False, dtype=np.dtype('float32'))

    return median_image.astype(np.dtype('uint8'))


def load_video(file_path: Path):

    stacked_array = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=np.dtype('uint8'))
    stacked_array = stacked_array[:, :, :, np.newaxis]

    video = cv.VideoCapture(str(file_path))
    video_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            continue

        if count == 0:
            stacked_array[:, :, :, 0] = frame
        else:
            frame = frame[:, :, :, np.newaxis]
            stacked_array = np.concatenate((stacked_array, frame), axis=3)
            print(np.shape(stacked_array))

        count += 1
        if count > video_frames:
            video.release()
            break

    return stacked_array, count


def process_video(video_path: Optional[Path] = None,
                  use_mean: Optional[bool] = True,
                  use_std: Optional[bool] = True,
                  use_median: Optional[bool] = False,
                  use_mode: Optional[bool] = False):

    if video_path is None:
        video_path = gf.get_path_dialog('Choose video file')
    ret = calculate_images(video_path, use_mean, use_std, use_median, use_mode)

    for key in ret:
        if ret[key] is not None:
            save_path = str(video_path.parent.joinpath(f'{key}.tif'))
            cv.imwrite(save_path, ret[key])


if __name__ == "__main__":

    process_video()
