import numpy as np
from scipy import stats as st
import cv2 as cv
from global_settings import *


def calculate_images(path: Path):
    mean_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=int)
    mode_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=int)
    median_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=int)

    video_files = path.glob("*.avi")

    for file in video_files:

        stacked_array, count = load_video(path, file)
        mean_image = calculate_mean_image(stacked_array)
        median_image = calculate_median_image(stacked_array)
        mode_image = calculate_mode_image(stacked_array)
        print(mode_image.shape)
        break

    return mean_image, median_image, mode_image


def calculate_mode_image(stacked_array):

    mode_image = st.mode(stacked_array, axis=-1, keepdims=False)

    return mode_image[0]


def calculate_mean_image(stacked_array):

    mean_image = np.mean(stacked_array, axis=-1)

    return mean_image.astype(int)


def calculate_median_image(stacked_array):

    median_image = np.median(stacked_array, axis=-1)

    return median_image.astype(int)


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


def process_video():

    mean_image, median_image, mode_image = calculate_images(VIDEO_PATH)
    cv.imwrite(VIDEO_PATH.joinpath('Mean.tif'), mean_image)
    cv.imwrite(VIDEO_PATH.joinpath('Median.tif'), median_image)
    cv.imwrite(VIDEO_PATH.joinpath('Mode.tif'), mode_image)


if __name__ == "__main__":

    mean_image1, median_image1, mode_image1 = calculate_images(VIDEO_PATH)
    cv.imwrite(VIDEO_PATH.joinpath('Mean.tif'), mean_image1)
    cv.imwrite(VIDEO_PATH.joinpath('Median.tif'), median_image1)
    cv.imwrite(VIDEO_PATH.joinpath('Mode.tif'), mode_image1)
