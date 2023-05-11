import read_data as rd
import numpy as np
from scipy import stats as st
import cv2 as cv
import os

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
video_path = rd.read_config_single('video path')
im_size_x = rd.read_config_single('image size x')  # Columns of pixels
im_size_y = rd.read_config_single('image size y')  # Rows of pixels
bit_depth = rd.read_config_single('bit depth')
bits = 2 ** bit_depth
max_DN = bits-1
min_DN = 0
channels = rd.read_config_single('channels')


def calculate_images(path):
    mean_image = np.zeros((im_size_y, im_size_x, channels), dtype=int)
    mode_image = np.zeros((im_size_y, im_size_x, channels), dtype=int)
    median_image = np.zeros((im_size_y, im_size_x, channels), dtype=int)
    files = os.listdir(path)
    for file in files:
        if file.endswith(".avi"):
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


def load_video(path, file):

    stacked_array = np.zeros((im_size_y, im_size_x, channels), dtype=np.dtype('uint8'))
    stacked_array = stacked_array[:, :, :, np.newaxis]

    video = cv.VideoCapture(os.path.join(path, file))
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

    mean_image, median_image, mode_image = calculate_images(video_path)
    cv.imwrite(os.path.join(video_path, 'Mean.tif'), mean_image)
    cv.imwrite(os.path.join(video_path, 'Median.tif'), median_image)
    cv.imwrite(os.path.join(video_path, 'Mode.tif'), mode_image)


if __name__ == "__main__":

    mean_image1, median_image1, mode_image1 = calculate_images(video_path)
    cv.imwrite(os.path.join(video_path, 'Mean.tif'), mean_image1)
    cv.imwrite(os.path.join(video_path, 'Median.tif'), median_image1)
    cv.imwrite(os.path.join(video_path, 'Mode.tif'), mode_image1)
