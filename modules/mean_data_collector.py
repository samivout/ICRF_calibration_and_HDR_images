import read_data as rd
import numpy as np
import cv2 as cv
import os

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
video_path = rd.read_config_single('video path')
im_size_x = rd.read_config_single('image size x')  # Columns of pixels
im_size_y = rd.read_config_single('image size y')  # Rows of pixels
mean_data_files = rd.read_config_list('camera mean data')
bit_depth = rd.read_config_single('bit depth')
bits = 2 ** bit_depth
max_DN = bits-1
min_DN = 0
channels = rd.read_config_single('channels')


def _process_videos(path):
    """
    Overview the data collection process. Load one video file at a time, process
    it and add the data to a collective data array after each video file.

    :param path: absolute path to the directory containing the video files from
        which the data will be collected.

    :return: Array containing the digital value distributions. First row
    contains the histogram of the pixels whose mean signal is zero. The array is
    shaped as 256*256*3 for 8 bit RGB images for example.
    """
    data_array = np.zeros((bits, bits, channels), dtype=int)
    files = os.listdir(path)

    for file in files:
        if file.endswith(".avi"):

            image_list = _load_images(path, file)
            channel_separated_images = _separate_channels(image_list)

            data_array += _calculate_mean_data(channel_separated_images)

    return data_array


def _load_images(path, file):
    """
    Capture a frame from a video file and add it to a list of images.

    :param path: absolute path to the directory of the video file

    :param file: the name of the video file to be processed

    :return: list of frames in the video. Each frame is a Numpy array.
    """
    image_list = []

    video = cv.VideoCapture(os.path.join(path, file))
    video_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            continue
        image_list.append(frame)
        count = count + 1
        if count > (video_frames - 1):
            video.release()
            break

    return image_list


def _separate_channels(images):
    """
    Stack the same channels of each image in the list into one array

    :param images: List of images as Numpy arrays

    :return: List containing a stack of images as Numpy arrays. One element in
        list per channel.
    """
    first_image = images[0]
    blue_arr = first_image[:, :, 0]
    green_arr = first_image[:, :, 1]
    red_arr = first_image[:, :, 2]
    blue_arr = blue_arr[:, :, np.newaxis]
    green_arr = green_arr[:, :, np.newaxis]
    red_arr = red_arr[:, :, np.newaxis]

    for n in range(1, len(images)):
        image = images[n]
        for c in range(channels):
            imCh = image[:, :, c, np.newaxis]
            if c == 0:
                blue_arr = np.append(blue_arr, imCh, axis=2)
            if c == 1:
                green_arr = np.append(green_arr, imCh, axis=2)
            if c == 2:
                red_arr = np.append(red_arr, imCh, axis=2)

    return [blue_arr, green_arr, red_arr]


def _calculate_mean_data(separated_channels_list):
    """
    Calculates the histogram of a pixel position in an image, for each channel
    separately. The histogram is summed into a bits*bits*channels data array on
    the row whose index equals the mean of that histogram.

    :param separated_channels_list: list with one element for each channel. The
    elements contain single channel images stacked into an im_size_y * im_size_x
    * number of images Numpy array.

    :return: the calculated bits*bits*channels shaped mean data array.
    """

    data_array = np.zeros((bits, bits, channels), dtype=int)
    weight = np.linspace(min_DN, max_DN, num=bits)

    for index, channel in enumerate(separated_channels_list):
        for row in range(im_size_y):
            for col in range(im_size_x):

                hist = np.histogram(
                    channel[row, col, :], bits, (min_DN, max_DN))[0]

                # row = round(np.sum(np.multiply(hist, weight)) / np.sum(hist))
                row = np.argmax(hist)

                data_array[row, :, index] += hist

    return data_array


def collect_mean_data():
    """
    Main function called from outside the module to run the mean data collection
    process and save the data into three different files.
    :return:
    """

    data_array = _process_videos(video_path)
    for index, file_name in enumerate(mean_data_files):

        np.savetxt(file_name, data_array[:, :, index], fmt='%i')


if __name__ == "__main__":

    data_matrix = _process_videos(video_path)
    for i, file in enumerate(mean_data_files):
        np.savetxt(file, data_matrix[:, :, i], fmt='%i')
