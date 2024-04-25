from ImageSet import ImageSet
import cv2 as cv
import numpy as np
from global_settings import *
import matplotlib.pyplot as plt
from typing import List
from numba import jit

camera_function_filename = 'pcalib.txt'
exposures_filename = 'times.txt'
save_path_dir = 'imagesets'


def get_files_and_exposures(data_dir: Path, image_type: str):

    exposure_list_file = data_dir.joinpath(exposures_filename)
    file_data_list = []

    with open(exposure_list_file) as f:
        for line in f:

            text = line.rstrip()
            text_elements = text.split()
            file_name = text_elements[0] + image_type
            exposure_time = text_elements[-1]

            file_data = (file_name, exposure_time)

            if not file_data_list:

                file_data_sublist = [file_data]
                file_data_list.append(file_data_sublist)
                continue

            number_of_sublists = len(file_data_list)
            for i, sublist in enumerate(file_data_list):
                data_element = sublist[0]
                if data_element[-1] == exposure_time:
                    sublist.append(file_data)
                    break
                if number_of_sublists - 1 - i == 0:
                    additional_list = [file_data]
                    file_data_list.append(additional_list)

    return file_data_list


def average_frames(file_data_list: List[List[tuple[str, str]]],
                   data_path: Path):

    save_path = data_path.joinpath(save_path_dir)

    for sublist in file_data_list:

        frame_stack = None
        number_of_frames = 0
        exposure_time = 0

        for data_element in sublist:

            if frame_stack is None:

                frame_stack = cv.imread(str(data_path.joinpath(data_element[0]))).astype(np.float64) / MAX_DN
                number_of_frames += 1
                exposure_time = data_element[-1]

            else:

                frame_stack += cv.imread(str(data_path.joinpath(data_element[0]))).astype(np.float64) / MAX_DN
                number_of_frames += 1

        frame_stack /= number_of_frames
        externalSet = ImageSet()

        externalSet.acq = frame_stack
        externalSet.path = save_path.joinpath(f'5x {exposure_time}ms gamma BF.tif')
        externalSet.save_8bit()

    return


def collect_noise_data(file_data_list: List[List[tuple[str, str]]],
                       data_path: Path,
                       use_mean: bool):

    data_array = np.zeros((BITS, BITS), dtype=np.uint64)
    save_path = data_path.joinpath(save_path_dir)

    for sublist in file_data_list:

        frame_stack = None
        number_of_frames = 0

        for data_element in sublist:

            frame = cv.imread(str(data_path.joinpath(data_element[0]))).astype(np.uint8)
            frame = frame[:, :, 0]
            frame = np.expand_dims(frame, axis=2)

            if frame_stack is None:

                frame_stack = frame
                number_of_frames += 1

            else:

                frame_stack = np.concatenate([frame_stack, frame], axis=2)
                number_of_frames += 1

        data_array += calculate_mean_data(frame_stack, use_mean)

    if use_mean:
        np.savetxt(save_path.joinpath('mean_data.txt'), data_array, fmt='%i')
    else:
        np.savetxt(save_path.joinpath('modal_data.txt'), data_array, fmt='%i')

    return


@jit(nopython=True)
def calculate_mean_data(frame_stack, use_mean):
    """
    Calculates the histogram of a pixel position in an image, for each channel
    separately. The histogram is summed into a bits*bits*channels data array on
    the row whose index equals the mean of that histogram.

    :param frame_stack: list with one element for each channel. The
    elements contain single channel images stacked into an im_size_y * im_size_x
    * number of images Numpy array.

    :return: the calculated bits*bits*channels shaped mean data array.
    """

    data_array = np.zeros((BITS, BITS), dtype=np.uint64)
    if use_mean:
        weight = np.linspace(MIN_DN, MAX_DN, num=BITS)

    rows, cols, frames = np.shape(frame_stack)

    for row in range(rows):
        for col in range(cols):

            hist = np.histogram(
                frame_stack[row, col, :], BITS, (MIN_DN, MAX_DN))[0]

            if use_mean:
                row = round(np.sum(np.multiply(hist, weight)) / np.sum(hist))
            else:
                row = np.argmax(hist)

            data_array[row, :] = np.add(data_array[row, :], hist)

    return data_array


def process_camera_function(camera_function_dir: Path):

    camera_function = rd.read_data_from_txt(str(camera_function_filename), str(camera_function_dir)) / MAX_DN
    original_data_points = np.size(camera_function)
    name = 'ICRF_external.png'

    x_new = np.linspace(0, 1, num=BITS)
    x_old = np.linspace(0, 1, num=original_data_points)
    interpolated_camera_function = np.zeros((BITS, CHANNELS), dtype=float)

    if original_data_points != DATAPOINTS:

        for c in range(CHANNELS):
            y_old = camera_function
            interpolated_camera_function[:, c] = np.interp(x_new, x_old, y_old)

    else:
        for c in range(CHANNELS):
            interpolated_camera_function[:, c] = camera_function

    np.savetxt(camera_function_dir.joinpath('ICRF_external.txt'), interpolated_camera_function)

    plt.plot(x_new, interpolated_camera_function[:, 0], color='b')
    plt.plot(x_new, interpolated_camera_function[:, 1], color='g')
    plt.plot(x_new, interpolated_camera_function[:, 2], color='r')
    plt.savefig(camera_function_dir.joinpath(name), dpi=300)
    plt.clf()

    return

if __name__ == "__main__":

    # external_data_path1 = Path(r'E:\calib_narrowGamma_sweep3')
    # process_camera_function(external_data_path1)
    # file_data_list = get_files_and_exposures(external_data_path1, '.jpg')
    # average_frames(file_data_list, external_data_path1)

    # external_data_path2 = Path(r'E:\calib_wideGamma_sweep1')
    # process_camera_function(external_data_path2)
    # file_data_list = get_files_and_exposures(external_data_path2, '.png')
    # average_frames(file_data_list, external_data_path2)

    # external_data_path3 = Path(r'E:\calib_wideGamma_sweep2')
    # process_camera_function(external_data_path3)
    # file_data_list = get_files_and_exposures(external_data_path3, '.png')
    # average_frames(file_data_list, external_data_path3)

    # external_data_path1 = Path(r'E:\calib_narrowGamma_sweep3')
    # file_data_list = get_files_and_exposures(external_data_path1, '.jpg')
    # collect_noise_data(file_data_list, external_data_path1, True)

    external_data_path1 = Path(r'E:\calib_wideGamma_sweep1')
    file_data_list = get_files_and_exposures(external_data_path1, '.png')
    collect_noise_data(file_data_list, external_data_path1, False)

    external_data_path2 = Path(r'E:\calib_wideGamma_sweep2')
    file_data_list = get_files_and_exposures(external_data_path2, '.png')
    collect_noise_data(file_data_list, external_data_path2, False)

    external_data_path3 = Path(r'E:\calib_narrowGamma_sweep3')
    file_data_list = get_files_and_exposures(external_data_path3, '.jpg')
    collect_noise_data(file_data_list, external_data_path3, False)

