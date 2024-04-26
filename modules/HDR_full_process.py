import matplotlib.pyplot as plt
from ImageSet import ImageSet
from numba import jit, prange
import numpy as np
import cv2 as cv
from typing import Optional
from typing import List
import general_functions as gf
from global_settings import *
import copy
import math

lower = LOWER_LIN_LIM/MAX_DN
upper = UPPER_LIN_LIM/MAX_DN


def get_dark_frame(imageSet: ImageSet, darkSet_list: List[ImageSet]):

    if imageSet.exp >= DARK_THRESHOLD:
        lesser_exp = False
        greater_exp = False
        lesser_index = 0
        greater_index = 0

        for i, darkSet in enumerate(darkSet_list):

            if darkSet.exp < imageSet.exp:
                lesser_exp = True
                lesser_index = i

            if darkSet.exp > imageSet.exp:
                greater_exp = True
                greater_index = i

            if imageSet.exp == darkSet.exp:
                return darkSet

            if lesser_exp is True and greater_exp is True:
                interpolated_darkSet = gf.interpolate_frames(
                    darkSet_list[lesser_index], darkSet_list[greater_index],
                    imageSet.exp)
                return interpolated_darkSet

    return None


def bad_pixel_filter(imageSet: ImageSet, darkSet: ImageSet):
    """
    Replace hot pixels with surrounding mean value.
    :param imageSet: ImageSet object of image being corrected.
    :param darkSet: ImageSet object of dark frame used to map bad pixels.
    :return: Corrected ImageSet.
    """
    kernel = np.ones((5, 5), dtype=np.dtype('float64')) / 25
    convolved_image = cv.filter2D(imageSet.acq, -1, kernel)

    @jit(nopython=True, parallel=True)
    def the_loop(acq, dark):
        for c in prange(CHANNELS):
            for i in range(IM_SIZE_Y):
                for j in range(IM_SIZE_X):

                    if dark[i, j, c] > 0.02:
                        acq[i, j, c] = convolved_image[i, j, c]

        return acq

    imageSet.acq = the_loop(imageSet.acq, darkSet.acq)

    return imageSet


def flat_field_mean(flatSet, flag):
    """
    Calculates the mean brightness of an image inside a centered ROI.
    :param flatSet: imageSet object of a flat field image.
    :param flag: binary flag, 0 for acquired image, 1 for STD image.
    :return: list of mean image brightness inside ROI for each channel.
    """
    if flag == 0:
        flat_field = flatSet.acq
    if flag == 1:
        flat_field = flatSet.std

    # Define ROI for calculating flat field spatial mean
    ROI_dx = math.floor(IM_SIZE_X * 0.047)
    ROI_dy = math.floor(IM_SIZE_Y * 0.047)

    red_mean = np.mean(
        flat_field[10 * ROI_dx:11 * ROI_dx, 10 * ROI_dy:11 * ROI_dy, 2])
    green_mean = np.mean(
        flat_field[10 * ROI_dx:11 * ROI_dx, 10 * ROI_dy:11 * ROI_dy, 1])
    blue_mean = np.mean(
        flat_field[10 * ROI_dx:11 * ROI_dx, 10 * ROI_dy:11 * ROI_dy, 0])

    return [blue_mean, green_mean, red_mean]


def flat_field_correction(imageSet: ImageSet, flatSet: ImageSet):

    # Determine flat field means
    b, g, r = flat_field_mean(flatSet, 0)
    ub, ug, ur = flat_field_mean(flatSet, 1)
    acq = imageSet.acq
    ff = flatSet.acq
    u_acq = imageSet.std
    u_ff = flatSet.std

    # Uncertainty
    u_acq_term = np.divide(u_acq ** 2, ff ** 2)
    u_acq_term = multiply_per_channel([b ** 2, g ** 2, r ** 2],
                                      u_acq_term)

    u_ff_term = np.divide(acq ** 2, ff ** 4)
    u_ff_term = np.multiply(u_ff_term, u_ff ** 2)
    u_ff_term = multiply_per_channel([b ** 2, g ** 2, r ** 2],
                                     u_ff_term)

    u_ffm_term = np.divide(acq ** 2, ff ** 2)
    u_ffm_term = multiply_per_channel([ub ** 2, ug ** 2, ur ** 2],
                                      u_ffm_term)

    imageSet.std = np.sqrt(u_acq_term + u_ff_term + u_ffm_term)

    # Flat field correction
    imageSet.acq = imageSet.acq / flatSet.acq
    imageSet.acq = multiply_per_channel([b, g, r], imageSet.acq)

    return imageSet


def multiply_per_channel(flat_field_mean_list, image):
    """
    Multiplies each image channel with a separate scalar value.
    :param flat_field_mean_list: List of the flat field means.
    :param image: Image, whose channels are to be multiplied. Image is a Numpy
        array.
    :return: Image, whose channels have been multiplied by the flat field means.
    """
    image[:, :, 2] = image[:, :, 2] * flat_field_mean_list[2]
    image[:, :, 1] = image[:, :, 1] * flat_field_mean_list[1]
    image[:, :, 0] = image[:, :, 0] * flat_field_mean_list[0]

    return image


def weight(x: float):

    # Previously used value: 30.
    # Normal distribution: np.e ** (-0.5 * ((x-0.5) / 0.1) ** 2)
    y = np.e ** (-30 * (x-0.5) ** 2)
    dydx = -2 * 30 * (x - 0.5) * y

    return y, dydx


vectorized_weight = np.vectorize(weight)


def calculate_HDR_image(acq_sublist: List[ImageSet], dark_list: List[ImageSet],
                        sum_of_weights: np.ndarray, square_sum_of_weights: np.ndarray,
                        ICRF: np.ndarray, ICRF_diff: np.ndarray):

    darkSet = get_dark_frame(acq_sublist[0], dark_list)
    imageSet_HDR = copy.deepcopy(acq_sublist[0])

    HDR_arr = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=np.dtype('float64'))
    HDR_std = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=np.dtype('float64'))

    for imageSet in acq_sublist:

        imageSet.load_acq()
        imageSet.load_std()
        delta_b = copy.deepcopy(imageSet.std)

        if darkSet is not None:

            imageSet = bad_pixel_filter(imageSet, darkSet)

        w, dw = vectorized_weight(imageSet.acq)
        imageSet = gf.linearize_ImageSet(imageSet, ICRF, ICRF_diff, False, True)
        g = imageSet.acq
        dg = imageSet.std
        t = imageSet.exp

        HDR_arr += (w * g) / (sum_of_weights * t)
        HDR_std += (((dw * g + w * dg)/sum_of_weights - (dw * w * g)/square_sum_of_weights) * delta_b / t) ** 2

        del imageSet.acq
        del imageSet.std

    HDR_std = np.sqrt(HDR_std)
    imageSet_HDR.acq = HDR_arr
    imageSet_HDR.std = HDR_std

    return imageSet_HDR


def precalculate_sum_of_weights(acq_sublist: List[ImageSet]):

    sum_of_weights = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=(np.dtype('float64')))

    for imageSet in acq_sublist:

        imageSet.load_acq()
        sum_of_weights += vectorized_weight(imageSet.acq)[0]
        del imageSet.acq

    squared_sum_of_weight = sum_of_weights ** 2

    return sum_of_weights, squared_sum_of_weight


def process_HDR_images(image_path: Optional[Path] = DEFAULT_ACQ_PATH,
                       fix_artifacts: Optional[bool] = True,
                       ICRF: Optional[np.ndarray] = None,
                       STD_data: Optional[np.ndarray] = None,
                       gaussian_blur: Optional[bool] = False):

    white_point_correction_multipliers = [0.93017, 0.95174, 1.14395]

    if ICRF is None:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)

    ICRF_diff = np.zeros_like(ICRF)
    x_range = np.linspace(0, 1, BITS)
    dx = 2/(BITS - 1)
    for c in range(CHANNELS):

        ICRF_diff[:, c] = np.gradient(ICRF[:, c], dx)
        plt.plot(x_range, ICRF_diff[:, c])

    np.savetxt(OUTPUT_DIRECTORY.joinpath('ICRF_diff.txt'), ICRF_diff)
    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_diff.png'), dpi=300)

    # Initialize image lists and name lists
    acq_list = gf.create_imageSets(image_path)
    acq_sublists = gf.separate_to_sublists(acq_list)
    del acq_list

    dark_list = gf.create_imageSets(DARK_PATH)
    flat_list = gf.create_imageSets(FLAT_PATH)

    for sublist in acq_sublists:

        sum_of_weights, square_sum_of_weights = precalculate_sum_of_weights(sublist)
        HDR_imageSet = calculate_HDR_image(sublist, dark_list, sum_of_weights, square_sum_of_weights,
                                           ICRF, ICRF_diff)

        try:
            flatSet = next(flatSet for flatSet in flat_list if
                           HDR_imageSet.mag == flatSet.mag and HDR_imageSet.ill == flatSet.ill)
            flatSet.load_acq()
            flatSet.load_std()
            HDR_imageSet = flat_field_correction(HDR_imageSet, flatSet)

        except StopIteration:
            continue

        for c in range(CHANNELS):

            HDR_imageSet.acq[:, :, c] = HDR_imageSet.acq[:, :, c] * white_point_correction_multipliers[c]
            HDR_imageSet.std[:, :, c] = HDR_imageSet.std[:, :, c] * white_point_correction_multipliers[c]

        HDR_imageSet.save_32bit(OUT_PATH.joinpath(HDR_imageSet.path.name), is_HDR=True, separate_channels=False)
        print(f'Saved {HDR_imageSet.path}')

    return


if __name__ == "__main__":

    x = np.linspace(0, 1, num=BITS)
    sum = 0
    sum2 = 0
    for element in x:
        sum += weight(element)
    print(sum)

    print('Run script from actual main file!')
