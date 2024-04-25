from ImageSet import ImageSet
import numpy as np
import cv2 as cv
import math
from numba import jit, prange
from typing import List
from typing import Optional
import general_functions as gf
from global_settings import *


def dark_correction(imageSet: ImageSet, darkSet_list: List[ImageSet]):
    """
    Performs dark frame subtraction on acquired images based on exposure time.
    :param imageSet: ImageSet object.
    :param darkSet_list: List of dark frames as ImageSet objects.
    :return: ImageSet with dark frame correction.
    """
    # Bias frame subtraction
    # bias = darkSet_list[0]
    # imageSet.acq = imageSet.acq - bias.acq

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
                imageSet = bad_pixel_filter(imageSet, darkSet)
                break

            if lesser_exp is True and greater_exp is True:
                interpolated_darkSet = gf.interpolate_frames(
                    darkSet_list[lesser_index], darkSet_list[greater_index],
                    imageSet.exp)
                imageSet = bad_pixel_filter(imageSet, interpolated_darkSet)
                break

    return imageSet


def bad_pixel_filter(imageSet: ImageSet, darkSet: ImageSet):
    """
    Replace hot pixels with surrounding mean value.
    :param imageSet: ImageSet object of image being corrected.
    :param darkSet: ImageSet object of dark frame used to map bad pixels.
    :return: Corrected ImageSet.
    """
    kernel = np.ones((5, 5), dtype=np.float64) / 25
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


def fixed_pattern_correction(imageSet, flatSet_list):
    """
    Performs fixed pattern correction based on captured flat field images.
    :param imageSet: List of acquired images as imageSet objects.
    :param flatSet_list: List of flat field images as imageSet objects.
    :return: List of fixed-pattern-corrected imageSet objects.
    """
    for flatSet in flatSet_list:

        if imageSet.mag == flatSet.mag:
            if imageSet.ill == flatSet.ill:
                imageSet.acq = np.divide(imageSet.acq, flatSet.acq)

                # Determine flat field means
                flat_field_mean_list = flat_field_mean(flatSet, 0)

                # Multiply images by the flat field spatial mean and clip
                # 32-bit digital numbers to [0,1] range
                # imageSet.acq = np.clip(
                #     multiply_per_channel(flat_field_mean_list,
                #                          imageSet.acq), 0, 1)
                multiply_per_channel(flat_field_mean_list, imageSet.acq)

    return imageSet


def flat_field_mean(flatSet, flag):
    """
    Calculates the mean brightness of an image inside a centered ROI.
    :param flatSet: imageSet object of a flat field image.
    :param flag: binary flag, 0 for acquired image, 1 for STD image.
    :return: list of mean image brightness inside ROI for each channel.
    """
    flat_field = np.zeros((IM_SIZE_X, IM_SIZE_Y), dtype=float)
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


def uncertainty(acqSet: ImageSet, flatList: List[ImageSet]):
    """
    Calculates the uncertainty associated with an acquired image and the
    applied corrections.
    :param acqSet: ImageSet object to be corrected.
    :param flatList: List containing the ImageSet objects of flat frames used in
        fixed pattern correction.
    :return: List containing the ImageSet objects of the main images of interest
        with an .std attribute containing the uncertainty image.
    """

    try:
        flatSet = next(flatSet for flatSet in flatList if
                       acqSet.mag == flatSet.mag and acqSet.ill == flatSet.ill)

        acq = acqSet.acq
        ff = flatSet.acq
        u_acq = acqSet.std
        u_ff = flatSet.std
        b, g, r = flat_field_mean(flatSet, 0)
        ub, ug, ur = flat_field_mean(flatSet, 1)

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

        acqSet.std = np.sqrt(u_acq_term + u_ff_term + u_ffm_term)

    except StopIteration:

        return acqSet

    return acqSet


def uncertainty_full(acqSet: ImageSet, darkList: List[ImageSet], flatList: List[ImageSet]):
    """
    Calculates the uncertainty associated with an acquired image and the
    applied corrections.
    :param acqSet: ImageSet object to be corrected.
    :param darkList: List containing the ImageSet objects of dark frames used in
        dark frame correction.
    :param flatList: List containing the ImageSet objects of flat frames used in
        fixed pattern correction.
    :return: List containing the ImageSet objects of the main images of interest
        with an .std attribute containing the uncertainty image.
    """

    try:
        flatSet = next(flatSet for flatSet in flatList if
                       acqSet.mag == flatSet.mag and acqSet.ill == flatSet.ill)
        darkSet = darkList[0]

        acq = acqSet.acq
        ff = flatSet.acq
        d = darkSet.acq
        u_acq = acqSet.std
        u_ff = flatSet.std
        u_d = darkSet.std
        b, g, r = flat_field_mean(flatSet, 0)
        ub, ug, ur = flat_field_mean(flatSet, 1)

        u_acq_term = cv.divide(u_acq ** 2, ff ** 4)
        u_acq_term = multiply_per_channel([b ** 2, g ** 2, r ** 2],
                                          u_acq_term)

        u_d_term = cv.divide(u_d ** 2, ff ** 4)
        u_d_term = multiply_per_channel([b ** 2, g ** 2, r ** 2],
                                        u_d_term)

        u_ff_term = cv.divide((acq - d) ** 2, ff ** 4)
        u_ff_term = cv.multiply(u_ff_term, u_ff ** 2)
        u_ff_term = multiply_per_channel([b ** 2, g ** 2, r ** 2],
                                         u_ff_term)

        u_ffm_term = cv.divide((acq - d) ** 2, ff ** 2)
        u_ffm_term = multiply_per_channel([ub ** 2, ug ** 2, ur ** 2],
                                          u_ffm_term)

        acqSet.std = np.clip(u_acq_term + u_d_term + u_ff_term + u_ffm_term,
                             0, 1)

    except StopIteration:

        return acqSet

    return acqSet


def calibrate_flats():
    """
    Function to calibrate flat frames, i.e. bias subtraction.
    :return:
    """
    darkList = gf.create_imageSets(DARK_PATH)
    darkList.sort(key=lambda darkSet: darkSet.exp)
    og_flatList = gf.create_imageSets(OG_FLAT_PATH)

    bias = darkList[0]
    bias.load_acq()
    bias.load_std()

    for flatSet in og_flatList:
        flatSet.load_acq()
        flatSet.load_std()

        flatSet.acq = flatSet.acq - bias.acq
        flatSet.std = np.sqrt(flatSet.std ** 2 + bias.std ** 2)

        flatSet.save_8bit(FLAT_PATH)


def calibrate_dark_frames():
    """
    Function that handles calibration of raw dark frames, i.e. bias subtraction.
    :return:
    """
    og_darkList = gf.create_imageSets(OG_DARK_PATH)
    og_darkList.sort(key=lambda darkSet: darkSet.exp)
    bias = og_darkList[0]
    bias.load_acq()
    bias.load_std()

    number_of_darks = len(og_darkList)
    for i in range(1, number_of_darks):
        og_darkList[i].load_acq()
        og_darkList[i].load_std()
        og_darkList[i].acq = og_darkList[i].acq - bias.acq

        og_darkList[i].std = np.sqrt(og_darkList[i].std ** 2 + bias.std ** 2)
        og_darkList[i].save_8bit(DARK_PATH)


def image_correction(acq_list: Optional[List[ImageSet]] = None,
                     dark_list: Optional[List[ImageSet]] = None,
                     flat_list: Optional[List[ImageSet]] = None,
                     save_to_file: Optional[bool] = False):
    """
    Main handler of image corrections for a batch of acquired images.
    :param acq_list: List containing images to correct as ImageSet objects.
    :param dark_list: List of dark frames as ImagesSet objects.
    :param flat_list: List of flat frames as ImageSet objects.
    :param save_to_file: Boolean representing if images are to be saved on disk
    :return: None or acq_list
    """
    # Initialize image lists and name lists
    acq_list_preloaded = False
    if acq_list is None:
        acq_list = gf.create_imageSets(ACQ_PATH)
    if acq_list[0].acq is not None:
        acq_list_preloaded = True

    dark_list_preloaded = False
    if dark_list is None:
        dark_list = gf.create_imageSets(DARK_PATH)
    if dark_list[0].acq is not None:
        dark_list_preloaded = True

    flat_list_preloaded = False
    if flat_list is None:
        flat_list = gf.create_imageSets(FLAT_PATH)
    if flat_list[0].acq is not None:
        flat_list_preloaded = True

    if not dark_list_preloaded:
        for darkSet in dark_list:
            darkSet.load_acq()
            # darkSet.load_std() # Not used in current version

    if not flat_list_preloaded:
        for flatSet in flat_list:
            flatSet.load_acq()
            flatSet.load_std()

    dark_list.sort(key=lambda darkSet: darkSet.exp)

    for acqSet in acq_list:
        if not acq_list_preloaded:
            acqSet.load_acq()
            acqSet.load_std()

        # Uncertainty
        acqSet = uncertainty(acqSet, flat_list)

        # Dark correction
        acqSet = dark_correction(acqSet, dark_list)

        # Flat field correction
        acqSet = fixed_pattern_correction(acqSet, flat_list)

        # Save the corrected images
        if save_to_file:
            acqSet.save_8bit(OUT_PATH.joinpath(acqSet.path.name))
            acqSet.acq = None
            acqSet.std = None

    # Pass corrected images if no need to save on disk.
    if not save_to_file:
        return acq_list

    return


if __name__ == "__main__":
    print('Run script from actual main file!')
