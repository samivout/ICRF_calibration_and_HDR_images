import ImageSet as IS
import numpy as np
import read_data as rd
import cv2 as cv
import math
import os

im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')
acq_path = rd.read_config_single('acquired images path')
dark_path = rd.read_config_single('dark frames path')
flat_path = rd.read_config_single('flat fields path')
out_path = rd.read_config_single('corrected output path')


def dark_correction(imageSet_list, imageSet_dark_list):
    """
    Performs dark frame subtraction on acquired images based on exposure time.

    :param imageSet_list: List of acquired imageSet objects.

    :param imageSet_dark_list: List of dark frames as imageSet objects.

    :return: list of imageSets with dark frame correction.
    """
    for imageSet in imageSet_list:
        for darkSet in imageSet_dark_list:

            if imageSet.exp == darkSet.exp:
                imageSet.acq = cv.subtract(imageSet.acq, darkSet.acq)

    return imageSet_list


def fixed_pattern_correction(imageSet_list, ImageSet_flat_list):
    """
    Performs fixed pattern correction based on captured flat field images.

    :param imageSet_list: List of acquired images as imageSet objects.

    :param ImageSet_flat_list: List of flat field images as imageSet objects.

    :return: List of fixed-pattern-corrected imageSet objects.
    """
    flatField = ImageSet_flat_list[0]
    for imageSet in imageSet_list:
        for flatSet in ImageSet_flat_list:

            if imageSet.mag == flatSet.mag:
                if imageSet.ill == flatSet.ill:
                    imageSet.acq = cv.divide(imageSet.acq, flatField.acq)

                    # Determine flat field means
                    flat_field_mean_list = flat_field_mean(flatField, 0)

                    # Multiply images by the flat field spatial mean and clip
                    # 32-bit digital numbers to [0,1] range
                    imageSet.acq = np.clip(
                        multiply_per_channel(flat_field_mean_list,
                                             imageSet.acq), 0, 1)

                    # Normalize image to 8-bit range for RGB saving
                    imageSet.acq = cv.normalize(imageSet.acq, None, 0, 255,
                                                cv.NORM_MINMAX, cv.CV_8U)

    return imageSet_list


def flat_field_mean(imageSet_flat, flag):
    """
    Calculates the mean brightness of an image inside a centered ROI.

    :param imageSet_flat: imageSet object of a flat field image.

    :param flag: binary flag, 0 for acquired image, 1 for STD image.

    :return: list of mean image brightness inside ROI for each channel.
    """
    flat_field = np.zeros((im_size_x, im_size_y), dtype=float)
    if flag == 0:
        flat_field = imageSet_flat.acq
    if flag == 1:
        flat_field = imageSet_flat.std

    # Define ROI for calculating flat field spatial mean
    rows, columns, channels = flat_field.shape
    ROI_dx = math.floor(rows * 0.0666)
    ROI_dy = math.floor(columns * 0.0666)

    red_mean = np.mean(
        flat_field[7 * ROI_dx:8 * ROI_dx, 7 * ROI_dy:8 * ROI_dy, 2])
    green_mean = np.mean(
        flat_field[7 * ROI_dx:8 * ROI_dx, 7 * ROI_dy:8 * ROI_dy, 1])
    blue_mean = np.mean(
        flat_field[7 * ROI_dx:8 * ROI_dx, 7 * ROI_dy:8 * ROI_dy, 0])

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


def uncertainty(acqList, darkList, flatList):
    """
    Calculates the uncertainty associated with an acquired image and the
    applied corrections.

    :param acqList: List containing the ImageSet objects of all the main images
    of interest subject to correction.

    :param darkList: List containing the ImageSet objects of dark frames used in
    dark frame correction.

    :param flatList: List containing the ImageSet objects of flat frames used in
    fixed pattern correction.

    :return: List containing the ImageSet objects of the main images of interest
    with an .std property containing the uncertainty image.
    """

    for acqSet in acqList:
        try:
            flatSet = next(flatSet for flatSet in flatList if
                           acqSet.mag == flatSet.mag)
            darkSet = next(darkSet for darkSet in darkList if
                           acqSet.exp == darkSet.exp)

            acq = acqSet.acq
            ff = flatSet.acq
            d = darkSet.acq
            u_acq = acqSet.std
            u_ff = flatSet.std
            u_d = darkSet.std
            r, g, b = flat_field_mean(flatSet, 0)
            ur, ug, ub = flat_field_mean(flatSet, 1)

            u_acq_term = cv.divide(u_acq ** 2, ff ** 4)
            u_acq_term = multiply_per_channel([r ** 2, g ** 2, b ** 2],
                                              u_acq_term)

            u_d_term = cv.divide(u_d ** 2, ff ** 4)
            u_d_term = multiply_per_channel([r ** 2, g ** 2, b ** 2],
                                            u_d_term)

            u_ff_term = cv.divide((acq - d) ** 2, ff ** 4)
            u_ff_term = cv.multiply(u_ff_term, u_ff ** 2)
            u_ff_term = multiply_per_channel([r ** 2, g ** 2, b ** 2],
                                             u_ff_term)

            u_ffm_term = cv.divide((acq - d) ** 2, ff ** 2)
            u_ffm_term = multiply_per_channel([ur ** 2, ug ** 2, ub ** 2],
                                              u_ffm_term)

            acqSet.std = np.clip(u_acq_term + u_d_term + u_ff_term + u_ffm_term,
                                 0, 1)

            acqSet.std = cv.normalize(acqSet.std, None, 0, 255,
                                      cv.NORM_MINMAX,
                                      cv.CV_8U)

        except StopIteration:
            return acqList

        return acqList


def main():
    # Initialize image lists and name lists
    acq_list = IS.load_images(acq_path)
    dark_list = IS.load_images(dark_path)
    flat_list = IS.load_images(flat_path)

    # Dark correction
    ideal_acq_list = dark_correction(acq_list, dark_list)
    ideal_flat_list = dark_correction(flat_list, dark_list)

    # Flat field correction
    ideal_acq_list = fixed_pattern_correction(ideal_acq_list, ideal_flat_list)

    # Save the corrected images
    for imageset in ideal_acq_list:
        cv.imwrite(os.path.join(out_path, imageset.name), imageset.acq)

    del ideal_acq_list

    acq_list = uncertainty(acq_list, dark_list, flat_list)

    for imageset in acq_list:
        if not (imageset.std is None):
            cv.imwrite(os.path.join(out_path, imageset.name.removesuffix(
                '.tif') + ' STD.tif'), imageset.std)


main()
