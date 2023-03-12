import ImageSet as IS
import image_calculation as ic
import numpy as np
import read_data as rd
import cv2 as cv
import math
import os

im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')
acq_path = rd.read_config_single('acquired images path')
flat_path = rd.read_config_single('flat fields path')
out_path = rd.read_config_single('corrected output path')
ICRF_calibrated_file = rd.read_config_single('calibrated ICRFs')
ICRF_calibrated = rd.read_data_from_txt(ICRF_calibrated_file)
ICRF_diff = np.zeros_like(ICRF_calibrated)
channels = rd.read_config_single('channels')
bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth-1
datapoints = rd.read_config_single('final datapoints')
STD_arr = rd.read_data_from_txt(rd.read_config_single('STD data'))


def separate_to_sublists(list_of_ImageSets):
    """
    Separates a list of ImageSet objects into sublists by their subject names,
    used magnification and used illumination type.
    :param list_of_ImageSets: list of ImageSet objects.
    :return: list of lists containing ImageSet objects.
    """
    list_of_sublists = []

    for imageSet in list_of_ImageSets:

        # Check if list_of_sublists is empty. If yes, create first sublist and
        # automatically add the first ImageSet object to it.
        if not list_of_sublists:

            sublist = [imageSet]
            list_of_sublists.append(sublist)
            continue

        number_of_sublists = len(list_of_sublists)
        for i in range(number_of_sublists):

            sublist = list_of_sublists[i]
            current_name = imageSet.name
            current_ill = imageSet.ill
            current_mag = imageSet.mag
            names_in_sublist = sublist[0].name
            ill_in_sublist = sublist[0].ill
            mag_in_sublist = sublist[0].mag
            if current_name == names_in_sublist and \
                    current_ill == ill_in_sublist and \
                    current_mag == mag_in_sublist:
                sublist.append(imageSet)
                break
            if number_of_sublists - 1 - i == 0:
                additional_list = [imageSet]
                list_of_sublists.append(additional_list)
                break

    return list_of_sublists


def hat_weight(x):

    if x <= 0.5:
        return x
    if x > 0.5:
        return 1-x


def create_HDR_absolute(list_of_ImageSets):

    hat_weight_vectorized = np.vectorize(hat_weight)

    number_of_imageSets = len(list_of_ImageSets)
    numerator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)
    denominator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)

    for i in range(number_of_imageSets):

        imageSet = list_of_ImageSets[i]
        weight_array = hat_weight_vectorized(imageSet.acq)
        numerator += weight_array * imageSet.acq * (1/imageSet.exp)
        denominator += weight_array

    HDR_acq = np.divide(numerator, denominator,
                        out=np.zeros_like(numerator),
                        where=denominator != 0)

    HDR_std = None

    number_of_STD_images = 0
    for imageSet in list_of_ImageSets:
        if imageSet.std is not None:
            number_of_STD_images += 1

    if number_of_STD_images == number_of_imageSets:

        numerator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)
        denominator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)

        for i in range(number_of_imageSets):
            imageSet = list_of_ImageSets[i]
            weight_array = hat_weight_vectorized(imageSet.acq)
            numerator += weight_array**2 * (imageSet.std**2) * (1 / imageSet.exp)
            denominator += weight_array

        HDR_std = np.sqrt(np.divide(numerator, denominator**2,
                            out=np.zeros_like(numerator),
                            where=denominator != 0))

    imageSet_HDR = IS.make_ImageSet(HDR_acq,
                                    HDR_std,
                                    list_of_ImageSets[0].file_name,
                                    None,
                                    list_of_ImageSets[0].mag,
                                    list_of_ImageSets[0].ill,
                                    list_of_ImageSets[0].name)

    return imageSet_HDR


def linearize_image_vectorized(imageSet):

    datapoint_step = int(datapoints / (max_DN + 1))
    sampled_ICRF = ICRF_calibrated[::datapoint_step, :]
    sampled_ICRF_diff = ICRF_diff[::datapoint_step, :]
    acq = (imageSet.acq * max_DN).astype(int)
    acq_new = np.zeros(np.shape(acq), dtype=float)
    std_new = np.zeros(np.shape(acq), dtype=float)
    for c in range(channels):

        # The ICRFs are in reverse order in the .txt file when compared
        # to how OpenCV opens the channels.
        acq_new[:, :, c] = sampled_ICRF[acq[:, :, c], channels-1-c]
        std_new[:, :, c] = sampled_ICRF_diff[acq[:, :, c], channels-1-c] * \
                           STD_arr[acq[:, :, c], channels-1-c]

    imageSet.acq = acq_new
    imageSet.std = std_new

    return imageSet


def process_HDR_images():

    # Determine the numerical derivative of the calibrated ICRFs.
    global ICRF_diff
    dx = 1/(datapoints-1)
    for c in range(channels):

        ICRF_diff[:, c] = np.gradient(ICRF_calibrated[:, c], dx)

    # Initialize image lists and name lists
    acq_list = IS.load_images(acq_path)
    flat_list = IS.load_images(flat_path)
    acq_list = ic.fixed_pattern_correction(acq_list, flat_list)
    linearized_list = []
    del flat_list

    for imageSet in acq_list:
        linearized_list.append(linearize_image_vectorized(imageSet))
    del acq_list
    '''
    IS.save_images_8bit(linearized_list, out_path)
    IS.save_images_32bit(linearized_list, out_path)
    '''
    list_of_linearized_sublists = separate_to_sublists(linearized_list)
    del linearized_list

    HDR_list = []
    for sublist in list_of_linearized_sublists:
        HDR_list.append(create_HDR_absolute(sublist))

    IS.save_images_8bit(HDR_list, out_path)
    IS.save_images_32bit(HDR_list, out_path)

    '''
    IS.save_images_8bit(linearized_list, out_path)
    IS.save_images_32bit(linearized_list, out_path)
    
    list_of_sublists = separate_to_sublists(linearized_list)
    del linearized_list

    HDR_images = []
    for sublist in list_of_sublists:
        HDRset = create_HDR(sublist)
        HDR_images.append(HDRset)
    '''
    '''
    acq_sublists_by_name = separate_to_sublists(acq_list)
    del acq_list

    linearized_list = []
    for sublist in acq_sublists_by_name:
        HDRset = create_HDR(sublist)
        linearized_list.append(HDRset)
    '''
    return


if __name__ == "__main__":

    print('Run script from actual main file!')
