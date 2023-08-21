import math

import read_data as rd
import numpy as np
import cv2 as cv
import os
import re
from typing import Optional

bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth-1
STD_arr = rd.read_data_from_txt(rd.read_config_single('STD data'))
channels = rd.read_config_single('channels')
im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')

'''
In an arbitrary order the name should contain (exposure time)ms, (illumination
type as bf or df), (magnification)x and (your image name). Each descriptor
should be separated by a space and within a descriptor there should be no white
space. For example: '5ms BF sample_1 50x.tif'. Additionally if the image is an
uncertainty image, it should contain a separate 'STD' descriptor in it. Only
.tif support for now. Flat field images should have a name 'flat' in them
and dark frames should have 'dark' in them.
'''


class ImageSet(object):

    def __init__(self, acq: Optional, std: Optional, file_name, exp, mag, ill, name, path):
        self.acq = acq
        self.std = std
        self.file_name = file_name
        self.exp = exp
        self.mag = mag
        self.ill = ill
        self.name = name
        self.path = path

        if acq is None:
            self.acq = None

        if std is None:
            self.std = None

    def load_acq(self):

        self.acq = cv.imread(os.path.join(self.path, self.file_name)).astype(np.float32) / max_DN

    def load_std(self, is_original: Optional[bool] = True):

        try:
            if is_original:
                self.std = cv.imread(os.path.join(self.path, self.file_name.removesuffix(
                    '.tif') + ' STD.tif')).astype(np.float32) / (max_DN * math.sqrt(67))
            else:
                self.std = cv.imread(
                    os.path.join(self.path, self.file_name.removesuffix(
                        '.tif') + ' STD.tif')).astype(np.float32) / max_DN
        except FileNotFoundError:
            self.std = calculate_numerical_STD((self.acq * max_DN).astype(np.dtype('uint8')))
        except AttributeError:
            self.std = calculate_numerical_STD((self.acq * max_DN).astype(np.dtype('uint8')))


def create_imageSets(path):
    """
    Load all images of a given path into a list of ImageSet objects but without
    acq or std images.

    :param path: Absolute path to the directory of images

    :return: List of ImageSet objects.
    """

    list_of_ImageSets = []
    files = os.listdir(path)
    for file in files:
        if file.endswith(".tif"):
            file_name_array = file.removesuffix('.tif').split()
            if not ("STD" in file_name_array):

                # print(file_name_array)
                acq = None
                std = None
                exp = None
                ill = None
                mag = None
                name = None

                for element in file_name_array:
                    if element.casefold() == 'bf' or element.casefold() == 'df':
                        ill = element
                    elif re.match("^[0-9]+.*[xX]$", element):
                        mag = element
                    elif re.match("^[0-9]+.*ms$", element):
                        exp = float(element.removesuffix('ms'))
                    else:
                        name = element
                imageSet = ImageSet(acq, std, file, exp, mag, ill, name, path)
                list_of_ImageSets.append(imageSet)

    return list_of_ImageSets


def calculate_numerical_STD(acq):

    STD_image = np.zeros((im_size_y, im_size_x, channels), dtype=(np.dtype('float32')))

    for c in range(channels):

        STD_image[:, :, c] = STD_arr[acq[:, :, c], channels - 1 - c]

    return STD_image


def save_image_8bit(imageSet, path):

    bit8_image = imageSet.acq
    max_float = np.amax(bit8_image)

    if max_float > 1:
        bit8_image /= max_float

    bit8_image = (bit8_image*max_DN).astype(np.dtype('uint8'))
    path_name = os.path.join(path, imageSet.file_name)
    cv.imwrite(path_name, bit8_image)

    if imageSet.std is not None:

        bit8_image = (imageSet.std*max_DN).astype(np.dtype('uint8'))
        cv.imwrite(path_name.removesuffix('.tif')+' STD.tif', bit8_image)

    return


def save_image_32bit(imageSet, path):

    bit32_image = imageSet.acq
    path_name = os.path.join(path, imageSet.file_name)
    cv.imwrite(path_name.removesuffix('.tif')+' blue.tif',
               bit32_image[:, :, 0])
    cv.imwrite(path_name.removesuffix('.tif') + ' green.tif',
               bit32_image[:, :, 1])
    cv.imwrite(path_name.removesuffix('.tif') + ' red.tif',
               bit32_image[:, :, 2])

    if imageSet.std is not None:

        bit32_image = imageSet.std
        cv.imwrite(path_name.removesuffix('.tif') + ' STD blue.tif',
                   bit32_image[:, :, 0])
        cv.imwrite(path_name.removesuffix('.tif') + ' STD green.tif',
                   bit32_image[:, :, 1])
        cv.imwrite(path_name.removesuffix('.tif') + ' STD red.tif',
                   bit32_image[:, :, 2])

    return
