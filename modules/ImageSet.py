import read_data as rd
import numpy as np
import cv2 as cv
import os
import re

bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth-1
STD_arr = rd.read_data_from_txt(rd.read_config_single('STD data'))
channels = rd.read_config_single('channels')

'''
In an arbitrary order the name should contain (exposure time)ms, (illumination
type as bf or df), (magnification)x and (your image name). Each descriptor
should be separated by a space and within a descriptor there should be no white
space. For example: '5ms BF sample_1 50x.tif'. Additionally if the image is an
uncertainty image, it should contain a separate 'STD' descriptor in it. Only
.tif support for now. Flat field images should have a name 'flat' in them
and dark frames should have 'dark' in them.
'''

im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')


class ImageSet(object):
    acq = np.zeros((im_size_x, im_size_y, channels), dtype=float)
    std = np.zeros((im_size_x, im_size_y, channels), dtype=float)
    file_name = ""

    def __init__(self, acq, std, file_name, exp, mag, ill, name):
        self.acq = acq
        self.std = std
        self.file_name = file_name
        self.exp = exp
        self.mag = mag
        self.ill = ill
        self.name = name

    @property
    def setAcq(self):
        return self.acq

    @property
    def setStd(self):
        return self.std

    @property
    def setFile_name(self):
        return self.file_name

    @property
    def setExp(self):
        return self.exp

    @property
    def setMag(self):
        return self.mag

    @property
    def setIll(self):
        return self.ill

    @property
    def setName(self):
        return self.name

    @setAcq.setter
    def setAcq(self, image):
        self.acq = image

    @setStd.setter
    def setStd(self, image):
        self.std = image

    @setFile_name.setter
    def setFile_name(self, file_name):
        self.file_name = file_name

    @setExp.setter
    def setExp(self, exp):
        self.exp = exp

    @setMag.setter
    def setMag(self, mag):
        self.mag = mag

    @setIll.setter
    def setIll(self, ill):
        self.ill = ill

    @setName.setter
    def setName(self, name):
        self.name = name


def make_ImageSet(acq, std, file_name, exp, mag, ill, name):
    imageSet = ImageSet(acq, std, file_name, exp, mag, ill, name)
    return imageSet


def load_images(path):
    """
    Load all images of a given path into a list of ImageSet objects.

    :param path: Absolute path to the directory of images

    :return: List of ImageSet objects.
    """

    list_of_ImageSets = []
    files = os.listdir(path)
    for file in files:
        if file.endswith(".tif"):
            file_name_array = file.removesuffix('.tif').split()
            if not ("STD" in file_name_array):

                acq = cv.imread(os.path.join(path, file))
                try:
                    # print(os.path.isfile(os.path.join(path, file.removesuffix(
                    #    '.tif') + ' STD.tif')))
                    std = cv.imread(os.path.join(path, file.removesuffix(
                        '.tif') + ' STD.tif')).astype(np.float32) / max_DN

                except FileNotFoundError:
                    std = calculate_numerical_STD(acq)
                except AttributeError:
                    std = calculate_numerical_STD(acq)

                acq = acq.astype(np.float32) / max_DN

                # print(file_name_array)
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
                imageSet = make_ImageSet(acq, std, file, exp, mag, ill, name)
                list_of_ImageSets.append(imageSet)

    return list_of_ImageSets


def calculate_numerical_STD(acq):

    STD_image = np.zeros((im_size_y, im_size_x, channels), dtype=float)

    for c in range(channels):

        STD_image[:, :, c] = STD_arr[acq[:, :, c], channels - 1 - c]

    return STD_image


def save_images_8bit(list_of_imageSets, path):

    for imageSet in list_of_imageSets:

        bit8_image = imageSet.acq
        max_float = np.amax(bit8_image)
        bit8_image /= max_float
        bit8_image = (bit8_image*max_DN).astype(int)
        path_name = os.path.join(path, imageSet.file_name)
        cv.imwrite(path_name, bit8_image)

        if imageSet.std is not None:

            bit8_image = (imageSet.std*max_DN).astype(int)
            cv.imwrite(path_name.removesuffix('.tif')+' STD.tif', bit8_image)

    return


def save_images_32bit(list_of_imageSets, path):

    for imageSet in list_of_imageSets:

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
