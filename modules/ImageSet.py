import read_data as rd
import numpy as np
import cv2 as cv
import os

bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth

'''
Image naming convention: -Exposure time- -Relevant descriptors- -Magnification-
-STD-. Exposure time in ms, dark frames should have Dark in their descriptors,
BF and DF modes can be noted, magnification in 50x for example.
'''

im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')


class ImageSet(object):
    acq = np.zeros((im_size_x, im_size_y), dtype=float)
    std = np.zeros((im_size_x, im_size_y), dtype=float)
    name = ""

    def __init__(self, acq, std, name, exp, mag, ill):
        self.acq = acq
        self.std = std
        self.name = name
        self.exp = exp
        self.mag = mag
        self.ill = ill

    @property
    def setAcq(self):
        return self.acq

    @property
    def setStd(self):
        return self.std

    @property
    def setName(self):
        return self.name

    @property
    def setExp(self):
        return self.exp

    @property
    def setMag(self):
        return self.mag

    @property
    def setIll(self):
        return self.ill

    @setAcq.setter
    def setAcq(self, image):
        self.acq = image

    @setStd.setter
    def setStd(self, image):
        self.std = image

    @setName.setter
    def setName(self, name):
        self.name = name

    @setExp.setter
    def setExp(self, exp):
        self.exp = exp

    @setMag.setter
    def setMag(self, mag):
        self.mag = mag

    @setIll.setter
    def setIll(self, ill):
        self.ill = ill


def _make_ImageSet(acq, std, name, exp, mag, ill):
    imageSet = ImageSet(acq, std, name, exp, mag, ill)
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
            if not ("STD" in file):

                file_name_array = file.removesuffix('.tif').split()
                acq = cv.imread(os.path.join(path, file)).astype(
                    np.float32) / max_DN
                try:
                    std = cv.imread(os.path.join(path, file.removesuffix(
                        '.tif') + ' STD.tif')).astype(np.float32) / max_DN

                except FileNotFoundError:
                    std = None
                except AttributeError:
                    std = None

                print(file_name_array)
                exp = file_name_array[0]
                ill = file_name_array[1]
                mag = file_name_array[-1]
                imageSet = _make_ImageSet(acq, std, file, exp, mag, ill)
                list_of_ImageSets.append(imageSet)

    return list_of_ImageSets
