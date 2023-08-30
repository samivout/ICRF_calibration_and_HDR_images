import math
import numpy as np
import cv2 as cv
from typing import Optional
from global_settings import *

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

        self.acq = cv.imread(os.path.join(self.path, self.file_name)).astype(np.float32) / MAX_DN

    def load_std(self, is_original: Optional[bool] = True,
                 STD_data: Optional[np.ndarray] = None
                 ):

        try:
            if is_original:
                self.std = cv.imread(os.path.join(self.path, self.file_name.removesuffix(
                    '.tif') + ' STD.tif')).astype(np.float32) / (MAX_DN * math.sqrt(67))
            else:
                self.std = cv.imread(
                    os.path.join(self.path, self.file_name.removesuffix(
                        '.tif') + ' STD.tif')).astype(np.float32) / MAX_DN
        except (FileNotFoundError, AttributeError) as e:
            self.std = calculate_numerical_STD((self.acq * MAX_DN).astype(np.dtype('uint8')),
                                               STD_data)


def calculate_numerical_STD(acq: np.ndarray, STD_data: np.ndarray):

    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)
    STD_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=(np.dtype('float32')))

    for c in range(CHANNELS):

        STD_image[:, :, c] = STD_data[acq[:, :, c], CHANNELS - 1 - c]

    return STD_image
