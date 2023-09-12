import math
import numpy as np
import cv2 as cv
import re
from typing import Optional
from typing import List
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

    def __init__(self, file_path: Path):
        self.acq = None
        self.std = None
        self.path = file_path
        self.exp = None
        self.mag = None
        self.ill = None
        self.subject = None
        self.channels = None

        file_name_array = file_path.name.removesuffix('.tif').split()

        for element in file_name_array:
            if element.casefold() == 'bf' or element.casefold() == 'df':
                self.ill = element
            elif re.match("^[0-9]+.*[xX]$", element):
                self.mag = element
            elif re.match("^[0-9]+.*ms$", element):
                self.exp = float(element.removesuffix('ms'))
            else:
                self.subject = element

    def copy(self, channel: Optional[int | List[int]] = None):
        """
        Create a copy of an ImageSet object with specified channels
        Args:
            channel: integer or list of integers representing the channel(s).
                0=B, 1=G, 2=R. None to copy no channels.

        Returns:
            A new ImageSet object with copied data from the base ImageSet.
        """
        channels = None
        if type(channel) is int:
            channels = [channel]
        if type(channel) is list:
            channels = channel

        extracted_imageSet = ImageSet(self.path)

        if channels is None:
            return extracted_imageSet

        if self.acq is not None:
            extracted_imageSet.acq = np.take(self.acq, channels, axis=2)
            extracted_imageSet.channels = channels
        if self.std is not None:
            extracted_imageSet.std = np.take(self.std, channels, axis=2)

        return extracted_imageSet

    def load_acq(self):

        self.acq = cv.imread(str(self.path)).astype(np.float32) / MAX_DN
        self.channels = [0, 1, 2]

    def load_std(self, STD_data: Optional[np.ndarray] = None):
        """
        Loads the error image of an ImageSet object to memory.
        Args:
            STD_data: Numpy array representing the STD data of pixel values.
        """

        std_path = str(self.path).removesuffix('.tif') + ' STD.tif'
        try:
            self.std = cv.imread(std_path).astype(np.float32) / (MAX_DN * math.sqrt(67))

        except (FileNotFoundError, AttributeError) as e:
            self.std = calculate_numerical_STD((np.around(self.acq * MAX_DN)).astype(np.dtype('uint8')),
                                               STD_data)

    def save_32bit(self, save_path: Optional[Path] = None):
        """
            Saves an ImageSet object's acquired image and error image to disk to given
            path into separate BGR channels in 32-bit format.
            Args:
                save_path: Full absolute path to save location, including filename.
        """
        if save_path is None:
            file_path = self.path.parent.joinpath('32bit', self.path.name)
        else:
            file_path = save_path

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        file_path = str(file_path)

        bit32_image = self.acq
        cv.imwrite(file_path.removesuffix('.tif') + ' blue.tif', bit32_image[:, :, 0])
        cv.imwrite(file_path.removesuffix('.tif') + ' green.tif', bit32_image[:, :, 1])
        cv.imwrite(file_path.removesuffix('.tif') + ' red.tif', bit32_image[:, :, 2])

        if self.std is not None:
            bit32_image = self.std
            cv.imwrite(file_path.removesuffix('.tif') + ' STD blue.tif',
                       bit32_image[:, :, 0])
            cv.imwrite(file_path.removesuffix('.tif') + ' STD green.tif',
                       bit32_image[:, :, 1])
            cv.imwrite(file_path.removesuffix('.tif') + ' STD red.tif',
                       bit32_image[:, :, 2])

    def save_8bit(self, save_path: Optional[Path] = None):
        """
        Saves an ImageSet object's acquired image and error image to disk to given
        path in an 8-bits per channel format.
        Args:
            save_path: Dir path where to save images, name is supplied by the ImageSet
                object
        """
        if save_path is None:
            file_path = self.path.parent.joinpath('8bit', self.path.name)
        else:
            file_path = save_path

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        file_path = str(file_path)

        bit8_image = self.acq
        max_float = np.amax(bit8_image)

        if max_float > 1:
            bit8_image /= max_float

        bit8_image = (np.around(bit8_image * MAX_DN)).astype(np.dtype('uint8'))
        cv.imwrite(file_path, bit8_image)

        if self.std is not None:
            bit8_image = (np.around(self.std * MAX_DN * math.sqrt(67))).astype(np.dtype('uint8'))
            cv.imwrite(file_path.removesuffix('.tif') + ' STD.tif', bit8_image)


def calculate_numerical_STD(acq: np.ndarray, STD_data: np.ndarray):

    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)
    STD_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=(np.dtype('float32')))

    for c in range(CHANNELS):

        STD_image[:, :, c] = STD_data[acq[:, :, c], CHANNELS - 1 - c]

    return STD_image
