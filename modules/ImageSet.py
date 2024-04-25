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

    def __init__(self, file_path: Optional[Path] = None, acq: Optional[np.ndarray] = None):
        self.acq = acq
        self.std = None
        self.path = file_path
        self.exp = None
        self.mag = None
        self.ill = None
        self.subject = None
        self.channels = None

        if self.path is not None:
            file_name_array = file_path.name.removesuffix('.tif').split()

            for element in file_name_array:
                if element.casefold() == 'bf' or element.casefold() == 'df':
                    self.ill = element
                elif re.match("^[0-9]+.*[xX]$", element):
                    self.mag = element
                elif re.match("^[0-9]+.*ms$", element):
                    self.exp = float(element.removesuffix('ms'))/1000
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

    def load_acq(self, bit32: Optional[bool] = False):

        if not bit32:
            self.acq = cv.imread(str(self.path)).astype(np.float64) / MAX_DN
        else:
            self.acq = cv.imread(str(self.path), cv.IMREAD_UNCHANGED)
        self.channels = [0, 1, 2]

    def load_single_acq_to_multiple(self):

        acq = cv.imread(str(self.path)).astype(np.float64) / MAX_DN
        self.acq = np.concatenate((acq, acq))
        self.acq = np.concatenate((self.acq, acq))
        self.channels = [0, 1, 2]

    def load_std(self, STD_data: Optional[np.ndarray] = None, bit32: Optional[bool] = False):
        """
        Loads the error image of an ImageSet object to memory.
        Args:
            bit32: whether the image to load is already in float or not
            STD_data: Numpy array representing the STD data of pixel values.
        """

        std_path = str(self.path).removesuffix('.tif') + ' STD.tif'
        try:
            # if not bit32:
            #     self.std = np.sqrt(cv.imread(std_path).astype(np.float64) / (7 * (AVERAGED_FRAMES - 1) * MAX_DN * AVERAGED_FRAMES))
            if not bit32:
                self.std = cv.imread(std_path, cv.IMREAD_UNCHANGED).astype(np.float64)
            else:
                self.std = cv.imread(std_path, cv.IMREAD_UNCHANGED).astype(np.float64)

        except (FileNotFoundError, AttributeError) as e:
            self.std = calculate_numerical_STD((np.around(self.acq * MAX_DN)).astype(np.dtype('uint8')),
                                               STD_data)

    def to_lbgr(self):
        """
        Converts imageSet object's images from sRGB to linear RGB.
        """

        lows = (self.acq <= 0.04045)
        highs = np.logical_not(lows)

        self.acq[lows] = self.acq[lows] / 12.92
        self.acq[highs] = ((self.acq[highs] + 0.055) / 1.055) ** 2.4

        np.clip(self.acq, 0, 1)

    def to_sbgr(self):
        """
        Covnerts imageSet object's images from linear RGB to sRGB.
        """

        lows = (self.acq <= 0.0031308)
        highs = np.logical_not(lows)

        self.acq[lows] = self.acq[lows] * 12.92
        self.acq[highs] = 1.055 * self.acq[highs] ** (1 / 2.4) - 0.055

        np.clip(self.acq, 0, 1)

    def save_32bit(self, save_path: Optional[Path] = None,
                   is_HDR: Optional[bool] = False,
                   separate_channels: Optional[bool] = False):
        """
        Saves an ImageSet object's acquired image and error image to disk to given
        path into separate BGR channels in 32-bit format.
        Args:
            is_HDR: whether to add HDR to the end of the filename or not.
            save_path: Full absolute path to save location, including filename.
        """
        if save_path is None:
            file_path = self.path.parent.joinpath('32bit', self.path.name)
        else:
            file_path = save_path

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        file_path = str(file_path)

        if is_HDR:
            acq_file_suffix = ' HDR.tif'
            std_file_suffix = ' HDR STD.tif'
        else:
            acq_file_suffix = '.tif'
            std_file_suffix = 'STD.tif'

        if not separate_channels:

            bit32_image = self.acq.astype(np.dtype('float64'))
            cv.imwrite(file_path.removesuffix('.tif') + acq_file_suffix, bit32_image)

            if self.std is not None:
                bit32_image = self.std.astype(np.dtype('float64'))
                cv.imwrite(file_path.removesuffix('.tif') + std_file_suffix, bit32_image)

        else:
            channel_name = ['blue', 'green', 'red']
            for c in range(CHANNELS):

                bit32_image = self.acq[:, :, c]
                cv.imwrite(file_path.removesuffix('.tif')
                           + acq_file_suffix.replace('.tif', f' {channel_name[c]}.tif'), bit32_image)

                if self.std is not None:
                    bit32_image = self.std[:, :, c]
                    cv.imwrite(file_path.removesuffix('.tif')
                               + std_file_suffix.replace('.tif', f' {channel_name[c]}.tif'), bit32_image)

    def save_8bit(self, save_path: Optional[Path] = None,
                  sbgr: Optional[bool] = True):
        """
        Saves an ImageSet object's acquired image and error image to disk to given
        path in an 8-bits per channel format.
        Args:
            sbgr: Whether to save image as sbgr or lbgr.
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
        if sbgr:
            pass
            # self.to_sbgr()

        bit8_image = self.acq
        max_float = np.amax(bit8_image)

        if max_float > 1:
            bit8_image /= max_float

        bit8_image = (np.around(bit8_image * MAX_DN)).astype(np.dtype('uint8'))
        cv.imwrite(file_path, bit8_image)

        if self.std is not None:
            # bit8_image = (np.around((self.std ** 2) * MAX_DN * AVERAGED_FRAMES * (AVERAGED_FRAMES - 1) * 7)).astype(np.dtype('uint8'))
            bit8_image = self.std
            cv.imwrite(file_path.removesuffix('.tif') + ' STD.tif', bit8_image)


def calculate_numerical_STD(acq: np.ndarray, STD_data: np.ndarray):

    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)
    STD_image = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=(np.dtype('float64')))

    for c in range(CHANNELS):

        STD_image[:, :, c] = STD_data[acq[:, :, c], c] / np.sqrt(AVERAGED_FRAMES)

    return STD_image
