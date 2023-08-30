import cv2 as cv
import numpy as np
import re
import copy
from ImageSet import ImageSet
from typing import List
from typing import Optional
from global_settings import*


def separate_to_sublists(list_of_ImageSets: List[ImageSet]):
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


def divide_imageSets(num: ImageSet, den: ImageSet,
                     use_std: Optional[bool] = False,
                     lower: Optional[float] = 0,
                     upper: Optional[float] = 1):
    """
    Calculates the division of two images using ImageSet objects. Can also
    calculate error of the division. At pixels whose value for either the
    numerator or denominator fall outside the lower and upper limits, the result
    pixel value will be set to zero.
    Args:
        num: the numerator ImageSet
        den: the denominator ImageSet
        use_std: whether to calculate error or not, defaults to False
        lower: lower limit of values to include in division
        upper: upper limit of values to include in division

    Returns:
        Numpy array of the division result and if desired, the error of the
        division.
    """

    division = np.divide(num.acq, den.acq,
                         out=np.zeros_like(num.acq),
                         where=((lower < den.acq) & (den.acq < upper) &
                                (lower < num.acq) & (num.acq < upper)))

    if not use_std:
        return division

    u_x = np.divide(num.std, den.acq, out=np.zeros_like(num.acq),
                    where=((lower < den.acq) & (den.acq < upper) &
                           (lower < num.acq) & (num.acq < upper)))

    u_y = np.divide(num.acq * den.std, den.acq ** 2,
                    out=np.zeros_like(num.acq),
                    where=((lower < den.acq) & (den.acq < upper) &
                           (lower < num.acq) & (num.acq < upper)))

    division_std = np.sqrt(u_x ** 2 + u_y ** 2)

    return division, division_std


def choose_evenly_spaced_points(array: np.ndarray, num_points: int):
    """
    Select points evenly in a Numpy array.
    Args:
        array: Input array
        num_points: 'number' of points to choose

    Returns: Sampled array
    """
    # Calculate the step size between points
    step = max(1, int(array.shape[0] / (num_points - 1)))

    # Select the evenly spaced points
    points = array[::step, ::step]

    return points


def interpolate_frames(x0, x1, exp):
    """
    Simple linear interpolation of two frames by exposure time.
    :param x0: Lower exposure ImageSet object.
    :param x1: Higher exposure ImageSet object.
    :param exp: exposure time at which to perform the interpolation.
    :return: Interpolated ImageSet.
    """
    interpolated_imageSet = copy.deepcopy(x0)
    interpolated_imageSet.acq = (x0.acq * (x1.exp - exp) + x1.acq * (
            exp - x0.exp)) / (x1.exp - x0.exp)

    return interpolated_imageSet


def linearize_image_vectorized(imageSet: ImageSet,
                               ICRF: np.ndarray,
                               ICRF_diff: Optional[np.ndarray] = None,
                               STD_arr: Optional[np.ndarray] = None):
    """
    Linearizes an input image using an ICRF. Additionally, if a derivative of
    the ICRF and an array representing the standard deviations of a pixel value
    is supplied, an error image is calculated for the linearized image.
    Args:
        imageSet: imageSet object whose, acquired image is linearized
        ICRF: ICRF as numpy array
        ICRF_diff: derivative of ICRF as numpy array
        STD_arr: camera STD data as numpy array

    Returns: the input ImageSet object with linearized acq and possibly new
        error image.
    """

    use_std = False
    if ICRF_diff is not None and STD_arr is not None:
        use_std = True

    acq = (imageSet.acq * MAX_DN).astype(int)
    acq_new = np.zeros(np.shape(acq), dtype=np.dtype('float32'))
    if use_std:
        std_new = np.zeros(np.shape(acq), dtype=np.dtype('float32'))
    for c in range(CHANNELS):

        # The ICRFs are in reverse order in the .txt file when compared
        # to how OpenCV opens the channels.
        acq_new[:, :, c] = ICRF[acq[:, :, c], c]
        if use_std:
            std_new[:, :, c] = ICRF_diff[acq[:, :, c], c] * STD_arr[acq[:, :, c], c]

    imageSet.acq = acq_new
    if use_std:
        imageSet.std = std_new

    return imageSet


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


def save_image_8bit(imageSet: ImageSet, path):
    """
    Saves an ImageSet object's acquired image and error image to disk to given
    path in an 8-bits per channel format.
    Args:
        imageSet: ImageSet object, whose images to save.
        path: Dir path where to save images, name is supplied by the ImageSet
            object
    """
    bit8_image = imageSet.acq
    max_float = np.amax(bit8_image)

    if max_float > 1:
        bit8_image /= max_float

    bit8_image = (bit8_image * MAX_DN).astype(np.dtype('uint8'))
    path_name = os.path.join(path, imageSet.file_name)
    cv.imwrite(path_name, bit8_image)

    if imageSet.std is not None:

        bit8_image = (imageSet.std * MAX_DN).astype(np.dtype('uint8'))
        cv.imwrite(path_name.removesuffix('.tif')+' STD.tif', bit8_image)

    return


def save_image_32bit(imageSet: ImageSet, path):
    """
    Saves an ImageSet object's acquired image and error image to disk to given
    path into separate BGR channels in 32-bit format.
    Args:
        imageSet: ImageSet object, whose images to save.
        path: Dir path where to save images, name is supplied by the ImageSet
            object
    """
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
