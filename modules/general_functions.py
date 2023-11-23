import numpy as np
import copy
import webview
from ImageSet import ImageSet
from typing import List
from typing import Optional
from scipy.ndimage.filters import gaussian_filter
from joblib import delayed, parallel
import cv2 as cv
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
            current_name = imageSet.subject
            current_ill = imageSet.ill
            current_mag = imageSet.mag
            names_in_sublist = sublist[0].subject
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


def copy_nested_list(list_of_sublists: List[List[ImageSet]],
                     channel: Optional[int | List[int]] = None):
    """
    Create a copy of a nested list of ImageSet lists with single channels.
    Args:
        list_of_sublists: nested list to be copied.
        channel: integer or list of integers representing the channel(s) to be
            copied. None to copy no channels.

    Returns: nested list of ImageSet objects with single channel acq and std.
    """
    ret_list = []

    for sublist in list_of_sublists:

        ret_sublist = []

        for imageSet in sublist:

            extracted_set = imageSet.copy(channel)
            ret_sublist.append(extracted_set)

        ret_list.append(ret_sublist)

    return ret_list


def divide_imageSets(im0: ImageSet, im1: ImageSet | float,
                     use_std: Optional[bool] = False,
                     lower: Optional[float] = None,
                     upper: Optional[float] = None):
    """
    Calculates the division of two images using ImageSet objects. Can also
    calculate error of the division. At pixels whose value for either the
    numerator or denominator fall outside the lower and upper limits, the result
    pixel value will be set to NaN.
    Args:
        im0: the numerator ImageSet
        im1: the denominator ImageSet or float constant.
        use_std: whether to calculate error or not, defaults to False
        lower: lower limit of values to include in division
        upper: upper limit of values to include in division

    Returns:
        New ImageSet object with num as the copy base.
    """
    x0 = im0.acq
    both_imageSets = type(im1) is ImageSet

    if both_imageSets:
        x1 = im1.acq
        if lower is not None and upper is not None:
            where = (lower < x1) & (x1 < upper) & (lower < x0) & (x0 < upper)
        else:
            where = True
    else:
        x1 = im1
        if lower is not None and upper is not None:
            where = (lower < x0) & (x0 < upper)
        else:
            where = True

    division = np.divide(x0, x1, out=np.full_like(x0, np.nan), where=where)
    division_std = None

    if use_std:

        x0_std = im0.std

        u_x0 = np.divide(x0_std, x1, out=np.full_like(x0, np.nan),
                         where=where)

        if both_imageSets:
            x1_std = im1.std
            u_x1 = np.divide(x0 * x1_std, x1 ** 2,
                             out=np.full_like(x0, np.nan), where=where)
            division_std = np.sqrt(u_x0 ** 2 + u_x1 ** 2)
        else:
            division_std = u_x0

    divisionSet = im0.copy()
    divisionSet.acq = division
    divisionSet.std = division_std

    return divisionSet


def multiply_imageSets(im0: ImageSet, im1: ImageSet | float,
                       use_std: Optional[bool] = False,
                       lower: Optional[float] = None,
                       upper: Optional[float] = None):
    """
    Calculates the product of two images using ImageSet objects. Can also
    calculate error of the division. At pixels whose value for either the
    numerator or denominator fall outside the lower and upper limits, the result
    pixel value will be set to NaN.
    Args:
        im0: The first ImageSet.
        im1: The second ImageSet.
        use_std: whether to calculate error or not, defaults to False.
        lower: lower limit of values to include in calculation.
        upper: upper limit of values to include in calculation.

    Returns:
        New ImageSet object with x0 as the copy base.
    """
    x0 = im0.acq
    both_imageSets = type(im1) is ImageSet

    if both_imageSets:
        x1 = im1.acq
        if lower is not None and upper is not None:
            where = (lower < x1) & (x1 < upper) & (lower < x0) & (x0 < upper)
        else:
            where = True
    else:
        x1 = im1
        if lower is not None and upper is not None:
            where = (lower < x0) & (x0 < upper)
        else:
            where = True

    product = np.multiply(x0, x1, out=np.full_like(x0, np.nan), where=where)
    product_std = None

    if use_std:

        x0_std = im0.std
        u_x0 = np.multiply(x0_std, x1, out=np.full_like(x0, np.nan), where=where)

        if both_imageSets:
            x1_std = im1.std
            u_x1 = np.multiply(x0, x1_std, out=np.full_like(x0, np.nan), where=where)
            product_std = np.sqrt(u_x0 ** 2 + u_x1 ** 2)
        else:
            product_std = u_x0

    divisionSet = im0.copy()
    divisionSet.acq = product
    divisionSet.std = product_std

    return divisionSet


def add_imageSets(im0: ImageSet, im1: ImageSet | float,
                  use_std: Optional[bool] = False,
                  lower: Optional[float] = None,
                  upper: Optional[float] = None):
    """
    Calculates the sum of two ImageSet objects with errors if desired. Pixels
    whose value falls outside the lower and upper limits are set to NaN.
    Args:
        im0: first ImageSet
        im1: second ImageSet
        use_std: Whether to use errors or not.
        lower: Lower limit of values to include in calculation.
        upper: Upper limit of values to include in calculation.

    Returns:
        New ImageSet object with x0 as copy base.
    """
    x0 = im0.acq
    both_imageSets = type(im1) is ImageSet

    if both_imageSets:
        x1 = im1.acq
        if lower is not None and upper is not None:
            where = (lower < x1) & (x1 < upper) & (lower < x0) & (x0 < upper)
        else:
            where = True
    else:
        x1 = im1
        if lower is not None and upper is not None:
            where = (lower < x0) & (x0 < upper)
        else:
            where = True

    addition = np.add(x0, x1, out=np.full_like(x0, np.nan), where=where)
    addition_std = None

    if use_std:
        x0_std = im0.std
        if not both_imageSets:
            addition_std = x0_std
        else:
            x1_std = im1.std
            addition_std = np.sqrt(np.add(x0_std ** 2, x1_std ** 2,
                                   out=np.full_like(x0, np.nan), where=where))

    additionSet = im0.copy()
    additionSet.acq = addition
    additionSet.std = addition_std

    return additionSet


def subtract_imageSets(im0: ImageSet, im1: ImageSet | float,
                       use_std: Optional[bool] = False,
                       lower: Optional[float] = None,
                       upper: Optional[float] = None):
    """
    Calculates the subtraction of two ImageSet objects with errors if desired.
    Pixels whose value falls outside the lower and upper limits are set to NaN.
    Args:
        im0: The subtractee ImageSet
        im1: The subtracting ImageSet
        use_std: Whether to use errors or not.
        lower: Lower limit of values to include in calculation.
        upper: Upper limit of values to include in calculation.

    Returns:
        New ImageSet object with x0 as copy base.
    """
    x0 = im0.acq
    both_imageSets = type(im1) is ImageSet

    if both_imageSets:
        x1 = im1.acq
        if lower is not None and upper is not None:
            where = (lower < x1) & (x1 < upper) & (lower < x0) & (x0 < upper)
        else:
            where = True
    else:
        x1 = im1
        if lower is not None and upper is not None:
            where = (lower < x0) & (x0 < upper)
        else:
            where = True

    subtraction = np.subtract(x0, x1, out=np.full_like(x0, np.nan), where=where)
    subtraction_std = None

    if use_std:
        x0_std = im0.std
        if not both_imageSets:
            subtraction_std = x0_std
        else:
            x1_std = im1.std
            subtraction_std = np.sqrt(np.add(x0_std ** 2, x1_std ** 2,
                                             out=np.full_like(x0, np.nan),
                                             where=where))

    subtractionSet = im0.copy()
    subtractionSet.acq = subtraction
    subtractionSet.std = subtraction_std

    return subtractionSet


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


def linearize_ImageSet(imageSet: ImageSet,
                       ICRF: np.ndarray,
                       ICRF_diff: Optional[np.ndarray] = None,
                       gaussian_blur: Optional[bool] = True):
    """
    Linearizes an input image using an ICRF. Additionally, if a derivative of
    the ICRF and an array representing the standard deviations of a pixel value
    is supplied, an error image is calculated for the linearized image.
    Args:
        imageSet: imageSet object whose, acquired image is linearized.
        ICRF: ICRF as numpy array.
        ICRF_diff: derivative of ICRF as numpy array.
        STD_arr: camera STD data as numpy array.
        gaussian_blur: whether to apply blur to the result or not.

    Returns: the input ImageSet object with linearized acq and possibly new
        error image.
    """
    acq = imageSet.acq
    std = imageSet.std
    channels = np.shape(imageSet.acq)[2]

    use_std = False
    if ICRF_diff is not None and std is not None:
        use_std = True

    acq_new = np.zeros(np.shape(imageSet.acq), dtype=np.dtype('float32'))
    if use_std:
        std_new = np.zeros(np.shape(imageSet.acq), dtype=np.dtype('float32'))

    def channel_loop(acq_c: np.ndarray, ICRF_c: np.ndarray, std_c: np.ndarray,
                     ICRF_diff_c: np.ndarray, gauss: bool):

        ret = linearize_channel(acq_c, ICRF_c, std_c, ICRF_diff_c, gauss)

        return ret

    if use_std:
        sub_results = parallel.Parallel(n_jobs=channels, prefer="threads")(delayed(channel_loop)(acq[:, :, c], ICRF[:, c], std[:, :, c], ICRF_diff[:, c], gaussian_blur) for c in range(channels))
    else:
        sub_results = parallel.Parallel(n_jobs=channels, prefer="threads")(delayed(channel_loop)(acq_c=acq[:, :, c], ICRF_c=ICRF[:, c], std_c=None, ICRF_diff_c=None, gauss=gaussian_blur) for c in range(channels))

    for c in range(channels):
        channel_res = sub_results[c]
        acq_new[:, :, c] = channel_res[0]
        if use_std:
            std_new[:, :, c] = channel_res[1]

    del channel_res
    del sub_results

    imageSet.acq = acq_new
    if use_std:
        imageSet.std = std_new

    return imageSet


def linearize_channel(channel: np.ndarray,
                      ICRF: np.ndarray,
                      channel_std: Optional[np.ndarray] = None,
                      ICRF_diff: Optional[np.ndarray] = None,
                      gaussian_blur: Optional[bool] = True):

    channel = (np.around(channel * MAX_DN)).astype(int)
    linear_channel = (ICRF[channel]).astype(np.dtype('float32'))
    if gaussian_blur:
        linear_channel = gaussian_filter(linear_channel, sigma=1)

    use_std = False
    if ICRF_diff is not None and channel_std is not None:
        use_std = True

    if not use_std:
        return [linear_channel]

    linear_std = (ICRF_diff[channel] * channel_std).astype(np.dtype('float32'))

    if gaussian_blur:
        linear_std = gaussian_filter(linear_std, sigma=1)

    return [linear_channel, linear_std]


def create_imageSets(path: Path):
    """
    Load all images of a given path into a list of ImageSet objects but without
    acq or std images.

    :param path: Absolute path to the directory of images

    :return: List of ImageSet objects.
    """

    list_of_ImageSets = []
    image_files = path.glob("*.tif")
    for file in image_files:
        if not ("STD" in file.name):

            imageSet = ImageSet(file)
            list_of_ImageSets.append(imageSet)

    return list_of_ImageSets


def show_image(input_image: Optional[ImageSet] = None):

    if input_image is None:
        image_path = get_filepath_dialog('Choose image to show')
        input_image = ImageSet(image_path)

    input_image.load_acq()
    input_image.path = Path(input_image.path.parent.joinpath(input_image.path.name.replace('.tif', 'sbgr.tif')))
    input_image.save_8bit(sbgr=False)
    cv.namedWindow(input_image.path.name, cv.WINDOW_NORMAL)
    cv.imshow(input_image.path.name, input_image.acq)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_filepath_dialog(title: str):
    file = None

    def open_file_dialog(w):
        nonlocal file
        try:
            file = w.create_file_dialog(webview.OPEN_DIALOG)[0]
        except TypeError:
            pass  # user exited file dialog without picking
        finally:
            w.destroy()
    window = webview.create_window(title, hidden=True)
    webview.start(open_file_dialog, window)
    # file will either be a string or None
    if file is not None:
        return Path(file)

    return

