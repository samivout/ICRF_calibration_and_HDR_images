import ImageSet as IS
from ImageSet import ImageSet
import image_correction as ic
import numpy as np
import read_data as rd
from typing import Optional
from typing import List
import general_functions as gf

im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')
default_acq_path = rd.read_config_single('acquired images path')
flat_path = rd.read_config_single('flat fields path')
dark_path = rd.read_config_single('dark frames path')
out_path = rd.read_config_single('corrected output path')
ICRF_calibrated_file = rd.read_config_single('calibrated ICRFs')
ICRF = None
ICRF_diff = None
channels = rd.read_config_single('channels')
bit_depth = rd.read_config_single('bit depth')
bits = 2**bit_depth
max_DN = bits-1
min_DN = 0
datapoints = rd.read_config_single('final datapoints')
data_multiplier = datapoints/bits
STD_arr = rd.read_data_from_txt(rd.read_config_single('STD data'))


def hat_weight(x: float):

    if x <= 0.5:
        return x
    if x > 0.5:
        return 1-x


def create_HDR_absolute(list_of_ImageSets: List[ImageSet]):

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

    imageSet_HDR = ImageSet(HDR_acq,
                            HDR_std,
                            list_of_ImageSets[0].file_name,
                            None,
                            list_of_ImageSets[0].mag,
                            list_of_ImageSets[0].ill,
                            list_of_ImageSets[0].name,
                            None)

    return imageSet_HDR


def linearize_image_vectorized(imageSet):

    global ICRF
    global ICRF_diff

    acq = (imageSet.acq * max_DN).astype(int)
    acq_new = np.zeros(np.shape(acq), dtype=np.dtype('float32'))
    std_new = np.zeros(np.shape(acq), dtype=np.dtype('float32'))
    for c in range(channels):

        # The ICRFs are in reverse order in the .txt file when compared
        # to how OpenCV opens the channels.
        acq_new[:, :, c] = ICRF[acq[:, :, c], c]
        std_new[:, :, c] = ICRF_diff[acq[:, :, c], c] * \
            STD_arr[acq[:, :, c], c]

    imageSet.acq = acq_new
    imageSet.std = std_new

    return imageSet


def process_HDR_images(acq_path: Optional[str] = default_acq_path,
                       save_linear: Optional[bool] = False,
                       save_HDR: Optional[bool] = True,
                       save_8bit: Optional[bool] = True,
                       save_32bit: Optional[bool] = False,
                       pass_linear: Optional[bool] = False,
                       fix_artifacts: Optional[bool] = True,
                       ICRF_arr: Optional[np.ndarray] = None):

    # Determine the numerical derivative of the calibrated ICRFs.
    global ICRF_diff
    global ICRF

    if ICRF_arr is None:
        ICRF = rd.read_data_from_txt(ICRF_calibrated_file)
    else:
        ICRF = ICRF_arr

    ICRF_diff = np.zeros_like(ICRF)
    dx = 1/(bits-1)
    for c in range(channels):

        ICRF_diff[:, c] = np.gradient(ICRF[:, c], dx)

    # Initialize image lists and name lists
    acq_list = IS.create_imageSets(acq_path)
    acq_sublists = gf.separate_to_sublists(acq_list)
    del acq_list

    if fix_artifacts:
        flat_list = IS.create_imageSets(flat_path)
        for flatSet in flat_list:
            flatSet.load_acq()
            flatSet.load_std()

        dark_list = IS.create_imageSets(dark_path)
        for darkSet in dark_list:
            darkSet.load_acq()
            darkSet.load_std()

    for sublist in acq_sublists:

        if fix_artifacts:
            sublist = ic.image_correction(sublist, dark_list, flat_list)

        for imageSet in sublist:

            if not fix_artifacts:
                imageSet.load_acq()
                imageSet.load_std()

            imageSet = linearize_image_vectorized(imageSet)
            if save_linear:
                if save_8bit:
                    IS.save_image_8bit(imageSet, out_path)
                if save_32bit:
                    IS.save_image_32bit(imageSet, out_path)

    if fix_artifacts:
        del flat_list
        del dark_list

    if pass_linear:
        return acq_sublists

    for sublist in acq_sublists:

        if save_HDR and len(sublist) > 1:

            imageSet_HDR = create_HDR_absolute(sublist)
            if save_8bit:
                IS.save_image_8bit(imageSet_HDR, out_path)
            if save_32bit:
                IS.save_image_32bit(imageSet_HDR, out_path)

    return


if __name__ == "__main__":

    print('Run script from actual main file!')
