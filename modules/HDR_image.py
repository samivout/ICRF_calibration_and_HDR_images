from ImageSet import ImageSet
import image_correction as ic
import numpy as np
from typing import Optional
from typing import List
import general_functions as gf
from global_settings import *
import copy


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

    imageSet_HDR = copy.deepcopy(list_of_ImageSets[0])
    imageSet_HDR.acq = HDR_acq
    imageSet_HDR.std = HDR_std

    return imageSet_HDR


def process_HDR_images(image_path: Optional[Path] = DEFAULT_ACQ_PATH,
                       save_linear: Optional[bool] = False,
                       save_HDR: Optional[bool] = True,
                       save_8bit: Optional[bool] = True,
                       save_32bit: Optional[bool] = False,
                       pass_linear: Optional[bool] = False,
                       fix_artifacts: Optional[bool] = True,
                       ICRF: Optional[np.ndarray] = None,
                       STD_data: Optional[np.ndarray] = None,
                       gaussian_blur: Optional[bool] = True):

    if ICRF is None:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)

    ICRF_diff = np.zeros_like(ICRF)
    dx = 1/(BITS - 1)
    for c in range(CHANNELS):

        ICRF_diff[:, c] = np.gradient(ICRF[:, c], dx)

    # Initialize image lists and name lists
    acq_list = gf.create_imageSets(image_path)
    acq_sublists = gf.separate_to_sublists(acq_list)
    del acq_list

    if fix_artifacts:
        flat_list = gf.create_imageSets(FLAT_PATH)
        for flatSet in flat_list:
            flatSet.load_acq()
            flatSet.load_std()

        dark_list = gf.create_imageSets(DARK_PATH)
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

            imageSet = gf.linearize_ImageSet(imageSet, ICRF, ICRF_diff, gaussian_blur)
            if save_linear:
                if save_8bit:
                    imageSet.save_8bit(OUT_PATH.joinpath(imageSet.path.name))
                if save_32bit:
                    imageSet.save_32bit(OUT_PATH.joinpath(imageSet.path.name))
                print(f"Saved {imageSet.path.name}")

    if fix_artifacts:
        del flat_list
        del dark_list

    if pass_linear:
        return acq_sublists

    for sublist in acq_sublists:

        if save_HDR and len(sublist) > 1:

            imageSet_HDR = create_HDR_absolute(sublist)
            if save_8bit:
                imageSet_HDR.save_8bit(OUT_PATH.joinpath(imageSet_HDR.path.name))
            if save_32bit:
                imageSet_HDR.save_32bit(OUT_PATH.joinpath(imageSet_HDR.path.name))

    return


if __name__ == "__main__":

    print('Run script from actual main file!')
