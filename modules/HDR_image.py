import matplotlib.pyplot as plt

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
        return (1-x)


def create_HDR_absolute(list_of_ImageSets: List[ImageSet]):

    hat_weight_vectorized = np.vectorize(hat_weight)

    number_of_STD_images = 0
    for imageSet in list_of_ImageSets:
        if imageSet.std is not None:
            number_of_STD_images += 1

    number_of_imageSets = len(list_of_ImageSets)

    numerator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)
    denominator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)

    if number_of_STD_images == number_of_imageSets:
        std_numerator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)

    for imageSet in list_of_ImageSets:

        baseSet = ImageSet(imageSet.path)
        baseSet.load_acq()
        weight_array = hat_weight_vectorized(baseSet.acq)
        del baseSet

        numerator += weight_array * imageSet.acq * (1/imageSet.exp)
        denominator += weight_array

        if number_of_STD_images == number_of_imageSets:
            std_numerator += (weight_array ** 2) * (imageSet.std ** 2) * (1 / (imageSet.exp ** 2))

    HDR_acq = np.divide(numerator, denominator)
    HDR_std = np.sqrt(np.divide(std_numerator, denominator**2))

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
                       gaussian_blur: Optional[bool] = False):

    if ICRF is None:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    if STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)

    ICRF_diff = np.zeros_like(ICRF)
    x_range = np.linspace(0, 1, BITS)
    dx = 2/(BITS - 1)
    for c in range(CHANNELS):

        ICRF_diff[:, c] = np.gradient(ICRF[:, c], dx)
        plt.plot(x_range, ICRF_diff[:, c])

    np.savetxt(OUTPUT_DIRECTORY.joinpath('ICRF_diff.txt'), ICRF_diff)
    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_diff.png'), dpi=300)

    # Initialize image lists and name lists
    acq_list = gf.create_imageSets(image_path)
    acq_sublists = gf.separate_to_sublists(acq_list)
    del acq_list

    # Dark correction
    if fix_artifacts:
        dark_list = gf.create_imageSets(DARK_PATH)
        for darkSet in dark_list:
            darkSet.load_acq()
            # darkSet.load_std() # Not used in current version

        for sublist in acq_sublists:
            for imageSet in sublist:

                imageSet = ic.dark_correction(imageSet, dark_list)

        del dark_list

    # Linearization
    for sublist in acq_sublists:
        for i, imageSet in enumerate(sublist):

            imageSet.load_acq()
            imageSet.load_std()

            sublist[i] = gf.linearize_ImageSet(imageSet, ICRF, ICRF_diff, gaussian_blur)
            if save_linear and not fix_artifacts:
                if save_8bit:
                    imageSet.save_8bit(OUT_PATH.joinpath(imageSet.path.name))
                if save_32bit:
                    imageSet.save_32bit(OUT_PATH.joinpath(imageSet.path.name), separate_channels=True)
                print(f"Saved {imageSet.path.name}")

    if pass_linear:
        return acq_sublists

    HDR_list = []
    if save_HDR:
        for sublist in acq_sublists:
            if len(sublist) > 1:

                imageSet_HDR = create_HDR_absolute(sublist)
                HDR_list.append(imageSet_HDR)

    if HDR_list:
        if fix_artifacts:
            flat_list = gf.create_imageSets(FLAT_PATH)
            for flatSet in flat_list:
                flatSet.load_acq()
                flatSet.load_std()

            for i, HDRset in enumerate(HDR_list):

                HDR_list[i] = ic.fixed_pattern_correction(HDRset, flat_list)
                HDR_list[i] = ic.uncertainty(HDRset, flat_list)

            del flat_list

        for HDRset in HDR_list:
            if save_8bit:
                HDRset.save_8bit(OUT_PATH.joinpath(HDRset.path.name))
            if save_32bit:
                HDRset.save_32bit(OUT_PATH.joinpath(HDRset.path.name), is_HDR=True)

    return


if __name__ == "__main__":

    x = np.linspace(0, 1, num=BITS)
    sum = 0
    sum2 = 0
    for element in x:
        sum += hat_weight(element)
    print(sum)

    print('Run script from actual main file!')
