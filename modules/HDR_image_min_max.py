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
        return x**2
    if x > 0.5:
        return (1-x)**2


def create_HDR_absolute(list_of_ImageSets: List[ImageSet]):

    hat_weight_vectorized = np.vectorize(hat_weight)

    numerator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)
    denominator = np.zeros(np.shape(list_of_ImageSets[0].acq), dtype=float)

    for imageSet in list_of_ImageSets:

        baseSet = ImageSet(imageSet.path)
        baseSet.load_acq()
        weight_array = hat_weight_vectorized(baseSet.acq)
        del baseSet

        numerator += weight_array * imageSet.acq * (1/imageSet.exp)
        denominator += weight_array

    HDR_acq = np.divide(numerator, denominator)

    imageSet_HDR = copy.deepcopy(list_of_ImageSets[0])
    imageSet_HDR.acq = HDR_acq

    return imageSet_HDR


def process_HDR_images(image_path: Optional[Path] = DEFAULT_ACQ_PATH,
                       save_linear: Optional[bool] = False,
                       save_HDR: Optional[bool] = True,
                       save_8bit: Optional[bool] = True,
                       save_32bit: Optional[bool] = False,
                       pass_linear: Optional[bool] = False,
                       fix_artifacts: Optional[bool] = True,
                       ICRF: Optional[np.ndarray] = None,
                       gaussian_blur: Optional[bool] = False):

    if ICRF is None:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)

    # Initialize image lists and name lists
    acq_list = gf.create_imageSets(image_path)
    acq_sublists = gf.separate_to_sublists(acq_list)
    del acq_list

    cycle = ['min', 'mean', 'max']

    for term in cycle:
        for sublist in acq_sublists:
            for imageSet in sublist:

                imageSet.load_acq()

                if term == 'min':
                    imageSet.load_std()
                    imageSet.acq = imageSet.acq - imageSet.std
                    imageSet.std = None
                if term == 'max':
                    imageSet.load_std()
                    imageSet.acq = imageSet.acq + imageSet.std
                    imageSet.std = None

                imageSet = gf.linearize_ImageSet(imageSet, ICRF, gaussian_blur=gaussian_blur)
                if save_linear and not fix_artifacts:
                    if save_8bit:
                        imageSet.save_8bit(OUT_PATH.joinpath(imageSet.path.name.replace('.tif', f' {term}.tif')))
                    if save_32bit:
                        imageSet.save_32bit(OUT_PATH.joinpath(imageSet.path.name.replace('.tif', f' {term}.tif')), separate_channels=True)
                    print(f"Saved {imageSet.path.name}")

        if pass_linear:
            return acq_sublists

        if fix_artifacts:
            flat_list = gf.create_imageSets(FLAT_PATH)
            for flatSet in flat_list:
                flatSet.load_acq()
                flatSet.load_std()

            dark_list = gf.create_imageSets(DARK_PATH)
            for darkSet in dark_list:
                darkSet.load_acq()
                # darkSet.load_std() # Not used in current version

            for sublist in acq_sublists:
                sublist = ic.image_correction(sublist, dark_list, flat_list)

            del flat_list
            del dark_list

        if save_linear and fix_artifacts:
            for sublist in acq_sublists:
                for imageSet in sublist:
                    if save_8bit:
                        imageSet.save_8bit(OUT_PATH.joinpath(imageSet.path.name.replace('.tif', f' {term}.tif')))
                    if save_32bit:
                        imageSet.save_32bit(OUT_PATH.joinpath(imageSet.path.name.replace('.tif', f' {term}.tif')), separate_channels=True)
                    print(f"Saved {imageSet.path.name}")

        for sublist in acq_sublists:

            if save_HDR and len(sublist) > 1:

                imageSet_HDR = create_HDR_absolute(sublist)
                print(OUT_PATH.joinpath(imageSet_HDR.path.name.replace('.tif', f' {term}.tif')))
                if save_8bit:
                    imageSet_HDR.save_8bit(OUT_PATH.joinpath(imageSet_HDR.path.name.replace('.tif', f' {term}.tif')))
                if save_32bit:
                    imageSet_HDR.save_32bit(OUT_PATH.joinpath(imageSet_HDR.path.name.replace('.tif', f' {term}.tif')),
                                            is_HDR=True)

    return


if __name__ == "__main__":

    print('Run script from actual main file!')
