import read_data
from ImageSet import ImageSet
import numpy as np
from typing import List
from typing import Optional
import general_functions as gf
import matplotlib.pyplot as plt
from global_settings import *


def analyze_linearity(path: str,
                      sublists_of_imageSets: Optional[List[ImageSet]] = None,
                      pass_results: Optional[bool] = False,
                      calculate_std: Optional[bool] = True,
                      original_std: Optional[bool] = False,
                      STD_data: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        original_std: whether used error images are original or not
        calculate_std: whether to calculate error or not
        pass_results: whether to return the result or not
        path: Path of the images directory as string
        sublists_of_imageSets: Optionally pass sublist from previous calculations
        STD_data: Array of the camera pixel error data
        save_path: path to which save plot of data

    Returns:
    """
    if save_path is None:
        save_path = os.path.join(path, 'scatter.png')

    file_name = 'linearity_processed.txt'
    if sublists_of_imageSets is None:
        list_of_imageSets = gf.create_imageSets(path)
        sublists_of_imageSets = gf.separate_to_sublists(list_of_imageSets)
        del list_of_imageSets
        file_name = 'linearity_og.txt'

    results = []
    lower = 5 / MAX_DN
    upper = 250 / MAX_DN

    if STD_data is None:
        STD_data = read_data.read_data_from_txt(STD_FILE_NAME)

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        sublist.sort(key=lambda imageSet: imageSet.exp)
        for imageSet in sublist:
            if imageSet.acq is None:
                imageSet.load_acq()
            if calculate_std:
                if imageSet.std is None:
                    imageSet.load_std(is_original=original_std,
                                      STD_data=STD_data)

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]
                means = []
                stds = []
                if not calculate_std:
                    division = gf.divide_imageSets(x, y, calculate_std,
                                                   lower, upper)
                else:
                    division, division_std = gf.divide_imageSets(x, y, calculate_std,
                                                                 lower, upper)

                for c in range(CHANNELS):
                    nonzeros = np.nonzero(division[:, :, c])
                    acq_channel = division[:, :, c]
                    channel_mean = np.mean(acq_channel[nonzeros])
                    means.append(channel_mean)

                    if calculate_std:
                        std_channel = division_std[:, :, c]
                        channel_std = np.mean(std_channel[nonzeros])
                        stds.append(channel_std)
                    else:
                        stds = [0, 0, 0]

                ratio = x.exp / y.exp
                if ratio < 0.05:
                    break
                # result = f'{x.name} {x.exp}-{y.exp} = {round(x.exp/y.exp, 2)}\t{means}\t{stds}'
                result = [round(ratio, 3),
                          round((means[0]-ratio)/ratio, 3),
                          round((means[1]-ratio)/ratio, 3),
                          round((means[2]-ratio)/ratio, 3),
                          round(stds[0]/ratio, 3),
                          round(stds[1]/ratio, 3),
                          round(stds[2]/ratio, 3)]
                results.append(result)

    data_array = np.array(results)
    plt.errorbar(data_array[:, 0], data_array[:, 3], yerr=data_array[:, 6],
                 c='r', marker='o', linestyle='none', markersize=3)
    plt.errorbar(data_array[:, 0], data_array[:, 1], yerr=data_array[:, 4],
                 c='b', marker='o', linestyle='none', markersize=3)
    plt.errorbar(data_array[:, 0], data_array[:, 2], yerr=data_array[:, 5],
                 c='g', marker='o', linestyle='none', markersize=3)
    plt.savefig(save_path, dpi=300)
    plt.clf()

    column_means = np.round(np.nanmean(data_array, axis=0), decimals=3)
    results.append(column_means.tolist())

    if not pass_results:
        with open(os.path.join(path, file_name), 'w') as f:
            for row in results:
                f.write(f'{row}\n')
        return

    return results


def linearity_distribution(long_imageSet: ImageSet,
                           short_imageSet: ImageSet,
                           path: str,
                           num: int):
    """
    Analyze the linearity of a pair of images by producing a distribution of the
    relative errors to the expected ratio based on exposure times.
    Args:
        num: number of datapoints to use from image
        long_imageSet: longer exposure time image
        short_imageSet: shorter exposure time image
        path: path to save results.

    Returns:

    """

    return


if __name__ == "__main__":

    print('Run script from actual main file!')