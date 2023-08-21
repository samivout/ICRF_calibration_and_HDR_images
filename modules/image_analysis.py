import ImageSet as IS
from ImageSet import ImageSet
import numpy as np
import read_data as rd
import cv2 as cv
import math
from typing import List
from typing import Optional
import general_functions as gf
import os
import matplotlib.pyplot as plt

bit_depth = rd.read_config_single('bit depth')
bits = 2**bit_depth
max_DN = bits-1
min_DN = 0
channels = rd.read_config_single('channels')


def analyze_linearity(path: str,
                      sublists_of_imageSets: Optional[List[ImageSet]] = None,
                      pass_results: Optional[bool] = False,
                      calculate_std: Optional[bool] = True,
                      original_std: Optional[bool] = False):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        original_std: whether used error images are original or not
        calculate_std: whether to calculate error or not
        pass_results: whether to return the result or not
        path: Path of the images directory as string
        sublists_of_imageSets: Optionally pass sublist from previous calculations

    Returns:
    """
    file_name = 'linearity_processed.txt'
    if sublists_of_imageSets is None:
        list_of_imageSets = IS.create_imageSets(path)
        sublists_of_imageSets = gf.separate_to_sublists(list_of_imageSets)
        del list_of_imageSets
        file_name = 'linearity_og.txt'

    results = []
    lower = 5/max_DN
    upper = 250/max_DN

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        sublist.sort(key=lambda imageSet: imageSet.exp)
        for imageSet in sublist:
            if imageSet.acq is None:
                imageSet.load_acq()
            if calculate_std:
                if imageSet.std is None:
                    imageSet.load_std(is_original=original_std)

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]
                means = []
                stds = []
                division = np.divide(x.acq, y.acq,
                                     out=np.zeros_like(x.acq),
                                     where=((lower < y.acq) & (y.acq < upper) &
                                            (lower < x.acq) & (x.acq < upper)))

                if calculate_std:
                    u_x = np.divide(x.std, y.acq, out=np.zeros_like(x.acq),
                                    where=((lower < y.acq) & (y.acq < upper) &
                                           (lower < x.acq) & (x.acq < upper)))

                    u_y = np.divide(x.acq * y.std, y.acq ** 2,
                                    out=np.zeros_like(x.acq),
                                    where=((lower < y.acq) & (y.acq < upper) &
                                           (lower < x.acq) & (x.acq < upper)))

                    division_std = np.sqrt(u_x ** 2 + u_y ** 2)

                for c in range(channels):
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
    plt.savefig(os.path.join(path, 'scatter.png'), dpi=300)
    plt.clf()

    column_means = np.round(np.nanmean(data_array, axis=0), decimals=3)
    results.append(column_means.tolist())

    if not pass_results:
        with open(os.path.join(path, file_name), 'w') as f:
            for row in results:
                f.write(f'{row}\n')
        return

    return results


def choose_evenly_spaced_points(array, num_points):
    # Convert the 2D array to a numpy array for easier indexing
    array = np.array(array)

    # Calculate the step size between points
    step = max(1, int(array.shape[0] / (num_points - 1)))

    # Select the evenly spaced points
    points = array[::step, ::step]

    return points


if __name__ == "__main__":

    array = np.arange(0, 10552320, dtype=int).reshape((2748, 3840))

    num_points = 70
    result = choose_evenly_spaced_points(array, num_points)
    ratio_of_elements = np.size(result)/10552320
    print(ratio_of_elements)
    print('Run script from actual main file!')