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
                      pass_results: Optional[bool] = False):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
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
    lower = 1/max_DN
    upper = 253/max_DN

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        sublist.sort(key=lambda imageSet: imageSet.exp)
        for imageSet in sublist:
            imageSet.load_acq()

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

                for c in range(channels):
                    nonzeros = np.nonzero(division[:, :, c])
                    channel = division[:, :, c]
                    channel_mean = np.mean(channel[nonzeros])
                    channel_std = np.std(channel[nonzeros])
                    means.append(round(channel_mean, 3))
                    stds.append(round(channel_std, 3))

                # result = f'{x.name} {x.exp}-{y.exp} = {round(x.exp/y.exp, 2)}\t{means}\t{stds}'
                result = [round(x.exp / y.exp, 3), means[0], means[1], means[2]]
                results.append(result)

    data_array = np.array(results)
    plt.scatter(data_array[:, 0], data_array[:, 1], c='r')
    plt.scatter(data_array[:, 0], data_array[:, 2], c='g')
    plt.scatter(data_array[:, 0], data_array[:, 3], c='b')
    plt.savefig(os.path.join(path, 'scatter.png'), dpi=300)
    plt.clf()

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

    array = [[1, 2, 3, 4, 5],
             [6, 7, 8, 9, 10],
             [11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20]]

    num_points = 3
    result = choose_evenly_spaced_points(array, num_points)
    print(result)
    print('Run script from actual main file!')