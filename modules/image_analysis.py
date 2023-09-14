import read_data
from ImageSet import ImageSet
import numpy as np
from typing import List
from typing import Optional
import general_functions as gf
import matplotlib.pyplot as plt
from global_settings import *


def analyze_linearity(path: Path,
                      sublists_of_imageSets: Optional[List[ImageSet]] = None,
                      pass_results: Optional[bool] = False,
                      use_std: Optional[bool] = True,
                      STD_data: Optional[np.ndarray] = None,
                      save_path: Optional[Path] = None,
                      use_relative: Optional[bool] = True,
                      absolute_result: Optional[bool] = False):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        original_std: whether used error images are original or not
        use_std: whether to calculate error or not
        pass_results: whether to return the result or not
        path: Path of the images directory as string
        sublists_of_imageSets: Optionally pass sublist from previous calculations
        STD_data: Array of the camera pixel error data
        save_path: path to which save plot of data
        use_relative: whether to calculate as relative or absolute disparity.
        absolute_result: whether to calculate results as (div - ratio)/ratio
            or abs(div - ratio)/ratio.

    Returns:
    """
    if save_path is None:
        save_path = path.joinpath('scatter.png')

    file_name = 'linearity_processed.txt'
    if sublists_of_imageSets is None:
        list_of_imageSets = gf.create_imageSets(path)
        sublists_of_imageSets = gf.separate_to_sublists(list_of_imageSets)
        del list_of_imageSets
        file_name = 'linearity_og.txt'

    results = []
    lower = LOWER_LIN_LIM
    upper = UPPER_LIN_LIM

    if use_std and STD_data is None:
        STD_data = read_data.read_data_from_txt(STD_FILE_NAME)

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        sublist.sort(key=lambda imageSet: imageSet.exp)
        for imageSet in sublist:
            if imageSet.acq is None:
                imageSet.load_acq()
            if use_std:
                if imageSet.std is None:
                    imageSet.load_std(STD_data=STD_data)

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]
                means = []
                stds = []

                ratio = x.exp / y.exp
                if ratio < 0.05:
                    break

                y = gf.multiply_imageSets(y, ratio, use_std=use_std)
                linearSet = gf.subtract_imageSets(x, y, use_std=use_std, lower=lower,
                                                  upper=upper)
                if use_relative:
                    linearSet = gf.divide_imageSets(x, y, use_std=use_std, lower=lower,
                                                    upper=upper)

                for c in range(CHANNELS):
                    base_acq = linearSet.acq[:, :, c]
                    finite_indices = np.isfinite(base_acq)

                    if use_relative:
                        if absolute_result:
                            acq = abs(base_acq - 1)
                        else:
                            acq = base_acq - 1
                    else:
                        if absolute_result:
                            acq = abs(base_acq * MAX_DN)
                        else:
                            acq = base_acq * MAX_DN

                    channel_mean = np.mean(acq[finite_indices])
                    means.append(channel_mean)

                    if use_std:
                        base_std = linearSet.std[:, :, c]

                        if use_relative:
                            std = base_std
                        else:
                            std = base_std * MAX_DN

                        channel_std = np.mean(std[finite_indices])
                        stds.append(channel_std)
                    else:
                        stds = [0, 0, 0]

                result = [round(ratio, 3),
                          round(means[0], 3),
                          round(means[1], 3),
                          round(means[2], 3),
                          round(stds[0], 3),
                          round(stds[1], 3),
                          round(stds[2], 3)]
                results.append(result)
                print(f'{x.path.name}-{y.path.name}-{ratio}-{means}')

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
        with open(path.joinpath(file_name), 'w') as f:
            for row in results:
                f.write(f'{row}\n')
        return

    return results


def linearity_distribution(long_imageSet: Optional[ImageSet] = None,
                           short_imageSet: Optional[ImageSet] = None,
                           save_dir: Optional[Path] = None,
                           num: Optional[int] = None,
                           upper: Optional[float] = UPPER_LIN_LIM,
                           lower: Optional[float] = LOWER_LIN_LIM,
                           use_std: Optional[bool] = False,
                           STD_data: Optional[np.ndarray] = None,
                           save_image: Optional[bool] = False,
                           use_relative: Optional[bool] = True):
    """
    Analyze the linearity of a pair of images by producing a distribution of the
    relative errors to the expected ratio based on exposure times.
    Args:
        num: number of datapoints to use from image.
        long_imageSet: longer exposure time image.
        short_imageSet: shorter exposure time image.
        save_dir: path to save results.
        upper: upper limit of values to include in division operation.
        lower: lower limit of values to include in division operation.
        use_std: whether to include errors in calculation or not.
        STD_data: Numpy array representing the STD data of pixel values.
        save_image: whether to save division image result or not.
        use_relative: whether to calculate the linearity difference as an absolute
            or a relative value.
    """
    if long_imageSet is None:
        long_imageSet_path = gf.get_path_dialog('Choose long exposure image')
        long_imageSet = ImageSet(long_imageSet_path)
    if short_imageSet is None:
        short_imageSet_path = gf.get_path_dialog('Choose short exposure image')
        short_imageSet = ImageSet(short_imageSet_path)

    if long_imageSet.exp > short_imageSet.exp:
        x = short_imageSet
        y = long_imageSet
    else:
        x = long_imageSet
        y = short_imageSet

    if save_dir is None:
        save_dir = y.path.parent.joinpath('Linearity distribution')
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

    ratio = x.exp/y.exp

    if y.acq is None:
        y.load_acq()
    if x.acq is None:
        x.load_acq()

    if use_std:
        if STD_data is None:
            STD_data = read_data.read_data_from_txt(STD_FILE_NAME)
        if y.std is None:
            y.load_std(STD_data=STD_data)
        if x.std is None:
            x.load_std(STD_data=STD_data)

    if num is not None:
        y.acq = gf.choose_evenly_spaced_points(y.acq, num)
        x.acq = gf.choose_evenly_spaced_points(x.acq, num)
        if use_std:
            y.std = gf.choose_evenly_spaced_points(y.std, num)
            x.std = gf.choose_evenly_spaced_points(x.std, num)

    y = gf.multiply_imageSets(y, ratio, use_std=use_std)
    linearSet = gf.subtract_imageSets(x, y, use_std=use_std, lower=lower,
                                      upper=upper)
    if use_relative:
        linearSet = gf.divide_imageSets(x, y, use_std=use_std, lower=lower,
                                        upper=upper)

    linearSet.path = save_dir.joinpath(f'{x.exp}-{y.exp} {y.subject}.tif')

    if save_image:
        linearSet.save_32bit()

    for c in range(CHANNELS):

        hist_name = f'{x.exp}-{y.exp} {y.subject} {c}.png'
        save_path = save_dir.joinpath(hist_name)
        base_acq = linearSet.acq[:, :, c]
        finite_indices = np.isfinite(base_acq)

        if use_relative:
            acq = base_acq
        else:
            acq = (base_acq * MAX_DN)
        bins = 200

        hist, bin_edges = np.histogram(acq[finite_indices], bins=bins)
        width = abs(bin_edges[0] - bin_edges[1])
        hist = hist / np.sum(hist)
        if c == 0:
            plt.bar(bin_edges[:-1], hist, width=width, fc='b', ec=None,
                    alpha=0.9)
        if c == 1:
            plt.bar(bin_edges[:-1], hist, width=width, fc='g', ec=None,
                    alpha=0.9)
        if c == 2:
            plt.bar(bin_edges[:-1], hist, width=width, fc='r', ec=None,
                    alpha=0.9)

        if use_std:
            base_std = linearSet.std[:, :, c]
            upper_acq = (base_acq + base_std - ratio) / ratio
            lower_acq = (base_acq - base_std - ratio) / ratio
            hist, bin_edges = np.histogram(upper_acq[finite_indices], bins=bins)
            hist = hist / np.sum(hist)
            width = abs(bin_edges[0] - bin_edges[1])
            plt.bar(bin_edges[:-1], hist, width=width, fc='y', ec=None, alpha=0.5)
            hist, bin_edges = np.histogram(lower_acq[finite_indices], bins=bins)
            hist = hist / np.sum(hist)
            width = abs(bin_edges[0] - bin_edges[1])
            plt.bar(bin_edges[:-1], hist, width=width, fc='c', ec=None, alpha=0.5)

        plt.savefig(save_path, dpi=300)
        plt.clf()

    return


if __name__ == "__main__":

    linearity_distribution(lower=5/255, upper=250/255, use_std=False,
                           save_image=True, use_relative=True)

    print('Run script from actual main file!')
