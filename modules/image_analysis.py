import math
from ImageSet import ImageSet
import numpy as np
from typing import List
from typing import Optional
import general_functions as gf
import matplotlib.pyplot as plt
from scipy.odr import *
from scipy.stats import gaussian_kde
from scipy.stats import iqr
from global_settings import *
from itertools import combinations


def analyze_linearity(path: Path,
                      sublists_of_imageSets: Optional[List[ImageSet]] = None,
                      pass_results: Optional[bool] = False,
                      use_std: Optional[bool] = True,
                      STD_data: Optional[np.ndarray] = None,
                      save_path: Optional[Path] = None,
                      relative_scale: Optional[bool] = True,
                      absolute_scale: Optional[bool] = True,
                      absolute_result: Optional[bool] = False,
                      ICRF: Optional[np.ndarray] = None,
                      linearity_limit: Optional[int] = None):
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
        relative_scale: whether to report result on a relative scale.
        absolute_scale: whether to report result on an absolute scale
        absolute_result: whether to calculate results as (div - ratio)/ratio
            or abs(div - ratio)/ratio.
        ICRF: The ICRF used to linearize images subject to analysis. Used to
            determine the which pixel values to ignore in linearity analysis.

    Returns:
    """
    # Save plotted figures to dir of images or a specified dir.
    if save_path is None:
        save_path = path.joinpath('scatter.png')

    file_name = 'linearity_processed.txt'
    if sublists_of_imageSets is None:
        list_of_imageSets = gf.create_imageSets(path)
        sublists_of_imageSets = gf.separate_to_sublists(list_of_imageSets)
        del list_of_imageSets
        file_name = 'linearity_og.txt'

    results = []
    if linearity_limit is None:
        lower = np.array([LOWER_LIN_LIM, LOWER_LIN_LIM, LOWER_LIN_LIM], dtype=np.dtype('float64'))
        upper = np.array([UPPER_LIN_LIM, UPPER_LIN_LIM, UPPER_LIN_LIM], dtype=np.dtype('float64'))
    else:
        lower = np.array([linearity_limit, linearity_limit, linearity_limit], dtype=np.dtype('float64'))
        upper = np.array([MAX_DN - linearity_limit, MAX_DN - linearity_limit, MAX_DN - linearity_limit], dtype=np.dtype('float64'))

    if ICRF is None:
        lower = lower/MAX_DN
        upper = upper/MAX_DN
    else:
        for c in range(CHANNELS):
            lower[c] = ICRF[int(lower[c]), c]
            upper[c] = ICRF[int(upper[c]), c]

    # TODO: this is probably not needed at all. Depends on if I want to expand
    #   error analysis to include ImageSet.std images.
    if use_std and STD_data is None:
        STD_data = rd.read_data_from_txt(STD_FILE_NAME)

    for sublist in sublists_of_imageSets:

        # Can't do much if there are less than two images eh?
        if len(sublist) < 2:
            continue

        sublist.sort(key=lambda imageSet: imageSet.exp)
        for imageSet in sublist:
            if imageSet.acq is None:
                imageSet.load_acq()
            if False:  # use_std
                if imageSet.std is None:
                    imageSet.load_std(STD_data=STD_data)

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]
                abs_means = []
                abs_stds = []
                rel_means = []
                rel_stds = []

                ratio = x.exp / y.exp
                #if ratio < 0.05:
                #    break

                # Set values that are outside the given threshold to NaN.
                for c in range(CHANNELS):
                    y_channel = y.acq[:, :, c]
                    x_channel = x.acq[:, :, c]
                    range_mask = (y_channel < lower[c]) | (y_channel > upper[c])
                    y_channel[range_mask] = np.nan
                    y.acq[:, :, c] = y_channel
                    range_mask = (x_channel < lower[c]) | (x_channel > upper[c])
                    x_channel[range_mask] = np.nan
                    x.acq[:, :, c] = x_channel
                del range_mask

                y = gf.multiply_imageSets(y, ratio, use_std=False)
                absoluteSet = gf.subtract_imageSets(x, y, use_std=False)
                relativeSet = None

                if relative_scale:
                    relativeSet = gf.divide_imageSets(absoluteSet, y, use_std=False)

                for c in range(CHANNELS):
                    absolute_channel = absoluteSet.acq[:, :, c]
                    if relative_scale is not None:
                        relative_channel = relativeSet.acq[:, :, c]

                    # This section is used to check if enough of the pixels of
                    # a channel are finite values. If not, then channel is skipped.
                    finite_indices = np.isfinite(absolute_channel)
                    finite_count = np.count_nonzero(finite_indices)
                    if finite_count < PIXEL_COUNT*0.0001:
                        if relative_scale:
                            rel_means.append(np.nan)
                            if use_std:
                                rel_stds.append(np.nan)
                        if absolute_scale:
                            abs_means.append(np.nan)
                            if use_std:
                                abs_stds.append(np.nan)
                        print(f'Skipped with {finite_count} pixels.')
                        continue

                    rel_acq = None
                    abs_acq = None

                    if relative_scale:
                        if absolute_result:
                            rel_acq = abs(relative_channel)
                        else:
                            rel_acq = relative_channel
                    if absolute_scale:
                        if absolute_result:
                            abs_acq = abs(absolute_channel * MAX_DN)
                        else:
                            abs_acq = absolute_channel * MAX_DN

                    if abs_acq is not None:
                        channel_abs_mean = np.mean(abs_acq[finite_indices])
                        abs_means.append(channel_abs_mean)
                    if rel_acq is not None:
                        channel_rel_mean = np.mean(rel_acq[finite_indices])
                        rel_means.append(channel_rel_mean)

                    if use_std:
                        # base_std = absoluteSet.std[:, :, c]

                        abs_std = None
                        rel_std = None

                        if relative_scale:
                            rel_std = relative_channel
                        if absolute_scale:
                            abs_std = absolute_channel * MAX_DN

                        if abs_std is not None:
                            channel_abs_std = np.std(abs_std[finite_indices])
                            abs_stds.append(channel_abs_std)
                        if rel_std is not None:
                            channel_rel_std = np.std(rel_std[finite_indices])
                            rel_stds.append(channel_rel_std)

                    else:
                        abs_stds = [0, 0, 0]
                        rel_stds = [0, 0, 0]

                if not absolute_scale:
                    abs_means = [0, 0, 0]
                if not relative_scale:
                    rel_means = [0, 0, 0]

                result = [ratio,
                          abs_means[0],
                          abs_means[1],
                          abs_means[2],
                          abs_stds[0],
                          abs_stds[1],
                          abs_stds[2],
                          rel_means[0],
                          rel_means[1],
                          rel_means[2],
                          rel_stds[0],
                          rel_stds[1],
                          rel_stds[2],]
                results.append(result)
                print(f'{x.path.name}-{y.path.name}-{ratio}-{abs_means}')

    data_array = np.array(results)
    column_means = np.nanmean(data_array, axis=0)
    create_linearity_plots(data_array, save_path, True, column_means, True)
    create_linearity_plots(data_array, save_path, False, column_means, True)
    create_linearity_plots(data_array, save_path, True, column_means, False)
    create_linearity_plots(data_array, save_path, False, column_means, False)
    results.append(column_means.tolist())

    if not pass_results:
        with open(path.joinpath(file_name), 'w') as f:
            for row in results:
                f.write(f'{row}\n')
        return

    return results


def create_linearity_plots(data_array: np.ndarray, save_path: Path,
                           scale_switch: bool, means: np.ndarray,
                           fit_line: bool):

    if scale_switch:
        if fit_line:
            fig_path = save_path.parent.joinpath(save_path.name.replace(r'.png', r'_abs_fit.png'))
        else:
            fig_path = save_path.parent.joinpath(save_path.name.replace(r'.png', r'_abs.png'))
        ylabel = 'Absolute disparity'
    else:
        if fit_line:
            fig_path = save_path.parent.joinpath(save_path.name.replace(r'.png', r'_rel_fit.png'))
        else:
            fig_path = save_path.parent.joinpath(save_path.name.replace(r'.png', r'_rel.png'))
        ylabel = 'Relative disparity'

    x = data_array[:, 0]
    fig, axes = plt.subplots(1, CHANNELS, figsize=(20, 5))

    for c, ax in enumerate(axes):

        if c == 0:
            color = 'b'
        elif c == 1:
            color = 'g'
        else:
            color = 'r'

        if scale_switch:
            y = data_array[:, c+1]
            y_err = data_array[:, c+4]
            mean = means[c+1]
            std = means[c+4]
        else:
            y = data_array[:, c+7]
            y_err = data_array[:, c+10]
            mean = means[c+7]
            std = means[c+10]

        if fit_line:
            linear_model = Model(linear_function)
            fit = RealData(x, y, sy=y_err)
            odr = ODR(fit, linear_model, beta0=[0., 0.])
            odr_output = odr.run()
            line = linear_function(odr_output.beta, x)
            plot_title = f'A={odr_output.beta[0]:.4f} $\\pm$ {odr_output.sd_beta[0]:.4f},' \
                         f'B={odr_output.beta[1]:.4f} $\\pm$ {odr.output.sd_beta[1]:.4f}\n' \
                         f'Mean={mean:.4f}, $\\sigma$={std:.4f}'
        else:
            plot_title = f'Mean={mean:.4f}, $\\sigma$={std:.4f}'

        ax.errorbar(x, y, yerr=y_err, elinewidth=1,
                    c=color, marker='x', linestyle='none', markersize=3)
        if fit_line:
            ax.plot(x, line, c='black')
        ax.set_title(plot_title)

    axes[0].set(ylabel=ylabel)
    axes[1].set(xlabel='Exposure ratio')
    plt.savefig(fig_path, dpi=300)
    plt.clf()


def linear_function(B, x):
    return B[0] + B[1]*x


def linearity_distribution(long_imageSet: Optional[ImageSet] = None,
                           short_imageSet: Optional[ImageSet] = None,
                           save_dir: Optional[Path] = None,
                           num: Optional[int] = None,
                           upper: Optional[float] = UPPER_LIN_LIM,
                           lower: Optional[float] = LOWER_LIN_LIM,
                           use_std: Optional[bool] = False,
                           STD_data: Optional[np.ndarray] = None,
                           save_image: Optional[bool] = False,
                           use_relative: Optional[bool] = True,
                           use_ICRF: Optional[bool] = True):
    """
    Analyze the linearity of a pair of images by producing a distribution of the
    relative errors to the expected ratio based on exposure times.
    Args:
        use_ICRF: whether to use ICRF to adjust the rejection threshold or not.
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
        long_imageSet_path = gf.get_filepath_dialog('Choose long exposure image')
        long_imageSet = ImageSet(long_imageSet_path)
    if short_imageSet is None:
        short_imageSet_path = gf.get_filepath_dialog('Choose short exposure image')
        short_imageSet = ImageSet(short_imageSet_path)

    lower_arr = np.array([lower, lower, lower], dtype=np.dtype('float64'))
    upper_arr = np.array([upper, upper, upper], dtype=np.dtype('float64'))

    if use_ICRF:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)

        for c in range(CHANNELS):
            lower_arr[c] = ICRF[int(lower_arr[c]), c]
            upper_arr[c] = ICRF[int(upper_arr[c]), c]
    else:
        lower_arr /= MAX_DN
        upper_arr /= MAX_DN

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
            STD_data = rd.read_data_from_txt(STD_FILE_NAME)
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

    for c in range(CHANNELS):
        y_channel = y.acq[:, :, c]
        x_channel = x.acq[:, :, c]
        range_mask = (y_channel < lower_arr[c]) | (y_channel > upper_arr[c])
        y_channel[range_mask] = np.nan
        y.acq[:, :, c] = y_channel
        range_mask = (x_channel < lower_arr[c]) | (x_channel > upper_arr[c])
        x_channel[range_mask] = np.nan
        x.acq[:, :, c] = x_channel
    del range_mask

    y = gf.multiply_imageSets(y, ratio, use_std=use_std)
    linearSet = gf.subtract_imageSets(x, y, use_std=use_std)
    if use_relative:
        linearSet = gf.divide_imageSets(x, y, use_std=use_std)

    linearSet.path = save_dir.joinpath(f'{x.exp}-{y.exp} {y.subject}.tif')

    if save_image:
        linearSet.save_32bit(separate_channels=True)

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


def analyze_dark_frames(path: Path, threshold: float):

    darkSets = gf.create_imageSets(path)
    darkSets.sort(key=lambda darkSet: darkSet.exp)
    results = []

    for darkSet in darkSets:

        darkSet.load_acq()
        channel_results = [darkSet.exp]

        for c in range(CHANNELS):

            channel = darkSet.acq[:, :, c]
            mean = np.mean(channel)
            std = np.std(channel)
            percentage_above_threshold = (channel > threshold).sum()/channel.size
            channel_results.append(mean)
            channel_results.append(std)
            channel_results.append(percentage_above_threshold)

        results.append(channel_results)
        data_array = np.array(results)

        save_path = path.joinpath('dark_frame_analysis.csv')
        header = 'exp, bmean, bstd, brat, gmean, gstd, grat, rmean, rstd, rrat'
        np.savetxt(save_path, data_array, delimiter=',', header=header)

    return


def HDR_zone_analysis(imageSet_1: Optional[ImageSet] = None,
                      number_of_zones_per_side: Optional[int] = 6,
                      imageSet_2: Optional[ImageSet] = None,
                      only_cross_analysis: Optional[bool] = False):
    def compute_site_vals(imageSet: ImageSet, number_of_zones_per_side: int, ROI_dx: int, ROI_dy: int):

        site_means = np.zeros((number_of_zones_per_side, number_of_zones_per_side, CHANNELS),
                              dtype=np.dtype('float64'))
        site_stds = np.zeros((number_of_zones_per_side, number_of_zones_per_side, CHANNELS),
                             dtype=np.dtype('float64'))

        for i in range(number_of_zones_per_side):
            for j in range(number_of_zones_per_side):
                for c in range(CHANNELS):
                        site_slice = imageSet.acq[i * ROI_dx: (i + 1) * ROI_dx - 1, j * ROI_dy: (j + 1) * ROI_dy - 1, c]
                        site_means[i, j, c] = np.nanmean(site_slice)
                        site_slice = imageSet.std[i * ROI_dx: (i + 1) * ROI_dx - 1, j * ROI_dy: (j + 1) * ROI_dy - 1, c]
                        site_stds[i, j, c] = np.sqrt(np.nansum(site_slice ** 2)) / (np.count_nonzero(np.isfinite(site_slice)))

        return site_means, site_stds

    def compute_zone_ratios(site_means_1: np.ndarray, site_stds_1: np.ndarray):

        site_ratios_1 = []
        site_ratio_stds_1 = []
        indices = []

        for (i1, j1), (i2, j2) in combinations(((i, j) for i in range(number_of_zones_per_side) for j in range(number_of_zones_per_side)), 2):
            if i1 == i2 and j1 == j2:
                continue

            mean_ratios = []
            ratio_stds = []
            for c in range(CHANNELS):

                ratio = site_means_1[i1, j1, c] / site_means_1[i2, j2, c]
                mean_ratios.append(ratio)

                ratio_std = math.sqrt((site_stds_1[i1, j1, c] / site_means_1[i2, j2, c]) ** 2 +
                                      ((site_means_1[i1, j1, c]) * site_stds_1[i2, j2, c] / (
                                                  site_means_1[i2, j2, c] ** 2)) ** 2)
                ratio_stds.append(ratio_std)

            site_ratios_1.append(mean_ratios)
            site_ratio_stds_1.append(ratio_stds)
            indices.append([i1, j1, i2, j2])

        return np.array(site_ratios_1), np.array(site_ratio_stds_1), np.array(indices)

    def compute_cross_image_ratios(site_means_1: np.ndarray, site_stds_1: np.ndarray,
                                   site_means_2: np.ndarray,site_stds_2: np.ndarray):

        cross_site_mean_ratios = site_means_1 / site_means_2

        cross_site_ratios_std = np.sqrt((site_stds_1 / site_means_2) ** 2 + (site_means_1 * site_stds_2 / (site_means_2 ** 2)) ** 2)

        return cross_site_mean_ratios.reshape(number_of_zones_per_side ** 2, CHANNELS), cross_site_ratios_std.reshape(number_of_zones_per_side ** 2, CHANNELS)

    def custom_calc(x, y):
        return x/y

    imageSet_1_path = None
    imageSet_2_path = None

    if imageSet_1 is None:
        imageSet_1_path = gf.get_filepath_dialog('Choose image 1')

    if imageSet_2 is None:
        imageSet_2_path = gf.get_filepath_dialog('Choose image 1')

    if imageSet_1_path is not None:
        imageSet_1 = ImageSet(imageSet_1_path)
        imageSet_1.load_acq(bit32=True)
        imageSet_1.load_std(bit32=True)
    else:
        return

    if imageSet_2_path is not None:
        imageSet_2 = ImageSet(imageSet_2_path)

    cross_analysis = False
    if imageSet_2 is not None:
        cross_analysis = True
        imageSet_2.load_acq(bit32=True)
        imageSet_2.load_std(bit32=True)

    ROI_dx = math.floor(IM_SIZE_Y / number_of_zones_per_side)
    ROI_dy = math.floor(IM_SIZE_X / number_of_zones_per_side)

    print(f'dy: {ROI_dy}, dx: {ROI_dx}')

    site_means_1, site_stds_1 = compute_site_vals(imageSet_1, number_of_zones_per_side, ROI_dx, ROI_dy)
    if not only_cross_analysis:
        site_1_ratios, site_1_ratio_stds, indices = compute_zone_ratios(site_means_1, site_stds_1)
    if cross_analysis:
        site_means_2, site_stds_2 = compute_site_vals(imageSet_2, number_of_zones_per_side, ROI_dx, ROI_dy)
        if not only_cross_analysis:
            site_2_ratios, site_2_ratio_stds, _ = compute_zone_ratios(site_means_2, site_stds_2)
            site_ratio_diff = (site_1_ratios - site_2_ratios) / site_2_ratios
            site_ratio_std = np.sqrt((site_1_ratio_stds / site_2_ratios) ** 2 + (site_1_ratios * site_2_ratio_stds / (site_2_ratios ** 2)) ** 2)

        cross_site_ratios, cross_site_stds = compute_cross_image_ratios(site_means_1, site_stds_1, site_means_2, site_stds_2)

    if cross_analysis:
        if not only_cross_analysis:
            np.savetxt(OUTPUT_DIRECTORY.joinpath(f'{imageSet_1.path.name.replace(".tif", " ")}Site_ratio_diff.txt'), site_ratio_diff)
            np.savetxt(OUTPUT_DIRECTORY.joinpath(f'{imageSet_1.path.name.replace(".tif", " ")} Site_ratio_diff_std.txt'), site_ratio_std)
        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'{imageSet_1.path.name.replace(".tif", " ")} Cross_site_ratios.txt'), cross_site_ratios)
        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'{imageSet_1.path.name.replace(".tif", " ")} Cross_site_ratios_std.txt'), cross_site_stds)
    else:
        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'Site_1_ratios.txt'), site_1_ratios)
        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'Site_1_ratio_stds.txt'), site_1_ratio_stds)

    if not only_cross_analysis:
        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'{imageSet_1.path.name.replace(".tif", " ")} Site_indices.txt'), indices, fmt='%i')

    return


def channel_histograms(image_path: Optional[Path] = None,
                       title_x: Optional[str] = None,
                       title_y: Optional[str] = None,
                       log_scale: Optional[bool] = False):

    if image_path is None:
        image_path = gf.get_filepath_dialog('Choose data path')

    image = ImageSet(image_path)
    image.load_acq(bit32=True)
    image.load_std(bit32=True)
    data = image.acq
    error = image.std
    data = data.reshape(IM_SIZE_X * IM_SIZE_Y, CHANNELS)
    error = error.reshape(IM_SIZE_X * IM_SIZE_Y, CHANNELS)

    fig, axes = plt.subplots(1, CHANNELS, figsize=(20, 5))
    stats = np.zeros((CHANNELS, 6), dtype=np.dtype('float64'))

    for c, ax in enumerate(axes):

        if log_scale:
            ax.set_yscale('log')
        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        data_points = np.shape(data[:, c])[0]
        bin_width = 2 * iqr(data[:, c]) / (data_points ** (1 / 3))
        min_x = np.min(data[:, c])
        max_x = np.max(data[:, c])
        number_of_bins = int(np.ceil((max_x - min_x) / bin_width))

        x_range_hist = np.linspace(min_x, max_x, number_of_bins)
        histogram_w, bin_edges_w = np.histogram(data[:, c], bins=x_range_hist, weights=error[:, c])
        histogram, bin_edges = np.histogram(data[:, c], bins=x_range_hist)
        histogram_w = histogram_w / np.sum(histogram_w)
        histogram = histogram / np.sum(histogram)
        width_w = abs(bin_edges_w[0] - bin_edges_w[1])
        width = abs(bin_edges[0] - bin_edges[1])
        ax.bar(bin_edges[:-1], histogram, width=width, fc='0', alpha=0.5, label='Unweighted')
        ax.bar(bin_edges_w[:-1], histogram_w, width=width_w, fc=color, alpha=0.5, label='Weighted')

        mean_w, std_w = weighted_avg_and_std(data[:, c], error[:, c])
        mean_std_w = std_w / np.sqrt(data_points)
        mean = np.mean(data[:, c])
        std = np.std(data[:, c])
        mean_std = std / np.sqrt(data_points)

        stats[c, 0] = mean
        stats[c, 1] = mean_std
        stats[c, 2] = std
        stats[c, 3] = mean_w
        stats[c, 4] = mean_std_w
        stats[c, 5] = std_w

        ax.set_title(f'{color}: Mean = {mean_w: .4f} $\\pm$ {mean_std_w: .4f}, STD = {std_w: .4f}', fontsize=14)
        ax.legend(loc='upper right')

    stats = np.round(stats, 5)

    row_labels = ['Blue', 'Green', 'Red']
    stats = np.c_[row_labels, stats]
    col_headers_top = np.array(['', 'Unweighted', '', '', 'Weighted', '', ''])
    col_headers_mid = np.array(
        ['Channel', 'Mean', 'SD of mean', 'SD', 'Mean', 'SD of mean', 'SD'])
    table_array = np.vstack([col_headers_top, col_headers_mid, stats])
    np.savetxt(OUTPUT_DIRECTORY.joinpath(image_path.name.replace('.tif', ' stats.csv')), table_array, delimiter=',', fmt='%.15s')

    if title_x is not None:
        axes[1].set(xlabel=title_x)
        axes[1].xaxis.label.set_size(16)
    if title_y is not None:
        axes[0].set(ylabel=title_y)
        axes[0].yaxis.label.set_size(16)
    plt.savefig(OUTPUT_DIRECTORY.joinpath(image_path.name.replace('.tif', '.png')), dpi=300)
    plt.clf()

    return


def kernel_density_estimation(data_path: Optional[Path] = None,
                              error_path: Optional[Path] = None,
                              title_x: Optional[str] = None,
                              title_y: Optional[str] = None,
                              edge_color: Optional[str] = '0'):

    if data_path is None:
        data_path = gf.get_filepath_dialog('Choose data path')
    if error_path is None:
        error_path = gf.get_filepath_dialog('Choose error path')

    data = rd.read_data_from_txt(data_path.name, data_path.parent)
    error = rd.read_data_from_txt(error_path.name, error_path.parent)

    fig, axes = plt.subplots(1, CHANNELS, figsize=(20, 5))
    stats = np.zeros((CHANNELS, 9), dtype=np.dtype('float64'))

    for c, ax in enumerate(axes):

        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        data_points = np.shape(data[:, c])[0]
        bin_width = 2 * iqr(data[:, c]) / (data_points ** (1 / 3))
        min_x = np.min(data[:, c])
        max_x = np.max(data[:, c])
        number_of_bins = int(np.ceil((max_x - min_x) / bin_width))

        x_range = np.linspace(np.min(data[:, c]), np.max(data[:, c]), 1000)
        x_range_hist = np.linspace(min_x, max_x, number_of_bins)
        histogram_w, bin_edges_w = np.histogram(data[:, c], bins=x_range_hist, weights=error[:, c])
        histogram, bin_edges = np.histogram(data[:, c], bins=x_range_hist)
        histogram_w = histogram_w / np.sum(histogram_w)
        histogram = histogram / np.sum(histogram)
        width_w = abs(bin_edges_w[0] - bin_edges_w[1])
        width = abs(bin_edges[0] - bin_edges[1])
        ax.bar(bin_edges[:-1], histogram, width=width, fc='0', alpha=0.5, label='Unweighted')
        ax.bar(bin_edges_w[:-1], histogram_w, width=width_w, fc=color, ec=edge_color, alpha=0.5, label='Weighted')

        gkde = gaussian_kde(data[:, c], 0.3, weights=error[:, c])
        result = gkde.evaluate(x_range_hist)
        result = result / np.sum(result)

        mean_w, std_w = weighted_avg_and_std(data[:, c], error[:, c])
        mean_std_w = std_w / np.sqrt(data_points)
        mean = np.mean(data[:, c])
        std = np.std(data[:, c])
        mean_std = std / np.sqrt(data_points)
        mean_gke, std_gke = weighted_avg_and_std(x_range_hist, result)
        mean_std_gke = std_gke / np.sqrt(1000)

        stats[c, 0] = mean
        stats[c, 1] = mean_std
        stats[c, 2] = std
        stats[c, 3] = mean_w
        stats[c, 4] = mean_std_w
        stats[c, 5] = std_w
        stats[c, 6] = mean_gke
        stats[c, 7] = mean_std_gke
        stats[c, 8] = std_gke

        ax.set_title(f'{color}: Mean = {mean_w: .4f} $\\pm$ {mean_std_w: .4f}, STD = {std_w: .4f}', fontsize=14)

        ax.plot(x_range_hist, result, c='0', label='KDE')
        ax.legend(loc='upper right')

    stats = np.round(stats, 5)

    row_labels = ['Blue', 'Green', 'Red']
    stats = np.c_[row_labels, stats]
    col_headers_top = np.array(['', 'Unweighted', '', '', 'Weighted', '', '', 'KDE', '', ''])
    col_headers_mid = np.array(['Channel', 'Mean', 'SD of mean', 'SD', 'Mean', 'SD of mean', 'SD', 'Mean', 'SD of mean', 'SD'])
    table_array = np.vstack([col_headers_top, col_headers_mid, stats])
    np.savetxt(OUTPUT_DIRECTORY.joinpath(data_path.name.replace('.txt', ' stats.csv')), table_array, delimiter=',', fmt='%.15s')
    '''
    table = axes[1].table(table_array)
    fig.canvas.draw()
    
    gf.merge_cells(table, [{0, 0}, {1, 0}])
    gf.merge_cells(table, [{0, 1}, {0, 2}, {0, 3}])
    gf.merge_cells(table, [{0, 4}, {0, 5}, {0, 6}])
    gf.merge_cells(table, [{0, 7}, {0, 8}, {0, 9}])
    '''

    if title_x is not None:
        axes[1].set(xlabel=title_x)
        axes[1].xaxis.label.set_size(16)
    if title_y is not None:
        axes[0].set(ylabel=title_y)
        axes[0].yaxis.label.set_size(16)
    plt.savefig(OUTPUT_DIRECTORY.joinpath(data_path.name.replace('.txt', '.png')), dpi=300)
    plt.clf()

    return


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)

    return average, math.sqrt(variance)


if __name__ == "__main__":

    # linearity_distribution(lower=5, upper=250, use_std=False,
    #                        save_image=True, use_relative=False, use_ICRF=True)
    # analyze_dark_frames(DARK_PATH, 10/255)

    # HDR_zone_analysis(number_of_zones_per_side=30)
    # HDR_zone_analysis(number_of_zones_per_side=100, only_cross_analysis=True)
    # kernel_density_estimation(title_y='Normalized frequency', title_x='Ratio of cross-image zone mean pixel values',
    #                           edge_color=None)
    channel_histograms(title_y='Normalized frequency', title_x='Relative radiance')

    print('Run script from actual main file!')
