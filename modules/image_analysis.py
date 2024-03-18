import read_data
from ImageSet import ImageSet
import numpy as np
from typing import List
from typing import Optional
import general_functions as gf
import matplotlib.pyplot as plt
from scipy.odr import *
from global_settings import *


def analyze_linearity(path: Path,
                      sublists_of_imageSets: Optional[List[ImageSet]] = None,
                      pass_results: Optional[bool] = False,
                      use_std: Optional[bool] = True,
                      STD_data: Optional[np.ndarray] = None,
                      save_path: Optional[Path] = None,
                      relative_scale: Optional[bool] = True,
                      absolute_scale: Optional[bool] = True,
                      absolute_result: Optional[bool] = False,
                      ICRF: Optional[np.ndarray] = None):
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
    lower = np.array([LOWER_LIN_LIM, LOWER_LIN_LIM, LOWER_LIN_LIM], dtype=np.dtype('float32'))
    upper = np.array([UPPER_LIN_LIM, UPPER_LIN_LIM, UPPER_LIN_LIM], dtype=np.dtype('float32'))
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
        STD_data = read_data.read_data_from_txt(STD_FILE_NAME)

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
                    if finite_count < PIXEL_COUNT*0.10:
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

                result = [round(ratio, 3),
                          round(abs_means[0], 3),
                          round(abs_means[1], 3),
                          round(abs_means[2], 3),
                          round(abs_stds[0], 3),
                          round(abs_stds[1], 3),
                          round(abs_stds[2], 3),
                          round(rel_means[0], 3),
                          round(rel_means[1], 3),
                          round(rel_means[2], 3),
                          round(rel_stds[0], 3),
                          round(rel_stds[1], 3),
                          round(rel_stds[2], 3)]
                results.append(result)
                print(f'{x.path.name}-{y.path.name}-{ratio}-{abs_means}')

    data_array = np.array(results)
    column_means = np.round(np.nanmean(data_array, axis=0), decimals=3)
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
        long_imageSet_path = gf.get_filepath_dialog('Choose long exposure image')
        long_imageSet = ImageSet(long_imageSet_path)
    if short_imageSet is None:
        short_imageSet_path = gf.get_filepath_dialog('Choose short exposure image')
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


if __name__ == "__main__":

    linearity_distribution(lower=5/255, upper=250/255, use_std=False,
                           save_image=True, use_relative=False)
    # analyze_dark_frames(DARK_PATH, 10/255)

    print('Run script from actual main file!')
