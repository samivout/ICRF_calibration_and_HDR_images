import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from typing import Optional
import general_functions as gf
from scipy.signal import savgol_filter

import general_functions
from global_settings import *

'''
Load numerical data from a .txt file of the given path into a numpy array as float data type
'''


def readData(path):
    dataArr = np.loadtxt(path, dtype=float)

    return dataArr


def plot_noise_profiles_3d(mean_data_arr):

    data_step = int(DATAPOINTS / BITS)
    x0 = 0
    x1 = 255
    y0 = 0
    y1 = 255

    for c in range(CHANNELS):

        mean_data_channel = mean_data_arr[:, :, c]
        x = np.linspace(0, 1, num=256)
        y = np.linspace(0, 1, num=256)
        x = x[x0:x1]
        y = y[y0:y1]

        X, Y = np.meshgrid(x, y)

        mean_data_channel = normalize_rows_by_sum(mean_data_channel)
        sampled_data = mean_data_channel[:, ::data_step]
        data_to_plot = sampled_data[x0:x1, y0:y1]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, data_to_plot, rstride=1, cstride=1, cmap='viridis',
                        edgecolor='none')
        ax.view_init(45, -30)
        plt.savefig(OUTPUT_DIRECTORY.joinpath(f'3d_Profiles{c}.png'))
        plt.clf()

    return


def plot_noise_profiles_2d(mean_data_array):

    number_of_profiles = 20
    lower_bound = 0
    upper_bound = 255
    bound_diff = upper_bound - lower_bound
    if number_of_profiles >= bound_diff:
        row_step = 1
    else:
        row_step = int(bound_diff/number_of_profiles)

    sampled_mean_data = mean_data_array[lower_bound:upper_bound:row_step, ::DATAPOINT_MULTIPLIER, :]
    x_range = np.linspace(0, 255, BITS)

    for c in range(CHANNELS):

        normalized_data = normalize_rows_by_sum(sampled_mean_data[:, :, c])

        for i in range(1, number_of_profiles):

            normalized_row = normalized_data[i, :]
            mode_index = np.argmax(normalized_row)
            mode = normalized_row[mode_index]
            plt.xlim(lower_bound, upper_bound)
            if np.shape(normalized_row)[0] == 64:
                print('Ree')
            plt.plot(x_range, normalized_row)
            plt.vlines(mode_index, 0, mode)

        plt.savefig(OUTPUT_DIRECTORY.joinpath(f'Profiles{c}.png'), dpi=300)
        plt.clf()

    return


def plot_heatmap(mean_data_array):

    mean_data_array = normalize_rows_by_sum(mean_data_array)

    for c in range(CHANNELS):
        ax = sns.heatmap(mean_data_array[:, :, c], square=True, norm=LogNorm())
        plt.savefig(OUTPUT_DIRECTORY.joinpath(f'Heatmap{c}.png'), dpi=700)
        plt.clf()

    return


def plot_ICRF(ICRF_array, name):

    x_range = np.linspace(0, 1, DATAPOINTS)

    plt.ylabel('Normalized irradiance')
    plt.xlabel('Normalized brightness')
    plt.plot(x_range, ICRF_array[:, 0], color='b')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='r')
    plt.savefig(OUTPUT_DIRECTORY.joinpath(name), dpi=300)
    plt.clf()


def normalize_rows_by_max(mean_data_arr):

    mean_data_arr = mean_data_arr / np.amax(mean_data_arr, axis=1, keepdims=True)

    return mean_data_arr


def normalize_rows_by_sum(mean_data_arr):

    mean_data_arr = mean_data_arr / mean_data_arr.sum(axis=1, keepdims=True)

    return mean_data_arr


def print_mean_data_mode(mean_data_array):

    modes = np.zeros((BITS, CHANNELS), dtype=int)

    for c in range(CHANNELS):
        for i in range(MAX_DN):

            noise_profile = mean_data_array[i, ::4, c]
            modes[i, c] = np.argmax(noise_profile)

    np.savetxt(OUTPUT_DIRECTORY.joinpath('modes.txt'), modes, fmt='%i')

    return


def plot_PCA():

    for i, file in enumerate(PCA_FILES):

        PCA_array = rd.read_data_from_txt(file)
        image_name = str(file)
        image_name = image_name.replace('.txt', '.png')
        datapoints, components = np.shape(PCA_array)
        x_range = np.linspace(0, 1, num=datapoints)

        for component in range(components):
            plt.plot(x_range, PCA_array[:, component])

        plt.savefig(OUTPUT_DIRECTORY.joinpath(image_name), dpi=300)
        plt.clf()


def plot_dorf_PCA():

    data_array = rd.read_data_from_txt('dorf_PCA.txt')
    x_range = data_array[:, 0]

    for i in range(2, 7):
        plt.plot(x_range, data_array[:, i])
    plt.savefig(OUTPUT_DIRECTORY.joinpath('dorf_PCA.png'), dpi=300)
    plt.clf()

    plt.plot(x_range, data_array[:, 1])
    plt.savefig(OUTPUT_DIRECTORY.joinpath('dorf_mean_ICRF.png'), dpi=300)
    plt.clf()


def plot_ICRF_PCA():
    """ The inverse camera response function calculated in terms of the mean
    ICRF, PCA vectors and the PCA coefficients that are subject to optimization.
    Dimensions of the PCA_array and PCA_params must match so that matrix
    multiplication produces an array of equal dimensions to mean_ICRF.

    Args:
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance
            datapoints.
        PCA_array: Numpy array containing the principal component vectors.
        PCA_params: The PCA coefficients for each vector in PCA_array, these
            the values subject to optimization.

    Return:
        The new iteration of the inverse camera response function.
    """
    PCA_BLUE = [-0.10908332, -1.7415221, 0.66263865, -0.23043307,  0.15340393]
    PCA_GREEN = [-0.56790662, -0.44675708, 0.08047224, 0.16562418, -0.0744729]
    PCA_RED = [0.38280571, -1.45670034, 0.27022986, 0.43637866, -0.34930558]
    PCA_Components = [PCA_BLUE, PCA_GREEN, PCA_RED]
    x_range = np.linspace(0, 1, BITS)
    for i in range(len(MEAN_DATA_FILES)):

        initial_function = x_range ** 4
        PCA_file_name = PCA_FILES[i]
        PCA_array = rd.read_data_from_txt(PCA_file_name)

        product = np.matmul(PCA_array, PCA_Components[i])
        iterated_ICRF = initial_function + product

        if i == 0:
            plt.plot(x_range, iterated_ICRF, color='b')
        if i == 1:
            plt.plot(x_range, iterated_ICRF, color='g')
        if i == 2:
            plt.plot(x_range, iterated_ICRF, color='r')

    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_manual_plot.png'), dpi=300)

    return


def mean_data_plot():

    x_range = np.linspace(0, 1, BITS)
    dx = 1 / (BITS - 1)
    mean_data_array = np.zeros((BITS, DATAPOINTS, CHANNELS), dtype=int)
    mean_ICRF_array = np.zeros((DATAPOINTS, CHANNELS), dtype=float)

    for i in range(len(MEAN_DATA_FILES)):

        mean_file_name = MEAN_DATA_FILES[i]
        mean_ICRF_file_name = MEAN_ICRF_FILES[i]

        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        mean_ICRF_array[:, i] = rd.read_data_from_txt(mean_ICRF_file_name)

    ICRF_calibrated = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    ICRF_diff = np.zeros_like(ICRF_calibrated)

    for c in range(CHANNELS):

        if c == 0:
            color = 'blue'
        elif c == 1:
            color = 'green'
        else:
            color = 'red'

        ICRF_diff[:, c] = np.gradient(ICRF_calibrated[:, c], dx)
        plt.plot(x_range, ICRF_diff[:, c], color=color)

    np.savetxt(OUTPUT_DIRECTORY.joinpath('ICRF_diff.txt'), ICRF_diff)
    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_diff.png'), dpi=300)
    plt.clf()

    plot_ICRF(mean_ICRF_array, 'mean_ICRF.png')
    plot_ICRF(ICRF_calibrated, 'ICRF_calibrated.png')
    plot_noise_profiles_2d(mean_data_array)
    plot_noise_profiles_3d(mean_data_array)
    plot_heatmap(mean_data_array)
    print_mean_data_mode(mean_data_array)


def calculate_and_plot_mean_ICRF(filepath: Optional[Path] = None):

    if filepath is None:
        filepath = gf.get_filepath_dialog('Choose ICRF file')

    path = filepath.parent
    name = filepath.name.replace('.txt', 'png')
    ICRF = rd.read_data_from_txt(filepath, str(filepath.parent))

    mean_ICRF = np.mean(ICRF, axis=1)
    np.savetxt(path.joinpath('mean_ICRF.txt'), mean_ICRF)
    x_range = np.linspace(0, 1, DATAPOINTS)

    plt.ylabel('Normalized irradiance')
    plt.xlabel('Normalized brightness')
    plt.plot(x_range, mean_ICRF)
    plt.savefig(path.joinpath(name), dpi=300)
    plt.clf()

    return


def calculate_mean_ICRF(filepath_1: Optional[Path] = None,
                        filepath_2: Optional[Path] = None):

    if filepath_1 is None:
        filepath_1 = gf.get_filepath_dialog('Choose ICRF file')

    if filepath_2 is None:
        filepath_2 = gf.get_filepath_dialog('Choose ICRF file')

    ICRF_1 = rd.read_data_from_txt(filepath_1, str(filepath_1.parent))
    ICRF_2 = rd.read_data_from_txt(filepath_1, str(filepath_2.parent))

    ICRF_mean = (ICRF_1 + ICRF_2) / 2

    np.savetxt(OUTPUT_DIRECTORY.joinpath('combined_ICRF.txt'), ICRF_mean)

    return


def plot_two_ICRF_and_calculate_RMSE(filepath1: Optional[Path] = None,
                                     filepath2: Optional[Path] = None):

    if filepath1 is None:
        filepath1 = gf.get_filepath_dialog('Choose ICRF file')

    if filepath2 is None:
        filepath2 = gf.get_filepath_dialog('Choose ICRF file')

    path = filepath2.parent
    name = 'ICRF_RMSE.png'

    ICRF1 = rd.read_data_from_txt(filepath1, str(filepath1.parent))
    ICRF2 = rd.read_data_from_txt(filepath2, str(filepath2.parent))

    RMSE = np.sqrt(np.mean((ICRF1-ICRF2)**2))

    x_range = np.linspace(0, 1, DATAPOINTS)
    plot_title = f'RMSE: {RMSE:.4f}'

    plt.title(plot_title)
    plt.ylabel('Normalized irradiance')
    plt.xlabel('Normalized brightness')
    plt.plot(x_range, ICRF1, c='r')
    plt.plot(x_range, ICRF2, c='b')
    plt.savefig(path.joinpath(name), dpi=300)
    plt.clf()


def smoothen_ICRF(ICRF_path: Optional[Path] = None):

    x_range = np.linspace(0, 1, BITS)
    dx = 2 / (BITS - 1)

    if ICRF_path is None:
        ICRF_path = gf.get_filepath_dialog('Choose ICRF file')

    ICRF = rd.read_data_from_txt(ICRF_path.name, ICRF_path.parent)
    ICRF_smoothed = np.zeros_like(ICRF)
    ICRF_smoothed_diff = np.zeros_like(ICRF)

    fig, axes = plt.subplots(1, CHANNELS, figsize=(20, 5))

    for c, ax in enumerate(axes):

        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        ICRF_smoothed[:, c] = savgol_filter(ICRF[:, c], 6, 2)
        ICRF_smoothed[0, c] = 0
        ICRF_smoothed[-1, c] = 1
        ICRF_smoothed_diff[:, c] = np.gradient(ICRF_smoothed[:, c], dx)
        RMSE = np.sqrt(np.mean(ICRF_smoothed[:, c] - ICRF[:, c]) ** 2)
        ax.plot(x_range, ICRF_smoothed[:, c], color='red')
        ax.plot(x_range, ICRF[:, c], color='blue', alpha=0.6)
        ax.set_title(f'{color}: RMSE = {RMSE: .4f}')

    np.savetxt(OUTPUT_DIRECTORY.joinpath('ICRF_smoothed.txt'), ICRF_smoothed)
    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_smoothed.png'), dpi=300)
    plt.clf()

    for c in range(CHANNELS):

        if c == 0:
            color = 'Blue'
        elif c == 1:
            color = 'Green'
        else:
            color = 'Red'

        plt.plot(x_range, ICRF_smoothed_diff[:, c], color=color)

    plt.axvline(LOWER_LIN_LIM/MAX_DN, color='0', alpha=0.5)
    plt.axvline(UPPER_LIN_LIM/MAX_DN, color='0', alpha=0.5)
    plt.savefig(OUTPUT_DIRECTORY.joinpath('ICRF_smoothed_diff.png'), dpi=300)

    return



if __name__ == "__main__":

    # calculate_mean_ICRF()
    mean_data_plot()
    # plot_PCA()
    # plot_dorf_PCA()
    # plot_ICRF_PCA()
    # calculate_and_plot_mean_ICRF()
    # plot_two_ICRF_and_calculate_RMSE()
    # smoothen_ICRF()
