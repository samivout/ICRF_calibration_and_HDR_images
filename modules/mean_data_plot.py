import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import read_data as rd
import seaborn as sns

current_directory = os.path.dirname(__file__)
output_directory = os.path.join(os.path.dirname(current_directory), 'output')
datapoints = rd.read_config_single('final datapoints')
datapoint_multiplier = rd.read_config_single('datapoint multiplier')
channels = rd.read_config_single('channels')
bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth
min_DN = 0
mean_data_files = rd.read_config_list('camera mean data')
mean_ICRF_files = rd.read_config_list('mean ICRFs')

cwd = os.getcwd()

'''
Load numerical data from a .txt file of the given path into a numpy array as float data type
'''


def readData(path):
    dataArr = np.loadtxt(path, dtype=float)

    return dataArr


def plot_noise_profiles_3d(mean_data_arr):

    data_step = int(datapoints / max_DN)
    x0 = 0
    x1 = 255
    y0 = 0
    y1 = 255

    for c in range(channels):

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
        plt.savefig(os.path.join(output_directory, f'3d_Profiles{c}.png'))
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

    sampled_mean_data = mean_data_array[lower_bound:upper_bound:row_step, ::datapoint_multiplier, :]
    x_range = np.linspace(0, 255, max_DN)

    for c in range(channels):

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

        plt.savefig(os.path.join(output_directory, f'Profiles{c}.png'), dpi=300)
        plt.clf()

    return


def plot_heatmap(mean_data_array):

    mean_data_array = normalize_rows_by_sum(mean_data_array)

    for c in range(channels):
        ax = sns.heatmap(mean_data_array[:, :, c], square=True, norm=LogNorm())
        plt.savefig(os.path.join(output_directory, f'Heatmap{c}.png'), dpi=700)
        plt.clf()

    return


def plot_ICRF(ICRF_array):

    x_range = np.linspace(0, 1, datapoints)

    plt.plot(x_range, ICRF_array[:, 0], color='r')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='b')
    plt.savefig(os.path.join(output_directory, f'ICRF.png'), dpi=300)
    plt.clf()


def normalize_rows_by_max(mean_data_arr):

    mean_data_arr = mean_data_arr / np.amax(mean_data_arr, axis=1, keepdims=True)

    return mean_data_arr


def normalize_rows_by_sum(mean_data_arr):

    mean_data_arr = mean_data_arr / mean_data_arr.sum(axis=1, keepdims=True)

    return mean_data_arr


def print_mean_data_mode(mean_data_array):

    modes = np.zeros((max_DN, channels), dtype=int)

    for c in range(channels):
        for i in range(max_DN):

            noise_profile = mean_data_array[i, ::4, c]
            modes[i, c] = np.argmax(noise_profile)

    np.savetxt(os.path.join(output_directory, 'modes.txt'), modes, fmt='%i')

    return


def mean_data_plot():

    mean_data_array = np.zeros((max_DN, datapoints, channels), dtype=int)
    mean_ICRF_array = np.zeros((datapoints, channels), dtype=float)

    for i in range(len(mean_data_files)):

        mean_file_name = mean_data_files[i]
        mean_ICRF_file_name = mean_ICRF_files[i]

        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        mean_ICRF_array[:, i] = rd.read_data_from_txt(mean_ICRF_file_name)

    plot_ICRF(mean_ICRF_array)
    plot_noise_profiles_2d(mean_data_array)
    plot_noise_profiles_3d(mean_data_array)
    plot_heatmap(mean_data_array)
    print_mean_data_mode(mean_data_array)


if __name__ == "__main__":

    print('Run script from actual main file!')
