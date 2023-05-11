import numpy as np
import os
import math
import read_data as rd
from scipy.interpolate import interp1d

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
base_data_files = rd.read_config_list('camera base data')
bit_depth = rd.read_config_single('bit depth')
datapoints = rd.read_config_single('final datapoints')
channels = rd.read_config_single('channels')
bits = 2 ** bit_depth
max_DN = bits - 1
min_DN = 0
mean_data_files = rd.read_config_list('camera mean data')
cwd = os.getcwd()


def clean_data_edges(base_data_arr):

    for i in range(bits):

        dist = base_data_arr[i, :]

        m = i - 1
        if m <= 0:
            m = i

        while m > min_DN:
            if m <= min_DN + 1:
                break
            if dist[m] == 0 and dist[m - 1] == 0:
                dist[0:m] = 0
                break
            if dist[m - 1] >= dist[m] or dist[m + 1] <= dist[m]:
                dist[m] = math.floor((dist[m - 1] + dist[m + 1]) / 2)
            m -= 1

        m = i + 1
        if m >= bits:
            m = i

        while m < bits:
            if m >= max_DN - 1:
                break
            if dist[m] == 0 and dist[m + 1] == 0:
                dist[m: max_DN] = 0
                break
            if dist[m + 1] >= dist[m] or dist[m - 1] <= dist[m]:
                dist[m] = math.floor((dist[m + 1] + dist[m - 1]) / 2)
            m += 1

        m = min_DN + 1
        while m < i:
            if m >= max_DN - 1:
                break
            if dist[m] == 0 and dist[m - 1] != 0 and dist[m + 1] != 0:
                dist[m] = dist[m - 1]
            if (dist[m] == dist[m + 1] and dist[m] != 0 and dist[
                m + 1] != 0):
                dist[m + 1] += 1
                m -= 1
            m += 1

        m = max_DN - 1
        while m > i:
            if m <= min_DN + 1:
                break
            if dist[m] == 0 and dist[m - 1] != 0 and dist[m + 1] != 0:
                dist[m] = dist[m + 1]
            if (dist[m] == dist[m - 1] and dist[m] != 0 and dist[
                m - 1] != 0):
                dist[m - 1] += 1
                m += 1
            m -= 1

        base_data_arr[i, :] = dist

    return base_data_arr


def interpolate_data(clean_data_arr):

    interpolated_data = np.zeros((bits, datapoints), dtype=float)

    for i in range(bits):

        x = np.linspace(0, 1, num=bits)
        y = clean_data_arr[i, :]

        f = interp1d(x, y)

        x_new = np.linspace(0, 1, num=datapoints)
        interpolated_data[i, :] = f(x_new)

    return interpolated_data


def clean_and_interpolate_data():

    base_data_arr = np.zeros((bits, bits, channels), dtype=int)
    final_data_arr = np.zeros((bits, datapoints, channels), dtype=float)

    for c in range(channels):

        base_data_name = base_data_files[c]
        mean_data_name = mean_data_files[c]

        base_data_arr[:, :, c] = rd.read_data_from_txt(base_data_name)
        base_data_arr[:, :, c] = clean_data_edges(base_data_arr[:, :, c])
        final_data_arr[:, :, c] = interpolate_data(base_data_arr[:, :, c])

        np.savetxt(os.path.join(data_directory, mean_data_name),
                   final_data_arr[:, :, c], fmt="%d")


if __name__ == "__main__":

    print('Run script from actual main file!')