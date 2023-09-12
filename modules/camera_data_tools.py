import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Optional
from global_settings import *


def clean_data_edges(base_data_arr):

    for i in range(BITS):

        dist = base_data_arr[i, :]

        m = i - 1
        if m <= 0:
            m = i

        while m > MIN_DN:
            if m <= MIN_DN + 1:
                break
            if dist[m] == 0 and dist[m - 1] == 0:
                dist[0:m] = 0
                break
            if dist[m - 1] >= dist[m] or dist[m + 1] <= dist[m]:
                dist[m] = math.floor((dist[m - 1] + dist[m + 1]) / 2)
            m -= 1

        m = i + 1
        if m >= BITS:
            m = i

        while m < BITS:
            if m >= MAX_DN - 1:
                break
            if dist[m] == 0 and dist[m + 1] == 0:
                dist[m: MAX_DN] = 0
                break
            if dist[m + 1] >= dist[m] or dist[m - 1] <= dist[m]:
                dist[m] = math.floor((dist[m + 1] + dist[m - 1]) / 2)
            m += 1

        m = MIN_DN + 1
        while m < i:
            if m >= MAX_DN - 1:
                break
            if dist[m] == 0 and dist[m - 1] != 0 and dist[m + 1] != 0:
                dist[m] = dist[m - 1]
            if (dist[m] == dist[m + 1] and dist[m] != 0 and dist[
                m + 1] != 0):
                dist[m + 1] += 1
                m -= 1
            m += 1

        m = MAX_DN - 1
        while m > i:
            if m <= MIN_DN + 1:
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

    if BITS == DATAPOINTS:
        return clean_data_arr

    interpolated_data = np.zeros((BITS, DATAPOINTS), dtype=float)

    for i in range(BITS):

        x = np.linspace(0, 1, num=BITS)
        y = clean_data_arr[i, :]

        f = interp1d(x, y)

        x_new = np.linspace(0, 1, num=DATAPOINTS)
        interpolated_data[i, :] = f(x_new)

    return interpolated_data


def clean_and_interpolate_data(path: Optional[str] = None):

    base_data_arr = np.zeros((BITS, BITS, CHANNELS), dtype=int)
    final_data_arr = np.zeros((BITS, DATAPOINTS, CHANNELS), dtype=float)

    for c in range(CHANNELS):

        base_data_name = BASE_DATA_FILES[c]
        mean_data_name = MEAN_DATA_FILES[c]

        if not path:
            base_data_arr[:, :, c] = rd.read_data_from_txt(base_data_name)
        else:
            base_data_arr[:, :, c] = rd.read_data_from_txt(base_data_name, path)

        base_data_arr[:, :, c] = clean_data_edges(base_data_arr[:, :, c])
        final_data_arr[:, :, c] = interpolate_data(base_data_arr[:, :, c])

        np.savetxt(data_directory.joinpath(mean_data_name), final_data_arr[:, :, c], fmt="%d")


if __name__ == "__main__":

    print('Run script from actual main file!')