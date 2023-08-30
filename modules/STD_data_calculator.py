import numpy as np
import math
from global_settings import *


def calculate_STD(mean_data_array):

    STD_array = np.zeros(MAX_DN + 1, dtype=float)

    for i in range(MAX_DN + 1):

        bin_edges = np.linspace(0, 1, num=DATAPOINTS, dtype=float)
        hist = mean_data_array[i, :]
        nonzeros = np.nonzero(hist)
        hist = hist[nonzeros]
        bin_edges = bin_edges[nonzeros]
        counts = np.sum(hist)
        mean = np.sum(hist * bin_edges)/counts
        squared_variances = np.power((bin_edges - mean), 2) * hist
        STD = math.sqrt(np.sum(squared_variances)/counts)
        STD_array[i] = STD

    return STD_array


def process_STD_data():

    mean_data_array = np.zeros((MAX_DN + 1, DATAPOINTS, CHANNELS), dtype=int)
    STD_data = np.zeros((MAX_DN + 1, CHANNELS), dtype=float)
    for i in range(len(MEAN_DATA_FILES)):

        mean_file_name = MEAN_DATA_FILES[i]
        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        STD_data[:, i] = calculate_STD(mean_data_array[:, :, i])

    np.savetxt(os.path.join(data_directory, STD_FILE_NAME), STD_data)

    return STD_data


if __name__ == "__main__":

    print('Run script from actual main file!')
