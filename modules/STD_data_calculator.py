import numpy as np
import read_data as rd
import os
import math

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
im_size_x = rd.read_config_single('image size x')
im_size_y = rd.read_config_single('image size y')
mean_data_files = rd.read_config_list('camera mean data')
bit_depth = rd.read_config_single('bit depth')
max_DN = 2**bit_depth - 1
min_DN = 0
datapoints = rd.read_config_single('final datapoints')
channels = rd.read_config_single('channels')
STD_file_name = rd.read_config_single('STD data')


def calculate_STD(mean_data_array):

    STD_array = np.zeros(max_DN+1, dtype=float)

    for i in range(max_DN+1):

        bin_edges = np.linspace(0, 1, num=datapoints, dtype=float)
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

    mean_data_array = np.zeros((max_DN+1, datapoints, channels), dtype=int)
    STD_data = np.zeros((max_DN+1, channels), dtype=float)
    for i in range(len(mean_data_files)):

        mean_file_name = mean_data_files[i]
        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        STD_data[:, i] = calculate_STD(mean_data_array[:, :, i])

    np.savetxt(os.path.join(data_directory, STD_file_name), STD_data)

    return


if __name__ == "__main__":

    print('Run script from actual main file!')
