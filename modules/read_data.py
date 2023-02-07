import numpy as np
import os
import configparser

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')


def read_config_list(key):
    """
    Read list config data from the config.ini file in the data directory.

    :param key: The keyword for a particular config list entry.

    :return: Returns a list of strings, ints or floats.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(data_directory, 'config.ini'))
    sections = config.sections()
    data_list = []
    for section in sections:

        if key in config[section]:

            data_list = config[section][key].split(',')

            if section == 'Float data':

                data_list = [float(element) for element in data_list]

            if section == 'Integer data':

                data_list = [int(element) for element in data_list]

    return data_list


def read_config_single(key):
    """
    Read single config data from the config.ini file in the data directory.

    :param key: The keyword for a particular config entry.

    :return: Returns a single string, int, or float value.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(data_directory, 'config.ini'))
    sections = config.sections()
    single_item = ''
    for section in sections:

        if key in config[section]:

            single_item = config[section][key]

            if section == 'Float data':

                single_item = float(single_item)

            if section == 'Integer data':

                single_item = int(single_item)

    return single_item


def read_data_from_txt(file_name):
    """
    Load numerical data from a .txt file of given name in the data directory
    into a numpy array of float datatype.

    :param file_name: the name of the .txt file that contains the desired data
        to load.

    :return: Numpy array of the loaded data in float datatype.
    """
    data_array = np.loadtxt(os.path.join(data_directory, file_name),
                            dtype=float)

    return data_array


if __name__ == "__main__":

    test_file_name = 'invDorfBlueMean.txt'
    data_from_txt = read_data_from_txt(test_file_name)
    print(data_from_txt[0])
    print(data_from_txt[-1])
    heights = read_config_list('evaluation heights')
    print(heights)
    ICRF_list = read_config_list('ICRFs')
    print(ICRF_list)
