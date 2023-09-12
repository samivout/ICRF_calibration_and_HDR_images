import numpy as np
import configparser
from typing import Optional
from pathlib import Path

current_directory = Path(__file__).parent.resolve()
root_directory = current_directory.parent
data_directory = root_directory.joinpath('data')


def read_config_list(key):
    """
    Read list config data from the config.ini file in the data directory.

    :param key: The keyword for a particular config list entry.

    :return: Returns a list of strings, ints or floats.
    """
    config = configparser.ConfigParser()
    config.read(data_directory.joinpath('config.ini'))
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
    config.read(data_directory.joinpath('config.ini'))
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


def read_data_from_txt(file_name: str, path: Optional[str] = None):
    """
    Load numerical data from a .txt file of given name. Defaults to data
    directory but optionally can use other paths.

    Args:
        file_name: name of the file to load.
        path: path to the file, optional.

    Returns: numpy array of the txt file.
    """
    if path is None:
        load_path = data_directory
    else:
        load_path = Path(path)

    data_array = np.loadtxt(load_path.joinpath(file_name), dtype=float)

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
