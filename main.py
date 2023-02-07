from modules import ICRF_calibration_algorithm as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
from modules import read_data as rd
import os
import numpy as np
import matplotlib.pyplot as plt

current_directory = os.path.dirname(__file__)
output_directory = os.path.join(current_directory, 'output')
channels = rd.read_config_single('channels')
datapoints = rd.read_config_single('final datapoints')
evaluation_heights = rd.read_config_list('evaluation heights')
number_of_heights = len(evaluation_heights)
energy_array = np.zeros((number_of_heights, channels*2), dtype=float)
lower_limit = rd.read_config_single('lower PC coefficient limit')
upper_limit = rd.read_config_single('upper PC coefficient limit')
inGuess = rd.read_config_list('initial guess')


def process_CRFs():

    process_CRF_database.process_CRF_data()

    return


def process_PCA():

    principal_component_analysis.analyze_principal_components()

    return


def calibrate_ICRF():

    for index, height in enumerate(evaluation_heights):
        ICRF_array, initial_energy_array, final_energy_array = \
            ICRF.calibration(inGuess, height, lower_limit, upper_limit)

        energy_array[index, :channels] = initial_energy_array
        energy_array[index, channels:] = final_energy_array

        np.savetxt(os.path.join(output_directory,
                                f'ICRFs{height}.txt'), ICRF_array)

        x_range = np.linspace(0, 1, datapoints)
        plt.plot(x_range, ICRF_array[:, 0], color='b')
        plt.plot(x_range, ICRF_array[:, 1], color='g')
        plt.plot(x_range, ICRF_array[:, 2], color='r')
        plt.savefig(os.path.join(output_directory, f'ICRFs{height}.png'))
        plt.clf()

    np.savetxt(os.path.join(output_directory, 'EnergyArray.txt'), energy_array)

    return


def main():

    # process_CRFs()
    # process_PCA()
    calibrate_ICRF()


main()
