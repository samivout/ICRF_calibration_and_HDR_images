from modules import ICRF_calibration_algorithm as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
from modules import read_data as rd
from modules import image_calculation
from modules import HDR_image as HDR
from modules import STD_data_calculator as STD
from modules import mean_data_plot as mdl
from modules import camera_data_tools as cdt
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
in_guess = rd.read_config_list('initial guess')
bit_depth = rd.read_config_single('bit depth')
bits = 2**bit_depth
max_DN = bits-1
min_DN = 0


def process_CRFs():
    """
    Run the scripts that gather data from dorfCurves.txt and save the data into
    the ICRF and ICRF_mean files for each channel.

    :return:
    """
    process_CRF_database.process_CRF_data()

    return


def process_PCA():
    """
    Run the scripts that utilize the ICRF data to calculate the principal
    components that are utilized for the ICRF calibration algorithm.

    :return:
    """
    principal_component_analysis.analyze_principal_components()

    return


def calibrate_ICRF():
    """
    Run the ICRF calibration algorithm for all the given evaluation heights in
    the config.ini file. Initial and final energies, the optimized ICRFs and
    their plots are saved for each evaluation height for all camera channels.

    :return:
    """
    x_range = np.linspace(0, 1, bits)
    for index, height in enumerate(evaluation_heights):
        ICRF_array, initial_energy_array, final_energy_array = \
            ICRF.calibration(in_guess, height, lower_limit, upper_limit)

        energy_array[index, :channels] = initial_energy_array
        energy_array[index, channels:] = final_energy_array

        np.savetxt(os.path.join(output_directory,
                                f'ICRFs{height}.txt'), ICRF_array)

        plt.plot(x_range, ICRF_array[:, 0], color='r')
        plt.plot(x_range, ICRF_array[:, 1], color='g')
        plt.plot(x_range, ICRF_array[:, 2], color='b')
        plt.savefig(os.path.join(output_directory, f'ICRFs{height}.png'), dpi=300)
        plt.clf()

    np.savetxt(os.path.join(output_directory, 'EnergyArray.txt'), energy_array)

    return


def process_HDR():

    save_8bit = False
    save_32bit = True
    save_linear = False
    save_HDR = True
    HDR.process_HDR_images(save_linear, save_HDR, save_8bit, save_32bit)

    return


def process_STD():

    STD.process_STD_data()

    return


def process_mdl():

    mdl.mean_data_plot()

    return


def process_cdt():

    cdt.clean_and_interpolate_data()

    return


def main():

    # process_cdt()
    # process_STD()
    # process_CRFs()
    # process_PCA()
    # calibrate_ICRF()
    # process_mdl()
    # image_calculation.image_correction()
    process_HDR()
    # image_calculation.calibrate_dark_frames()
    # image_calculation.calibrate_flats()


main()
