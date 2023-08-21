from modules import ICRF_calibration_noise as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
from modules import read_data as rd
from modules import image_correction
from modules import HDR_image as HDR
from modules import STD_data_calculator as STD
from modules import mean_data_plot as mdl
from modules import camera_data_tools as cdt
from modules import image_analysis as ia
from modules import mean_data_collector as mdc
from modules import ICRF_calibration_exposure as ICRF_e
from typing import Optional
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(current_directory, 'data')
acq_path = rd.read_config_single('acquired images path')
out_path = rd.read_config_single('corrected output path')
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

        plt.plot(x_range, ICRF_array[:, 0], color='b')
        plt.plot(x_range, ICRF_array[:, 1], color='g')
        plt.plot(x_range, ICRF_array[:, 2], color='r')
        plt.savefig(os.path.join(output_directory, f'ICRFs{height}.png'), dpi=300)
        plt.clf()

        if height == evaluation_heights[-1]:
            np.savetxt(os.path.join(data_directory,
                                    f'ICRF_calibrated.txt'), ICRF_array)

    np.savetxt(os.path.join(output_directory, 'EnergyArray.txt'), energy_array)

    return


def calibrate_ICRF_e():

    x_range = np.linspace(0, 1, bits)
    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_e.calibration(lower_limit, upper_limit)

    np.savetxt(os.path.join(output_directory,
                            f'ICRF_exp.txt'), ICRF_array)

    plt.plot(x_range, ICRF_array[:, 0], color='b')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='r')
    plt.savefig(os.path.join(output_directory, f'ICRF_exp.png'), dpi=300)
    plt.clf()

    return


def process_mdl():

    mdl.mean_data_plot()

    return


def process_base_data(path: Optional[str] = None):

    cdt.clean_and_interpolate_data(path)
    STD.process_STD_data()
    process_CRF_database.process_CRF_data()
    principal_component_analysis.analyze_principal_components()

    return


def process_HDR():

    save_8bit = True
    save_32bit = False
    save_linear = True
    save_HDR = False
    pass_linear = False
    fix_artifacts = False
    HDR.process_HDR_images(save_linear=save_linear,
                           save_HDR=save_HDR,
                           save_8bit=save_8bit,
                           save_32bit=save_32bit,
                           pass_linear=pass_linear,
                           fix_artifacts=fix_artifacts)

    return


def large_linearity_analysis_noise(paths: Optional[List[str]] = [data_directory]):

    linearity_results = []

    save_8bit = False
    save_32bit = False
    save_linear = False
    save_HDR = False
    pass_linear = True
    fix_artifacts = False
    pass_results = True

    for path in paths:

        linearity_results.append(f'{path}\n\n')

        process_base_data(path)
        calibrate_ICRF()

        acq_sublists = HDR.process_HDR_images(save_linear=save_linear,
                                              save_HDR=save_HDR,
                                              save_8bit=save_8bit,
                                              save_32bit=save_32bit,
                                              pass_linear=pass_linear,
                                              fix_artifacts=fix_artifacts)

        sub_results = ia.analyze_linearity(acq_path,
                                           sublists_of_imageSets=acq_sublists,
                                           pass_results=pass_results)
        for row in sub_results:
            linearity_results.append(row)

    with open(os.path.join(out_path, 'large_linearity_results.txt'), 'w') as f:
        for row in linearity_results:
            f.write(f'{row}\n')
        f.close()

    return


def main():

    data_paths = [r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Mean',
                  r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Modal']
    # mdc.collect_mean_data()
    # process_base_data()
    calibrate_ICRF()
    # calibrate_ICRF_e()
    # process_mdl()
    # image_correction.image_correction(save_to_file=True)
    # process_HDR()
    # image_calculation.calibrate_dark_frames()
    # image_calculation.calibrate_flats()
    # ia.analyze_linearity(out_path, original_std=False)
    # ia.analyze_linearity(acq_path, original_std=True)
    # large_linearity_analysis(data_paths)


main()
