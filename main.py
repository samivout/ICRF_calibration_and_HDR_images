import general_functions
from modules import ICRF_calibration_noise as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
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
import numpy as np
import matplotlib.pyplot as plt
from global_settings import *

energy_array = np.zeros((NUMBER_OF_HEIGHTS, CHANNELS * 2), dtype=float)


def calibrate_ICRF(initial_function: Optional[np.ndarray] = None):
    """
    Run the ICRF calibration algorithm for all the given evaluation heights in
    the config.ini file. Initial and final energies, the optimized ICRFs and
    their plots are saved for each evaluation height for all camera channels.

    :return:
    """
    x_range = np.linspace(0, 1, BITS)
    for index, height in enumerate(EVALUATION_HEIGHTS):
        ICRF_array, initial_energy_array, final_energy_array = \
            ICRF.calibration(height, LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function)

        energy_array[index, :CHANNELS] = initial_energy_array
        energy_array[index, CHANNELS:] = final_energy_array

        np.savetxt(OUTPUT_DIRECTORY.joinpath(f'ICRFs{height}.txt'), ICRF_array)

        plt.plot(x_range, ICRF_array[:, 0], color='b')
        plt.plot(x_range, ICRF_array[:, 1], color='g')
        plt.plot(x_range, ICRF_array[:, 2], color='r')
        plt.savefig(OUTPUT_DIRECTORY.joinpath(f'ICRFs{height}.png'), dpi=300)
        plt.clf()

        if height == EVALUATION_HEIGHTS[-1]:
            np.savetxt(data_directory.joinpath(f'ICRF_calibrated.txt'), ICRF_array)

    np.savetxt(OUTPUT_DIRECTORY.joinpath('EnergyArray.txt'), energy_array)

    return


def calibrate_ICRF_e(initial_function: Optional[np.ndarray] = None,
                     data_spacing: Optional[int] = 150,
                     data_limits: Optional[int] = 2):

    lower_limit = data_limits
    upper_limit = MAX_DN - data_limits

    x_range = np.linspace(0, 1, BITS)
    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_e.calibration(LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function,
                           data_spacing, lower_limit, upper_limit)

    np.savetxt(OUTPUT_DIRECTORY.joinpath(f'ICRF_exp.txt'), ICRF_array)

    plt.plot(x_range, ICRF_array[:, 0], color='b')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='r')
    plt.savefig(OUTPUT_DIRECTORY.joinpath(f'ICRF_exp.png'), dpi=300)
    plt.clf()

    return


def process_mdl():

    mdl.mean_data_plot()

    return


def process_base_data(path: Optional[str] = None,
                      include_gamma: Optional[bool] = False,
                      color_split: Optional[bool] = True):

    cdt.clean_and_interpolate_data(path)
    STD.process_STD_data(pass_result=False)
    process_CRF_database.process_CRF_data(include_gamma, color_split)
    principal_component_analysis.analyze_principal_components()

    return


def process_HDR():

    save_8bit = True
    save_32bit = False
    save_linear = True
    save_HDR = False
    pass_linear = False
    fix_artifacts = False
    gaussian_blur = False
    HDR.process_HDR_images(save_linear=save_linear,
                           save_HDR=save_HDR,
                           save_8bit=save_8bit,
                           save_32bit=save_32bit,
                           pass_linear=pass_linear,
                           fix_artifacts=fix_artifacts,
                           gaussian_blur=gaussian_blur)

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

        sub_results = ia.analyze_linearity(ACQ_PATH,
                                           sublists_of_imageSets=acq_sublists,
                                           pass_results=pass_results)
        for row in sub_results:
            linearity_results.append(row)

    with open(OUT_PATH.joinpath('large_linearity_results.txt'), 'w') as f:
        for row in linearity_results:
            f.write(f'{row}\n')
        f.close()

    return


def main():

    data_paths = [r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Mean',
                  r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Modal',
                  r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD-LBGR\mean']
    analysis_path = Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Analysis\Modal data collector\File cache\Bias correction on\srgb mean')
    init_func = np.linspace(0, 1, BITS) ** 3
    ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    data_limits = 5
    data_spacing = 2

    # mdc.collect_mean_data()
    # process_base_data(data_paths[0], include_gamma=True, color_split=True)
    # calibrate_ICRF(init_func)
    calibrate_ICRF_e(init_func, data_spacing, data_limits)
    # process_mdl()
    # image_correction.image_correction(save_to_file=True)
    # process_HDR()
    # image_calculation.calibrate_dark_frames()
    # image_calculation.calibrate_flats()
    # ia.analyze_linearity(OUT_PATH, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True, ICRF=ICRF)
    # ia.analyze_linearity(ACQ_PATH, use_std=True, absolute_result=True, absolute_scale=True, relative_scale=True)
    # ia.analyze_linearity(analysis_path, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True)
    # large_linearity_analysis(data_paths)
    # general_functions.show_image()


main()
