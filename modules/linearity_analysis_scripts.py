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
import time
import timeit

current_directory = os.path.dirname(__file__)
root_directory = os.path.dirname(current_directory)
data_directory = os.path.join(root_directory, 'data')
acq_path = rd.read_config_single('acquired images path')
out_path = rd.read_config_single('corrected output path')
output_directory = os.path.join(root_directory, 'output')
channels = rd.read_config_single('channels')
datapoints = rd.read_config_single('final datapoints')
evaluation_heights = rd.read_config_list('evaluation heights')
number_of_heights = len(evaluation_heights)
energy_array = np.zeros((number_of_heights, channels*2), dtype=np.dtype('float32'))
lower_limit = rd.read_config_single('lower PC coefficient limit')
upper_limit = rd.read_config_single('upper PC coefficient limit')
in_guess = rd.read_config_list('initial guess')
bit_depth = rd.read_config_single('bit depth')
bits = 2**bit_depth
max_DN = bits-1
min_DN = 0
linear_scale = np.linspace(0, 1, bits)


def calibrate_ICRF_noise(height: int, initial_function: Optional[np.ndarray] = None):
    """
    Run the ICRF calibration algorithm for all the given evaluation heights in
    the config.ini file. Initial and final energies, the optimized ICRFs and
    their plots are saved for each evaluation height for all camera channels.

    :return:
    """

    energy_arr = np.zeros((channels*2), dtype=float)
    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF.calibration(in_guess, height, lower_limit, upper_limit,
                         initial_function)

    energy_arr[:channels] = initial_energy_array
    energy_arr[channels:] = final_energy_array

    return ICRF_array


def calibrate_ICRF_exposure(initial_function: Optional[np.ndarray] = None):

    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_e.calibration(lower_limit, upper_limit, initial_function)

    return ICRF_array


def process_base_data(path: Optional[str] = None, include_gamma: Optional[bool] = False):

    cdt.clean_and_interpolate_data(path)
    STD.process_STD_data()
    process_CRF_database.process_CRF_data(include_gamma)
    principal_component_analysis.analyze_principal_components()

    return


def save_and_plot_ICRF(ICRF_array: np.ndarray, name: str, path: str):

    np.savetxt(os.path.join(path, f"{name}.txt"), ICRF_array)

    plt.plot(linear_scale, ICRF_array[:, 0], color='b')
    plt.plot(linear_scale, ICRF_array[:, 1], color='g')
    plt.plot(linear_scale, ICRF_array[:, 2], color='r')
    plt.savefig(os.path.join(path, f'{name}.png'), dpi=300)

    plt.clf()

    np.savetxt(os.path.join(data_directory, f'ICRF_calibrated.txt'), ICRF_array)


def large_linearity_analysis(heights: Optional[List[int]] = [50],
                             data_paths: Optional[List[str]] = [data_directory],
                             include_gamma: Optional[bool] = False,
                             powers: Optional[List[float]] = None,
                             use_noise: Optional[bool] = True):

    save_8bit = False
    save_32bit = False
    save_linear = False
    save_HDR = False
    pass_linear = True
    fix_artifacts = False
    pass_results = True
    results = []

    for path in data_paths:

        results.append(f"{path}\n")

        process_base_data(path, include_gamma)
        time.sleep(0.5)

        for height in heights:

            start_time = timeit.default_timer()
            if use_noise:
                ICRF_array = calibrate_ICRF_noise(height)
                save_and_plot_ICRF(ICRF_array, f"ICRFn_mean_h{height}", path)
            else:
                ICRF_array = calibrate_ICRF_exposure()
                save_and_plot_ICRF(ICRF_array, f"ICRFe_mean", path)
            duration = timeit.default_timer() - start_time

            acq_sublists = HDR.process_HDR_images(save_linear=save_linear,
                                                  save_HDR=save_HDR,
                                                  save_8bit=save_8bit,
                                                  save_32bit=save_32bit,
                                                  pass_linear=pass_linear,
                                                  fix_artifacts=fix_artifacts,
                                                  ICRF_arr=ICRF_array)

            sub_results = ia.analyze_linearity(acq_path,
                                               sublists_of_imageSets=acq_sublists,
                                               pass_results=pass_results)
            results.append(f"Mean\t{height}\t{duration}\t{sub_results[-1]}")

            if powers is not None:
                for power in powers:

                    initial_function = linear_scale ** power
                    start_time = timeit.default_timer()
                    if use_noise:
                        ICRF_array = calibrate_ICRF_noise(height, initial_function)
                        save_and_plot_ICRF(ICRF_array,
                                           f"ICRFn_h{height}_p{power}",
                                           output_directory)
                    else:
                        ICRF_array = calibrate_ICRF_exposure()
                        save_and_plot_ICRF(ICRF_array,
                                           f"ICRFe_p{power}",
                                           output_directory)
                    duration = timeit.default_timer() - start_time


                    acq_sublists = HDR.process_HDR_images(
                        save_linear=save_linear,
                        save_HDR=save_HDR,
                        save_8bit=save_8bit,
                        save_32bit=save_32bit,
                        pass_linear=pass_linear,
                        fix_artifacts=fix_artifacts,
                        ICRF_arr=ICRF_array)

                    sub_results = ia.analyze_linearity(acq_path,
                                                       sublists_of_imageSets=acq_sublists,
                                                       pass_results=pass_results)
                    results.append(f"{power}\t{height}\t{duration}\t{sub_results[-1]}")
    if use_noise:
        file_name = f"large_linearity_results_noise.txt"
    else:
        file_name = f"large_linearity_results_exposure.txt"
    with open(os.path.join(out_path, file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


if __name__ == "__main__":
    data_paths = [
        r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Mean',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Modal',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Mean',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Modal'
    ]
    include_gamma = False
    # heights = [1, 2, 5, 10, 20, 40, 80, 160]
    heights = [50]
    powers = [2]
    use_noise = True
    large_linearity_analysis(heights=heights, data_paths=data_paths,
                             include_gamma=include_gamma, powers=powers,
                             use_noise=use_noise)
