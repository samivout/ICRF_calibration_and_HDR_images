from modules import ICRF_calibration_noise as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
from modules import HDR_image as HDR
from modules import STD_data_calculator as STD
from modules import camera_data_tools as cdt
from modules import image_analysis as ia
from modules import ICRF_calibration_exposure as ICRF_e
from typing import Optional
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from global_settings import *

linear_scale = np.linspace(0, 1, BITS)


def calibrate_ICRF_noise(height: int, initial_function: Optional[np.ndarray] = None):
    """
    Run the ICRF calibration algorithm for all the given evaluation heights in
    the config.ini file. Initial and final energies, the optimized ICRFs and
    their plots are saved for each evaluation height for all camera channels.

    :return:
    """

    energy_arr = np.zeros((CHANNELS * 2), dtype=float)
    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF.calibration(height, LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function)

    energy_arr[:CHANNELS] = initial_energy_array
    energy_arr[CHANNELS:] = final_energy_array

    return ICRF_array, energy_arr


def calibrate_ICRF_exposure(initial_function: Optional[np.ndarray] = None,
                            data_spacing: Optional[int] = 150,
                            lower_data_limit: Optional[int] = 2,
                            upper_data_limit: Optional[int] = 253):

    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_e.calibration(LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function,
                           data_spacing, lower_data_limit, upper_data_limit)

    return ICRF_array


def process_base_data(path: Optional[str] = None, include_gamma: Optional[bool] = False):

    cdt.clean_and_interpolate_data(path)
    STD_data = STD.process_STD_data()
    process_CRF_database.process_CRF_data(include_gamma)
    principal_component_analysis.analyze_principal_components()

    return STD_data


def save_and_plot_ICRF(ICRF_array: np.ndarray, name: str, path: Path):

    np.savetxt(path.joinpath(f"{name}.txt"), ICRF_array)

    plt.plot(linear_scale, ICRF_array[:, 0], color='b')
    plt.plot(linear_scale, ICRF_array[:, 1], color='g')
    plt.plot(linear_scale, ICRF_array[:, 2], color='r')
    plt.savefig(path.joinpath(f'{name}.png'), dpi=300)

    plt.clf()

    np.savetxt(data_directory.joinpath(f'ICRF_calibrated.txt'), ICRF_array)


def linearity_analysis_exposure(data_paths: Optional[List[Path]] = [data_directory],
                                include_gamma: Optional[bool] = False,
                                limits: Optional[List[int]] = None,
                                data_spacing: Optional[List[int]] = None):
    """
    Function that runs a large scale linearity analysis.
    Args:
        data_paths: List of paths from which to source camera noise data
        include_gamma: whether to include gamma functions from CRF database
        limits: list of pixel value limits used to set rejection range
        data_spacing: determines amount of pixels sampled in exposure-based
            ICRF calibration.
    """
    def linearity_cycle(path: Path, init_func: Optional[np.ndarray] = None):
        """
        Inner funciton to run a single cycle of linearity analysis.
        Args:
            path: the path from which the camera data distribution is sourced
                from.
            init_func: The initial function passed to ICRF calibration
        """
        start_time = timeit.default_timer()
        lower = MIN_DN + limit
        upper = MAX_DN - limit
        ICRF_array = calibrate_ICRF_exposure(initial_function=init_func,
                                             data_spacing=data_spacing,
                                             lower_data_limit=lower,
                                             upper_data_limit=upper)
        if init_func is None:
            save_and_plot_ICRF(ICRF_array, f"ICRFe_mean_l{limit}_s{spacing}", path)
            scatter_path = path.joinpath(f"SctrE_mean_l{limit}_s{spacing}.png")
        else:
            save_and_plot_ICRF(ICRF_array, f"ICRFe_l{limit}_s{spacing}", path)
            scatter_path = path.joinpath(f"SctrE_l{limit}_s{spacing}.png")

        duration = timeit.default_timer() - start_time

        acq_sublists = HDR.process_HDR_images(save_linear=save_linear,
                                              save_HDR=save_HDR,
                                              save_8bit=save_8bit,
                                              save_32bit=save_32bit,
                                              pass_linear=pass_linear,
                                              fix_artifacts=fix_artifacts,
                                              ICRF=ICRF_array,
                                              STD_data=STD_data)

        linearity_result = ia.analyze_linearity(OUT_PATH,
                                                sublists_of_imageSets=acq_sublists,
                                                pass_results=pass_results,
                                                STD_data=STD_data,
                                                save_path=scatter_path,
                                                relative_scale=use_relative,
                                                absolute_result=absolute_result)

        if init_func is None:
            results.append(f"Mean\t{limit}\t{spacing}\t{duration}\t{linearity_result[-1]}")
        else:
            results.append(f"Power\t{limit}\t{spacing}\t{duration}\t{linearity_result[-1]}")

        return

    save_8bit = False
    save_32bit = False
    save_linear = False
    save_HDR = False
    pass_linear = True
    fix_artifacts = False
    pass_results = True
    use_relative = True
    absolute_result = False
    results = []

    if limits is None:
        limits = [2]
    if data_spacing is None:
        data_spacing = [150]

    for data_path in data_paths:

        results.append(f"{data_path}\n")

        STD_data = process_base_data(data_path, include_gamma)
        time.sleep(0.5)

        for limit in limits:
            for spacing in data_spacing:

                initial_function = None
                linearity_cycle(data_path, initial_function)

        for limit in limits:
            for spacing in data_spacing:

                initial_function = linear_scale
                linearity_cycle(data_path, initial_function)

    file_name = f"large_linearity_results_exposure.txt"
    with open(OUT_PATH.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


def linearity_analysis_noise(heights: Optional[List[int]] = None,
                             data_paths: Optional[List[Path]] = [data_directory],
                             include_gamma: Optional[bool] = False):

    def linearity_cycle(path: Path, init_func: Optional[np.ndarray] = None):
        """
        Inner funciton to run a single cycle of linearity analysis.
        Args:
            path: the path from which the camera data distribution is sourced
                from.
            init_func: The initial function passed to ICRF calibration
        """
        start_time = timeit.default_timer()
        ICRF_array, energy_array = calibrate_ICRF_noise(height, initial_function=init_func)
        if init_func is None:
            save_and_plot_ICRF(ICRF_array, f"ICRFn_mean_h{height}", path)
            scatter_path = path.joinpath(f"SctrN_mean_h{height}.png")
        else:
            save_and_plot_ICRF(ICRF_array, f"ICRFn_h{height}", path)
            scatter_path = path.joinpath(f"SctrN_h{height}.png")

        duration = timeit.default_timer() - start_time

        acq_sublists = HDR.process_HDR_images(save_linear=save_linear,
                                              save_HDR=save_HDR,
                                              save_8bit=save_8bit,
                                              save_32bit=save_32bit,
                                              pass_linear=pass_linear,
                                              fix_artifacts=fix_artifacts,
                                              ICRF=ICRF_array,
                                              STD_data=STD_data)

        linearity_result = ia.analyze_linearity(OUT_PATH,
                                                sublists_of_imageSets=acq_sublists,
                                                pass_results=pass_results,
                                                STD_data=STD_data,
                                                save_path=scatter_path,
                                                relative_scale=use_relative,
                                                absolute_result=absolute_result)

        if init_func is None:
            results.append(f"Mean\t{height}\t{duration}\t{linearity_result[-1]}")
        else:
            results.append(f"Power\t{height}\t{duration}\t{linearity_result[-1]}")

        return

    save_8bit = False
    save_32bit = False
    save_linear = False
    save_HDR = False
    pass_linear = True
    fix_artifacts = False
    pass_results = True
    use_relative = True
    absolute_result = False
    results = []

    if heights is None:
        heights = [50]

    for data_path in data_paths:

        results.append(f"{data_path}\n")

        STD_data = process_base_data(data_path, include_gamma)
        time.sleep(0.5)

        for height in heights:

            initial_function = None
            linearity_cycle(data_path, initial_function)

        for height in heights:

            initial_function = linear_scale
            linearity_cycle(data_path, initial_function)

    file_name = f"large_linearity_results_noise.txt"

    with open(OUT_PATH.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


def run_linearity_analysis():

    data_paths = [
        Path(r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Mean'),
        Path(r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Modal'),
        #Path(r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Mean'),
        #Path(r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Modal')
    ]
    include_gamma = False
    heights = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    powers = [1, 3]
    powers = None


if __name__ == "__main__":

    run_linearity_analysis()
