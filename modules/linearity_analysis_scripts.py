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

energy_array = np.zeros((NUMBER_OF_HEIGHTS, CHANNELS * 2), dtype=np.dtype('float32'))
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
        ICRF.calibration(IN_PCA_GUESS, height, LOWER_PCA_LIM, UPPER_PCA_LIM,
                         initial_function)

    energy_arr[:CHANNELS] = initial_energy_array
    energy_arr[CHANNELS:] = final_energy_array

    return ICRF_array


def calibrate_ICRF_exposure(initial_function: Optional[np.ndarray] = None):

    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_e.calibration(LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function)

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


def large_linearity_analysis(heights: Optional[List[int]] = None,
                             data_paths: Optional[List[str]] = [data_directory],
                             include_gamma: Optional[bool] = False,
                             powers: Optional[List[float]] = None):
    """
    Function that runs a large scale linearity analysis.
    Args:
        heights: List of heights used for noise ICRF calibration. If none, then
            exposure version of ICRF calibration is used.
        data_paths: List of paths from which to source camera noise data
        include_gamma: whether to include gamma functions from CRF database
        powers: list of exponents to use as initial function, i.e. x^power form.
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
        if use_noise:
            ICRF_array = calibrate_ICRF_noise(height, initial_function=init_func)
            if init_func is None:
                save_and_plot_ICRF(ICRF_array, f"ICRFn_mean_h{height}", path)
                scatter_path = path.joinpath(f"SctrN_mean_h{height}.png")
            else:
                save_and_plot_ICRF(ICRF_array, f"ICRFn_p{power}_h{height}", path)
                scatter_path = path.joinpath(f"SctrN_p{power}_h{height}.png")
        else:
            ICRF_array = calibrate_ICRF_exposure(initial_function=init_func)
            if init_func is None:
                save_and_plot_ICRF(ICRF_array, f"ICRFe_mean", path)
                scatter_path = path.joinpath(f"SctrE_mean.png")
            else:
                save_and_plot_ICRF(ICRF_array, f"ICRFe_p{power}", path)
                scatter_path = path.joinpath(f"SctrE_p{power}.png")

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
                                                save_path=scatter_path)

        if init_func is None:
            results.append(f"Mean\t{height}\t{duration}\t{linearity_result[-1]}")
        else:
            results.append(f"{power}\t{height}\t{duration}\t{linearity_result[-1]}")

        return

    save_8bit = False
    save_32bit = False
    save_linear = False
    save_HDR = False
    pass_linear = True
    fix_artifacts = False
    pass_results = True
    results = []

    if heights is None:
        heights = [50]
        use_noise = False
    else:
        use_noise = True

    for data_path in data_paths:

        results.append(f"{data_path}\n")

        STD_data = process_base_data(data_path, include_gamma)
        time.sleep(0.5)

        for height in heights:

            initial_function = None
            linearity_cycle(data_path, initial_function)

            if powers is not None:
                for power in powers:

                    initial_function = linear_scale ** power
                    linearity_cycle(data_path, initial_function)

    if use_noise:
        file_name = f"large_linearity_results_noise.txt"
    else:
        file_name = f"large_linearity_results_exposure.txt"
    with open(OUT_PATH.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


def run_linearity_analysis():

    data_paths = [
        r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Mean',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\YD\Modal',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Mean',
        # r'D:\Koodailu\Test\ICRF_calibration_and_HDR_images\data\ND\Modal'
    ]
    include_gamma = False
    # heights = [1, 2, 5, 10, 20, 40, 80, 160]
    heights = [10, 20]
    powers = [1, 3]
    large_linearity_analysis(heights=heights, data_paths=data_paths,
                             include_gamma=include_gamma, powers=powers)


if __name__ == "__main__":

    run_linearity_analysis()
