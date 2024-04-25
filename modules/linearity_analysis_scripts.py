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
from typing import Union
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
                            data_limit: Optional[int] = 2):

    lower_data_limit = MIN_DN + data_limit
    upper_data_limit = MAX_DN - data_limit

    ICRF_array, initial_energy_array, final_energy_array, pixel_ratio = \
        ICRF_e.calibration(LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function,
                           data_spacing, (lower_data_limit, upper_data_limit))

    return ICRF_array, pixel_ratio


def process_base_data(path: Optional[str] = None, include_gamma: Optional[bool] = False,
                      color_split: Optional[bool] = True):

    cdt.clean_and_interpolate_data(path)
    STD_data = STD.process_STD_data()
    process_CRF_database.process_CRF_data(include_gamma, color_split)
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
                                data_spacing: Optional[List[Union[int, tuple[int, int]]]] = None,
                                run_number: Optional[int] = 1):
    """
    Function that runs a large scale linearity analysis.
    Args:
        data_paths: List of paths from which to source camera noise data
        include_gamma: whether to include gamma functions from CRF database
        limits: list of pixel value limits used to set rejection range
        data_spacing: determines amount of pixels sampled in exposure-based
            ICRF calibration.
        run_number: identifies the particular run and is used to save data
    """
    def linearity_cycle(path: Path, init_func: Optional[np.ndarray] = None,
                        data_limit: Optional[int] = 2,
                        data_spacing: Optional[Union[int, tuple[int, int]]] = 100):
        """
        Inner funciton to run a single cycle of linearity analysis.
        Args:
            path: the path from which the camera data distribution is sourced
                from.
            init_func: The initial function passed to ICRF calibration
            data_limit: Limit for rejecting pixel values in linearity analysis
            data_spacing: Sampling spacing used for the images in exposure
                based calibration.
        """
        save_path = path.joinpath(f'exp_measurement_{run_number}')
        save_path.mkdir(exist_ok=True)

        start_time = timeit.default_timer()
        ICRF_array, pixel_ratio = calibrate_ICRF_exposure(initial_function=init_func,
                                                          data_spacing=data_spacing,
                                                          data_limit=data_limit)
        if init_func is None:
            save_and_plot_ICRF(ICRF_array, f"ICRFe_mean_l{limit}_s{spacing}", save_path)
            scatter_path = save_path.joinpath(f"SctrE_mean_l{limit}_s{spacing}.png")
        else:
            save_and_plot_ICRF(ICRF_array, f"ICRFe_l{limit}_s{spacing}", save_path)
            scatter_path = save_path.joinpath(f"SctrE_l{limit}_s{spacing}.png")

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
                                                absolute_result=absolute_result,
                                                ICRF=ICRF_array,
                                                linearity_limit=data_limit)

        if init_func is None:
            results.append(f"{limit}, {pixel_ratio}, {duration}, {', '.join(str(x) for x in linearity_result[-1])}")
        else:
            results.append(f"{limit}, {pixel_ratio}, {duration}, {', '.join(str(x) for x in linearity_result[-1])}")

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

        # results.append(f"{data_path}\n")

        STD_data = process_base_data(data_path, include_gamma)
        time.sleep(0.5)

        for limit in limits:
            for spacing in data_spacing:

                initial_function = None
                linearity_cycle(data_path, initial_function, limit, spacing)

        for limit in limits:
            for spacing in data_spacing:

                initial_function = linear_scale**2
                linearity_cycle(data_path, initial_function, limit, spacing)

    file_name = f"large_linearity_results_exposure_{run_number}.txt"
    with open(OUT_PATH.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


def linearity_analysis_noise(data_paths: Optional[List[Path]] = [data_directory],
                             include_gamma: Optional[bool] = False,
                             heights: Optional[List[int]] = None,
                             run_number: Optional[int] = 1):

    def linearity_cycle(path: Path, init_func: Optional[np.ndarray] = None):
        """
        Inner funciton to run a single cycle of linearity analysis.
        Args:
            path: the path from which the camera data distribution is sourced
                from.
            init_func: The initial function passed to ICRF calibration
            run_number: identifies the particular run and is used to save data
        """
        save_path = path.joinpath(f'noise_measurement_{run_number}')
        save_path.mkdir(exist_ok=True)

        start_time = timeit.default_timer()
        ICRF_array, energy_array = calibrate_ICRF_noise(height, initial_function=init_func)
        if init_func is None:
            save_and_plot_ICRF(ICRF_array, f"ICRFn_mean_h{height}", save_path)
            scatter_path = save_path.joinpath(f"SctrN_mean_h{height}.png")
        else:
            save_and_plot_ICRF(ICRF_array, f"ICRFn_h{height}", save_path)
            scatter_path = save_path.joinpath(f"SctrN_h{height}.png")

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

        # for height in heights:
        #
        #     initial_function = None
        #     linearity_cycle(data_path, initial_function)

        for height in heights:

            initial_function = linear_scale**2
            linearity_cycle(data_path, initial_function)

    file_name = f"large_linearity_results_noise_{run_number}.txt"
    with open(OUT_PATH.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    return


def run_repeat_calibration(calibration_params: tuple, number_of_runs: Optional[int] = 5):

    save_dir = OUTPUT_DIRECTORY.joinpath(f'repeat_calibration_{number_of_runs}')
    scatter_path = save_dir.joinpath('linearity_results')
    save_dir.mkdir(exist_ok=True)
    scatter_path.mkdir(exist_ok=True)
    ICRF_stack = None
    fig, axes = plt.subplots(1, CHANNELS, figsize=(20, 5))
    fig_name = f'ICRF_repeat_{number_of_runs}.png'

    for i in range(number_of_runs):

        ICRF = calibrate_ICRF_exposure(*calibration_params)[0]

        if ICRF_stack is None:
            ICRF_stack = ICRF[:, :, np.newaxis]
        else:
            ICRF_stack = np.concatenate((ICRF_stack, ICRF[:, :, np.newaxis]), axis=2)

        for c, ax in enumerate(axes):

            ax.plot(linear_scale, ICRF[:, c], alpha=0.5)

    mean_ICRF = np.mean(ICRF_stack, axis=2)

    for c, ax in enumerate(axes):

        ax.plot(linear_scale, mean_ICRF[:, c], color='0')

        title = None
        if c == 0:
            title = 'Blue'
        if c == 1:
            title = 'Green'
        if c == 2:
            title = 'Red'

        ax.set_title(title)

    axes[0].set(ylabel='Relative exposure')
    axes[1].set(xlabel='Image brightness')
    plt.savefig(save_dir.joinpath(fig_name), dpi=300)

    mean_ICRF = mean_ICRF[:, :, np.newaxis]
    ICRF_stack = np.concatenate((ICRF_stack, mean_ICRF), axis=2)
    results = []

    for i in range(number_of_runs + 1):

        linearity_plot_path = scatter_path.joinpath(f'linearity_results_{i}.png')
        ICRF = ICRF_stack[:, :, i]
        acq_sublists = HDR.process_HDR_images(save_linear=False,
                                              save_HDR=False,
                                              save_8bit=False,
                                              save_32bit=False,
                                              pass_linear=True,
                                              fix_artifacts=False,
                                              ICRF=ICRF,
                                              STD_data=None)

        linearity_result = ia.analyze_linearity(OUT_PATH,
                                                sublists_of_imageSets=acq_sublists,
                                                pass_results=True,
                                                STD_data=None,
                                                save_path=linearity_plot_path,
                                                relative_scale=True,
                                                absolute_result=False,
                                                ICRF=ICRF)

        results.append(f"{', '.join(str(x) for x in linearity_result[-1])}")

    file_name = f"large_linearity_results_repeat_{number_of_runs}.txt"
    with open(save_dir.joinpath(file_name), 'w') as f:
        for row in results:
            f.write(f'{row}\n')
        f.close()

    for c in range(CHANNELS):

        if c == 0:
            file_name = 'ICRFs_blue.txt'
        if c == 1:
            file_name = 'ICRFs_green.txt'
        if c == 2:
            file_name = 'ICRFs_red.txt'

        np.savetxt(save_dir.joinpath(file_name), ICRF_stack[:, c, :])

    return


def mean_RMSE(path: Path):

    ICRF_file_names = ['ICRFs_blue.txt', 'ICRFs_green.txt', 'ICRFs_red.txt']
    results = [[], [], []]

    for i, ICRF_file in enumerate(ICRF_file_names):
        repeat_ICRFs = rd.read_data_from_txt(ICRF_file, str(path))

        datapoints, number_of_ICRFs = np.shape(repeat_ICRFs)

        for j in range(number_of_ICRFs - 1):

                RMSE = np.sqrt(np.mean(repeat_ICRFs[:, j] - repeat_ICRFs[:, -1]) ** 2)

                if i == 0:
                    results[0].append(RMSE)
                elif i == 1:
                    results[1].append(RMSE)
                else:
                    results[2].append(RMSE)

    results = np.array(results)
    col_means = np.mean(results, axis=1)
    col_stds = np.std(results, axis=1)
    np.savetxt(path.joinpath('mean_RMSE.txt'), col_means)
    np.savetxt(path.joinpath('std_RMSE.txt'), col_stds)

    return


def run_exposure_measurement(measurements: List[tuple]):

    for i, measurement in enumerate(measurements):
        linearity_analysis_exposure(*measurement)

    return


def run_noise_measurement(measurements: List[tuple]):

    for i, measurement in enumerate(measurements):
        linearity_analysis_noise(*measurement)

    return


def run_linearity_analysis():

    data_paths = [
        Path(r'E:\ICRF_calibration_and_HDR_images\data\YD\Mean'),
        #Path(r'E:\ICRF_calibration_and_HDR_images\data\YD\Modal'),
        #Path(r'E:\ICRF_calibration_and_HDR_images\data\ND\Mean'),
        #Path(r'E:\ICRF_calibration_and_HDR_images\data\ND\Modal')
    ]
    include_gamma = False
    heights = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    data_spacings = [18]
    '''
    data_limits = [5]
    
    exposure_measurement_parameters_1 = (
        data_paths, include_gamma, data_limits, data_spacings, 8)

    data_spacings = [7]

    exposure_measurement_parameters_2 = (
        data_paths, include_gamma, data_limits, data_spacings, 9)

    data_spacings = [6]

    exposure_measurement_parameters_3 = (
        data_paths, include_gamma, data_limits, data_spacings, 10)

    data_spacings = [18]
    data_limits = [1, 2, 3, 4, 5]

    exposure_measurement_parameters_4 = (
        data_paths, include_gamma, data_limits, data_spacings, 11)
    
    '''

    data_limits = [7, 10, 15]

    exposure_measurement_parameters_5 = (
        data_paths, include_gamma, data_limits, data_spacings, 12)

    exposure_measurements = [exposure_measurement_parameters_5]

    noise_measurement_parameters_1 = (data_paths, include_gamma, heights, 1)

    noise_measurements = [noise_measurement_parameters_1]

    run_exposure_measurement(exposure_measurements)
    # run_noise_measurement(noise_measurements)


if __name__ == "__main__":

    path = Path(r'E:\ICRF_calibration_and_HDR_images\output\repeat_calibration_20')
    mean_RMSE(path)

    path = Path(r'E:\ICRF_calibration_and_HDR_images\output\repeat_calibration_21')
    mean_RMSE(path)

    path = Path(r'E:\ICRF_calibration_and_HDR_images\output\repeat_calibration_22')
    mean_RMSE(path)

    path = Path(r'E:\ICRF_calibration_and_HDR_images\output\repeat_calibration_23')
    mean_RMSE(path)
    '''
    path = r'E:\ICRF_calibration_and_HDR_images\data\YD\Mean'
    run_linearity_analysis()
    
    process_base_data(path, True, True)
    time.sleep(3)
    repeat_params = (linear_scale**2, 300, 5)

    run_repeat_calibration(repeat_params, 21)

    process_base_data(path, False, False)
    time.sleep(3)
    run_repeat_calibration(repeat_params, 22)
    
    repeat_params = (linear_scale ** 2, 300, 5)
    process_base_data(path, True, False)
    time.sleep(3)
    run_repeat_calibration(repeat_params, 23)
    '''
