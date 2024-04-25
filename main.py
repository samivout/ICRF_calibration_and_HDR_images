import general_functions
from modules import ICRF_calibration_noise as ICRF
from modules import principal_component_analysis
from modules import process_CRF_database
from modules import general_functions as gf
from modules.ImageSet import ImageSet
from modules import image_correction
from modules import HDR_image as HDR
from modules import STD_data_calculator as STD
from modules import mean_data_plot as mdl
from modules import camera_data_tools as cdt
from modules import image_analysis as ia
from modules import mean_data_collector as mdc
from modules import ICRF_calibration_exposure as ICRF_e
from modules import ICRF_calibration_pair as ICRF_p
from modules import HDR_full_process
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
                     data_spacing: Optional[int | tuple[int, int]] = 150,
                     data_limits: Optional[tuple[int, int]] = (5, 250)):

    x_range = np.linspace(0, 1, BITS)
    ICRF_array, initial_energy_array, final_energy_array, pixel_ratio = \
        ICRF_e.calibration(LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function,
                           data_spacing, data_limits)

    np.savetxt(OUTPUT_DIRECTORY.joinpath(f'ICRF_exp.txt'), ICRF_array)

    plt.plot(x_range, ICRF_array[:, 0], color='b')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='r')
    plt.savefig(OUTPUT_DIRECTORY.joinpath(f'ICRF_exp.png'), dpi=300)
    plt.clf()

    return


def calibrate_ICRF_pair(initial_function: Optional[np.ndarray] = None,
                        data_limits: Optional[int] = 2,
                        data_spacing: Optional[int] = None):
    lower_limit = data_limits
    upper_limit = MAX_DN - data_limits

    long_imageSet_path = gf.get_filepath_dialog('Choose long exposure image')
    y = ImageSet(long_imageSet_path)
    short_imageSet_path = gf.get_filepath_dialog('Choose short exposure image')
    x = ImageSet(short_imageSet_path)

    x_range = np.linspace(0, 1, BITS)
    ICRF_array, initial_energy_array, final_energy_array = \
        ICRF_p.calibration(x, y, LOWER_PCA_LIM, UPPER_PCA_LIM, initial_function,
                           lower_limit, upper_limit, data_spacing)

    np.savetxt(OUTPUT_DIRECTORY.joinpath(f'ICRF_p.txt'), ICRF_array)

    plt.plot(x_range, ICRF_array[:, 0], color='b')
    plt.plot(x_range, ICRF_array[:, 1], color='g')
    plt.plot(x_range, ICRF_array[:, 2], color='r')
    plt.savefig(OUTPUT_DIRECTORY.joinpath(f'ICRF_p.png'), dpi=300)
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
    ret = HDR.process_HDR_images(save_linear=save_linear,
                                 save_HDR=save_HDR,
                                 save_8bit=save_8bit,
                                 save_32bit=save_32bit,
                                 pass_linear=pass_linear,
                                 fix_artifacts=fix_artifacts,
                                 gaussian_blur=gaussian_blur)

    return ret


def process_HDR_full():

    ret = HDR_full_process.process_HDR_images()

    return ret


def large_linearity_analysis_noise(paths: Optional[List[str]] = [data_directory]):
    linearity_results = []

    save_8bit = False
    save_32bit = False
    save_linear = True
    save_HDR = False
    pass_linear = True
    fix_artifacts = True
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
    data_paths = [
        Path(r'E:\ICRF_calibration_and_HDR_images\data\YD\Mean'),
        # Path(r'E:\ICRF_calibration_and_HDR_images\data\YD\Modal'),
        # Path(r'E:\ICRF_calibration_and_HDR_images\data\ND\Mean'),
        # Path(r'E:\ICRF_calibration_and_HDR_images\data\ND\Modal')
    ]
    analysis_path = Path(
        r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Analysis\Image correction class\Input\Images\Grid image cache')
    init_func = np.linspace(0, 1, BITS) ** 3
    ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    data_limit = (5, 250)
    data_spacing = 300
    data_limits = 2

    # mdc.collect_mean_data()
    # process_base_data(data_paths[0], include_gamma=False, color_split=True)
    # calibrate_ICRF(init_func)
    # calibrate_ICRF_e(init_func, data_spacing, data_limit)
    # calibrate_ICRF_pair(None, data_limits, data_spacing)
    # process_mdl()
    # image_correction.image_correction(save_to_file=True)
    # ret = process_HDR()
    ret = process_HDR_full()
    # image_calculation.calibrate_dark_frames()
    # image_calculation.calibrate_flats()
    # ia.analyze_linearity(OUT_PATH, ret, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True, ICRF=ICRF)
    # ia.analyze_linearity(ACQ_PATH, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True)
    # ia.analyze_linearity(analysis_path, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True)
    # ia.analyze_linearity(analysis_path, use_std=True, absolute_result=False, absolute_scale=True, relative_scale=True)
    # large_linearity_analysis(data_paths)
    # general_functions.show_image()


main()
