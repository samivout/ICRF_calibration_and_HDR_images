import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import general_functions as gf
from typing import Optional
from typing import List
from ImageSet import ImageSet
from joblib import delayed, parallel
from global_settings import *

use_mean_ICRF = False
ICRF = np.zeros((DATAPOINTS, CHANNELS), dtype=float)
linear_scale = np.linspace(0, 1, DATAPOINTS, dtype=float)
global_lower_data_limit = 0
global_upper_data_limit = 255


def _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params):
    """ The inverse camera response function calculated in terms of the mean
    ICRF, PCA vectors and the PCA coefficients that are subject to optimization.
    Dimensions of the PCA_array and PCA_params must match so that matrix
    multiplication produces an array of equal dimensions to mean_ICRF.

    Args:
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance
            datapoints.
        PCA_array: Numpy array containing the principal component vectors.
        PCA_params: The PCA coefficients for each vector in PCA_array, these
            the values subject to optimization.

    Return:
        The new iteration of the inverse camera response function.
    """
    global use_mean_ICRF
    if not use_mean_ICRF:
        mean_ICRF = np.linspace(0, 1, BITS) ** PCA_params[0]
        product = np.matmul(PCA_array, PCA_params[1:])
    else:
        product = np.matmul(PCA_array, PCA_params)

    iterated_ICRF = mean_ICRF + product

    return iterated_ICRF


def analyze_linearity(sublists_of_imageSets: List[List[ImageSet]],
                      use_relative: Optional[bool] = True):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        sublists_of_imageSets: Optionally pass sublist from previous calculations.
        use_relative: whether to utilize relative or absolute pixel values.
    Returns:
    """
    global global_upper_data_limit
    global global_lower_data_limit
    results = []
    lower = global_lower_data_limit/MAX_DN
    upper = global_upper_data_limit/MAX_DN

    '''
    range_step = (upper - lower) / 3
    low_range = [lower, range_step]
    mid_range = [range_step, 2*range_step]
    high_range = [2*range_step, upper]
    '''

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]

                ratio = x.exp / y.exp
                if ratio < 0.05:
                    break

                range_mask = (y.acq < lower) | (y.acq > upper)
                y.acq[range_mask] = np.nan
                range_mask = (x.acq < lower) | (x.acq > upper)
                x.acq[range_mask] = np.nan
                del range_mask

                y = gf.multiply_imageSets(y, ratio, use_std=False)
                linearSet = gf.subtract_imageSets(x, y, use_std=False)

                if use_relative:
                    linearSet = gf.divide_imageSets(linearSet, y, use_std=False)

                if use_relative:
                    acq = abs(linearSet.acq)
                else:
                    acq = abs(linearSet.acq * MAX_DN)
                '''
                where = (low_range[0] < x.acq) & (x.acq < low_range[1]) &\
                        (low_range[0] < y.acq) & (y.acq < low_range[1])
                low_channel_mean = np.nanmean(acq[where])

                where = (mid_range[0] < x.acq) & (x.acq < mid_range[1]) &\
                        (mid_range[0] < y.acq) & (y.acq < mid_range[1])
                mid_channel_mean = np.nanmean(acq[where])

                where = (high_range[0] < x.acq) & (x.acq < high_range[1]) & \
                        (high_range[0] < y.acq) & (y.acq < high_range[1])
                high_channel_mean = np.nanmean(acq[where])

                res_arr = np.array([low_channel_mean, mid_channel_mean, high_channel_mean])

                # result = max(low_channel_mean, mid_channel_mean, high_channel_mean)
                result = np.nanmean(res_arr)
                '''
                result = np.nanmean(acq)
                results.append(result)

    data_array = np.array(results)

    return data_array


def _energy_function(PCA_params, mean_ICRF, PCA_array, acq_sublists, channel):
    """ The energy function, whose value is subject to minimization. Iterates
    ICRF on a global scale.

    Args:
        PCA_params: The PCA coefficients for each vector in PCA_array, these
            the values subject to optimization.
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance
            datapoints.
        PCA_array: Numpy array containing the principal component vectors.

    Return:
        The mean skewness of value of all the distributions as a float.
    """
    global ICRF
    ICRF[:, channel] = _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params)
    ICRF[:, channel] += 1 - ICRF[-1, channel]
    ICRF[0, channel] = 0

    if np.max(ICRF[:, channel]) > 1 or np.min(ICRF[:, channel]) < 0:
        energy = np.inf
        return energy

    ICRF_ch = ICRF[:, channel]
    if not np.all(ICRF_ch[1:] > ICRF_ch[:-1]):
        energy = np.inf
        return energy

    acq_sublists_iter = gf.copy_nested_list(acq_sublists, channel)
    for sublist in acq_sublists_iter:
        for imageSet in sublist:

            gf.linearize_ImageSet(imageSet, ICRF[:, [channel]], ICRF_diff=None, gaussian_blur=False)

    linearity_data = analyze_linearity(acq_sublists_iter, use_relative=True)
    energy = np.nanmean(linearity_data)
    if np.isnan(energy):
        energy = np.Inf

    return energy


def _initial_energy_function(x, acq_sublists, channel):

    initial_function = np.linspace(0, 1, BITS) ** x
    initial_function = np.expand_dims(initial_function, axis=1)

    acq_sublists_iter = gf.copy_nested_list(acq_sublists, channel)
    for sublist in acq_sublists_iter:
        for imageSet in sublist:
            gf.linearize_ImageSet(imageSet, initial_function, ICRF_diff=None, gaussian_blur=False)

    linearity_data = analyze_linearity(acq_sublists_iter)
    energy = np.nanmean(linearity_data)
    if np.isnan(energy):
        energy = np.Inf

    return energy


def interpolate_ICRF(ICRF_array):
    if BITS == DATAPOINTS:
        return ICRF_array

    x_new = np.linspace(0, 1, num=BITS)
    x_old = np.linspace(0, 1, num=DATAPOINTS)
    interpolated_ICRF = np.zeros((BITS, CHANNELS), dtype=float)

    for c in range(CHANNELS):
        y_old = ICRF_array[:, c]
        interpolated_ICRF[:, c] = np.interp(x_new, x_old, y_old)

    return interpolated_ICRF


def calibration(lower_PCA_limit, upper_PCA_limit,
                initial_function: Optional[np.ndarray] = None,
                data_spacing: Optional[int] = 150,
                lower_data_limit: Optional[int] = 2,
                upper_data_limit: Optional[int] = 253):
    """ The main function running the ICRF calibration process that is called
    from outside the module.

       Args:
           initial_function: base function from which iteration starts.
               distances to the edges of the distribution have been calculated.
           lower_PCA_limit: a lower limit for the PCA coefficient values.
           upper_PCA_limit: an upper limit for the PCA coefficient values.
           data_spacing: used to determine the amount of pixels used in linearity
                analysis.
           lower_data_limit: pixel values under this are ignored in linearity
           upper_data_limit: pixel values above this are ignored in linearity

       Return:
            ICRF_array: a Numpy float array containing the optimized ICRFs of
                each channel.
            initial_energy_array: a Numpy float array containing the initial
                energies of each channel.
            final_energy_array: a Numpy float array containing the final
                energies of each channel.
       """
    global ICRF
    global use_mean_ICRF
    global global_upper_data_limit
    global global_lower_data_limit
    global_upper_data_limit = upper_data_limit
    global_lower_data_limit = lower_data_limit
    final_energy_array = np.zeros(CHANNELS, dtype=float)
    initial_energy_array = np.zeros(CHANNELS, dtype=float)

    if initial_function is None:
        use_mean_ICRF = True
        limits = [[lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit]]
        x0 = [0, 0, 0, 0, 0]
    else:
        use_mean_ICRF = False
        limits = [[0.1, 6],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit],
                  [lower_PCA_limit, upper_PCA_limit]]
        x0 = [1, 0, 0, 0, 0, 0]

    limit = 0.1

    # Initialize image lists and name lists

    acq_list = gf.create_imageSets(ACQ_PATH)
    for imageSet in acq_list:
        imageSet.load_acq()
        imageSet.acq = gf.choose_evenly_spaced_points(imageSet.acq, data_spacing)
    acq_sublists = gf.separate_to_sublists(acq_list)
    for sublist in acq_sublists:
        sublist.sort(key=lambda imageSet: imageSet.exp)
    del acq_list

    def solve_channel(PCA_file_name: str, mean_ICRF_file_name: str,
                      channel: int):

        global ICRF

        PCA_array = rd.read_data_from_txt(PCA_file_name)
        if use_mean_ICRF:
            mean_ICRF_array = rd.read_data_from_txt(mean_ICRF_file_name)
        else:
            mean_ICRF_array = initial_function

        number_of_iterations = 0
        # Access DifferentialEvolutionSolver directly to stop iteration if
        # solution has converged or energy function value is under given limit.
        with DifferentialEvolutionSolver(_energy_function, limits, args=(
                mean_ICRF_array, PCA_array, acq_sublists, channel),
                                         strategy='best1bin', tol=0.005,
                                         x0=x0) as solver:
            for step in solver:
                number_of_iterations += 1
                step = next(solver)  # Returns a tuple of xk and func evaluation
                func_value = step[1]  # Retrieves the func evaluation
                if number_of_iterations % 5 == 0:
                    print(
                        f'Channel {channel} value: {func_value} on step {number_of_iterations}')
                if solver.converged() or func_value < limit:
                    break

        result = solver.x
        ICRF[:, channel] = _inverse_camera_response_function(mean_ICRF_array,
                                                             PCA_array, result)

        del solver
        print(f'Channel {channel} result: f{result}, number of iterations: {number_of_iterations}')

    parallel.Parallel(n_jobs=CHANNELS, prefer="threads")\
        (delayed(solve_channel)(PCA_FILES[c], MEAN_ICRF_FILES[c], c) for c in range(CHANNELS))

    ICRF_array = ICRF

    # The ICRF might be shifted on the y-axis, so we adjust it back to [0,1]
    # here.
    ICRF_array[:, 0] += 1 - ICRF_array[-1, 0]
    ICRF_array[:, 1] += 1 - ICRF_array[-1, 1]
    ICRF_array[:, 2] += 1 - ICRF_array[-1, 2]

    ICRF_array[0, 0] = 0
    ICRF_array[0, 1] = 0
    ICRF_array[0, 2] = 0

    # Clipping values just in case. Shouldn't be needed as the ICRF should be
    # continuously increasing between [0,1] without going outside that interval.
    ICRF_array[ICRF_array < 0] = 0
    ICRF_array[ICRF_array > 1] = 1

    ICRF_interpolated = interpolate_ICRF(ICRF_array)

    return ICRF_interpolated, initial_energy_array, final_energy_array


if __name__ == "__main__":

    arr = np.arange(27).reshape((3, 3, 3)).astype(float)
    '''
    arr[:,:,1] = np.nan
    arr2 = arr[:,:,[1]]
    arr3 = np.zeros((DATAPOINTS, 1), dtype=float)
    print(np.shape(arr3))
    '''
    print(arr[:, :, 0])
    ch = [0]
    help = np.take(arr, ch, axis=2)
    print(help[:,:,0])

    print('Run script from main file!')
