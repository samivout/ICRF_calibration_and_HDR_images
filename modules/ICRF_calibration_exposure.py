import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import general_functions as gf
from typing import Optional
from global_settings import *

ICRF = np.zeros((DATAPOINTS, CHANNELS), dtype=float)
linear_scale = np.linspace(0, 1, DATAPOINTS, dtype=float)


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
    product = np.matmul(PCA_array, PCA_params)
    iterated_ICRF = mean_ICRF + product

    return iterated_ICRF


def analyze_linearity(sublists_of_imageSets):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        sublists_of_imageSets: Optionally pass sublist from previous calculations

    Returns:
    """
    results = []
    lower = 5 / MAX_DN
    upper = 250 / MAX_DN

    range_step = (upper - lower) / 3

    low_range = [lower, range_step]
    mid_range = [range_step, 2*range_step]
    high_range = [2*range_step, upper]

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

                division = np.divide(x.acq, y.acq,
                                     out=np.full_like(x.acq, np.nan),
                                     where=((lower < y.acq) & (y.acq < upper) &
                                            (lower < x.acq) & (x.acq < upper)))

                division = abs(division - ratio) / ratio

                where = (low_range[0] < x.acq) & (x.acq < low_range[1]) &\
                        (low_range[0] < y.acq) & (y.acq < low_range[1])
                low_channel_mean = np.nanmean(division[where])

                where = (mid_range[0] < x.acq) & (x.acq < mid_range[1]) &\
                        (mid_range[0] < y.acq) & (y.acq < mid_range[1])
                mid_channel_mean = np.nanmean(division[where])

                where = (high_range[0] < x.acq) & (x.acq < high_range[1]) & \
                        (high_range[0] < y.acq) & (y.acq < high_range[1])
                high_channel_mean = np.nanmean(division[where])

                res_arr = np.array([low_channel_mean, mid_channel_mean, high_channel_mean])

                # result = max(low_channel_mean, mid_channel_mean, high_channel_mean)
                result = np.nanmean(res_arr)
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

    acq_sublists_iter = gf.copy_nested_list(acq_sublists, channel)
    for sublist in acq_sublists_iter:
        for imageSet in sublist:

            gf.linearize_ImageSet(imageSet, ICRF[:, [channel]], gaussian_blur=True)

    linearity_data = analyze_linearity(acq_sublists_iter)
    energy = np.mean(linearity_data)
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


def calibration(lower_limit, upper_limit,
                initial_function: Optional[np.ndarray] = None):
    """ The main function running the ICRF calibration process that is called
    from outside the module.

       Args:
           initial_function: base function from which iteration starts.
           initial_guess: initial guess for the values of the PCA coefficients
                subject to optimization.
           evaluation_heights: the number of heights at which the datapoint
               distances to the edges of the distribution have been calculated.
           lower_limit: a lower limit for the PCA coefficient values.
           upper_limit: an upper limit for the PCA coefficient values.

       Return:
            ICRF_array: a Numpy float array containing the optimized ICRFs of
                each channel.
            initial_energy_array: a Numpy float array containing the initial
                energies of each channel.
            final_energy_array: a Numpy float array containing the final
                energies of each channel.
       """
    final_energy_array = np.zeros(CHANNELS, dtype=float)
    initial_energy_array = np.zeros(CHANNELS, dtype=float)

    limits = [[lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit]]

    # Initialize image lists and name lists

    acq_list = gf.create_imageSets(ACQ_PATH)
    for imageSet in acq_list:
        imageSet.load_acq()
        imageSet.acq = gf.choose_evenly_spaced_points(imageSet.acq, 250)
    acq_sublists = gf.separate_to_sublists(acq_list)
    for sublist in acq_sublists:
        sublist.sort(key=lambda imageSet: imageSet.exp)
    del acq_list

    for i in range(len(MEAN_DATA_FILES)):
        # Get the filenames from the attribute arrays.
        PCA_file_name = PRINCIPAL_COMPONENT_FILES[i]
        mean_ICRF_file_name = MEAN_ICRF_FILES[i]

        # Load mean data, principal component data and mean ICRF data into
        # numpy arrays.
        PCA_array = rd.read_data_from_txt(PCA_file_name)

        if initial_function is None:
            mean_ICRF_array = rd.read_data_from_txt(mean_ICRF_file_name)
        else:
            mean_ICRF_array = initial_function
        '''
        if i == 0:
            limit = 0.29
        if i == 1:
            limit = 0.75
        if i == 2:
            limit = 0.34
        '''
        limit = 0.1

        with DifferentialEvolutionSolver(_energy_function, limits, args=(
            mean_ICRF_array, PCA_array, acq_sublists, i),
                                         strategy='best1bin', tol=0.3) as solver:
            for step in solver:
                step = next(solver)  # Returns a tuple of xk and func evaluation
                func_value = step[1]  # Retrieves the func evaluation
                print(func_value)
                if solver.converged() or func_value < limit:
                    break

        result = solver.x
        del solver
        print(f'Result: f{result}')
        '''
        evaluation = _energy_function(solution, mean_ICRF_array[:, i],
                                      PCA_array[:, :, i], acq_sublists, i)
        final_energy_array[i] = evaluation
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
        '''
    ICRF_array = ICRF

    ICRF_array[:, 0] += 1 - ICRF_array[-1, 0]
    ICRF_array[:, 1] += 1 - ICRF_array[-1, 1]
    ICRF_array[:, 2] += 1 - ICRF_array[-1, 2]

    ICRF_array[0, 0] = 0
    ICRF_array[0, 1] = 0
    ICRF_array[0, 2] = 0

    ICRF_array[ICRF_array < 0] = 0

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
