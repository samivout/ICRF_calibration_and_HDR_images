import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from typing import Optional
from global_settings import *

ICRF = np.zeros((DATAPOINTS, CHANNELS), dtype=float)
linear_scale = np.linspace(0, 1, DATAPOINTS, dtype=float)
use_mean_ICRF = False


def _process_datapoint_distances(mean_data, number_of_heights):
    """
    Wrapper function for processing all distributions in the mean data array.
    Args:
        mean_data: Numpy array of the camera mean data
        number_of_heights: number of heights at which to evaluate the edge
            distances of each distribution.

    Returns: Numpy array containing the edge distances for each distribution.
        First column is left edge, middle column is mode value and right column
        is the right edge.
    """
    number_of_dists = np.shape(mean_data)[0]
    edge_distances = np.zeros((number_of_dists, number_of_heights, 3),
                              dtype=int)

    for i in range(number_of_dists-1):

        edge_distances[i, :, :] = _datapoint_distances(mean_data[i, :],
                                                       number_of_heights)

    return edge_distances


def _datapoint_distances(dist, number_of_heights):
    """ Calculates how far the edges of the pixel value distribution are from
    the mode of the distribution in terms of number of datapoints at n heights.

    Args:
        dist: array containing the distribution of a pixel value.
        number_of_heights: the number of heights at which the datapoint
            distances to the edges of the distribution should be calculated.

    Return:
        integer array of the distances from distribution mode to the left and
        right edges of the distribution. n rows and 3 columns. Each row
        corresponds to an evaluation height, first column contains the left edge
        distance, second column contains the distribution mode and the third
        column contains the right edge distance.
    """
    dist_length = len(dist)

    # Calculate the d+/d- distances for each distribution and save to array.

    # Determine the location/index of the distribution mode.
    mode = np.argmax(dist)
    second_mode = None
    if mode != dist_length-1:
        try:
            second_mode = np.argmax(dist[mode+1:-1])
        except ValueError:
            second_mode = None
    if second_mode is not None:
        if dist[mode] == dist[second_mode]:

            print('Second equal mode found!')

    # Determine height step based on distribution peak count and desired
    # number of evaluation heights.
    if dist[mode] <= 1:
        return 0
    if number_of_heights >= dist[mode]:
        number_of_heights = dist[mode].astype(int)
        step_height = 1
    else:
        step_height = math.floor((dist[mode] / number_of_heights))

    heights = np.linspace(0, step_height * number_of_heights,
                          num=number_of_heights, dtype=int)
    distances = np.ones((number_of_heights, 3), dtype=int) * mode

    # Calculate the d+ distances for the given heights.
    h = 0
    while h < number_of_heights:
        m = mode
        while dist[m] > 0:
            m += 1

            # Mark the current height as invalid if reach max 1023 datapoint
            # and break loop.
            if m >= dist_length-1:
                break

            # Break loop if stepping outside of distribution.
            if heights[h] >= dist[m]:
                break

            # Increment the number of taken steps from the distribution
            # mode towards the edge.
            if heights[h] < dist[m]:
                distances[h, 2] += 1

        h += 1

    # Calculate the d- distances for the given heights.
    h = 0
    while h < number_of_heights:
        m = mode
        while dist[m] > 0:
            m -= 1

            # Mark the current height as invalid if reach min 0 datapoint
            # and break loop.
            if m <= 0:
                break

            # Break loop if stepping outside of distribution.
            if heights[h] >= dist[m]:
                break

            # Increment the number of taken steps from the distribution
            # mode towards the edge.
            if heights[h] < dist[m]:
                distances[h, 0] -= 1

        h += 1

    return distances


def _skewness_evaluation(d, iterated_ICRF):
    """ Evaluate the skewness of the distribution by the equation in the article
    'Radiometric calibration from noise distributions'.

    Args:
        d: the distances array calculated in the function datapoint_distances.
        iterated_ICRF: the current iteration of the ICRF function under
            optimization.

    Return:
        The sum of the skewness values of each height as a float.
    """
    evaluation_heights = np.shape(d)[0]
    dIrr = np.zeros(np.shape(d), dtype=float)
    bool_mask = np.ones(evaluation_heights, dtype=bool)

    for j in range(0, evaluation_heights):

        if d[j, 1] - d[j, 0] == 0 or d[j, 2] - d[j, 1] == 0:
            bool_mask[j] = False

        for i in range(0, 3):
            dIrr[j, i] = iterated_ICRF[d[j, i]]

    mode = dIrr[:, 1]
    dp = np.absolute(dIrr[:, 2] - mode)
    dn = np.absolute(mode - dIrr[:, 0])

    skewArr = ((dn[bool_mask] - dp[bool_mask])
               / (dn[bool_mask] + dp[bool_mask])) ** 2
    skew = np.sum(skewArr) / evaluation_heights

    return skew


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


def _energy_function(PCA_params, mean_ICRF, PCA_array, evaluation_heights,
                     edge_distances, channel):
    """ The energy function, whose value is subject to minimization. Iterates
    ICRF on a global scale.

    Args:
        PCA_params: The PCA coefficients for each vector in PCA_array, these
            the values subject to optimization.
        mean_ICRF: Numpy array containing the [0,1] normalized mean irradiance
            datapoints.
        PCA_array: Numpy array containing the principal component vectors.
        evaluation_heights: the number of heights at which the datapoint
            distances to the edges of the distribution have been calculated.
        edge_distances: the distances array calculated in the function
            datapoint_distances.

    Return:
        The mean skewness of value of all the distributions as a float.
    """
    energy = 0
    global ICRF

    ICRF[:, channel] = _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params)
    ICRF[:, channel] += 1 - ICRF[-1, channel]
    ICRF[0, channel] = 0

    lower_ignore = 2
    upper_ignore = 2

    if np.max(ICRF[:, channel]) > 1 or np.min(ICRF[:, channel]) < 0:
        energy = np.inf
        return energy

    for i in range(MIN_DN + lower_ignore, BITS - upper_ignore):
        energy += _skewness_evaluation(edge_distances[i, :, :], ICRF[:, channel])

    energy /= (BITS - upper_ignore - lower_ignore - MIN_DN)

    return energy


def interpolate_ICRF(ICRF_array: np.ndarray):
    """
    Function for interpolating a different number of datapoints to an ICRF if
    the config.ini final datapoints differ from the number of bits.
    Args:
        ICRF_array: ICRF as numpy array.

    Returns: Interpolated ICRF
    """
    if BITS == DATAPOINTS:
        return ICRF_array

    x_new = np.linspace(0, 1, num=BITS)
    x_old = np.linspace(0, 1, num=DATAPOINTS)
    interpolated_ICRF = np.zeros((BITS, CHANNELS), dtype=float)

    for c in range(CHANNELS):

        y_old = ICRF_array[:, c]
        interpolated_ICRF[:, c] = np.interp(x_new, x_old, y_old)

    return interpolated_ICRF


def calibration(evaluation_heights, lower_limit, upper_limit,
                initial_function: Optional[np.ndarray] = None):
    """ The main function running the ICRF calibration process that is called
    from outside the module.

       Args:
           initial_function: base function from which iteration starts.
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
    global ICRF
    global use_mean_ICRF
    final_energy_array = np.zeros(CHANNELS, dtype=float)
    initial_energy_array = np.zeros(CHANNELS, dtype=float)

    if initial_function is None:
        use_mean_ICRF = True
        limits = [[lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit]]
        x0 = [0, 0, 0, 0, 0]
    else:
        use_mean_ICRF = False
        limits = [[0.1, 6],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit],
                  [lower_limit, upper_limit]]
        x0 = [3, 0, 0, 0, 0, 0]

    for i in range(len(MEAN_DATA_FILES)):
        # Get the filenames from the attribute arrays.
        mean_file_name = MEAN_DATA_FILES[i]
        PCA_file_name = PCA_FILES[i]
        mean_ICRF_file_name = MEAN_ICRF_FILES[i]

        # Load mean data, principal component data and mean ICRF data into
        # numpy arrays.
        mean_data_array = rd.read_data_from_txt(mean_file_name)
        PCA_array = rd.read_data_from_txt(PCA_file_name)

        if use_mean_ICRF:
            mean_ICRF_array = rd.read_data_from_txt(mean_ICRF_file_name)
        else:
            mean_ICRF_array = initial_function

        edge_distances = _process_datapoint_distances(mean_data_array,
                                                      evaluation_heights)

        initialEnergy = _energy_function(x0, mean_ICRF_array,
                                         PCA_array, evaluation_heights,
                                         edge_distances, i)
        initial_energy_array[i] = initialEnergy
        print(initialEnergy)

        # Access DifferentialEvolutionSolver directly to stop iteration if
        # solution has converged or energy function value is under given limit.
        with DifferentialEvolutionSolver(_energy_function, limits, args=(
                mean_ICRF_array, PCA_array, evaluation_heights, edge_distances, i),
                                         strategy='best1bin', tol=0.01,
                                         x0=x0) as solver:
            for step in solver:
                step = next(solver)  # Returns a tuple of xk and func evaluation
                func_value = step[1]  # Retrieves the func evaluation
                print(func_value)
                if solver.converged():
                    break

        result = solver.x
        ICRF[:, i] = _inverse_camera_response_function(mean_ICRF_array,
                                                       PCA_array, result)
        del solver
        print(f'Result: f{result}')
        final_energy_array[i] = func_value
        '''
        result = differential_evolution(_energy_function, limits, args=(
            mean_ICRF_array, PCA_array, evaluation_heights, edge_distances),
                                        strategy='best1bin', disp=False,
                                        maxiter=10000)

        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        solution = result['x']
        evaluation = _energy_function(solution, mean_ICRF_array, PCA_array,
                                      evaluation_heights, edge_distances)
        final_energy_array[i] = evaluation
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
        ICRF[:, i] = _inverse_camera_response_function(mean_ICRF_array,
                                                       PCA_array, solution)
        '''
    ICRF_array = ICRF

    ICRF_array[:, 0] += 1 - ICRF_array[-1, 0]
    ICRF_array[:, 1] += 1 - ICRF_array[-1, 1]
    ICRF_array[:, 2] += 1 - ICRF_array[-1, 2]

    ICRF_array[0, 0] = 0
    ICRF_array[0, 1] = 0
    ICRF_array[0, 2] = 0

    ICRF_interpolated = interpolate_ICRF(ICRF_array)

    return ICRF_interpolated, initial_energy_array, final_energy_array


if __name__ == "__main__":

    eval_heights = 10
    number_of_dists = 10
    data = np.random.normal(128 / MAX_DN, 0.0246, 10000000)
    hist, bins = np.histogram(data, bins=1024, range=(0, 1))
    number_of_dists -= 1
    for d in range(number_of_dists):
        data = np.random.normal(128 / MAX_DN, 0.0246, 10000000)
        new_hist = np.histogram(data, bins=1024, range=(0, 1))[0]
        hist = np.vstack((hist, new_hist))

    perfect_hist_counts = np.array([0, 0, 0, 5, 10, 20, 25, 20, 10, 5, 0, 0, ])
    perfect_linear_scale = np.linspace(0, 1, np.shape(perfect_hist_counts)[0],
                                       dtype=float)
    perfect_edge_dists = _datapoint_distances(perfect_hist_counts, 20)
    perfect_skew = _skewness_evaluation(perfect_edge_dists,
                                        perfect_linear_scale)
    print(perfect_skew)

    skew_list = []
    edge_dists = _process_datapoint_distances(hist, eval_heights)
    for i in range(number_of_dists):
        skew = _skewness_evaluation(edge_dists[i, :, :], linear_scale)
        skew_list.append(skew)
        plt.stairs(hist[i, :], bins)
        plt.title(f'Skew: {skew}')
        path = Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Analysis\ICRF calibration and HDR images\test output').joinpath(f'Dist{ i}.png')
        plt.savefig(path, dpi=500)
        plt.clf()

    print(skew_list)
