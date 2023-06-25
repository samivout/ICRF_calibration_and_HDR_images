import matplotlib.pyplot as plt
import numpy as np
import math
import read_data as rd
from scipy.optimize import differential_evolution
import os

current_directory = os.path.dirname(__file__)
output_directory = os.path.join(os.path.dirname(current_directory), 'output')
datapoints = rd.read_config_single('final datapoints')
channels = rd.read_config_single('channels')
bit_depth = rd.read_config_single('bit depth')
bits = 2 ** bit_depth
max_DN = bits - 1
min_DN = 0
num_of_PCA_params = rd.read_config_single('number of principal components')
mean_data_files = rd.read_config_list('camera mean data')
principal_component_files = rd.read_config_list('principal components')
mean_ICRF_files = rd.read_config_list('mean ICRFs')

ICRF = np.zeros((datapoints, 1), dtype=float)
linear_scale = np.linspace(0, 1, datapoints, dtype=float)


def _process_datapoint_distances(mean_data, number_of_heights):

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
    product = np.matmul(PCA_array, PCA_params)
    iterated_ICRF = mean_ICRF + product

    return iterated_ICRF


def _energy_function(PCA_params, mean_ICRF, PCA_array, evaluation_heights,
                     edge_distances):
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

    ICRF = _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params)
    lower_ignore = 2
    upper_ignore = 2

    for i in range(min_DN + lower_ignore, bits - upper_ignore):
        energy += _skewness_evaluation(edge_distances[i, :, :], ICRF)

    energy /= (bits - upper_ignore - lower_ignore - min_DN)

    return energy


def interpolate_ICRF(ICRF_array):

    if bits == datapoints:
        return ICRF_array

    x_new = np.linspace(0, 1, num=bits)
    x_old = np.linspace(0, 1, num=datapoints)
    interpolated_ICRF = np.zeros((bits, channels), dtype=float)

    for c in range(channels):

        y_old = ICRF_array[:, c]
        interpolated_ICRF[:, c] = np.interp(x_new, x_old, y_old)

    return interpolated_ICRF


def calibration(initial_guess, evaluation_heights, lower_limit, upper_limit):
    """ The main function running the ICRF calibration process that is called
    from outside the module.

       Args:
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
    mean_data_array = np.zeros((bits, datapoints, channels), dtype=int)
    PCA_array = np.zeros((datapoints, num_of_PCA_params, channels), dtype=float)
    ICRF_array = np.zeros((datapoints, channels), dtype=float)
    mean_ICRF_array = np.zeros((datapoints, channels), dtype=float)
    final_energy_array = np.zeros(channels, dtype=float)
    initial_energy_array = np.zeros(channels, dtype=float)

    limits = [[lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit]]

    for i in range(len(mean_data_files)):
        # Get the filenames from the attribute arrays.
        mean_file_name = mean_data_files[i]
        PCA_file_name = principal_component_files[i]
        mean_ICRF_file_name = mean_ICRF_files[i]

        # Load mean data, principal component data and mean ICRF data into
        # numpy arrays.
        mean_data_array[:, :, i] = rd.read_data_from_txt(mean_file_name)
        PCA_array[:, :, i] = rd.read_data_from_txt(PCA_file_name)
        mean_ICRF_array[:, i] = rd.read_data_from_txt(mean_ICRF_file_name)

        # mean_ICRF_array[:, i] = np.linspace(0, 1, datapoints)
        # mean_ICRF_array[:, i] = mean_ICRF_array[:, i]**4

        edge_distances = _process_datapoint_distances(mean_data_array[:, :, i],
                                                      evaluation_heights)

        initialEnergy = _energy_function(initial_guess, mean_ICRF_array[:, i],
                                         PCA_array[:, :, i], evaluation_heights,
                                         edge_distances)
        initial_energy_array[i] = initialEnergy
        print(initialEnergy)

        result = differential_evolution(_energy_function, limits, args=(
            mean_ICRF_array[:, i], PCA_array[:, :, i],
            evaluation_heights, edge_distances), strategy='best1bin', disp=False)

        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        solution = result['x']
        evaluation = _energy_function(solution, mean_ICRF_array[:, i],
                                      PCA_array[:, :, i], evaluation_heights,
                                      edge_distances)
        final_energy_array[i] = evaluation
        print('Solution: f(%s) = %.5f' % (solution, evaluation))

        ICRF_array[:, i] = ICRF

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
    data = np.random.normal(128 / max_DN, 0.0246, 10000000)
    hist, bins = np.histogram(data, bins=1024, range=(0, 1))
    number_of_dists -= 1
    for d in range(number_of_dists):
        data = np.random.normal(128 / max_DN, 0.0246, 10000000)
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
        plt.savefig(os.path.join(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Analysis\ICRF calibration and HDR images\test output', f'Dist{ i}.png'), dpi=500)
        plt.clf()

    print(skew_list)
