import copy

import matplotlib.pyplot as plt
import ImageSet as IS
from ImageSet import ImageSet
import numpy as np
import math
import read_data as rd
from scipy.optimize import differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import os
import general_functions as gf
from typing import Optional

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
acq_path = rd.read_config_single('acquired images path')

ICRF = np.zeros((datapoints, 1), dtype=float)
linear_scale = np.linspace(0, 1, datapoints, dtype=float)


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


def linearize_image_vectorized(imageSet, channel):

    acq = (imageSet.acq * max_DN).astype(int)
    acq_new = np.zeros(np.shape(acq), dtype=float)
    # std_new = np.zeros(np.shape(acq), dtype=float)

    # The ICRFs are in reverse order in the .txt file when compared
    # to how OpenCV opens the channels.
    acq_new[:, :, channel] = ICRF[acq[:, :, channel]]
    # std_new[:, :, channel] = ICRF_diff[acq[:, :, channel]] * \
    #    STD_arr[acq[:, :, channel]]

    imageSet.acq = acq_new
    # imageSet.std = std_new

    return imageSet


def analyze_linearity(sublists_of_imageSets, channel):
    """
    Analyze the linearity of images taken at different exposures.
    Args:
        sublists_of_imageSets: Optionally pass sublist from previous calculations

    Returns:
    """
    results = []
    lower = 5/max_DN
    upper = 250/max_DN

    for sublist in sublists_of_imageSets:

        if len(sublist) < 2:
            continue

        for i in range(len(sublist)):
            for j in range(i+1, len(sublist)):

                if i == j:
                    continue

                x = sublist[i]
                y = sublist[j]
                # stds = []
                division = np.divide(x.acq[:, :, channel], y.acq[:, :, channel],
                                     out=np.zeros_like(x.acq[:, :, channel]),
                                     where=((lower < y.acq[:, :, channel]) & (y.acq[:, :, channel] < upper) &
                                            (lower < x.acq[:, :, channel]) & (x.acq[:, :, channel] < upper)))

                nonzeros = np.nonzero(division)
                channel_mean = np.mean(division[nonzeros])
                # channel_std = np.std(division[nonzeros])
                # stds.append(channel_std)

                ratio = x.exp / y.exp
                if ratio < 0.05:
                    break
                result = (abs(channel_mean-ratio))/ratio
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
    ICRF = _inverse_camera_response_function(mean_ICRF, PCA_array, PCA_params)
    ICRF[:] += 1 - ICRF[-1]
    ICRF[0] = 0
    if np.max(ICRF) > 1 or np.min(ICRF) < 0:
        energy = np.inf
        return energy

    acq_sublists_iter = copy.deepcopy(acq_sublists)
    for sublist in acq_sublists_iter:
        for imageSet in sublist:

            linearize_image_vectorized(imageSet, channel)

    linearity_data = analyze_linearity(acq_sublists_iter, channel)
    energy = np.mean(linearity_data)
    if np.isnan(energy):
        energy = np.Inf

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


def choose_evenly_spaced_points(array, num_points):
    # Convert the 2D array to a numpy array for easier indexing
    array = np.array(array)

    # Calculate the step size between points
    step = max(1, int(array.shape[0] / (num_points - 1)))

    # Select the evenly spaced points
    points = array[::step, ::step]

    return points


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
    ICRF_array = np.zeros((datapoints, channels), dtype=float)
    final_energy_array = np.zeros(channels, dtype=float)
    initial_energy_array = np.zeros(channels, dtype=float)

    limits = [[lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit], [lower_limit, upper_limit],
              [lower_limit, upper_limit]]

    # Initialize image lists and name lists

    acq_list = IS.create_imageSets(acq_path)
    for imageSet in acq_list:
        imageSet.load_acq()
        imageSet.acq = choose_evenly_spaced_points(imageSet.acq, 30)
    acq_sublists = gf.separate_to_sublists(acq_list)
    for sublist in acq_sublists:
        sublist.sort(key=lambda imageSet: imageSet.exp)
    del acq_list

    for i in range(len(mean_data_files)):
        # Get the filenames from the attribute arrays.
        PCA_file_name = principal_component_files[i]
        mean_ICRF_file_name = mean_ICRF_files[i]

        # Load mean data, principal component data and mean ICRF data into
        # numpy arrays.
        PCA_array = rd.read_data_from_txt(PCA_file_name)

        if initial_function is None:
            mean_ICRF_array = rd.read_data_from_txt(mean_ICRF_file_name)
        else:
            mean_ICRF_array = initial_function

        limit = 0.01

        with DifferentialEvolutionSolver(_energy_function, limits, args=(
            mean_ICRF_array, PCA_array, acq_sublists, i),
                                         strategy='best1bin', tol=0.0001) as solver:
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

    print('Run script from main file!')
