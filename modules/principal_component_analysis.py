import os
import numpy as np
import read_data as rd
from sklearn.decomposition import PCA

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
ICRF_files = rd.read_config_list('ICRFs')
number_of_ICRFs = np.shape(ICRF_files)[0]
mean_ICRF_files = rd.read_config_list('mean ICRFs')
PCA_files = rd.read_config_list('principal components')
number_of_components = rd.read_config_single('number of principal components')
datapoints = rd.read_config_single('final datapoints')


def _calculate_principal_components(covariance_array):

    PCA_array = PCA(n_components=number_of_components)
    PCA_array.fit(covariance_array)
    result = PCA_array.transform(covariance_array)

    return result


def _calculate_covariance_matrix(data_array, mean_data_array):
    """
    Calculate the covariance matrix according to the article 'What is the space
    of camera response functions'.

        Args:
            data_array: the ICRF data obtained from the original DoRF file for
                each channel separately.
            mean_data_array: mean ICRF data for each channel calculated from the
                collection of ICRFs in the original DoRF file.

        Return:
            the covariance matrix calculated for the given data.
    """
    covariance_array = np.zeros((datapoints, datapoints), dtype=float)

    for i in range(datapoints):
        for j in range(datapoints):

            running_sum = 0

            for k in range(number_of_ICRFs + 1):

                running_sum += (data_array[k, i] - mean_data_array[i]) * (
                        data_array[k, j] - mean_data_array[j])

            covariance_array[i, j] = running_sum

    # eigen_value, eigen_vector = la.eig(covariance_array)

    return covariance_array  # , eigen_value.real, eigen_vector.real


def analyze_principal_components():
    """ Main function to be called outside the module, used to run the process
    of obtaining principal components for the ICRF data for each channel
    separately.
    """
    for i in range(len(ICRF_files)):

        file_name = ICRF_files[i]
        mean_file_name = mean_ICRF_files[i]
        ICRF_array = rd.read_data_from_txt(file_name)
        mean_ICRF_array = rd.read_data_from_txt(mean_file_name)

        covariance_matrix = _calculate_covariance_matrix(ICRF_array,
                                                         mean_ICRF_array)

        PCA_array = _calculate_principal_components(covariance_matrix)

        np.savetxt(os.path.join(data_directory, PCA_files[i]), PCA_array)

    return


if __name__ == "__main__":

    test_file_name = ICRF_files[0]
    test_mean_file_name = mean_ICRF_files[0]
    test_inverse_curve = rd.read_data_from_txt(test_file_name)
    test_mean_inverse_curve = rd.read_data_from_txt(test_mean_file_name)

    test_covariance = _calculate_covariance_matrix(test_inverse_curve,
                                                   test_mean_inverse_curve)

    test_PCA = _calculate_principal_components(test_covariance)
    print(test_PCA[0])
