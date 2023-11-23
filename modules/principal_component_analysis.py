import numpy as np
from sklearn.decomposition import PCA
from global_settings import *


def _calculate_principal_components(covariance_array):

    PCA_array = PCA(n_components=NUM_OF_PCA_PARAMS)
    PCA_array.fit(covariance_array)
    result = PCA_array.transform(covariance_array)

    # Scale to unit vector and shift to start and end at zero.
    for n in range(NUM_OF_PCA_PARAMS):
        norm = np.linalg.norm(result[:, n])
        result[:, n] /= norm
        result[:, n] -= result[0, n]

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
    covariance_array = np.zeros((DATAPOINTS, DATAPOINTS), dtype=float)
    number_of_ICRFs = np.shape(data_array)[0]

    for i in range(DATAPOINTS):
        for j in range(DATAPOINTS):

            running_sum = 0

            for k in range(number_of_ICRFs):

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
    for i in range(len(ICRF_FILES)):

        file_name = ICRF_FILES[i]
        mean_file_name = MEAN_ICRF_FILES[i]
        ICRF_array = rd.read_data_from_txt(file_name)
        mean_ICRF_array = rd.read_data_from_txt(mean_file_name)

        covariance_matrix = _calculate_covariance_matrix(ICRF_array,
                                                         mean_ICRF_array)

        PCA_array = _calculate_principal_components(covariance_matrix)

        np.savetxt(data_directory.joinpath(PCA_FILES[i]), PCA_array)

    return


if __name__ == "__main__":

    test_file_name = ICRF_FILES[0]
    test_mean_file_name = MEAN_ICRF_FILES[0]
    test_inverse_curve = rd.read_data_from_txt(test_file_name)
    test_mean_inverse_curve = rd.read_data_from_txt(test_mean_file_name)

    test_covariance = _calculate_covariance_matrix(test_inverse_curve,
                                                   test_mean_inverse_curve)

    test_PCA = _calculate_principal_components(test_covariance)
    print(test_PCA[0])
