import read_data as rd
import numpy as np
import os

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(os.path.dirname(current_directory), 'data')
dorf_file = rd.read_config_single('source DoRF data')
dorf_datapoints = rd.read_config_single('original DoRF datapoints')
final_datapoints = rd.read_config_single('final datapoints')
ICRF_files = rd.read_config_list('ICRFs')
mean_ICRF_files = rd.read_config_list('mean ICRFs')


def _read_dorf_data(file_name):
    """ Load numerical data from a .txt file of the given name from the data
    directory. The dorfCurves.txt contains measured irradiance vs. digital
    number data for various cameras. In this function all the data is read in
    and split into separate Numpy float arrays for each color channel.

        Args:
            file_name: the name of the .txt file containing the dorf data.

        Return:
            list of numpy float arrays, one for each color channel.
    """
    file = os.path.join(data_directory, file_name)
    red_curves = np.zeros((1, dorf_datapoints), dtype=float)
    blue_curves = np.zeros((1, dorf_datapoints), dtype=float)
    green_curves = np.zeros((1, dorf_datapoints), dtype=float)
    number_of_lines = 0
    is_red = False
    is_green = False
    is_blue = False
    with open(file) as f:
        for line in f:
            text = line.rstrip().casefold()
            number_of_lines += 1
            if (number_of_lines + 5) % 6 == 0:
                if text.endswith('red') or text[-1] == 'r' or text[-2] == 'r':
                    is_red = True
                    continue
                elif text.endswith('green') or text[-1] == 'g' or text[-2] == 'g':
                    is_green = True
                    continue
                elif text.endswith('blue') or text[-1] == 'b' or text[-2] == 'b':
                    is_blue = True
                    continue
                else:
                    is_red = True
                    is_green = True
                    is_blue = True

            if number_of_lines % 6 == 0:
                line_to_arr = np.fromstring(text, dtype=float, sep=' ')
                if is_red:
                    red_curves = np.vstack([red_curves, line_to_arr])
                    is_red = False
                if is_green:
                    green_curves = np.vstack([green_curves, line_to_arr])
                    is_green = False
                if is_blue:
                    blue_curves = np.vstack([blue_curves, line_to_arr])
                    is_blue = False

    f.close()

    # Remove the initial row of zeros from the arrays.
    red_curves = np.delete(red_curves, 0, 0)
    green_curves = np.delete(green_curves, 0, 0)
    blue_curves = np.delete(blue_curves, 0, 0)

    list_of_curves = [red_curves, green_curves, blue_curves]

    return list_of_curves


def _invert_and_interpolate_data(list_of_curves, new_datapoints):
    """ Invert the camera response functions obtained from the dorfCurves.txt
    file. Numpy interpolation is used to obtain the same digital value
    datapoints for all curves, as originally the evenly spaced datapoints were
    in the irradiance domain.

            Args:
                list_of_curves: list containing all the CRFs, separated based
                on color channel, in Numpy float arrays. Original data is spaced
                into 1024 points evenly from 0 to 1.
                new_datapoints: number of datapoints to be used in the digital
                value range.

            Return:
                list of numpy float arrays, one for each color channel,
                containing the inverted camera response functions, or ICRFs.
    """
    list_of_processed_curves = []
    x_old = np.linspace(0, 1, dorf_datapoints)
    x_new = np.linspace(0, 1, new_datapoints)

    for index, arr in enumerate(list_of_curves):
        rows = arr.shape[0]
        y_new = np.zeros(new_datapoints, dtype=float)

        for i in range(rows):
            y = arr[i]
            y_inv = np.interp(x_old, y, x_old)

            if dorf_datapoints != new_datapoints:

                interpolated_row = np.interp(x_new, x_old, y_inv)
                y_new = np.vstack([y_new, interpolated_row])

        y_new = np.delete(y_new, 0, 0)
        list_of_processed_curves.append(y_new)

    return list_of_processed_curves


def _calculate_mean_curve(list_of_curves):
    """ Calculate the mean function from a collection of CRFs or ICRFs

            Args:
                list_of_curves: list containing the Numpy float arrays of the
                CRFs or ICRFs from which a mean function will be calculated for
                each channel.
            Return:
                list containing the Numpy float arrays for the mean CRFs or
                ICRFs for each color channel.

    """

    for index, curves in enumerate(list_of_curves):

        list_of_curves[index] = np.mean(curves, axis=0)

    return list_of_curves


def process_CRF_data():
    """ Main function to be called outside the module, used to run the process
    of obtaining the CRFs from dorfCurves.txt, invert them and determine a mean
    ICRF, for each color channel separately.
    """
    list_of_curves = _read_dorf_data(dorf_file)
    processed_curves = _invert_and_interpolate_data(list_of_curves, final_datapoints)
    list_of_mean_curves = processed_curves.copy()
    list_of_mean_curves = _calculate_mean_curve(list_of_mean_curves)

    for i in range(len(ICRF_files)):

        np.savetxt(os.path.join(data_directory, ICRF_files[i]),
                   processed_curves[i])
        np.savetxt(os.path.join(data_directory, mean_ICRF_files[i]),
                   list_of_mean_curves[i])

    return


if __name__ == "__main__":
    test_curves = _read_dorf_data(dorf_file)
    test_curves = _invert_and_interpolate_data(test_curves, final_datapoints)
    test_mean_curves = test_curves.copy()
    test_mean_curves = _calculate_mean_curve(test_mean_curves)

    for i in range(len(ICRF_files)):
        np.savetxt(os.path.join(data_directory, ICRF_files[i]),
                   test_curves[i])
        np.savetxt(os.path.join(data_directory, mean_ICRF_files[i]),
                   test_mean_curves[i])
