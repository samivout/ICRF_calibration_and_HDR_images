import read_data as rd
from pathlib import Path

# From read_data.py
data_directory = rd.data_directory
module_directory = rd.current_directory

# From HDR_image.py
IM_SIZE_X = rd.read_config_single('image size x')
IM_SIZE_Y = rd.read_config_single('image size y')
DEFAULT_ACQ_PATH = Path(rd.read_config_single('acquired images path'))
FLAT_PATH = Path(rd.read_config_single('flat fields path'))
OG_FLAT_PATH = Path(rd.read_config_single('original flat fields path'))
DARK_PATH = Path(rd.read_config_single('dark frames path'))
OG_DARK_PATH = Path(rd.read_config_single('original dark frames path'))
OUT_PATH = Path(rd.read_config_single('corrected output path'))
ICRF_CALIBRATED_FILE = rd.read_config_single('calibrated ICRFs')
CHANNELS = rd.read_config_single('channels')
BIT_DEPTH = rd.read_config_single('bit depth')
BITS = 2 ** BIT_DEPTH
MAX_DN = BITS - 1
MIN_DN = 0
DATAPOINTS = rd.read_config_single('final datapoints')
DATA_MULTIPLIER = DATAPOINTS / BITS
DATAPOINT_MULTIPLIER = rd.read_config_single('datapoint multiplier')
STD_FILE_NAME = rd.read_config_single('STD data')

# From camera_data_tools.py
MEAN_DATA_FILES = rd.read_config_list('camera mean data')
BASE_DATA_FILES = rd.read_config_list('camera base data')

# From process_CRF_database.py
DORF_FILE = rd.read_config_single('source DoRF data')
DORF_DATAPOINTS = rd.read_config_single('original DoRF datapoints')
ICRF_FILES = rd.read_config_list('ICRFs')
MEAN_ICRF_FILES = rd.read_config_list('mean ICRFs')

# From ICRF_calibration_exposure.py
OUTPUT_DIRECTORY = module_directory.parent.joinpath('output')
NUM_OF_PCA_PARAMS = rd.read_config_single('number of principal components')
PRINCIPAL_COMPONENT_FILES = rd.read_config_list('principal components')
ACQ_PATH = Path(rd.read_config_single('acquired images path'))

# From image_correction.py
DARK_THRESHOLD = rd.read_config_single('dark threshold')

# From linearity_analysis_scripts.py
EVALUATION_HEIGHTS = rd.read_config_list('evaluation heights')
NUMBER_OF_HEIGHTS = len(EVALUATION_HEIGHTS)
LOWER_PCA_LIM = rd.read_config_single('lower PC coefficient limit')
UPPER_PCA_LIM = rd.read_config_single('upper PC coefficient limit')
IN_PCA_GUESS = rd.read_config_list('initial guess')

# From mean_data_collector.py
VIDEO_PATH = Path(rd.read_config_single('video path'))

# New
LOWER_LIN_LIM = rd.read_config_single('lower linearity limit')/MAX_DN
UPPER_LIN_LIM = rd.read_config_single('upper linearity limit')/MAX_DN
