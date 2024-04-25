import numpy as np
from scipy import stats as st
import cv2 as cv
import general_functions as gf
from ImageSet import ImageSet
from typing import Optional, List, Union
from global_settings import *


def calculate_images_simple(path: Path, use_mean: bool, use_variance: bool,
                            use_median: bool, use_mode: bool):

    video, count = load_video(path)

    mean_image = None
    std_image = None
    median_image = None
    mode_image = None

    if use_mean:
        mean_image = calculate_mean_image(video)
    if use_variance:
        std_image = calculate_std(video)
    if use_median:
        median_image = calculate_median_image(video)
    if use_mode:
        mode_image = calculate_mode_image(video)

    ret = {'mean': mean_image, 'std': std_image, 'median': median_image, 'mode': mode_image}

    return ret


def calculate_mode_image(video: np.ndarray):

    mode_image = st.mode(video, axis=-1, keepdims=False)

    return np.around(mode_image[0]).astype(np.dtype('uint8'))


def calculate_std(video: np.ndarray):

    std_image = np.var(video, axis=-1, keepdims=False, dtype=np.dtype('float64')) * 10

    return np.around(std_image).astype(np.dtype('uint8'))


def calculate_mean_image(video: np.ndarray):

    mean_image = np.mean(video, axis=-1, keepdims=False, dtype=np.dtype('float64'))

    return np.around(mean_image).astype(np.dtype('uint8'))


def calculate_median_image(video: np.ndarray):

    median_image = np.median(video, axis=-1, keepdims=False, dtype=np.dtype('float64'))

    return np.around(median_image).astype(np.dtype('uint8'))


def load_video(file_path: Path):

    stacked_array = np.zeros((IM_SIZE_Y, IM_SIZE_X, CHANNELS), dtype=np.dtype('uint8'))
    stacked_array = stacked_array[:, :, :, np.newaxis]

    video = cv.VideoCapture(str(file_path))
    video_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            continue

        if count == 0:
            stacked_array[:, :, :, 0] = frame
        else:
            frame = frame[:, :, :, np.newaxis]
            stacked_array = np.concatenate((stacked_array, frame), axis=3)
            print(np.shape(stacked_array))

        count += 1
        if count > video_frames:
            video.release()
            break

    return stacked_array, count


def _to_lbgr(frame: np.ndarray):

    lows = (frame <= 0.04045)
    highs = np.logical_not(lows)

    frame[lows] = frame[lows] / 12.92
    frame[highs] = ((frame[highs] + 0.055) / 1.055) ** 2.4

    # frame = np.clip(frame, 0, 1)

    return frame


def _to_sbgr(frame: np.ndarray):

    lows = (frame <= 0.0031308)
    highs = np.logical_not(lows)

    frame[lows] = frame[lows] * 12.92
    frame[highs] = 1.055 * frame[highs] ** (1 / 2.4) - 0.055

    # frame = np.clip(frame, 0, 1)

    return frame


def welford_algorithm(file_paths: Union[Path, List[Path]], linearize, lbgr):

    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    if linearize:
        ICRF = rd.read_data_from_txt(ICRF_CALIBRATED_FILE)
    video = cv.VideoCapture(str(file_paths[0]))
    video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = 0
    mean = np.zeros((video_height, video_width, CHANNELS), dtype=np.dtype('float64'))
    m2 = np.zeros((video_height, video_width, CHANNELS), dtype=np.dtype('float64'))

    for file_path in file_paths:

        video = cv.VideoCapture(str(file_path))
        video_frame_count = 0
        video_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                continue

            frame = (frame/MAX_DN).astype(np.dtype('float64'))
            if linearize:
                frameSet = ImageSet(acq=frame)
                frameSet = gf.linearize_ImageSet(frameSet, ICRF, gaussian_blur=False)
                frame = frameSet.acq
            if lbgr:
                frame = _to_lbgr(frame)

            video_frame_count += 1
            total_frame_count += 1
            delta = frame - mean
            mean = mean + delta / total_frame_count
            m2 = m2 + delta * (frame - mean)

            if video_frame_count > video_frames:
                video.release()
                break

    m2 = np.sqrt(m2 / (total_frame_count - 1)) / np.sqrt(total_frame_count)
    mean = mean * MAX_DN
    # m2 = m2 * MAX_DN * 7

    mean = (np.around(mean)).astype(np.dtype('uint8'))
    # m2 = (np.around(m2)).astype(np.dtype('uint8'))
    ret = {'mean': mean, 'std': m2}

    return ret


def process_video(video_paths: Optional[List[Path]] = None,
                  linearize: Optional[bool] = False,
                  use_mean: Optional[bool] = True,
                  use_std: Optional[bool] = True,
                  use_median: Optional[bool] = False,
                  use_mode: Optional[bool] = False):

    if video_paths is None:
        video_paths = gf.get_filepath_dialog('Choose video file')
    # ret = calculate_images_simple(video_path, use_mean, use_std, use_median, use_mode)
    ret = welford_algorithm(video_paths, linearize, lbgr=False)

    for key in ret:
        if ret[key] is not None:
            save_path = str(video_paths.parent.joinpath(video_paths.name.replace('.avi', f'.{key}.tif')))
            cv.imwrite(save_path, ret[key])


def process_directory(dir_path: Optional[Path] = None,
                      linearize: Optional[bool] = False,
                      separately: Optional[bool] = True):

    temp_path = Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Analysis\Image correction class\Input\Output')

    if dir_path is None:
        video_path = gf.get_filepath_dialog('Choose video file for dir')
        dir_path = video_path.parent

    video_files = list(dir_path.glob("*.avi"))

    if not separately:
        ret = welford_algorithm(video_files, linearize, lbgr=False)

        for key in ret:
            if ret[key] is not None:
                save_path = str(dir_path.joinpath(f'total_{key}.tif'))
                cv.imwrite(save_path, ret[key])

    else:
        for path in video_files:
            print(f'Starting video file {path}')
            ret = welford_algorithm(path, linearize, False)
            print('Finished file')

            for key in ret:
                if ret[key] is not None:
                    save_dir = temp_path.joinpath(key)
                    # save_dir = path.parent.joinpath(key)
                    save_dir.mkdir(exist_ok=True)
                    if key == 'std':
                        save_path = str(save_dir.joinpath(path.name.replace('.avi', f' STD.tif')))
                    else:
                        save_path = str(save_dir.joinpath(path.name.replace('.avi', f'.tif')))
                    cv.imwrite(save_path, ret[key])

    return


def process_directory_simple(dir_path: Optional[Path] = None,
                             use_mean: Optional[bool] = True,
                             use_variance: Optional[bool] = True,
                             use_median: Optional[bool] = False,
                             use_mode: Optional[bool] = False):

    if dir_path is None:
        video_path = gf.get_filepath_dialog('Choose video file for dir')
        dir_path = video_path.parent

    video_files = list(dir_path.glob("*.avi"))

    for path in video_files:
        ret = calculate_images_simple(path, use_mean, use_variance, use_median, use_mode)

        for key in ret:
            if ret[key] is not None:
                save_dir = dir_path.joinpath(key)
                save_dir.mkdir(exist_ok=True)
                if key == 'variance':
                    save_path = str(save_dir.joinpath(path.name.replace('.avi', f'.{key}.tif')))
                else:
                    save_path = str(save_dir.joinpath(path.name.replace('.avi', f'.tif')))
                cv.imwrite(save_path, ret[key])


if __name__ == "__main__":

    # process_video(linearize=True, use_mean=True, use_std=True, use_median=True, use_mode=True)
    process_directory(linearize=False, separately=True)
    # process_directory_simple(None, False, False, False, True)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\5x\BF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\5x\DF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\10x\BF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\10x\DF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\20x\BF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\20x\DF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\50x\BF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\50x\DF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\50x LWD\BF'), True, False)
    # process_directory(Path(r'E:\Kouluhommat\MSc SPV\Mittaukset jne\Flatfield movies\50x LWD\DF'), True, False)

