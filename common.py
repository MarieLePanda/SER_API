import numpy as np
from sklearn.model_selection import train_test_split
from utilities import get_data, get_feature_vector_from_mfcc

def extract_data(flatten: bool, sample_rate: int,path_to_test_data: str, feat_path: str, audio_duration: int, data_details_file: str):
    x_test, test_wavs = get_data(ip_file=data_details_file, data_path=path_to_test_data, flatten=flatten, feat_path=feat_path, sample_rate=sample_rate, audio_dur=audio_duration, mfcc_len= 39)

    return np.array(x_test), np.array(test_wavs)


def get_feature_vector(file_path, flatten, sample_rate,audio_dur):
    return get_feature_vector_from_mfcc(file_path=file_path, mfcc_len=13, flatten=flatten, audio_dur=audio_dur,sample_rate=sample_rate)

