"""
This file contains functions to read the data files from the given folders and
generate Mel Frequency Cepestral Coefficients features for the given audio
files as training samples.
"""
import os
import sys
from typing import Tuple
import librosa, pickle
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

# mean_signal_length = 32000  # Empirically calculated for the given data set 66150


def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
                                 audio_dur: float, sample_rate: int, mfcc_len: int =39) -> np.ndarray:
    """
    Make feature vector from MFCC for the given wav file.

    Args:
        file_path (str): path to the .wav file that needs to be read.
        flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
        mfcc_len (int): Number of cepestral co efficients to be consider.

    Returns:
        numpy.ndarray: feature vector of the wav file made from mfcc.
    """
    # fs, signal = wav.read(file_path)
    signal, fs = librosa.load(file_path, sr=sample_rate)
    s_len = len(signal)
    # audio_dur = 2  # in sec
    mean_signal_length = int(audio_dur * fs)  # Empirically calculated for the given data set 66150
    i = 0
    # print(fs)
    # quit()
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mean_signal_length:
        print(i + 1) # files less than 3 sec duration
        i += 1
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    # print('mfcc ',np.array(mel_coefficients.shape))

    # print(mel_coefficients.shape)
    if flatten:
        # Flatten the data
        # mel_coefficients = np.ravel(mel_coefficients)
        mel_coefficients = np.average(mel_coefficients, axis=0)
        print('mel_coefficients after average: ', mel_coefficients.shape)
    file_name = file_path.split('/')[-1]
    basename = file_name.split('.')[0]

    return mel_coefficients



def get_data(ip_file: str, data_path: str, flatten: bool, audio_dur: int, sample_rate: int, feat_path: str, mfcc_len:int =39) -> \
        tuple[np.ndarray, np.ndarray]:
    """Extract data for training and testing.

    1. Iterate through all the folders.
    2. Read the audio files in each folder.
    3. Extract Mel frequency cepestral coefficients for each file.
    4. Generate feature vector for the audio files as required.

    Args:
        data_path (str): path to the data set folder
        flatten (bool): Boolean specifying whether to flatten the data or not.
        mfcc_len (int): Number of mfcc features to take for each frame.
        class_labels (tuple): class labels that we care about.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Two numpy arrays, one with mfcc and
        other with labels.


    """
    data = []
    labels = []
    names = []
    emo_names = []
    df = pd.read_excel(ip_file, 'Sheet1')
    sent_labels = df["label"].to_list()
    filename_list = df["filename"].to_list()
    sentiment_list = df["sentiment"].to_list()
    tag_list = df["flag"].to_list()
    for i in range(len(sent_labels)):
        if tag_list[i] == 'test':
            filepath = data_path + '/' + filename_list[i]
            print('Extracting features for testing set')

            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                          mfcc_len=13,
                                                          flatten=flatten,
                                                          audio_dur=audio_dur,
                                                          sample_rate=sample_rate,
                                                          feat_path=feat_path)

            data.append(feature_vector)
            names.append(filename_list[i])

    return np.array(data), np.array(names)
