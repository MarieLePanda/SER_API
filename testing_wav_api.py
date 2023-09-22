# 11th february 2022, Gauri Prajapati, Nuance
# To predict sentiment (neutral,negative) based on audio features.
# inputs: trained model fullpath
#         testing audio fullpath
#         sampling rate of audio (default=8000 Hz)
#         audio duration to be considered (default=3 sec)
#         type of the model (CNN4, CNN5, CNN6, CNN5_Dense5, LSTM) (default='CNN5_Dense5')
# outputs: - printing wavfile name, sentiment, probability of predicted sentiment, time taken for testing
#          - returns wavfile_name, label, score, time_taken
# python testing_wav_api.py --model_address --full_wav_path --sample_rate --audio_duration --model_type

import keras
from common import get_feature_vector
import numpy as np
import argparse
from datetime import date
import time
from flask import Flask, request
import argparse

app = Flask(__name__)

today = date.today()
def get_sentiment(model_address,wav_path,sample_rate,audio_duration,model_type):
    start_time = time.time()
    if model_type=='LSTM':
        to_flatten = False
        x_test = get_feature_vector(file_path=wav_path,
                                                        flatten=to_flatten,
                                                        sample_rate=sample_rate,
                                                        audio_dur=audio_duration)
        #print('total test utts', x_test.shape)
        #print("starting model loading")
        reconstructed_model = keras.models.load_model(model_address)
        #print('x_test loaded ', x_test.shape)
        x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])  # for LSTM
        #print('x_test: ',x_test)
    elif model_type=='CNN4' or model_type=='CNN5' or model_type=='CNN5_dense5' or model_type=='CNN6':
        to_flatten = False
        x_test = get_feature_vector(file_path=wav_path,
                                              flatten=to_flatten,
                                              sample_rate=sample_rate,
                                              audio_dur=audio_duration)
        #print(x_test.shape)

        #print("starting model loading")
        reconstructed_model = keras.models.load_model(model_address)
        #print('x_test loaded ', x_test.shape)
        x_test = x_test.reshape(1, x_test.shape[0],x_test.shape[1] , 1)  # for CNN
        #print('x_test: ', x_test.shape)

    pred_probs = reconstructed_model.predict(np.array(x_test))
    #print(pred_probs)
    pred = np.argmax(pred_probs)

    if pred == 0:
        label = "female_angry"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[0]
    elif pred == 1:
        label = "female_happy"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[1]
    elif pred == 2:
        label = "female_sad"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[2]
    elif pred == 3:
        label = "male_angry"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[3]
    elif pred == 4:
        label = "male_happy"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[4]
    else:
        label = "male_sad"
        pred_probs = pred_probs.flatten()
        prob = (list(pred_probs))[5]


    wav_name = wav_path.split('/')[-1]
    #print('Prediction Done')
    #print("wavfile name: %s"%wav_name)
    #print("Predicted sentiment: %s" %label)
    #print("Confidence score for the predicted sentiment: %.3f" % prob)
    #print("Time taken for prediction %.3f seconds"%(time.time() - start_time))
    prob = round(prob, 3)
    time_taken_sec = round((time.time() - start_time), 3)
    # #print("predicted sentiment for %s is %s with probability of %.3f."% (wav_name,label,prob))
    # #print("time taken for prediction of sentiment for %s is %.3f seconds."% (wav_name,(time.time() - start_time)))
    return wav_name, label, prob, time_taken_sec

#if __name__ == "__main__":
#    my_parser = argparse.ArgumentParser()
#    my_parser.add_argument('--model_address', type=str, help='path to store the trained model')
#    my_parser.add_argument('--full_wav_path', type=str, help='path to wavfile')
#    my_parser.add_argument('--sample_rate', type=int, help='sampling rate in Hz', default=8000)
#    my_parser.add_argument('--audio_duration', type=float, help='duration of audio on which model is trained', default=3)
#    my_parser.add_argument('--model_type', type=str, help='CNN LSTM ANN CNN4 CNN5 CNN6 CNN5_dense5', default='LSTM')
#    args = my_parser.parse_args()
#    get_sentiment(args.model_address,args.full_wav_path, args.sample_rate,args.audio_duration,args.model_type)


@app.route('/get_sentiment', methods=['POST'])
def api_get_sentiment():
    if 'datafile' not in request.files:
        #print("Heho")
        #print(request.files)
        return 'No datafile part in the request', 400

    datafile = request.files['datafile']
    model_address = request.form.get('model_address', "best_saved_lstm_001lr_100ep_30pat_3sec_2class_mfccs-chroma-mel-contrast-tonnetz_20221004-32-0.74.h5")
    sample_rate = int(request.form.get('sample_rate', 8000))
    audio_duration = float(request.form.get('audio_duration', 3))
    model_type = request.form.get('model_type', 'LSTM')

    # Save the data file to a temporary location and pass its path to get_sentiment
    datafile.save('/tmp/datafile')
    full_wav_path = '/tmp/datafile'

    result = get_sentiment(model_address, full_wav_path, sample_rate, audio_duration, model_type)

    return {
        'result': str(result)
    }

if __name__ == "__main__":
    app.run(debug=True)