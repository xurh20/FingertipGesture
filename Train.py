import os, joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from Plot import getAveragePath, getPathAngles

from keras.preprocessing import sequence
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, BatchNormalization, Dense, Input, LSTM, Bidirectional, Masking, ConvLSTM2D, Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras.backend import concatenate
from keras.callbacks import ModelCheckpoint

candidates = [chr(y) for y in range(97, 123)]
BASE_DIR = "new_data/"
TARGET_DIR = "avg_new_data"
MAX_LEN = 300
CHAR_NUM = 26


def loadData():
    x_train, y_train = [], []
    for dir in os.listdir(BASE_DIR):
        for file in os.listdir(BASE_DIR + "/" + dir):
            x = np.load(BASE_DIR + "/" + dir + "/" + file)
            y = file.split('_')[0]
            x_train.append(x)
            y_train.append(candidates.index(y))
    return x_train, y_train


def loadAngleData():
    x_train, y_train = [], []
    for dir in os.listdir(BASE_DIR):
        for file in os.listdir(BASE_DIR + "/" + dir):
            x = np.load(BASE_DIR + "/" + dir + "/" + file)
            y = file.split('_')[0]
            x_train.append(getPathAngles(*getAveragePath(x)))
            y_train.append(candidates.index(y))
    return x_train, y_train


def trainLSTM():
    x, y = loadAngleData()
    x_train = pad_sequences(x, MAX_LEN, dtype=np.float64)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_train = to_categorical(y, num_classes=CHAR_NUM)
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(
        MAX_LEN,
        1,
    )))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(CHAR_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, validation_split=0.2)
    # model.save('deal-with-data/model/rnn_model.h5')


if __name__ == '__main__':
    trainLSTM()