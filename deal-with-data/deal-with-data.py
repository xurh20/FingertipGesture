import json
import numpy as np
import random
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, BatchNormalization, Dense, Input, LSTM, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import sys, os

candidates = [chr(y) for y in range(97, 123)]
BASE_DIR = "../new_data/"
MAX_LEN = 200
CHAR_NUM = 26


def loadData():
    x_train, y_train = [], []
    for dir in os.listdir(BASE_DIR):
        for file in os.listdir(BASE_DIR + "/" + dir):
            x = np.load(BASE_DIR + "/" + dir + "/" + file)
            y = file.split('_')[0]
            x_train.append(x)
            y_train.append(candidates.index(y))
    return np.array(x_train), np.array(y_train)


def RNN_model_1(x_train_padded_seqs, y_train):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(
        MAX_LEN,
        980,
    )))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(CHAR_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # print(y_train)
    y_train = to_categorical(y_train, num_classes=CHAR_NUM)
    model.fit(x_train_padded_seqs,
              y_train,
              epochs=15,
              batch_size=40,
              validation_split=0.2)


if __name__ == "__main__":
    x_train, y_train = loadData()
    for i in range(len(x_train)):
        first_dimention = len(x_train[i])
        # for k in range(len(x_train[i])):
        # tmp = [j for arr in x_train[i][k] for j in arr]
        # x_train[i][k] = 1
        # x_train[i][k] = x_train[i][k].re\
        x_train[i] = x_train[i].reshape(first_dimention, 980)

    # print(x_train)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, dtype="float32")
    # print(x_train)
    # print(y_train)
    y_train = np.array(y_train)
    # print(y_train)
    RNN_model_1(x_train, y_train)