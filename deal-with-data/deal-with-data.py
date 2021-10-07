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

candidates = [chr(y) for y in range(97, 123)]
BASE_DIR = "../alphabet_data/"
TRAIN_NUM = 40
VALID_NUM = 20
CHAR_NUM = 6
MAX_LEN = 200

def loadData(num):
    # num 表示字母数量
    trainData = []
    validData = []
    trainAns = []
    validAns = []
    label_i = list(range(TRAIN_NUM + VALID_NUM))
    random.shuffle(label_i)
    print(label_i)
    for char in range(0, num):
        for i in range(0, TRAIN_NUM):
            trainData.append(np.load(BASE_DIR + candidates[char] + "_" + str(label_i[i]) + ".npy"))
            trainAns.append(char)
        for i in range(TRAIN_NUM, TRAIN_NUM + VALID_NUM):
            validData.append(np.load(BASE_DIR + candidates[char] + "_" + str(label_i[i]) + ".npy"))
            validAns.append(char)
    print(len(trainData))
    print(len(trainData[0]))
    return trainData, trainAns, validData, validAns

def RNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, x_valid, y_valid):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(200, 980,)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(CHAR_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(y_train)
    y_train = to_categorical(y_train, num_classes=CHAR_NUM)
    y_valid = to_categorical(y_valid, num_classes=CHAR_NUM)
    filepath = './model/lstm_model_weights.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                            mode='max')
    callbacks_list = [checkpoint]
    model.fit(x_train_padded_seqs, y_train, epochs=15, batch_size=40, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
    print(x_test_padded_seqs)
    y_predict = np.argmax(model.predict(x_test_padded_seqs), axis=-1)
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('Micro', metrics.precision_score(y_test, y_predict, average='micro'))
    print('Macro', metrics.precision_score(y_test, y_predict, average='macro'))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
    confusion_matrix = metrics.confusion_matrix(y_test, np.rint(y_predict))
    print(confusion_matrix)
    # model.save_weights('./model/lstm_model_weights.h5')
    json_string = model.to_json()
    open('../model/lstm_model_architecture.json','w').write(json_string)

    # model = model_from_json(open('./model/lstm_model_architecture.json').read())  
    # model.load_weights('./model/lstm_model_weights.h5')
    # y_predict = np.argmax(model.predict(x_test_padded_seqs), axis=-1)
    # print(y_predict)
    # print('准确率', metrics.accuracy_score(y_test, y_predict))
    ans = [metrics.accuracy_score(y_test, y_predict), metrics.precision_score(y_test, y_predict, average='micro'), metrics.precision_score(y_test, y_predict, average='macro'), metrics.f1_score(y_test, y_predict, average='weighted')]
    return ans


if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid = loadData(CHAR_NUM)
    for i in range(len(x_train)):
        first_dimention = len(x_train[i])
        # for k in range(len(x_train[i])):
            # tmp = [j for arr in x_train[i][k] for j in arr]
            # x_train[i][k] = 1
            # x_train[i][k] = x_train[i][k].re\
        x_train[i] = x_train[i].reshape(first_dimention, 980)
    for i in range(len(x_valid)):
        first_dimention = len(x_valid[i])
    #     for k in range(len(x_valid[i])):
    #         x_valid[i][k] = [j for arr in x_valid[i][k] for j in arr]
        x_valid[i] = x_valid[i].reshape(first_dimention, 980)
    # print(x_train)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, dtype="float32")
    x_valid = pad_sequences(x_valid, maxlen=MAX_LEN, dtype="float32")
    # print(x_train)
    # print(y_train)
    y_train = np.array(y_train)
    # print(y_train)
    y_valid = np.array(y_valid)
    RNN_model_1(x_train, y_train, x_valid, y_valid, x_valid, y_valid)