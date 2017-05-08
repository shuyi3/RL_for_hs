from __future__ import print_function
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from simple_dqn.encoder import *
from os import listdir
from os.path import isfile, join
import logit_eval
import random
import sys
import zmq

batch_size = 32
nb_classes = 2
# nb_epoch = 200
nb_epoch = 150
data_augmentation = True

# input image dimensions
board_rows, board_cols = 7, 7
board_channels = 87

hand_rows, hand_cols = 10, 7
hand_channels = 119
hidden_size = 1024

class CNNPhaseActionPolicyFast:

    def __init__(self):
        max_mana = 10
        weight_folder = 'HL_policy_weight_fcn_nBN\\'
        cnn_weight_list = [None] * 10
        self.model_list = [None] * 10
        weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
        for file_name in weight_files:
            phase = int(file_name[7])
            cnn_weight_list[phase] = file_name

        for i in range(max_mana):
            merged_model = compile_fcn_model(nn_type='policy')
            merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
            self.model_list[i] = merged_model

    def predict_policy(self, feature_list):
        idx = int(feature_list[0][0])
        state_feature = feature_list[1]
        return self.model_list[idx].predict(state_feature)

    def predict_classes(self, mana, ft_list):
        return self.model_list[mana].predict_classes(ft_list)

    def run_server(self):
        port = "5556"
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % port)

        X = [np.zeros((1, 26), dtype=np.float32),
               np.zeros((1, 9 * 17 * 5), dtype=np.float32),
               np.zeros((1, 18 * 23), dtype=np.float32),
               np.zeros((1, 23), dtype=np.float32)]

        def process(chunk):
            return map(lambda x: map(int, x.split(',')), chunk.split('.'))

        while True:
            frm = socket.recv(copy=False)
            vals = map(process, frm.bytes.split('|'))
            mana = vals[0][0][0]
            for i in xrange(len(vals)): X[i][vals[i][0][0]][vals[i][0][1]][vals[i][0][2]] = 1
            res = np.argmax(self.predict_classes(mana, X)[0])
            socket.send(res)

class CNNPhaseActionPolicy:

    def __init__(self):
        max_mana = 10
        weight_folder = 'action_policy_weight\\'
        cnn_weight_list = [None] * 10
        self.model_list = [None] * 10
        weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
        for file_name in weight_files:
            phase = int(file_name[11])
            cnn_weight_list[phase] = file_name

        for i in range(max_mana):
            merged_model = compile_cnn_model(nn_type='policy')
            merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
            self.model_list[i] = merged_model

    def predict_policy(self, feature_list):
        idx = int(feature_list[0][0])
        state_feature = feature_list[1]
        return self.model_list[idx].predict(state_feature)

    def predict_classes(self, feature_list):
        idx = int(feature_list[0][0])
        state_feature = feature_list[1]
        return self.model_list[idx].predict_classes(state_feature)

class CNNPhasePolicy:

    def __init__(self):
        max_mana = 10
        # weight_folder = 'HL_policy_weight_fcn_nBN1\\'
        weight_folder = 'HL_policy_weight_cnn_nBN_adad\\'
        cnn_weight_list = [None] * 10
        self.model_list = [None] * 10
        weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
        for file_name in weight_files:
            phase = int(file_name[7])
            cnn_weight_list[phase] = file_name

        for i in range(max_mana):
            merged_model = compile_cnn_model(nn_type='policy')
            merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
            self.model_list[i] = merged_model

    def predict_policy(self, feature_list):
        idx = int(feature_list[0][0])
        state_feature = feature_list[1]
        # return self.model_list[idx].predict(state_feature)
        return self.model_list[idx].predict_proba(state_feature)


class CNNPhaseEval:

    def __init__(self):
        max_mana = 10
        weight_folder = 'MSEvalue_phase_weight\\'
        cnn_weight_list = [None] * 10
        self.model_list = [None] * 10
        weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
        for file_name in weight_files:
            phase = int(file_name[11])
            cnn_weight_list[phase] = file_name

        for i in range(max_mana):
            merged_model = compile_cnn_model(nn_type='value')
            merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
            self.model_list[i] = merged_model

    def predict(self, feature_list):
        idx = int(feature_list[0][0])
        state_feature = feature_list[1]
        return self.model_list[idx].predict(state_feature)

def mask_hand_feature(hand_feature, mask):

    new_hand = []
    for i in xrange(len(hand_feature)):
        n = np.multiply(hand_feature[i], mask[i])
        new_hand.append(n)

    new_hand = np.array(new_hand)
    return new_hand

def compile_cnn_model(nn_type ='policy', weight_file = None):

    # global_feature_shape = (44,)
    global_feature_shape = (26,)

    global_model = Sequential()
    global_model.add(Dense(64, input_shape=global_feature_shape))
    global_model.add(BatchNormalization())
    global_model.add(LeakyReLU(0.2))

    board_feature_shape = (9, 17, 5)
    # board_feature_shape = (13 * 17 * 5, )

    board_model = Sequential()
    board_model.add(Convolution2D(96, 3, 5, border_mode='same',
                                  input_shape=board_feature_shape))
    board_model.add(LeakyReLU(0.2))
    board_model.add(MaxPooling2D(pool_size=(2, 2)))

    board_model.add(Convolution2D(96, 3, 3, border_mode='same'))
    board_model.add(LeakyReLU(0.2))
    board_model.add(Convolution2D(96, 3, 3, border_mode='same'))
    board_model.add(LeakyReLU(0.2))
    board_model.add(Convolution2D(96, 3, 3, border_mode='same'))
    board_model.add(LeakyReLU(0.2))
    board_model.add(Convolution2D(96, 3, 3, border_mode='same'))
    board_model.add(LeakyReLU(0.2))
    board_model.add(Convolution2D(96, 3, 3, border_mode='same'))
    board_model.add(LeakyReLU(0.2))
    board_model.add(Flatten())

    hand_feature_shape = (19, 23,)

    # hand_model = Sequential()
    # hand_model.add(Dense(128, input_shape=hand_feature_shape))
    # hand_model.add(LeakyReLU(0.2))

    # board_model.add(Dense(128))
    # board_model.add(LeakyReLU(0.2))

    # hand_feature_shape = (9, 23)
    # hand_feature_shape = (18, 23)
    # hand_feature_shape = (9 * 23, )

    hand_model = Sequential()
    hand_model.add(Convolution1D(input_length = 19, nb_filter=96, filter_length=3, input_dim=23, border_mode='same')) #need length here
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    # hand_model.add(MaxPooling1D(pool_length=2))
    # hand_model.add(Dropout(0.25))

    hand_model.add(Convolution1D(nb_filter=96, filter_length=3, border_mode='same'))
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    hand_model.add(Convolution1D(nb_filter=96, filter_length=3, border_mode='same'))
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    hand_model.add(Convolution1D(nb_filter=96, filter_length=3, border_mode='same'))
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    hand_model.add(Convolution1D(nb_filter=96, filter_length=3, border_mode='same'))
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    hand_model.add(Convolution1D(nb_filter=96, filter_length=3, border_mode='same'))
    # hand_model.add(BatchNormalization())
    hand_model.add(LeakyReLU(0.2))
    # hand_model.add(MaxPooling1D(pool_length=2))
    # hand_model.add(Dropout(0.25))
    hand_model.add(Flatten())
    # hand_model.add(Dense(128))
    # hand_model.add(BatchNormalization())
    # hand_model.add(LeakyReLU(0.2))

    # play_feature_shape = (23,)
    #
    # play_model = Sequential()
    # play_model.add(Dense(64, input_shape=play_feature_shape))
    # # play_model.add(BatchNormalization())
    # play_model.add(LeakyReLU(0.2))

    merged_model = Sequential()
    merged_model.add(Merge([global_model, board_model, hand_model], mode='concat', concat_axis=1))
    # merged_model.add(Merge([global_model, board_model, hand_model, play_model], mode='concat', concat_axis=1))
    # merged_model.add(Merge([global_model, board_model, hand_model, future_model], mode='concat', concat_axis=1))
    merged_model.add(LeakyReLU(0.2))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(256))
    merged_model.add(LeakyReLU(0.2))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(128))
    merged_model.add(LeakyReLU(0.2))
    merged_model.add(Dropout(0.5))
    if nn_type == 'policy':
        # merged_model.add(Dense(23))
        merged_model.add(Dense(23))
    # elif nn_type == 'target':
    #     merged_model.add(Dense(17))
    else:
        merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))
    # merged_model.add(Dense(11))
    # merged_model.add(Activation('softmax'))

    if weight_file:
        merged_model.load_weights(weight_file)

    merged_model.summary()
    #
    # sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    # rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)

    merged_model.compile(loss='binary_crossentropy',
    # merged_model.compile(loss='categorical_crossentropy',
    # merged_model.compile(loss='mse',
    #               optimizer='Adam',
                  # optimizer=sgd,
                  # optimizer=rmsprop,
                     optimizer='adadelta',
                     metrics=['accuracy'])

    return merged_model


# def compile_cnn_model(nn_type ='policy', weight_file = None):
#
#     # global_feature_shape = (44,)
#     global_feature_shape = (26,)
#
#     global_model = Sequential()
#     global_model.add(Dense(hidden_size/2, input_shape=global_feature_shape, init='glorot_uniform'))
#     global_model.add(Activation('relu'))
#
#     board_feature_shape = (19, 17, 5)
#     # board_feature_shape = (13 * 17 * 5, )
#
#     board_model = Sequential()
#     board_model.add(Convolution2D(96, 2, 2, border_mode='same',
#                                   input_shape=board_feature_shape))
#     board_model.add(Activation('relu'))
#     board_model.add(MaxPooling2D(pool_size=(2, 2)))
#     board_model.add(Dropout(0.25))
#
#     board_model.add(Convolution2D(192, 2, 2, border_mode='same'))
#     board_model.add(Activation('relu'))
#     board_model.add(MaxPooling2D(pool_size=(2, 2)))
#     board_model.add(Dropout(0.25))
#
#     board_model.add(Activation('relu'))
#     board_model.add(Flatten())
#     board_model.add(Dense(512))
#     board_model.add(Activation('relu'))
#
#     # hand_feature_shape = (9, 23)
#     hand_feature_shape = (9, 46)
#     # hand_feature_shape = (9 * 23, )
#
#     hand_model = Sequential()
#     hand_model.add(Convolution1D(64, 2, border_mode='same',
#                                   input_shape=hand_feature_shape))
#     hand_model.add(Activation('relu'))
#     hand_model.add(MaxPooling1D(pool_length=2))
#     hand_model.add(Dropout(0.25))
#
#     hand_model.add(Convolution1D(128, 2, border_mode='same'))
#     hand_model.add(Activation('relu'))
#     hand_model.add(MaxPooling1D(pool_length=2))
#     hand_model.add(Dropout(0.25))
#
#     hand_model.add(Flatten())
#     hand_model.add(Dense(256))
#     hand_model.add(Activation('relu'))
#
#     play_feature_shape = (46,)
#
#     play_model = Sequential()
#     play_model.add(Dense(hidden_size / 2, input_shape=play_feature_shape, init='glorot_uniform'))
#     play_model.add(Activation('relu'))
#
#     merged_model = Sequential()
#     merged_model.add(Merge([global_model, board_model, hand_model, play_model], mode='concat', concat_axis=1))
#     # merged_model.add(Merge([global_model, board_model, hand_model, future_model], mode='concat', concat_axis=1))
#     merged_model.add(Activation('relu'))
#     merged_model.add(Dropout(0.5))
#     merged_model.add(Dense(512, init='glorot_uniform'))
#     merged_model.add(Activation('relu'))
#     merged_model.add(Dropout(0.5))
#     merged_model.add(Dense(256, init='glorot_uniform'))
#     merged_model.add(Activation('relu'))
#     merged_model.add(Dropout(0.5))
#     if nn_type == 'policy':
#         merged_model.add(Dense(23))
#         # merged_model.add(Dense(40))
#     # elif nn_type == 'target':
#     #     merged_model.add(Dense(17))
#     else:
#         merged_model.add(Dense(1))
#     # merged_model.add(Activation('sigmoid'))
#     # merged_model.add(Activation('sigmoid'))
#     # merged_model.add(Dense(11))
#     merged_model.add(Activation('softmax'))
#
#     if weight_file:
#         merged_model.load_weights(weight_file)
#
#     merged_model.summary()
#     #
#     sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
#     rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)
#
#     # merged_model.compile(loss='binary_crossentropy',
#     merged_model.compile(loss='categorical_crossentropy',
#     # merged_model.compile(loss='mse',
#                   optimizer='adadelta',
#                   # optimizer=sgd,
#                   # optimizer=rmsprop,
#                   metrics=['accuracy'])
#
#     return merged_model

def compile_linear_model(nn_type ='policy', weight_file = None):

    hidden_size = 64
    global_feature_shape = (26,)

    global_model = Sequential()
    global_model.add(Dense(hidden_size, input_shape=global_feature_shape, init='glorot_uniform'))
    global_model.add(Activation('relu'))

    # board_feature_shape = (19, 17, 5)
    board_feature_shape = (19 * 17 * 5,)

    board_model = Sequential()
    board_model.add(Dense(hidden_size, input_shape=board_feature_shape))
    board_model.add(Activation('relu'))

    hand_feature_shape = (9 * 46, )

    hand_model = Sequential()
    hand_model.add(Dense(hidden_size, input_shape=hand_feature_shape, init='glorot_uniform'))
    hand_model.add(Activation('relu'))

    play_feature_shape = (46,)

    play_model = Sequential()
    play_model.add(Dense(hidden_size, input_shape=play_feature_shape, init='glorot_uniform'))
    hand_model.add(Dropout(0.25))
    play_model.add(Activation('relu'))

    merged_model = Sequential()
    merged_model.add(Merge([global_model, board_model, hand_model, play_model], mode='concat', concat_axis=1))
    merged_model.add(Activation('relu'))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(hidden_size/2, init='glorot_uniform'))
    merged_model.add(Activation('relu'))
    merged_model.add(Dropout(0.5))

    if nn_type == 'policy':
        merged_model.add(Dense(23))
    elif nn_type == 'target':
        merged_model.add(Dense(17))
    else:
        merged_model.add(Dense(1))
    merged_model.add(Activation('softmax'))

    if weight_file:
        merged_model.load_weights(weight_file)

    merged_model.summary()
    #
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)

    merged_model.compile(
                        # loss='binary_crossentropy',
                         loss='categorical_crossentropy',
                         optimizer='adadelta',
                         # optimizer=sgd,
                         # optimizer=rmsprop,
                         metrics=['accuracy'])

    #
    # history = model.fit(X_train, Y_train,
    #                     batch_size=batch_size, nb_epoch=nb_epoch,
    #                     verbose=1, validation_data=(X_test, Y_test), callbacks=[weight_save_callback])

    return merged_model

# def compile_fcn_model(nn_type ='policy', weight_file = None):
#
#     # global_feature_shape = (44,)
#     hidden_size = 256
#     global_feature_shape = (26,)
#
#     global_model = Sequential()
#     global_model.add(Dense(hidden_size/4, input_shape=global_feature_shape, init='glorot_uniform'))
#     global_model.add(Activation('relu'))
#
#     # board_feature_shape = (19, 17, 5)
#     board_feature_shape = (19 * 17 * 5, )
#
#     board_model = Sequential()
#     board_model.add(Dense(hidden_size/2, input_shape=board_feature_shape))
#     board_model.add(Activation('relu'))
#
#     hand_feature_shape = (9 * 46, )
#
#     hand_model = Sequential()
#     hand_model.add(Dense(hidden_size/4, input_shape=hand_feature_shape, init='glorot_uniform'))
#     hand_model.add(Activation('relu'))
#
#     play_feature_shape = (46,)
#
#     play_model = Sequential()
#     play_model.add(Dense(hidden_size / 4, input_shape=play_feature_shape, init='glorot_uniform'))
#     hand_model.add(Dropout(0.25))
#     play_model.add(Activation('relu'))
#
#     merged_model = Sequential()
#     merged_model.add(Merge([global_model, board_model, hand_model, play_model], mode='concat', concat_axis=1))
#     merged_model.add(Activation('relu'))
#     merged_model.add(Dropout(0.5))
#     merged_model.add(Dense(hidden_size / 4, init='glorot_uniform'))
#     merged_model.add(Activation('relu'))
#     merged_model.add(Dropout(0.5))
#
#     if nn_type == 'policy':
#         merged_model.add(Dense(23))
#     elif nn_type == 'target':
#         merged_model.add(Dense(17))
#     else:
#         merged_model.add(Dense(1))
#     merged_model.add(Activation('softmax'))
#
#     if weight_file:
#         merged_model.load_weights(weight_file)
#
#     merged_model.summary()
#     #
#     sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
#     rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)
#
#     merged_model.compile(
#                         # loss='binary_crossentropy',
#                          loss='categorical_crossentropy',
#                          optimizer='adadelta',
#                          # optimizer=sgd,
#                          # optimizer=rmsprop,
#                          metrics=['accuracy'])
#
#     #
#     # history = model.fit(X_train, Y_train,
#     #                     batch_size=batch_size, nb_epoch=nb_epoch,
#     #                     verbose=1, validation_data=(X_test, Y_test), callbacks=[weight_save_callback])
#
#     return merged_model

def compile_fcn_model(nn_type ='policy', weight_file = None):

    # global_feature_shape = (44,)
    hidden_size = 256
    global_feature_shape = (26,)

    global_model = Sequential()
    global_model.add(Dense(hidden_size/4, input_shape=global_feature_shape))
    global_model.add(Activation('relu'))

    # board_feature_shape = (19, 17, 5)
    board_feature_shape = (9 * 17 * 5, )

    board_model = Sequential()
    board_model.add(Dense(hidden_size/2, input_shape=board_feature_shape))
    board_model.add(LeakyReLU(0.2))

    # hand_feature_shape = (18 * 23, )
    hand_feature_shape = (9 * 23, )

    hand_model = Sequential()
    hand_model.add(Dense(hidden_size/4, input_shape=hand_feature_shape))
    hand_model.add(LeakyReLU(0.2))

    play_feature_shape = (23,)

    play_model = Sequential()
    play_model.add(Dense(hidden_size / 4, input_shape=play_feature_shape))
    play_model.add(Dropout(0.25))
    play_model.add(LeakyReLU(0.2))

    merged_model = Sequential()
    merged_model.add(Merge([global_model, board_model, hand_model, play_model], mode='concat', concat_axis=1))
    merged_model.add(LeakyReLU(0.2))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(hidden_size / 4))
    merged_model.add(LeakyReLU(0.2))
    merged_model.add(Dropout(0.5))

    if nn_type == 'policy':
        merged_model.add(Dense(23))
    elif nn_type == 'target':
        merged_model.add(Dense(17))
    else:
        merged_model.add(Dense(1))
    # merged_model.add(Activation('softmax'))
    merged_model.add(Activation('sigmoid'))

    if weight_file:
        merged_model.load_weights(weight_file)

    merged_model.summary()
    #
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)

    merged_model.compile(
                        loss='binary_crossentropy',
                         # loss='categorical_crossentropy',
                         # optimizer='adadelta',
                        optimizer='adam',
                        # optimizer=sgd,
                         # optimizer=rmsprop,
                        metrics=['accuracy'])

    return merged_model

def compile_inter_model(nn_type = 'policy', weight_file = None):

    hidden_size = 256

    global_feature_shape = (28,)
    global_model = Sequential()
    global_model.add(Dense(hidden_size, input_shape=global_feature_shape, W_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.activity_l1(0.01)))
    global_model.add(Activation('relu'))

    play_feature_shape = (5 * 23,)
    play_model = Sequential()
    play_model.add(Dense(hidden_size, input_shape=play_feature_shape, W_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.activity_l1(0.01)))
    play_model.add(Activation('relu'))

    effect_feature_shape = (15 * 23,)
    effect_model = Sequential()
    effect_model.add(Dense(hidden_size, input_shape=effect_feature_shape, W_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.activity_l1(0.01)))
    effect_model.add(Activation('relu'))

    merged_model = Sequential()
    merged_model.add(
        Merge([global_model, play_model, effect_model], mode='concat',
              concat_axis=1))

    merged_model.add(Dense(hidden_size / 2, init='glorot_uniform'))
    merged_model.add(Activation('relu'))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(hidden_size / 2, init='glorot_uniform'))
    merged_model.add(Activation('relu'))
    merged_model.add(Dropout(0.5))
    if nn_type == 'policy':
        merged_model.add(Dense(23))
    elif nn_type == 'target':
        merged_model.add(Dense(17))
    else:
        merged_model.add(Dense(1))
    # merged_model.add(Activation('sigmoid'))
    # merged_model.add(Dense(11))
    merged_model.add(Activation('softmax'))

    if weight_file:
        merged_model.load_weights(weight_file)

    merged_model.summary()
    #
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.003, rho=0.9, epsilon=1e-08)

    # merged_model.compile(loss='binary_crossentropy',
    merged_model.compile(loss='categorical_crossentropy',
                         # merged_model.compile(loss='mse',
                         optimizer='adadelta',
                         # optimizer=sgd,
                         # optimizer=rmsprop,
                         metrics=['accuracy'])

    return merged_model

def test_policy_phase(file_list):

    max_mana = 10

    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        value_list = []

        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')
            value_list += hf.get('sf_value')

    own_board = np.array(own_board)
    enemy_board = np.array(enemy_board)
    own_hand = np.array(own_hand)
    enemy_hand = np.array(enemy_hand)
    global_feature = np.array(global_feature)
    target_feature = np.array(target_feature)
    playcard_feature = np.array(playcard_feature)[:, 3:]
    own_deck = np.array(own_deck)
    enemy_deck = np.array(enemy_deck)
    value_list = np.array(value_list)

    playcard_feature = np.clip(playcard_feature, 0, 1)
    result_list = np.array(global_feature)[:, 2]
    global_feature = np.array(global_feature)[:, 3:]

    policy_target = np.concatenate([playcard_feature, target_feature], axis=1)

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=False)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=False)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    # feature_list = [phase_feature, board_feature, handdeck_feature]
    # print(feature_list[2].shape)
    # print(phase_feature[:2, :100])

    total_len = len(phase_feature)
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    total_len_test = 0
    import collections
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr]]
        # phase_target[i] = policy_target[bool_arr]
        phase_target[i] = playcard_feature[bool_arr]
        print(i, len(phase_target[i]))
        total_len_test += len(phase_target[i])

        value_phase = value_list[bool_arr]

        # hash_target = collections.defaultdict(list)
        # for j in phase_target[i]:
        #     ht = j[:-6]
        #     hash_target[np.sum(j[:-6])].append(str(ht))
        #
        # print('phase:', i)
        # for k in hash_target:
        #     print('mana:', k, 'len:', len(hash_target[k]))
        #     vals = sorted(collections.Counter(hash_target[k]).values(), reverse=True)
        #     print(vals[:30])

        move_val_dic = {}
        for j,ht in enumerate(phase_target[i]):
            hash = str(ht)
            value = value_phase[j]
            if hash not in move_val_dic:
                move_val_dic[hash] = [value, 1.0]
            else:
                val, count = move_val_dic[hash]
                count += 1
                val = ((count - 1) * val + value)/count
                move_val_dic[hash] = [val, count]

        count_list = sorted(move_val_dic.values(), reverse=True)
        print('phase:', i)
        for val, count in count_list:
            print('move count:', count, 'value:', val)

        # hash_target = set([str(j) for j in phase_target[i]])
        # vals = sorted(collections.Counter(hash_target).values(), reverse=True)
        # print(len(vals))
        # print("===========================")
        # for i in range(len(vals)):
        #     print(vals[i])

        # hash_target = [str(j) for j in phase_target]
        # print('phase:',j, ', size:', len(set(hash_target)))
        # print('trg:', set(hash_target))


    # print(total_len, total_len_test)
    #
    # for i in range(max_mana):
    #     feature_list, target_list = phase_data[i], phase_target[i]
    #     percent = 0.9 if i != 9 else 0.75
    #     rand_state = random.randint(1, sys.maxint)
    #     print('rand_state:', rand_state)
    #     X_train, Y_train, X_test, Y_test = train_test_split(feature_list, target_list, percent = percent, random_state=rand_state)
    #
    #     # merged_model = compile_fcn_model(type='value')
    #     merged_model = compile_cnn_model(nn_type='policy')
    #     print('start trainning: ' + str(i))
    #     print('X_train:', X_train[0].shape, Y_train.shape)
    #     print('X_test:', X_test[0].shape, Y_test.shape)
    #     weight_save_callback = ModelCheckpoint('policy_phase_weight/cnn_weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
    #                                            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    #     print('end trainning: ' + str(i))
    #
    #     history = merged_model.fit(X_train, Y_train,
    #                                batch_size=batch_size, nb_epoch=nb_epoch,
    #                                verbose=1, validation_data=(X_test, Y_test),
    #                                callbacks=[weight_save_callback])


def train_policy_phase(file_list):

    max_mana = 10
    weight_folder = 'policy_phase_weight\\'

    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        israndom_feature = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')
            israndom_feature += hf.get('is_random')


    f = lambda x: x == 0
    israndom_feature = np.array(israndom_feature)
    nrandom_arr = np.array([f(row) for row in israndom_feature])

    own_board = np.array(own_board)[nrandom_arr]
    enemy_board = np.array(enemy_board)[nrandom_arr]
    own_hand = np.array(own_hand)[nrandom_arr]
    enemy_hand = np.array(enemy_hand)[nrandom_arr]
    global_feature = np.array(global_feature)[nrandom_arr]
    target_feature = np.array(target_feature)[nrandom_arr]
    playcard_feature = np.array(playcard_feature)[:, 3:][nrandom_arr]
    own_deck = np.array(own_deck)[nrandom_arr]
    enemy_deck = np.array(enemy_deck)[nrandom_arr]

    playcard_feature = np.clip(playcard_feature, 0, 1)
    result_list = np.array(global_feature)[:, 2]
    global_feature = np.array(global_feature)[:, 3:]

    policy_target = np.concatenate([playcard_feature, target_feature], axis=1)

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=False)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=False)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    # feature_list = [phase_feature, board_feature, handdeck_feature]
    # print(feature_list[2].shape)
    # print(phase_feature[:2, :100])

    total_len = len(phase_feature)
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    total_len_test = 0
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr]]
        phase_target[i] = policy_target[bool_arr]
        print(i, len(phase_target[i]))
        total_len_test += len(phase_target[i])

    print(total_len, total_len_test)

    # cnn_weight_list = [None] * 10
    # weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
    # for file_name in weight_files:
    #     phase = int(file_name[11])
    #     cnn_weight_list[phase] = file_name
    #
    # print(cnn_weight_list)

    for i in range(8, max_mana):
        feature_list, target_list = phase_data[i], phase_target[i]
        # percent = 0.9 if i != 9 else 0.75
        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, Y_train, X_test, Y_test = train_test_split(feature_list, target_list, percent = percent, random_state=rand_state)

        # merged_model = compile_fcn_model(type='value')
        merged_model = compile_cnn_model(nn_type='policy')
        print('start trainning: ' + str(i))
        print('X_train:', X_train[0].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(weight_folder + 'cnn_weight_1' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        print('end trainning: ' + str(i))
        # merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
        # last_epoch = int(cnn_weight_list[i][13:].split('-')[0])
        # print('last_epoch:', last_epoch)
        last_epoch = 0

        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch - last_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback])

def train_value_phase_mse(file_list):

    max_mana = 10
    weight_folder = 'MSEvalue_phase_weight\\'


    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        value_list = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')
            value_list += hf.get('sf_value')

    own_board = np.array(own_board)
    enemy_board = np.array(enemy_board)
    own_hand = np.array(own_hand)
    enemy_hand = np.array(enemy_hand)
    global_feature = np.array(global_feature)
    target_feature = np.array(target_feature)
    playcard_feature = np.array(playcard_feature)[:, 3:]
    own_deck = np.array(own_deck)
    enemy_deck = np.array(enemy_deck)

    playcard_feature = np.clip(playcard_feature, 0, 1)
    # result_list = np.array(global_feature)[:, 2]
    result_list = np.array(value_list)
    global_feature = np.array(global_feature)[:, 3:]

    policy_target = np.concatenate([playcard_feature, target_feature], axis=1)

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=False)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=False)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    # feature_list = [phase_feature, board_feature, handdeck_feature]
    # print(feature_list[2].shape)
    # print(phase_feature[:2, :100])

    total_len = len(phase_feature)
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    total_len_test = 0
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr], policy_target[bool_arr]]
        phase_target[i] = result_list[bool_arr]
        print(i, len(phase_target[i]))
        total_len_test += len(phase_target[i])

    print(total_len, total_len_test)

    for i in range(max_mana):
        feature_list, result_list = phase_data[i], phase_target[i]
        percent = 0.9 if i != 9 else 0.75
        X_train, Y_train, X_test, Y_test = train_test_split(feature_list, result_list, percent = percent, random_state=42)

        # merged_model = compile_fcn_model(type='value')
        merged_model = compile_cnn_model(nn_type='value')
        print('start trainning: ' + str(i))
        print('X_train:', X_train[0].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(weight_folder + 'cnn_weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

        print('end trainning: ' + str(i))

        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback])


def train_value_phase(file_list):

    max_mana = 10
    weight_folder = 'value_phase_weight\\'


    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')

    own_board = np.array(own_board)
    enemy_board = np.array(enemy_board)
    own_hand = np.array(own_hand)
    enemy_hand = np.array(enemy_hand)
    global_feature = np.array(global_feature)
    target_feature = np.array(target_feature)
    playcard_feature = np.array(playcard_feature)[:, 3:]
    own_deck = np.array(own_deck)
    enemy_deck = np.array(enemy_deck)

    playcard_feature = np.clip(playcard_feature, 0, 1)
    result_list = np.array(global_feature)[:, 2]
    global_feature = np.array(global_feature)[:, 3:]

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=False)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=False)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    # feature_list = [phase_feature, board_feature, handdeck_feature]
    # print(feature_list[2].shape)
    # print(phase_feature[:2, :100])

    total_len = len(phase_feature)
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    total_len_test = 0
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr]]
        phase_target[i] = result_list[bool_arr]
        print(i, len(phase_target[i]))
        total_len_test += len(phase_target[i])

    print(total_len, total_len_test)

    for i in range(max_mana):
        feature_list, result_list = phase_data[i], phase_target[i]
        percent = 0.9 if i != 9 else 0.33
        X_train, Y_train, X_test, Y_test = train_test_split(feature_list, result_list, percent = percent, random_state=42)

        # merged_model = compile_fcn_model(type='value')
        merged_model = compile_cnn_model(nn_type='value')
        print('start trainning: ' + str(i))
        print('X_train:', X_train[0].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(weight_folder + 'cnn_weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

        print('end trainning: ' + str(i))

        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback])

def test_value_phase(file_list):

    max_mana = 10

    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        endturn_feature = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')
            endturn_feature += hf.get('endturn_feature')

    own_board = np.array(own_board)
    enemy_board = np.array(enemy_board)
    own_hand = np.array(own_hand)
    enemy_hand = np.array(enemy_hand)
    global_feature = np.array(global_feature)
    target_feature = np.array(target_feature)
    playcard_feature = np.array(playcard_feature)[:, 3:]
    own_deck = np.array(own_deck)
    enemy_deck = np.array(enemy_deck)

    playcard_feature = np.clip(playcard_feature, 0, 1)
    result_list = np.array(global_feature)[:, 2]
    global_feature = np.array(global_feature)[:, 3:]

    endturn_feature = np.array(endturn_feature)

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=False)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=False)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)

    total_len = len(phase_feature)
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    phase_data_endturn = [None] * max_mana
    f = lambda x: x[0] == i
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr]]
        phase_data_endturn[i] = endturn_feature[bool_arr]
        phase_target[i] = result_list[bool_arr]

    weight_folder = 'value_phase_weight'
    cnn_weight_list = [None] * 10
    weight_files = [f for f in listdir(weight_folder) if isfile(join(weight_folder, f))]
    for file_name in weight_files:
        phase = int(file_name[11])
        cnn_weight_list[phase] = file_name

    print(cnn_weight_list)

    for i in range(max_mana):

        merged_model = compile_cnn_model(nn_type='value')
        merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))

        feature_list, feature_list_endturn, result_list = phase_data[i], phase_data_endturn[i], phase_target[i]

        percent = 0.2 if i != 9 else 0.1
        X_train, Y_train, X_test, Y_test = train_test_split(feature_list, result_list, percent = percent, random_state=13)
        X_train_endturn, Y_train_endturn, X_test, Y_test = train_test_split([feature_list_endturn], result_list, percent = percent, random_state=13)

        win_ind = np.where(Y_train==1)[0]
        loss_ind = np.where(Y_train==0)[0]
        print('win:', len(win_ind), 'loss:', len(loss_ind))
        min_len = min(len(win_ind), len(loss_ind))
        print(min_len)
        win_ind = win_ind[:min_len]
        loss_ind = loss_ind[:min_len]
        np.random.shuffle(win_ind)
        np.random.shuffle(loss_ind)

        cnn_score = 0
        sf_score = 0
        count = 0
        for i in range(min_len):
            win_idx = win_ind[i]
            win_score = merged_model.predict([X_train[0][win_idx:win_idx+1], X_train[1][win_idx:win_idx+1], X_train[2][win_idx:win_idx+1]])
            for j in range(min_len):
                loss_idx = loss_ind[j]
                loss_score = merged_model.predict([X_train[0][loss_idx:loss_idx+1], X_train[1][loss_idx:loss_idx+1], X_train[2][loss_idx:loss_idx+1]])
                if win_score > loss_score:
                    cnn_score += 1
                count += 1
                if count >= 50000:
                    break
            if count >= 50000:
                break

        count = 0
        for i in range(min_len):
            win_idx = win_ind[i]
            win_score = logit_eval.sigmoid_predict(X_train_endturn[0][win_idx])
            for j in range(min_len):
                loss_idx = loss_ind[j]
                loss_score = logit_eval.sigmoid_predict(X_train_endturn[0][loss_idx])
                if win_score > loss_score:
                    sf_score += 1
                count += 1
                if count >= 50000:
                    break
            if count >= 50000:
                break

        print('cnn_score:{}/{}'.format(cnn_score, count))
        print('sf_score:{}/{}'.format(sf_score, count))

def train_policy():
    with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        own_board = np.array(hf.get('own_board'))
        enemy_board = np.array(hf.get('enemy_board'))
        own_hand = np.array(hf.get('own_hand'))
        enemy_hand = np.array(hf.get('enemy_hand'))
        own_future = np.array(hf.get('own_future'))
        enemy_future = np.array(hf.get('enemy_future'))
        global_feature = np.array(hf.get('global'))
        target_feature = np.array(hf.get('target_feature'))
        playcard_feature = np.array(hf.get('playcard_feature'))[:, 3:]

        playcard_feature = np.clip(playcard_feature, 0, 1)
        print(np.max(playcard_feature))

        result_list = np.array(global_feature)[:, 2]
        global_feature = np.array(global_feature)[:, 3:]

        # own_board = np.delete(own_board, (2), axis=1)

        # print('Shape of the array own_board: \n', own_board.shape)
        # print('Shape of the array enemy_board: \n', enemy_board.shape)
        # # print('Shape of the array hero: \n', hero_list.shape)
        # print('Shape of the array own_hand: \n', own_hand.shape)
        # print('Shape of the array enemy_hand: \n', enemy_hand.shape)
        # print('Shape of the array own_future: \n', own_future.shape)
        # print('Shape of the array enemy_future: \n', enemy_future.shape)
        # print('Shape of the array global_tensor: \n', global_feature.shape)
        print('Shape of the array target: \n', target_feature.shape)
        print('Shape of the array result: \n', result_list.shape)
        print('Shape of the array playcard_feature: \n', playcard_feature.shape)

        # test_0 = [one_hot_hero(global_feature[0].reshape(1, -1)),
        #           one_hot_board(own_board[0].reshape(1, -1), enemy_board[0].reshape(1, -1), flatten=False),
        #           one_hot_hand(own_hand[0].reshape(1, -1), enemy_hand[0].reshape(1, -1), flatten=False)]

        # test
        bf_board = one_hot_board_s(own_board[0], enemy_board[0], flatten=False)
        bf_hero = one_hot_hero_s(global_feature[0])
        bf_hand = one_hot_hand_s(own_hand[0], enemy_hand[0], flatten=False)

        data_length = own_board.shape[0]
        global_feature = one_hot_hero(global_feature)
        board_feature = one_hot_board(own_board, enemy_board, flatten=False)
        board_feature_f = one_hot_board(own_board, enemy_board, flatten=True)
        hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=False)
        hand_feature_f = one_hot_hand(own_hand, enemy_hand, flatten=True)
        print('Shape of the array global_tensor: \n', global_feature.shape)

        # test
        print(np.array_equal(global_feature[0], bf_hero))
        print(np.array_equal(board_feature[0], bf_board))
        print(np.array_equal(hand_feature[0], bf_hand))

        # hand_future = one_hot_future(own_future, enemy_future)

        print('board_feature:', board_feature.shape)
        # print('hand_feature:', hand_feature.shape)

        # print(global_feature.shape)
        # print(board_feature.shape)
        # print(hand_feature.shape)

        # feature_list = [global_feature, board_feature, hand_feature, hand_future]
        masked_hand = mask_hand_feature(hand_feature, playcard_feature)
        feature_list_f = [global_feature, board_feature_f, hand_feature_f]
        feature_list = [global_feature, board_feature, hand_feature]
        # feature_list = [global_feature, board_feature, masked_hand]

        print('hand_feature:', hand_feature.shape)
        print('playcard_feature:', playcard_feature.shape)


        # print(test_0)
        # print(feature_list[0])
        # X_train, y_train, X_test, y_test = train_test_split(feature_list, result_list)
        # X_train, Y_train, X_test, Y_test = train_test_split(feature_list, target_feature)

    batch_size = 64
    nb_classes = 2
    nb_epoch = 350
    hidden_size = 1024

    # Y_train = np_utils.to_categorical(y_train, 11)
    # Y_test = np_utils.to_categorical(y_test, 11)

    width = 5706

    # X_train, Y_train, X_test, Y_test = train_test_split(feature_list, playcard_feature, random_state=42)
    X_train, Y_train, X_test, Y_test = train_test_split(feature_list, target_feature, random_state=42)
    # X_train, Y_train, X_test, Y_test = train_test_split(feature_list, result_list, random_state=13)
    # merged_model = compile_cnn_model()
    # merged_model = compile_cnn_model(type='target', weight_file='cnn_target_weight/weights.181-0.05.hdf5')
    # merged_model = compile_cnn_model(type='policy', weight_file='cnn_policy_weight/weights.409-0.08.hdf5')
    merged_model = compile_cnn_model(nn_type='value')
    merged_model.summary()

    # weight_save_callback = ModelCheckpoint('cnn_target_weight_1/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #                                        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # weight_save_callback = ModelCheckpoint('cnn_policy_weight/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #                                        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # history = merged_model.fit(X_train, Y_train,
    #                     batch_size=batch_size, nb_epoch=nb_epoch,
    #                     verbose=1, validation_data=(X_test, Y_test), callbacks=[weight_save_callback])
    # verbose=1, validation_data=(X_test, Y_test))
    #
    # X_train, Y_train, X_test, Y_test = train_test_split(feature_list_f, playcard_feature, random_state=42)
    # # X_train, Y_train, X_test, Y_test = train_test_split(feature_list_f, target_feature, random_state=42)
    # # X_train, Y_train, X_test, Y_test = train_test_split(feature_list_f, result_list, random_state=13)
    #
    # merged_model = compile_fcn_model()
    # history = merged_model.fit(X_train, Y_train,
    #                            batch_size=batch_size, nb_epoch=nb_epoch,
    #                            verbose=1, validation_data=(X_test, Y_test))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    e_score = merged_model.predict_classes(X_test)
    score = merged_model.predict_proba(X_test)
    print(score.shape)
    f = lambda x: 1.0 if x >= 0.30 else 0.0
    c = 0
    min_p = []
    c_f = 0
    for i in range(len(score)):
        n = sum(Y_test[i])
        print('======================================================================================================')
        print('top n = ', n)
        # print(score[i])
        # print(Y_test[i])
        e_s = np.array([f(s) for s in score[i]])
        eind = np.argpartition(score[i], -n)[-n:]
        ind = np.argpartition(Y_test[i], -n)[-n:]
        if np.array_equal(e_s, Y_test[i]):
            c_f += 1
        min_p.append(np.min(score[i][eind]))
        if set(ind) == set(eind):
            c += 1
        else:
            print(ind)
            print(eind)
            print(score[i][eind])
            print(score[i][ind])

    print('final:', float(c) / len(score))
    print('final:', float(c_f) / len(score))
    print('mean:', np.mean(min_p))

def train_action_policy(file_list, model_type = 'cnn'):

    max_mana = 10
    if model_type == 'cnn':
        flatten = False
        weight_folder = 'action_policy_weight\\'
    else:
        flatten = True
        weight_folder = 'action_policy_weight_fcn\\'

    for file_name in file_list:
        own_board = []
        enemy_board = []
        own_hand = []
        enemy_hand = []
        own_deck = []
        enemy_deck = []
        global_feature = []
        target_feature = []
        playcard_feature = []
        israndom_feature = []
        playable_feature = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            own_board += hf.get('own_board')
            enemy_board += hf.get('enemy_board')
            own_hand += hf.get('own_hand')
            enemy_hand += hf.get('enemy_hand')
            global_feature += hf.get('global')
            target_feature += hf.get('target_feature')
            playcard_feature += hf.get('playcard_feature')
            own_deck += hf.get('own_deck')
            enemy_deck += hf.get('enemy_deck')
            israndom_feature += hf.get('is_random')
            playable_feature += hf.get('playable_feature')

    f = lambda x: x == 0
    israndom_feature = np.array(israndom_feature)
    # nrandom_arr = np.array([f(row) for row in israndom_feature])
    nrandom_arr = np.array([True for row in israndom_feature])

    own_board = np.array(own_board)[nrandom_arr]
    enemy_board = np.array(enemy_board)[nrandom_arr]
    own_hand = np.array(own_hand)[nrandom_arr]
    enemy_hand = np.array(enemy_hand)[nrandom_arr]
    global_feature = np.array(global_feature)[nrandom_arr]
    target_feature = np.array(target_feature)[nrandom_arr]
    playcard_feature = np.array(playcard_feature)[:, 3:][nrandom_arr]
    own_deck = np.array(own_deck)[nrandom_arr]
    enemy_deck = np.array(enemy_deck)[nrandom_arr]
    playable_feature = np.array(playable_feature)[nrandom_arr]

    # playcard_feature = np.clip(playcard_feature, 0, 1)
    result_list = np.array(global_feature)[:, 2]
    global_feature = np.array(global_feature)[:, 3:]
    # policy_target = np.concatenate([playcard_feature, target_feature], axis=1)

    # test
    phase_feature = one_hot_hero_phase(global_feature)
    board_feature = one_hot_board(own_board, enemy_board, flatten=flatten)
    hand_feature = one_hot_hand(own_hand, enemy_hand, flatten=flatten)
    deck_feature = one_hot_hand(own_deck, enemy_deck, flatten=flatten)
    # feature_list = [global_feature, board_feature, hand_feature, hand_future]
    # masked_hand = mask_hand_feature(hand_feature, playcard_feature)
    if model_type == 'cnn':
        handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    else:
        handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=1)

    # feature_list = [phase_feature, board_feature, handdeck_feature]
    # print(feature_list[2].shape)
    # print(phase_feature[:2, :100])


    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in phase_feature])
        phase_data[i] = [phase_feature[bool_arr,1:], board_feature[bool_arr], handdeck_feature[bool_arr], playable_feature[bool_arr]]
        phase_target[i] = playcard_feature[bool_arr]

    # np.set_printoptions(linewidth=300)
    for i in xrange(max_mana):
        feature_list, target_list = phase_data[i], phase_target[i]
        X = [[],[],[],[]]
        y = []
        for j in xrange(len(target_list)):
            # if j > 10: break
            prev = np.zeros(23)
            for k in xrange(10):
                # print('play seq', target_list[j][k])
                play_idx = target_list[j][k]
                if target_list[j][k] == 0:
                     break
                X[0].append(feature_list[0][j])
                X[1].append(feature_list[1][j])
                X[2].append(feature_list[2][j])
                X[3].append(np.concatenate([feature_list[3][j][k], np.array(prev)], axis=0))
                new_move = np.zeros(23)
                new_move[play_idx] = 1
                y.append(new_move)
                # print('phase_feature:', feature_list[0][j])
                # print('board_feature:', feature_list[1][j])
                # print('handdeck_feature:', feature_list[2][j])
                # print('prv', prev)
                # print('plb:', feature_list[3][j][k])
                # print('mov', new_move)
                # print('1 move ---------')
                prev += new_move
            # print('==============================')
            # if j > 50:
            #     print(len(y))
            #     break
            # break
        feature_list = [np.array(X[0]), np.array(X[1]), np.array(X[2]), np.array(X[3])]
        target_list = np.array(y)
        # print(feature_list[0].shape, feature_list[1].shape, feature_list[2].shape, feature_list[3].shape)
        # print(target_list.shape)


        # percent = 0.9 if i != 9 else 0.75
        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, Y_train, X_test, Y_test = train_test_split(feature_list, target_list, percent = percent, random_state=rand_state)

        if model_type == 'cnn':
            merged_model = compile_cnn_model(nn_type='policy')
        else:
            merged_model = compile_fcn_model(nn_type='policy')

        print('start trainning: ' + str(i))
        print('X_train:', X_train[0].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(weight_folder + 'cnn_weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')

        # merged_model.load_weights(join(weight_folder, cnn_weight_list[i]))
        # last_epoch = int(cnn_weight_list[i][13:].split('-')[0])
        # print('last_epoch:', last_epoch)

        last_epoch = 0
        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch - last_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback, early_stopping])
        print('end trainning: ' + str(i))

def train_action_policy_inter(file_list):
    max_mana = 10
    weight_folder = 'inter_action_policy_weight\\'

    for file_name in file_list:
        glob = []
        own_card = []
        own_playable = []
        own_played = []
        effect = []
        effect_after = []
        target = []
        # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        with h5py.File(file_name, 'r') as hf:
            glob += hf.get('Global')
            own_card += hf.get('OwnCardFeature')
            own_playable += hf.get('OwnPlayableFeature')
            own_played += hf.get('OwnPlayedFeature')
            effect += hf.get('EffectFeature')
            effect_after += hf.get('CanPlayAfter')
            target += hf.get('Target')


    glob = [heroOneHot([g[1], g[0], g[3], g[2]]) + [g[4], g[5]] for g in glob]
    glob = np.array(glob)

    left = one_hot_hero_phase(glob[:,:-2])
    glob = np.concatenate([left, glob[:,-2:]],axis=1)

    own_card = np.array(own_card)
    own_card = np.clip(own_card, 0, 2)

    own_playable = np.array(own_playable)
    own_played = np.array(own_played)
    effect = np.array(effect)
    effect_after = np.array(effect_after)
    target = np.array(target)

    temp = np.zeros((len(target), 23))
    temp[np.arange(len(target)), target] = 1
    target = temp

    # for i in xrange(100):
    #     print(target[i])
    #     print(own_played[i])

    def onehot(e, max):
        b = np.zeros((max, 23))
        b[e, np.arange(23)] = 1
        return b

    def encode_onehot(ft_list):
        cmax = np.max(ft_list) + 1
        return np.array([onehot(e, cmax) for e in ft_list])

    own_card = encode_onehot(own_card)
    effect = encode_onehot(effect)
    effect_after = encode_onehot(effect_after)

    own_playable = np.expand_dims(own_playable, 1)
    own_played = np.expand_dims(own_played, 1)

    played_feature = np.concatenate([own_card, own_playable, own_played], axis=1)
    effect_feature = np.concatenate([own_card, effect, effect_after], axis=1)

    print(played_feature.shape)
    print(effect_feature.shape)

    played_feature = played_feature.reshape(-1, 5 * 23)
    effect_feature = effect_feature.reshape(-1, 15 * 23)

    # print(hand_feature.shape)
    # print(target.shape)

    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in range(max_mana):
        bool_arr = np.array([f(row) for row in glob])
        phase_data[i] = [glob[bool_arr, 1:], played_feature[bool_arr], effect_feature[bool_arr]]
        phase_target[i] = target[bool_arr]
        #
        # print(phase_data[i][0].shape)
        # print(phase_data[i][1].shape)

        # percent = 0.9 if i != 9 else 0.75
        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, Y_train, X_test, Y_test = train_test_split(phase_data[i], phase_target[i], percent=percent,
                                                            random_state=rand_state)

        merged_model = compile_inter_model()

        print('start training: ' + str(i))
        print('X_train:', X_train[0].shape, X_train[1].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, X_test[1].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(
            weight_folder + 'weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        last_epoch = 0
        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch - last_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback, early_stopping])
        print('end trainning: ' + str(i))

def train_HL_policy(file_list):

    max_mana = 10
    weight_folder = 'HL_policy_weight_fcn_nBN_adad\\'

    glob = []
    own_board = []
    own_card = []
    own_play = []
    target = []
    for file_name in file_list:
        # # with h5py.File('new_board_action_v1.hdf5', 'r') as hf:
        # outFile.create_dataset("globalList",data=np.array(self.global_ft, dtype=np.float32))
        # outFile.create_dataset("boardList",data=np.array(self.board_ft, dtype=np.float32))
        # outFile.create_dataset("handList", data=np.array(self.hand_ft, dtype=np.float32))
        # outFile.create_dataset("playList", data=np.array(self.play_ft, dtype=np.float32))
        # outFile.create_dataset("playList", data=np.array(self.target, dtype=np.float32))
        with h5py.File(file_name, 'r') as hf:
            glob += hf.get('globalList')
            own_board += hf.get('boardList')
            own_card += hf.get('handList')
            own_play += hf.get('playList')
            target += hf.get('target')
            # effect_after += hf.get('CanPlayAfter')
            # target += hf.get('Target')

    glob = np.array(glob)
    own_board = np.array(own_board)
    own_card = np.array(own_card)
    own_play = np.array(own_play)
    target = np.array(target)

    print(glob.shape)
    own_mana = glob[:,:10]
    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    print(own_mana_feature.shape)
    print(glob[:,10:].shape)
    glob = np.concatenate([own_mana_feature, glob[:,10:]], axis=1)
    own_play = own_play[:,0,:]

    print(own_mana_feature.shape)
    print(np.max(own_mana_feature))
    print(own_mana_feature[:100])

    target = target[:, :23]
    #flatten
    # own_card = own_card.reshape(-1, 18 * 23)
    own_play = np.expand_dims(own_play, 1)
    print('own_play:', own_play.shape)
    own_card = np.concatenate([own_card, own_play], axis=1)

    print(glob.shape)
    print(own_board.shape)
    print(own_card.shape)
    print(own_play.shape)
    print(target.shape)

    np.set_printoptions(linewidth=300)

    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in range(max_mana-1)[::-1]:
        # if i > 4: continue
        bool_arr = np.array([f(row) for row in glob])
        # phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr], own_play[bool_arr]]
        phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr]]
        phase_target[i] = target[bool_arr]
        #
        # print(phase_data[i][0].shape)
        # print(phase_data[i][1].shape)

        # percent = 0.9 if i != 9 else 0.75
        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, Y_train, X_test, Y_test = train_test_split(phase_data[i], phase_target[i], percent=percent,
                                                            random_state=rand_state)

        merged_model = compile_cnn_model()

        print('start training: ' + str(i))
        print('X_train:', X_train[0].shape, X_train[1].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, X_test[1].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(
            weight_folder + 'weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        last_epoch = 0
        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch - last_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback, early_stopping])
        print('end trainning: ' + str(i))

def train_HL_policy_fast(file_list):
    max_mana = 10
    weight_folder = 'HL_policy_weight_fcn_nBN\\'

    glob = []
    own_board = []
    own_card = []
    own_play = []
    target = []
    for file_name in file_list:

        with h5py.File(file_name, 'r') as hf:
            glob += hf.get('globalList')
            own_board += hf.get('boardList')
            own_card += hf.get('handList')
            own_play += hf.get('playList')
            target += hf.get('target')

    glob = np.array(glob)
    own_board = np.array(own_board)
    own_card = np.array(own_card)
    own_play = np.array(own_play)
    target = np.array(target)

    print(glob.shape)
    own_mana = glob[:, :10]
    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    print(own_mana_feature.shape)
    print(glob[:, 10:].shape)
    glob = np.concatenate([own_mana_feature, glob[:, 10:]], axis=1)
    # own_play = own_play[:, 0, :]

    print(own_mana_feature.shape)
    print(np.max(own_mana_feature))
    print(own_mana_feature[:100])

    target = target[:, :23]

    print(glob.shape)
    print(own_board.shape)
    print(own_card.shape)
    print(own_play.shape)
    print(target.shape)

    # flatten
    own_board = own_board.reshape(-1, 9 * 17 * 5)
    own_card = own_card.reshape(-1, 9 * 23)

    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in range(max_mana)[::-1]:
        bool_arr = np.array([f(row) for row in glob])
        phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr], own_play[bool_arr]]
        phase_target[i] = target[bool_arr]
        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, Y_train, X_test, Y_test = train_test_split(phase_data[i], phase_target[i], percent=percent,
                                                            random_state=rand_state)

        merged_model = compile_fcn_model()

        print('start training: ' + str(i))
        print('X_train:', X_train[0].shape, X_train[1].shape, Y_train.shape)
        print('X_test:', X_test[0].shape, X_test[1].shape, Y_test.shape)
        weight_save_callback = ModelCheckpoint(
            weight_folder + 'weight_' + str(i) + '.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        last_epoch = 0
        history = merged_model.fit(X_train, Y_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch - last_epoch,
                                   verbose=1, validation_data=(X_test, Y_test),
                                   callbacks=[weight_save_callback, early_stopping])
        print('end trainning: ' + str(i))

def testHLAgent(file_list):

    hlAgent = CNNPhasePolicy()
    glob = []
    own_board = []
    own_card = []
    own_play = []
    target = []
    for file_name in file_list:

        with h5py.File(file_name, 'r') as hf:
            glob += hf.get('globalList')
            own_board += hf.get('boardList')
            own_card += hf.get('handList')
            own_play += hf.get('playList')
            target += hf.get('target')
            # effect_after += hf.get('CanPlayAfter')
            # target += hf.get('Target')

    glob = np.array(glob)
    own_board = np.array(own_board)
    own_card = np.array(own_card)
    own_play = np.array(own_play)
    target = np.array(target)

    print(glob.shape)
    own_mana = glob[:, :10]
    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    print(own_mana_feature.shape)
    print(glob[:, 10:].shape)
    glob = np.concatenate([own_mana_feature, glob[:, 10:]], axis=1)
    own_play = own_play[:, 0, :]

    print(own_mana_feature.shape)
    print(np.max(own_mana_feature))
    print(own_mana_feature[:100])

    target = target[:, :23]

    # flatten
    own_board = own_board.reshape(-1, 9 * 17 * 5)
    own_card = own_card.reshape(-1, 18 * 23)

    print(glob.shape)
    print(own_board.shape)
    print(own_card.shape)
    print(own_play.shape)
    print(target.shape)

    np.set_printoptions(linewidth=300)

    max_mana = 10
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in [3, 4, 5, 6, 7, 9]:
    # for i in range(max_mana)[::-1]:
    #     if i !=7: continue
        bool_arr = np.array([f(row) for row in glob])
        phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr], own_play[bool_arr]]
        phase_target[i] = target[bool_arr]

        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, y_train, X_test, y_test = train_test_split(phase_data[i], phase_target[i], percent=0.9,
                                                            random_state=rand_state)

        prediction = hlAgent.predict_policy([[i], X_test])
        print('prediction shape:', prediction.shape)

        correct = 0
        for k in range(len(y_test)):
            test = y_test[k]
            pred = prediction[k]
            n_played = np.sum(test)
            # print(n_played)
            # print(pred)
            pred = pred.argsort()[int(-n_played):][::-1]
            test = test.argsort()[int(-n_played):][::-1]
            if set(test) == set(pred):
                correct += 1

        print('correct:', correct)
        print('correct rate:', float(correct)/len(y_test))

        # threshold = np.arange(0.1, 0.9, 0.1)
        # from sklearn.metrics import matthews_corrcoef
        #
        # acc = []
        # accuracies = []
        # best_threshold = np.zeros(prediction.shape[1])
        # for i in range(prediction.shape[1]):
        #     y_prob = np.array(prediction[:, i])
        #     for j in threshold:
        #         y_pred = [1 if prob >= j else 0 for prob in y_prob]
        #         acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        #     acc = np.array(acc)
        #     index = np.where(acc == acc.max())
        #     accuracies.append(acc.max())
        #     best_threshold[i] = threshold[index[0][0]]
        #     acc = []
        #
        # print("best thresholds", best_threshold)
        #
        # y_pred = np.array(
        #     [[1 if prediction[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
        #
        # print("-" * 40)
        # print("Matthews Correlation Coefficient")
        # print("Class wise accuracies")
        # print(accuracies)
        #
        # print("other statistics\n")
        # total_correctly_predicted = len([i for i in range(len(y_test)) if np.array_equal(y_test[i], y_pred[i])])
        # print("Fully correct output")
        # print(total_correctly_predicted)
        # print(y_test.shape[0])
        # print(float(total_correctly_predicted) / y_test.shape[0])

        # for j in xrange(10):
        #     answer = Y_test[j]
        #     pred =  prediction[j]
        #     print('prediction:', prediction[j])
        #     string = [card_list[k] for k in xrange(len(card_list))  if pred[k] == 1]
        #     print(string)
        #     print('answer:', answer)
        #     string = [card_list[k] for k in xrange(len(card_list))  if answer[k] == 1]
        #     print(string)

def testCNNHLAgent(file_list):

    hlAgent = CNNPhasePolicy()
    max_mana = 10
    weight_folder = 'HL_policy_weight_fcn_nBN_adad\\'

    glob = []
    own_board = []
    own_card = []
    own_play = []
    target = []
    for file_name in file_list:
        with h5py.File(file_name, 'r') as hf:
            glob += hf.get('globalList')
            own_board += hf.get('boardList')
            own_card += hf.get('handList')
            own_play += hf.get('playList')
            target += hf.get('target')

    glob = np.array(glob)
    own_board = np.array(own_board)
    own_card = np.array(own_card)
    own_play = np.array(own_play)
    target = np.array(target)

    print(glob.shape)
    own_mana = glob[:, :10]
    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    print(own_mana_feature.shape)
    print(glob[:, 10:].shape)
    glob = np.concatenate([own_mana_feature, glob[:, 10:]], axis=1)
    own_play = own_play[:, 0, :]

    print(own_mana_feature.shape)
    print(np.max(own_mana_feature))
    print(own_mana_feature[:100])

    target = target[:, :23]
    # flatten
    # own_card = own_card.reshape(-1, 18 * 23)
    own_play = np.expand_dims(own_play, 1)
    print('own_play:', own_play.shape)
    own_card = np.concatenate([own_card, own_play], axis=1)

    print(glob.shape)
    print(own_board.shape)
    print(own_card.shape)
    print(own_play.shape)
    print(target.shape)

    np.set_printoptions(linewidth=300)

    max_mana = 10
    phase_data = [None] * max_mana
    phase_target = [None] * max_mana
    f = lambda x: x[0] == i
    for i in [3, 4, 5, 6, 7, 8]:
    # for i in range(max_mana)[::-1]:
    #     if i !=7: continue
        bool_arr = np.array([f(row) for row in glob])
        # phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr], own_play[bool_arr]]
        phase_data[i] = [glob[bool_arr, 1:], own_board[bool_arr], own_card[bool_arr]]
        phase_target[i] = target[bool_arr]

        percent = 0.9
        rand_state = random.randint(1, sys.maxint)
        print('rand_state:', rand_state)
        X_train, y_train, X_test, y_test = train_test_split(phase_data[i], phase_target[i], percent=percent,
                                                            random_state=rand_state)

        prediction = hlAgent.predict_policy([[i], X_test])
        print('prediction shape:', prediction.shape)

        correct = 0
        for k in range(len(y_test)):
            test = y_test[k]
            pred = prediction[k]
            n_played = np.sum(test)
            # print(n_played)
            # print(pred)
            pred = pred.argsort()[int(-n_played):][::-1]
            test = test.argsort()[int(-n_played):][::-1]
            if set(test) == set(pred):
                correct += 1

        print('correct:', correct)
        print('correct rate:', float(correct)/len(y_test))

        # threshold = np.arange(0.1, 0.9, 0.1)
        # from sklearn.metrics import matthews_corrcoef
        #
        # acc = []
        # accuracies = []
        # best_threshold = np.zeros(prediction.shape[1])
        # for i in range(prediction.shape[1]):
        #     y_prob = np.array(prediction[:, i])
        #     for j in threshold:
        #         y_pred = [1 if prob >= j else 0 for prob in y_prob]
        #         acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        #     acc = np.array(acc)
        #     index = np.where(acc == acc.max())
        #     accuracies.append(acc.max())
        #     best_threshold[i] = threshold[index[0][0]]
        #     acc = []
        #
        # print("best thresholds", best_threshold)
        #
        # y_pred = np.array(
        #     [[1 if prediction[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
        #
        # print("-" * 40)
        # print("Matthews Correlation Coefficient")
        # print("Class wise accuracies")
        # print(accuracies)
        #
        # print("other statistics\n")
        # total_correctly_predicted = len([i for i in range(len(y_test)) if np.array_equal(y_test[i], y_pred[i])])
        # print("Fully correct output")
        # print(total_correctly_predicted)
        # print(y_test.shape[0])
        # print(float(total_correctly_predicted) / y_test.shape[0])

        # for j in xrange(10):
        #     answer = Y_test[j]
        #     pred =  prediction[j]
        #     print('prediction:', prediction[j])
        #     string = [card_list[k] for k in xrange(len(card_list))  if pred[k] == 1]
        #     print(string)
        #     print('answer:', answer)
        #     string = [card_list[k] for k in xrange(len(card_list))  if answer[k] == 1]
        #     print(string)


if __name__ == '__main__':

    file_folder = ''
    # file_list = [file_folder + 'svs_result_1.1.txt_ft.hdf5', file_folder + 'svs_result_2.1.txt_ft.hdf5', file_folder + 'svs_result_3.1.txt_ft.hdf5']
    # file_list = [file_folder + 'svs_result_1.1.txtnorm.hdf5', file_folder + 'svs_result_2.1.txtnorm.hdf5', file_folder + 'svs_result_3.1.txtnorm.hdf5']
    # file_list = [file_folder + 'svs_result_1.txtinter.hdf5', file_folder + 'svs_result_2.txtinter.hdf5', file_folder + 'svs_result_3.txtinter.hdf5']
    # file_list = [file_folder + 'svs_result_4.txtinter.hdf5']
    # file_list = [file_folder + 'svs_result_2.txt.2.txt.3.txtHL.hdf5']
    file_list = [file_folder + 'svs_result_3.txt.2.txt.3.txtHL.hdf5', file_folder + 'svs_result_1.txt.2.txt.3.txtHL.hdf5', file_folder + 'svs_result_2.txt.2.txt.3.txtHL.hdf5']
    train_HL_policy_fast(file_list)
    # cnnFast = CNNPhaseActionPolicyFast()
    # cnnFast.run_server()
