from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random
import numpy as np
import h5py
from simple_dqn.encoder import train_test_split
from keras.layers import Merge
from keras.optimizers import RMSprop
from data_encoder import load_encoded_data
from data_encoder import encode_et_feature
import cPickle as pickle

global_data_dim = 16
hand_data_dim = 46

class LogitEval:

    def __init__(self, weight_file=None, batch_size=256):
        self.exp = []
        self.target = []
        self.state_str = []
        self.model = self.load_model(weight_file)
        self.batch_size = batch_size
        self.discount = 1
        # self.log_file = open('learing_log.txt', 'w')
        # self.log_file.write('test!\n' + 'test\n')
        # self.log_file = h5py.File('learing_log.hdf5')

    def load_model(self, weight_file):
        reg = l1l2(l1=0.01, l2=0.01)
        rmsprop = RMSprop(lr=0.0002)
        global_model = Sequential()
        global_model.add(Dense(1, activation='relu', input_dim=global_data_dim, W_regularizer=reg))
        global_model.add(Dense(128, activation='relu'))

        hand_model = Sequential()
        hand_model.add(Dense(1, activation='relu', input_dim=hand_data_dim, W_regularizer=reg))
        hand_model.add(Dense(128, activation='relu'))

        model = Sequential()
        model.add(Merge([global_model, hand_model], mode='concat', concat_axis=1))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=rmsprop, loss='mse', metrics=['mse'])

        if weight_file:
            model.load_weights(weight_file)
        return model

    def save_one_episode(self, match, result, state_str_list):
        self.exp.append(match)
        self.target.append(result)
        self.state_str.append(state_str_list)

    def save_exp(self, match_list, result_list, state_str_list):
        self.exp += match_list
        self.target += result_list
        self.state_str += state_str_list

    def get_data_from_exp(self, size):

        S = [np.zeros(shape = (size, 16)), np.zeros(shape = (size, 46))]
        NS = [np.zeros(shape = (size, 16)), np.zeros(shape = (size, 46))]
        T = np.zeros(shape = (size, 1))
        R = np.zeros(shape = (size, 1))
        # S_str = []
        # NS_str = []

        for i in xrange(size):
            episode = random.randint(0, len(self.exp)-1)
            num_frames = len(self.exp[episode])
            frame = random.randint(0, num_frames - 1)
            S[0][i] = self.exp[episode][frame][0]
            S[1][i] = self.exp[episode][frame][1]
            # S_str.append(self.state_str[episode][frame])
            if frame + 2 <= num_frames - 1:
                NS[0][i] = self.exp[episode][frame + 2][0]
                NS[1][i] = self.exp[episode][frame + 2][1]
                # NS_str.append(self.state_str[episode][frame + 2])
            # else:
                # NS_str.append('game end!')
            T[i] = 1 if frame + 2 > num_frames - 1 else 0
            R[i] = self.target[episode][frame] if frame + 2 > num_frames - 1 else 0

        # now = self.model.predict(S)
        future = (1 - T) * self.model.predict(NS)

        X_train = S
        y_train = R + self.discount * future

        # return X_train, y_train, S_str, now, NS_str
        return X_train, y_train

    def train_model(self, nb_epoch=100, X_train=None, y_train=None, X_test=None, y_test=None, save_weight=True):
        #random pick N samples

        if not X_train and not y_train:
            if len(self.target) < 20:
                print 'len exp:', len(self.target), '<', 20
                return [-1]
            # X_train, y_train, S_str, now, NS_str = self.get_data_from_exp(self.batch_size)
            X_train, y_train = self.get_data_from_exp(self.batch_size)

        weight_save_callback = ModelCheckpoint('simple_dqn/endturn_ft_weight/weights_new.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

        callbacks = [early_stopping]
        # callbacks = []
        if save_weight:
            callbacks.append(weight_save_callback)
        if X_test != None and y_test != None:
            self.model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
        else:
            self.model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_split=0.1, verbose=1, callbacks=callbacks)

        return [1]
        # return 1, S_str, now.tolist(), NS_str, y_train.tolist()

    def advance(self, X_list, verbose=0):
        res = self.model.predict([np.array(ft) for ft in X_list])
        if verbose == 1:
            print 'res:', res
        return res.argmax()

def train_model(x_train, y_train, x_test, y_test):

    logit_eval = LogitEval()
    logit_eval.save_exp(x_train, y_train)
    logit_eval.batch_size = len(y_train)
    logit_eval.train_model(nb_epoch=300, X_test=x_test, y_test=y_test)
    return logit_eval

def load_exp(file_list):

    exp = []
    result = []

    for file_name in file_list:
        endturn_feature = []
        global_feature = []
        with h5py.File(file_name, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())

            own_hand = np.array(hf.get('own_hand'))
            enemy_hand = np.array(hf.get('enemy_hand'))
            endturn_feature += hf.get('endturn_feature')
            global_feature += hf.get('global')

        own_hand = np.array(own_hand)
        enemy_hand = np.array(enemy_hand)
        hand_feature = np.concatenate([own_hand, enemy_hand], axis=1)
        endturn_feature = np.array(endturn_feature)
        result_list = np.array(global_feature)[:, 2]
        match_no = np.array(global_feature)[:, :3]
        global_feature = np.array(global_feature)[:, 3:]

        vis_match = set()

        for i in xrange(len(match_no)):
            if match_no[i][0] not in vis_match:
                if len(vis_match) > 0:
                    assert len(match) == len(m_result)
                    exp.append(match)
                    result.append(m_result)
                vis_match.add(match_no[i][0])
                match = []
                m_result = []
            match.append([endturn_feature[i], hand_feature[i]])
            m_result.append(result_list[i])
        exp.append(match)
        result.append(m_result)

    return exp, result


def load_data(file_list):

    for file_name in file_list:
        endturn_feature = []
        global_feature = []
        with h5py.File(file_name, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())

            own_hand = np.array(hf.get('own_hand'))
            enemy_hand = np.array(hf.get('enemy_hand'))
            endturn_feature += hf.get('endturn_feature')
            global_feature += hf.get('global')

        own_hand = np.array(own_hand)
        enemy_hand = np.array(enemy_hand)
        hand_feature = np.concatenate([own_hand, enemy_hand], axis=1)
        endturn_feature = np.array(endturn_feature)
        result_list = np.array(global_feature)[:, 2]
        global_feature = np.array(global_feature)[:, 3:]

        endturn_feature[:,0] = -endturn_feature[:,0]
        print 'Shape of endturn_feature:', endturn_feature.shape
        print 'Shape of result_list:', result_list.shape

        X_train, y_train, X_test, y_test = train_test_split([endturn_feature, hand_feature], result_list, percent=0.2)

        print 'X_train:', X_train[0].shape
        print 'y_train:', y_train.shape

        print 'X_test:', X_test[0].shape
        print 'y_test:', y_test.shape

        return X_train, y_train, X_test, y_test

def sigmoid_predict(feature):
    value = sum(feature)
    # max_mana = feature[2]
    # if max_mana/5 < 4:
    #     k = -0.02
    # elif max_mana/5 < 7:
    #     k = -0.015
    # else:
    #     k = -0.01
    k = -0.015

    squashed = 1.0 / (1.0 + np.exp(k * value))
    # if squashed >= 0.5:
    #     return 1
    # else:
    #     return 0
    return squashed

def train_supervised():
    x_train, y_train, x_test, y_test = load_data(['sf_vs_sf_result.txt_ft.hdf5', 'sf_vs_sf_result_1.txt_ft.hdf5', 'sf_vs_sf_result_2.txt_ft.hdf5'])

    for i in xrange(len(x_train)):
        x_train[i] = x_train[i].tolist()
    train_model(x_train, y_train.tolist(), x_test, y_test)

def train_supervised_reg():
    x_train, y_train, x_test, y_test = gen_target()

    for i in xrange(len(x_train)):
        x_train[i] = x_train[i].tolist()
    train_model(x_train, y_train, x_test, np.array(y_test))

def test_predict():
    logit_eval = LogitEval(weight_file='endturn_ft_weight/weights_new.hdf5')
    f1 = np.zeros((1, 16))
    f2 = np.zeros(46)
    f = [f1, f2]
    print logit_eval.model.predict(f)

def gen_target():
    x_train, y_train, x_test, y_test = load_data(['sf_vs_sf_result.txt_ft.hdf5', 'sf_vs_sf_result_1.txt_ft.hdf5', 'sf_vs_sf_result_2.txt_ft.hdf5'])
    y_train = [sigmoid_predict(sample) for sample in x_train[0]]
    y_test = [sigmoid_predict(sample) for sample in x_test[0]]
    return x_train, y_train, x_test, y_test

def train_supervised_regression():

    exp, target = load_exp(['sf_vs_sf_result.txt_ft.hdf5', 'sf_vs_sf_result_1.txt_ft.hdf5', 'sf_vs_sf_result_2.txt_ft.hdf5'])
    X_train = [np.array([x[0] for m in exp for x in m]), np.array([x[1] for m in exp for x in m])]
    y_train = np.array([sigmoid_predict(sample) for sample in X_train[0]])
    X_train, y_train, X_test, y_test = train_test_split(X_train, y_train, percent=0.8)
    logit_eval = LogitEval()
    logit_eval.train_model(nb_epoch=100, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def read_batch_samples():
    file_path = 'C:/Users/bugdx123/Documents/batch_sample.txt'
    lines = open(file_path, 'r').readlines()
    for l in lines:
        # s = l.split(',')
        # for a in s:
        #     print s
        s = eval(l)
        length = len(s[1])
        for i in range(length):
            print s[1][i]
            print s[2][i]
            print s[3][i]
            print s[4][i]

if __name__ == '__main__':
    # train_supervised_regression()
    # l = LogitEval()
    read_batch_samples()