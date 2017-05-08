import random
import numpy
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Dropout
from keras.optimizers import RMSprop, Adadelta
from keras import backend as K
from theano import printing
from theano.gradient import disconnected_grad

memory_size = 1000

class Agent:
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, mbsz=64, discount=0.9, memory=memory_size,
                 save_name='basic', save_freq=10):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.mbsz = mbsz
        self.discount = discount
        self.memory = memory
        self.save_name = 'weight'
        self.states = []
        # self.actions = []
        self.rewards = []
        self.experience = []
        self.i = 1
        self.save_freq = save_freq
        self.build_functions()

    def build_model(self):
        S = Input(shape=self.state_size)
        # h = Dropout(0.1)(S)
        h = Dense(1024, activation='relu')(S)
        # h = Dropout(0.25)(h)
        h = Dense(1024, activation='relu')(h)
        # h = Dropout(0.25)(h)
        h = Dense(1024, activation='relu')(h)
        # h = Dropout(0.25)(h)
        h = Dense(1024, activation='relu')(h)
        # h = Dropout(0.25)(h)
        h = Dense(512, activation='relu')(h)
        # h = Dropout(0.25)(h)
        # h = Convolution2D(16, 8, 8, subsample=(4, 4),
        #     border_mode='same', activation='relu')(S)
        # h = Convolution2D(32, 4, 4, subsample=(2, 2),
        #     border_mode='same', activation='relu')(h)
        # h = Flatten()(h)
        h = Dense(256, activation='relu')(h)
        V = Dense(1)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"


    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        # A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        self.build_model()
        self.model.summary()
        # self.value_fn = K.function([S, K.learning_phase()], self.model(S))
        self.value_fn = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = disconnected_grad(self.model(NS))
        # future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        future_value = (1-T) * VNS
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        # cost = ((VS[:, A] - target)**2).mean()
        cost = ((VS - target)**2).mean()
        # opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        # self.train_fn = K.function([S, NS, A, R, T, K.learning_phase()], cost, updates=updates)
        # self.train_fn = K.function([S, NS, A, R, T], cost, updates=updates)
        self.train_fn = K.function([S, NS, R, T], cost, updates=updates)

    def new_episode(self):
        self.states.append([])
        # self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        # self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def end_episode(self):
        pass

    def act(self, state, next_states):
        # values = self.value_fn([state[None, :]])
        values = []
        for next_state in next_states:
            values.append(self.value_fn([next_state]))
        values = numpy.array(values)
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(len(next_states))
        else:
            action = values.argmax()
        # self.actions[-1].append(action)
        # return action, values
        self.states[-1].append([state, next_states[action]])
        print 'store a trasision'
        return action

    def store_transition(self, S, NS, R):
        self.states[-1].append([S, NS])
        print 'store a trasision'
        self.rewards[-1].append(R)
        print 'store a reward'
        return self.iterate()

    def update_transition(self, NS):
        self.states[-1][-1][-1] = NS

    def observe(self, reward):
        self.rewards[-1].append(reward)
        print 'store a reward'
        return self.iterate()

    def iterate(self):
        N = len(self.states)
        S = numpy.zeros((self.mbsz,) + self.state_size)
        NS = numpy.zeros((self.mbsz,) + self.state_size)
        # A = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
        R = numpy.zeros((self.mbsz, 1), dtype=numpy.float32)
        T = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
        for i in xrange(self.mbsz):
            episode = random.randint(max(0, N-memory_size), N-1)
            num_frames = len(self.states[episode])
            frame = random.randint(0, num_frames-1)
            S[i] = self.states[episode][frame][0]
            NS[i] = self.states[episode][frame][1]
            T[i] = 1 if frame == num_frames - 1 else 0
            # if frame < num_frames - 1:
            #     NS[i] = self.states[episode][frame+1]
            # A[i] = self.actions[episode][frame]
            R[i] = self.rewards[episode][frame]
        # cost = self.train_fn([S, NS, A, R, T])
        cost = self.train_fn([S, NS, R, T])

        return cost



















