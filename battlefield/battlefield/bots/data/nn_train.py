import os
import gzip
import cPickle as pickle
import json
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, save_model, Model
from keras.layers import Dense, LSTM, Masking, GRU, CuDNNGRU, CuDNNLSTM, Input, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(PARENT_DIR)

from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction, possible_hidden_states, starting_hidden_states
VOTE_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vote_data.npz'))
PROPOSE_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'propose_data.npz'))


def create_vote_model():
    model = Sequential([
        Masking(input_shape=(None, 79)),
        Dense(128, activation='relu'),
        LSTM(64),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
    return model


def create_propose_model():
    inputs = Input(shape=(None, 79))
    x = Dense(192, activation='relu')(inputs)
    x = LSTM(192, return_sequences=True)(x)
    x = LSTM(192)(x)
    x = Dense(128, activation='relu')(x)
    outs = [
        Dense(1, activation='sigmoid', name="propose_{}".format(i))(x)
        for i in range(5)
    ]

    model = Model(inputs=inputs, outputs=outs)
    model.compile(optimizer='adam', loss={
        'propose_{}'.format(i): 'binary_crossentropy'
        for i in range(5)
    }, metrics=[ 'accuracy' ])
    return model


def create_kl_propose_model():
    model = Sequential([
        Masking(input_shape=(None, 79)),
        Dense(256, activation='relu'), # input_shape=(None, 79)),
        Dropout(0.4),
        LSTM(256, return_sequences=False),
        Dropout(0.4),
        Dense(5, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['kullback_leibler_divergence'])
    return model


def train():
    print "Loading data"
    data = np.load(PROPOSE_FILENAME)
    X = data['arr_0']
    y = data['arr_1']
    print "Shuffling data"
    X, y = shuffle(X, y)
    y = y / np.sum(y, axis=1).reshape((len(X), 1))
    # y = y.T
    # actual_y = { 'propose_{}'.format(i): y[i] for i in range(5) }
    model = create_kl_propose_model()
    model.summary()
    model.fit(x=X, y=y, batch_size=1024, epochs=100, validation_split=0.1, callbacks=[ModelCheckpoint('propose_model-{epoch:02d}-{val_loss:.4f}.h5', save_best_only=True)])


if __name__ == "__main__":
    train()
