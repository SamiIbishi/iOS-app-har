#!/usr/bin/env python

# Keras
import keras.backend as k
from keras import optimizers
from keras import losses, regularizers
from keras import initializers
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import plot_model, to_categorical
from keras.utils.vis_utils import model_to_dot


def create_model(input_shape, 
                num_classes=6,
                num_mem_units=90, 
                num_hidden_units=96, 
                dropout=False,
                batchnormalization=True,
                init='glorot_uniform',
                model_name='Activity Engine'):

    # Start sequentially defining model 
    model = Sequential(name=model_name)
    model.author = 'Goalplay'
    model.short_description = 'Activity Recognition with goalplayer training'

    # Reshape input
    model.add(Reshape(input_shape, input_shape=(input_shape[0] * input_shape[1],)))

    ## Temporal feature extraction - multiple LSTM layer
    model.add(LSTM(num_mem_units,
                name = 'LSTM_Layer1',
                return_sequences=True,
                kernel_initializer=init,
                input_shape=input_shape))
    model.add(LSTM(num_mem_units,
                name = 'LSTM_Layer2',
                return_sequences=True,
                kernel_initializer=init))
    model.add(LSTM(num_mem_units,
                name = 'LSTM_Layer3',
                return_sequences=False,
                kernel_initializer=init))

    ## Classification - fully-connected network
    # Dense layer 1
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(num_hidden_units, 
                    name = 'Feature_Layer1',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    if batchnormalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Dense layer 2
    if dropout:
        model.add(Dropout(0.2))
    model.add(Dense(num_hidden_units,
                    name = 'Feature_Layer2',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    if batchnormalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    ## Output layer
    model.add(Dense(num_classes, activation='softmax', name = 'Output_Layer'))

    return model
