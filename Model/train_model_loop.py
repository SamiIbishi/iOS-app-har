#!/usr/bin/env python

#########################################################
########### Imports #####################################
#########################################################

# General
import os
import time

# Python libraries
import numpy as np, collections

# Tensorflow
import tensorflow as tf

# Keras
import keras.backend as k
from keras import optimizers
from keras import losses, regularizers
from keras import initializers
from keras.utils import plot_model, to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

# CoreML 
import coremltools

# Scikit-learn
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, GridSearchCV

# Seaborn
import seaborn as sn

# Pandas
import pandas as pd

# Locale methods and properties
from model import create_compiled_model
from callbacks import get_callbacks 
from auxilary_functions import time_stamp
from load_data import load_and_store_data, load_stored_data 
from data_augmentation import normalize_data, standardize_data
from plot_trainings_history import plot_loss_acc
from plot_confusion_matrix import print_confusion_matrix 
from plot_data_distribution import plot_data_distribution
from plot_pose_sequence import plot_trainings_data, plot_test_data

#########################################################
########### Load Data ###################################
#########################################################

# Output classes
LABELS = [    
    "STAND_BY",
    "LEFT_DIVE",
    "RIGHT_DIVE",
    "HIGH_CATCH",
]

# Path to stored keypoints
data_path = "./stored_data/4_classes/"

# Framrates/Poses-per-sequence: '5' / '10' / '15' / '30' / '45' / '60' / '75' / '90'
fps_array = ['5', '10' , '15' , '30' , '45' , '60' , '75' , '90']

# Get time stamp 
time_stamp = time_stamp()

for fps in fps_array:

    # Load data from .npy files 
    # X = [samples, timesteps, features]
    # y = [samples, labels]
    X_raw, y_raw = load_stored_data(dir_path=data_path, num_poses=fps)

    # Augment data
    X, y, _, _, _ = standardize_data(X_raw, y_raw)
    #X, y = normalize_data(X_raw, y_raw)

    # Split dataset into train and test set (and shuffle them)
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y, test_size = 0.1)

    # Classes
    num_classes = len(LABELS)

    # Get one hot vector from labels
    y_train = to_categorical(y_train_temp, num_classes=num_classes)
    y_test = to_categorical(y_test_temp, num_classes=num_classes)

    # Plot some debugging information about the data
    print("_______________________________________")
    print("---------------------------------------")
    print("Bsic information regardin the data:")
    print("_______________________________________")
    print("Trainings data...")
    print("... input shape=" + str(X_train.shape))
    print("... target shape=" + str(y_train.shape))
    print("")
    print("Test data...")
    print("... input shape=" + str(X_test.shape))
    print("... target shape=" + str(y_test.shape))
    print("")
    print("---------------------------------------")
    print("_______________________________________")

    #########################################################
    ########### Constants ###################################
    #########################################################

    # Input Data
    n_timesteps = X_train.shape[1] # n-timesteps per series per series
    n_features = X_train.shape[2] # n input parameters per timestep

    # LSTM Neural Network's internal structure
    input_shape = (n_timesteps, n_features)
    num_mem_units = 256
    num_hidden_units = 256

    # Training - Hyperparameter  
    learning_rate = [0.001, 0.0005, 0.0001]
    init = ['glorot_uniform', 'uniform']
    optim = [optimizers.RMSprop(lr=learning_rate[0], decay=1), 
                optimizers.Adam(lr=learning_rate[0], decay=0.5),
                optimizers.Nadam(lr=learning_rate[0])]
    num_epochs = 128
    batch_size = 64
    dropout = [True, False]

    # Ether class weight or sample weights to overcome the unbalanced data (NOT both, select one)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_temp), y_train_temp)
    #sample_weights = class_weight.compute_sample_weight('balanced', y)

    #########################################################
    ########### Callbacks ###################################
    #########################################################

    # Tensorboard
    logdir = './training_history/logs/model_selection_3/'

    # Create directory which will contain trained models 
    model_directory = './training_history/saved_models/model_selection_3/'

    # Generate callback list 
    callbacks = get_callbacks(model_directory, logdir, time_stamp, fps)

    #########################################################
    ########### DL Model ####################################
    #########################################################

    # Get compiled model 
    model = create_compiled_model(input_shape=input_shape,
                                    num_classes=num_classes,
                                    num_mem_units=num_mem_units, 
                                    num_hidden_units=num_hidden_units,  
                                    init=init[0],
                                    dropout=dropout[0],
                                    optimizer=optim[0],
                                    loss=losses.categorical_crossentropy,)

    # Print model details/summary
    model.summary()

    # Reshape model fo CoreML model 
    reshaped_X_train = X_train.reshape(X_train.shape[0],-1)
    reshaped_X_test = X_test.reshape(X_test.shape[0],-1)

    # Train model on trainings data 
    history = model.fit(reshaped_X_train, 
                        y_train, 
                        verbose=2, 
                        shuffle=True, 
                        epochs=num_epochs, 
                        callbacks=callbacks,
                        validation_split=0.1, 
                        batch_size=batch_size, 
                        class_weight=class_weights)

    # Plot evaluation results
    print('\nUsed metrics: ' + str(model.metrics_names) + '\n')
    print('Evaluation on trainings data: ' + str(model.evaluate(reshaped_X_train, y_train)) + '\n')
    print('Evaluation on test data: ' + str(model.evaluate(reshaped_X_test, y_test)) + '\n')

    str_test_acc = str(model.evaluate(reshaped_X_test, y_test)[1])

    #########################################################
    ########### Converting and storing model ################
    #########################################################

    # Save trained model
    saved_model = model_directory + "/activity_recognition_" + fps + "fps_model_" + str_test_acc + "_test_acc.h5"
    model.save(saved_model)

    # Convert trained model into CoreML
    # coreml_model = coremltools.converters.keras.convert(saved_model, 
                                                        # class_labels=LABELS, 
                                                        # input_names=['pose'])

    # Store CoreML model 
    # coreml_model.save(model_directory + "/activity_recognition_" + fps + "fps_model_" + str_test_acc + "_test_acc.mlmodel")

    # Delete elemental objects (just to be sure that those are loaded new for each iteration)
    del  X_raw, y_raw # model, coreml_model,