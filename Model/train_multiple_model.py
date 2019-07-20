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
    "LONG_LEFT_DIVE",
    "RIGHT_DIVE",
    "LONG_RIGHT_DIVE",
    "HIGH_CATCH",
]

num_classes = len(LABELS)

# Path to stored keypoints
data_path = "./stored_data/"

# Framrates/Poses-per-sequence: '5' / '10' / '15' / '30' / '45' / '60' / '75' / '90'
# Load data from .npy files 
# X = [samples, timesteps, features]
# y = [samples, labels]
X_5fps, y_5fps = load_stored_data(dir_path=data_path, num_poses='5')
X_10fps, y_10fps = load_stored_data(dir_path=data_path, num_poses='10')
X_15fps, y_15fps = load_stored_data(dir_path=data_path, num_poses='15')
X_30fps, y_30fps = load_stored_data(dir_path=data_path, num_poses='30')
X_45fps, y_45fps = load_stored_data(dir_path=data_path, num_poses='45')
X_60fps, y_60fps = load_stored_data(dir_path=data_path, num_poses='60')
X_75fps, y_75fps = load_stored_data(dir_path=data_path, num_poses='75')
X_90fps, y_90fps = load_stored_data(dir_path=data_path, num_poses='90')

# Split dataset into train and test set (and shuffle them)
X_train_5fps, X_test_5fps, y_train_temp_5fps, y_test_temp_5fps = train_test_split(X_5fps, y_5fps, test_size = 0.1, random_state = 42)
X_train_10fps, X_test_10fps, y_train_temp_10fps, y_test_temp_10fps = train_test_split(X_10fps, y_10fps, test_size = 0.1, random_state = 42)
X_train_15fps, X_test_15fps, y_train_temp_15fps, y_test_temp_15fps = train_test_split(X_15fps, y_15fps, test_size = 0.1, random_state = 42)
X_train_30fps, X_test_30fps, y_train_temp_30fps, y_test_temp_30fps = train_test_split(X_30fps, y_30fps, test_size = 0.1, random_state = 42)
X_train_45fps, X_test_45fps, y_train_temp_45fps, y_test_temp_45fps = train_test_split(X_45fps, y_45fps, test_size = 0.1, random_state = 42)
X_train_60fps, X_test_60fps, y_train_temp_60fps, y_test_temp_60fps = train_test_split(X_60fps, y_60fps, test_size = 0.1, random_state = 42)
X_train_75fps, X_test_75fps, y_train_temp_75fps, y_test_temp_75fps = train_test_split(X_75fps, y_75fps, test_size = 0.1, random_state = 42)
X_train_90fps, X_test_90fps, y_train_temp_90fps, y_test_temp_90fps = train_test_split(X_90fps, y_90fps, test_size = 0.1, random_state = 42)

# Get one hot vector from labels
y_train_5fps = to_categorical(y_train_temp_5fps, num_classes=num_classes)
y_train_10fps = to_categorical(y_train_temp_10fps, num_classes=num_classes)
y_train_15fps = to_categorical(y_train_temp_15fps, num_classes=num_classes)
y_train_30fps = to_categorical(y_train_temp_30fps, num_classes=num_classes)
y_train_45fps = to_categorical(y_train_temp_45fps, num_classes=num_classes)
y_train_60fps = to_categorical(y_train_temp_60fps, num_classes=num_classes)
y_train_75fps = to_categorical(y_train_temp_75fps, num_classes=num_classes)
y_train_90fps = to_categorical(y_train_temp_90fps, num_classes=num_classes)

y_test_5fps = to_categorical(y_test_temp_5fps, num_classes=num_classes)
y_test_10fps = to_categorical(y_test_temp_10fps, num_classes=num_classes)
y_test_15fps = to_categorical(y_test_temp_15fps, num_classes=num_classes)
y_test_30fps = to_categorical(y_test_temp_30fps, num_classes=num_classes)
y_test_45fps = to_categorical(y_test_temp_45fps, num_classes=num_classes)
y_test_60fps = to_categorical(y_test_temp_60fps, num_classes=num_classes)
y_test_75fps = to_categorical(y_test_temp_75fps, num_classes=num_classes)
y_test_90fps = to_categorical(y_test_temp_90fps, num_classes=num_classes)

# Plot some debugging information about the data
print("_______________________________________")
print("---------------------------------------")
print("Bsic information regardin the data:")
print("_______________________________________")
print("Trainings data...")
print("5fps: input shape=" + str(X_train_5fps.shape) + ", target shape=" + str(y_train_5fps.shape))
print("10fps: input shape=" + str(X_train_10fps.shape) + ", target shape=" + str(y_train_10fps.shape))
print("15fps: input shape=" + str(X_train_15fps.shape) + ", target shape=" + str(y_train_15fps.shape))
print("30fps: input shape=" + str(X_train_30fps.shape) + ", target shape=" + str(y_train_30fps.shape))
print("45fps: input shape=" + str(X_train_45fps.shape) + ", target shape=" + str(y_train_45fps.shape))
print("60fps: input shape=" + str(X_train_60fps.shape) + ", target shape=" + str(y_train_60fps.shape))
print("75fps: input shape=" + str(X_train_75fps.shape) + ", target shape=" + str(y_train_75fps.shape))
print("90fps: input shape=" + str(X_train_90fps.shape) + ", target shape=" + str(y_train_90fps.shape))
print("")
print("Test data...")
print("5fps: input shape=" + str(X_test_5fps.shape) +", target shape=" + str(y_test_5fps.shape))
print("")
print("---------------------------------------")
print("_______________________________________")


list_of_data = [ (X_train_5fps, X_test_5fps, y_train_5fps, y_test_5fps, "5"), 
                    (X_train_10fps, X_test_10fps, y_train_10fps, y_test_10fps, "10"),
                    (X_train_15fps, X_test_15fps, y_train_15fps, y_test_15fps, "15"),
                    (X_train_30fps, X_test_30fps, y_train_30fps, y_test_30fps, "30"),
                    (X_train_45fps, X_test_45fps, y_train_45fps, y_test_45fps, "45"),
                    (X_train_60fps, X_test_60fps, y_train_60fps, y_test_60fps, "60"),
                    (X_train_75fps, X_test_75fps, y_train_75fps, y_test_75fps, "75"), 
                    (X_train_90fps, X_test_90fps, y_train_90fps, y_test_90fps, "90")
                ]

#########################################################
########### Callbacks ###################################
#########################################################

# Get time stamp 
time_stamp = time_stamp()

# Tensorboard
logdir = './training_history/logs/' + time_stamp + "/"

# Create directory which will contain trained models 
model_directory = "./training_history/saved_models/" + time_stamp + "/"

# Record ressults of multiple model (training)
record_driectory = "./training_history/saved_models/" + time_stamp + "/result_documentation.txt"
if not os.path.exists(record_driectory):
        os.makedirs(record_driectory)

f_results = open("result_documentation.txt","w+") 

#########################################################
########### Constants ###################################
#########################################################

# Training - Hyperparameter  
learning_rate = [0.001, 0.0005, 0.0001]
init = ['glorot_uniform', 'uniform']
optim = [optimizers.RMSprop(lr=learning_rate[1], decay=0.5), 
              optimizers.Adam(lr=learning_rate[1], decay=0.5),
              optimizers.Nadam(lr=learning_rate[1])]
num_epochs = 24
batch_size = 512
dropout = [True, False]

# LSTM Neural Network's internal structure
num_mem_units = 182
num_hidden_units = 96

for (X_train, X_test, y_train, y_test, fps) in list_of_data:

    # Input Data
    n_timesteps = X_train.shape[1] # n-timesteps per series per series
    n_features = X_train.shape[2] # n input parameters per timestep
    input_shape = (n_timesteps, n_features)

    # Ether class weight or sample weights to overcome the unbalanced data (NOT both, select one)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_temp_5fps), y_train_temp_5fps)
    #sample_weights = class_weight.compute_sample_weight('balanced', y)

    # Generate callback list 
    callbacks = get_callbacks(model_directory, logdir + fps, time_stamp)

    #########################################################
    ########### DL Model ####################################
    #########################################################

    # Get compiled model 
    model = create_compiled_model(input_shape=input_shape,
                                    num_classes=num_classes,
                                    num_mem_units=num_mem_units, 
                                    num_hidden_units=num_hidden_units,  
                                    dropout=dropout[0],
                                    init=init[0],
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
    print('Used metrics: ' + str(model.metrics_names))
    print('Evaluation on trainings data' + str(model.evaluate(reshaped_X_train, y_train)))
    print('Evaluation on test data' + str(model.evaluate(reshaped_X_test, y_test)))

    f_results.write("Model: "activity_recognition_model" + fps + "fps.h5", This is line %d\r\n" % (i+1))
    #########################################################
    ########### Converting and storing model ################
    #########################################################

    # Save trained model
    saved_model = model_directory + "/activity_recognition_model" + fps + "fps.h5"
    model.save(saved_model)

    # Convert trained model into CoreML
    coreml_model = coremltools.converters.keras.convert(saved_model, 
                                                        class_labels=LABELS, 
                                                        input_names=['pose'])

    # Store CoreML model 
    coreml_model.save(model_directory + "/activity_recognition_model" + fps + "fps.mlmodel")