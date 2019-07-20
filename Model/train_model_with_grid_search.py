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
labels = [    
    "STAND_BY",
    "LEFT_DIVE",
    "LONG_LEFT_DIVE",
    "RIGHT_DIVE",
    "LONG_RIGHT_DIVE",
    "HIGH_CATCH",
]

# Path to stored keypoints
data_path = "./stored_data/"

# Framrates/Poses-per-sequence: '5' / '10' / '15' / '30' / '45' / '60' / '75' / '90'
fps = '90'

# Load data from .npy files 
# X = [samples, timesteps, features]
# y = [samples, labels]
X_raw, y_raw = load_stored_data(dir_path=data_path, num_poses=fps)

# Augment data
X, y = standardize_data(X_raw, y_raw)
#X, y = normalize_data(X_raw, y_raw)

# Split dataset into train and test set (and shuffle them)
X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Get one hot vector from labels
y_train = to_categorical(y_train_temp, num_classes=6)
y_test = to_categorical(y_test_temp, num_classes=6)

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
num_classes = len(labels)
num_mem_units = [48, 96, 182]
num_hidden_units = [48, 96, 182]

# Training - Hyperparameter  
learning_rate = [0.001, 0.0005, 0.0001]
init = ['glorot_uniform', 'uniform']
optim = [optimizers.RMSprop(lr=learning_rate[0], decay=0.5), 
              optimizers.Adam(lr=learning_rate[0], decay=0.5),
              optimizers.Nadam(lr=learning_rate[0])]
num_epochs = 50
batch_size = 24
dropout = [True, False]

# Ether class weight or sample weights to overcome the unbalanced data (NOT both, select one)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_temp), y_train_temp)
#sample_weights = class_weight.compute_sample_weight('balanced', y)

#########################################################
########### DL Model ####################################
#########################################################

# Parameter grid 
param_grid = dict(init=init, optimizer=optim, dropout=dropout)

# Get compiled model 
model = KerasClassifier(build_fn=create_compiled_model, input_shape=input_shape, epochs=num_epochs,
                        batch_size=batch_size, verbose=0)

# Reshape model fo CoreML model 
reshaped_X_train = X_train.reshape(X_train.shape[0],-1)
reshaped_X_test = X_test.reshape(X_test.shape[0],-1)

# Generate grid search instance
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,
                    n_jobs=1, refit=True, verbose=2)

# Train model on trainings data 
grid_result = grid_search.fit(reshaped_X_train, y_train)

#########################################################
########### Converting and storing model ################
#########################################################

# Get time stamp 
time_stamp = time_stamp()

str_cv_result = 'time_stamp/results_' + fps + 'fps.csv'
cv_result = pd.DataFrame(grid_search.cv_results_)
cv_result.sort_values('rank_test_score')
cv_result.to_csv(r'./grid_search_' + str_cv_result)

str_best_param = 'time_stamp/best_param_' + fps + 'fps.csv'
cv_best_param = pd.DataFrame(grid_search.best_params_, index=[0])
cv_best_param.to_csv(r'/grid_search_' + str_best_param)