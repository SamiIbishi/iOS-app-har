#!/usr/bin/env python

import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger

def get_callbacks(model_directory, logdir, time_stamp, fps):

    ## Check whether the target directoy exist and 
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
        
    ## Create callback for tensorboard/trainings-history with the path to the logs directory
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    ## Create callback which stores model-weights checkpoints
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, min_lr=0.001)
    csv_logger = CSVLogger(model_directory + '/activity_recognition_' + fps + 'fps_model_trainigs.log')

    return [tensorboard, reduce_lr, csv_logger]