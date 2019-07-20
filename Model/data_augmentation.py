#!/usr/bin/env python

import os
import numpy as np

#########################################################
########### AUGMENT DATA ##############################
#########################################################

def normalize_data(X_raw, y_raw):
  """
  => Nomarlize the feature vector from -1 to 1
  Divide eacht feature vector (26,) by the half of the width and height.
  Then subtract the feature vector by 1 such that the new origin is located in (0,0).
  """
  # Type of augmentation
  data_augmentation = "normalize"

  # Scale feature values 0 to 2
  divisor = np.tile([801, 450], 13) / 2
  X_norm = X_raw / divisor[None, None,:]

  # Shift feature vector by -1 in x and y direction 
  subtractor = np.ones((26,))
  X = np.subtract(X_norm, subtractor)

  return X, y_raw, data_augmentation


def standardize_data(X_raw, y_raw):
  """
  => Standardize the feature vector from -1 to 1
  Divide eacht feature vector (26,) by the half of its standard deviation.
  Then subtract the feature vector by its means toward the new origin in (0,0).
  """
  # Type of augmentation
  data_augmentation = "standardize"

  # Calculate the mean over all feature vectors
  X_mean = X_raw.mean(axis=(0,1))  
  
  # Calculate the standard deviation over all feature vectors
  X_std = X_raw.std(axis=(0,1)) / 2

  # Scale the data by half of its standard deviation and shift all feature vector to its own mean  
  X = np.subtract(X_raw, X_mean) / X_std

  # Get limits/range for plotting purpose
  range_mean = X_mean.reshape((-1,2)).mean(axis=0)  
  range_std = np.max(X_std.reshape((-1,2)), axis=0)

  return X, y_raw, range_mean, range_std, data_augmentation