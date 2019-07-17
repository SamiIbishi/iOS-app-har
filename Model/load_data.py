#!/usr/bin/env python

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import sequence
from datetime import datetime
from constants import allowed_parts, full_basePath, full_exercises, target_labels
from collections import OrderedDict

#############################################################################################################################################################
########### Full data preprocessing #########################################################################################################################
#############################################################################################################################################################

#########################################################
########### Auxilary methods ############################
#########################################################

def time_stamp():
  return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def get_label_vector_by_id(exercise_id):
  return target_labels.get(exercise_id, 0)

# Sample pose for padding 
diveder_X = 1
diveder_Y = 1
sample_pose = [423.0/diveder_X, 128.0/diveder_Y, 
               436.0/diveder_X, 138.0/diveder_Y, 
               415.0/diveder_X, 137.0/diveder_Y, 
               440.0/diveder_X, 152.0/diveder_Y, 
               411.0/diveder_X, 149.0/diveder_Y, 
               440.0/diveder_X, 162.0/diveder_Y, 
               408.0/diveder_X, 162.0/diveder_Y, 
               428.0/diveder_X, 166.0/diveder_Y, 
               416.0/diveder_X, 166.0/diveder_Y, 
               425.0/diveder_X, 185.0/diveder_Y, 
               417.0/diveder_X, 185.0/diveder_Y, 
               422.0/diveder_X, 205.0/diveder_Y, 
               421.0/diveder_X, 205.0/diveder_Y]

#########################################################
########### SEQUENCE LOADER #############################
#########################################################

def extraxt_full_information_from_data(data, single=False):
  # [images, pose] - List of list which contain all poses of one exercise sequence
  sequences = [] # size/length: (numOfAllImagesInOneExercise, numOfBodyparts=24)

  # Extract label
  label = get_label_vector_by_id(next(iter(data.keys())).split('_')[0])
  
  # Extract trainings data
  for _ , value in data.items():
    bodyparts = []

    # Extract poses from current image 
    if value and single:
      [bodyparts.extend([bodypart['position']['x'], bodypart['position']['y']]) for bodypart in value['keypoints'] if bodypart['part'] in allowed_parts]
    elif value:
      [bodyparts.extend([bodypart['position']['x'], bodypart['position']['y']]) for bodypart in value[0]['keypoints'] if bodypart['part'] in allowed_parts]

    # Add new bodypose/skeleton to seuquence list
    if bodyparts:
      sequences.append(bodyparts)

  return sequences, label

#########################################################
########### DATA LOADER #################################
#########################################################

def load_all_data(num_poses):
  list_of_trainings_data = []
  list_of_labels = []

  for exercise in full_exercises:

    exPath = full_basePath + exercise
    paths = [x[0] for x in os.walk(exPath) if exPath != x[0] and not 'Videos' in x[0]]
    
    for videoPath in paths:
      path = videoPath + '/keypoints_single.json'
      
      if os.path.isfile(path) is True:
        with open(path) as f:
          data = json.load(f, object_pairs_hook=OrderedDict)
          if data:
            sequences, label = extraxt_full_information_from_data(data=data, single=True)
            list_of_trainings_data.append(sequences)
            list_of_labels.append(label)
      elif os.path.isfile(videoPath + '/keypoints.json'):
        with open(videoPath + '/keypoints.json') as f:
          data = json.load(f, object_pairs_hook=OrderedDict)
          del data['config']
          if data:
            sequences, label = extraxt_full_information_from_data(data=data)
            list_of_trainings_data.append(sequences)
            list_of_labels.append(label)

  # Sequence have different amount of timestamps -> post zero padding 
  full_padded_sequences = sequence.pad_sequences(list_of_trainings_data, maxlen=90, padding='pre', truncating='pre', value=sample_pose)
  #print('Full sequences: ' + str(len(full_padded_sequences)))

  # Convert list of trainings data to numpy array 
  trainings_data = np.asarray(full_padded_sequences, dtype=float)
  #print('Trainings data: ' + str(trainings_data.shape))

  # Convert list of trainings data to numpy array
  label_data = np.asarray(list_of_labels)
  #print('Label data: ' + str(label_data.shape))

  return trainings_data, label_data
