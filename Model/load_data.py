#!/usr/bin/env python

import os
import json
import numpy as np
from datetime import datetime
from constants import allowed_parts, full_basePath, full_exercises, target_labels
from collections import OrderedDict
from auxilary_functions import get_label_vector_by_id

#########################################################
########### SEQUENCE LOADER #############################
#########################################################

def extraxt_sequences_from_data(data, single=False):
  # [images, pose] - List of list which contain all poses of one exercise sequence
  sequence = [] # size/length: (numOfAllImagesInOneExercise, numOfBodyparts=24)

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
      sequence.append(bodyparts)

  return sequence, label

#########################################################
########### DATA LOADER #################################
#########################################################

def load_and_store_data(num_poses=90, sample_equal_dist=True, store_data=True, path = './', X_filename_prefix='X_data_90p', y_filename_prefix='y_data_90p'):
  list_of_trainings_data = []
  list_of_labels = []

  for exercise in full_exercises:
    # get all paths 
    exPath = full_basePath + exercise
    paths = [x[0] for x in os.walk(exPath) if exPath != x[0] and not 'Videos' in x[0]]
    
    for videoPath in paths:
      path = videoPath + '/keypoints_single.json'

      # get the "cleared" keypoints_single json file if it exist
      if os.path.isfile(path) is True:
        with open(path) as f:
          data = json.load(f, object_pairs_hook=OrderedDict)
          if data:
            sequence, label = extraxt_sequences_from_data(data=data, single=True)
            
            if len(sequence) >= 30:
              list_of_trainings_data.append(sequence)
              list_of_labels.append(label)

      # get keypoint files which can include multiple skeletons/poses per frame
      # we select always the first pose for each frame (problamtic: could be the wrong one)
      elif os.path.isfile(videoPath + '/keypoints.json'):
        with open(videoPath + '/keypoints.json') as f:
          data = json.load(f, object_pairs_hook=OrderedDict)
          del data['config']
          if data:
            sequence, label = extraxt_sequences_from_data(data=data)

            # drop all sequence which are shorter than 15 poses 
            if len(sequence) >= 30:
              list_of_trainings_data.append(sequence)
              list_of_labels.append(label)

  # equyalize length of each sequence and convert them to numpy arrays 
  list_of_arr_of_trainings_data = []
  for sequence in list_of_trainings_data:
    # get once the sequence length
    seq_len = len(sequence)

    if seq_len > num_poses:
      if sample_equal_dist:
        # get index of num_poses evenly spaced out elements from sequence
        # https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
        idx = np.round(np.linspace(0, seq_len-1, num_poses)).astype(int)
        list_of_arr_of_trainings_data.append(np.array(sequence)[idx])
      else:
        # current sequence in list is replaced by sequence consisting 
        # with only the last 90 poses of the original sequence 
        list_of_arr_of_trainings_data.append(np.array(sequence)[-num_poses:])  
    elif seq_len < num_poses:
      pre_missing_len = num_poses - seq_len
      # replicate the first element n-times
      temp = np.array(sequence[0])
      pre_arr = np.tile(temp, (pre_missing_len, 1))
      # insert application in front of the existing sequence such that sequnce length is equal to num_poses
      list_of_arr_of_trainings_data.append(np.array(np.concatenate((pre_arr, sequence))))
    else:
      # sequence has already the target length
      list_of_arr_of_trainings_data.append(np.array(sequence))
  
  X_data = np.array(list_of_arr_of_trainings_data)
  y_data = np.array(list_of_labels)

  print(X_data.shape)
  if store_data:
    np.save('./stored_data/4_classes/' + X_filename_prefix + '.npy', X_data)
    np.save('./stored_data/4_classes/' + y_filename_prefix + '.npy', y_data)

  # Convert list of trainings data to numpy array 
  #trainings_data = np.asarray(full_padded_sequences, dtype=float)
  #print('Trainings data: ' + str(trainings_data.shape))

  # Convert list of trainings data to numpy array
  #label_data = np.asarray(list_of_labels)
  #print('Label data: ' + str(label_data.shape))

  return X_data, y_data

def load_stored_data(dir_path='./stored_data/6_classes', num_poses="90"):
  full_X_path = dir_path + 'X_data_' + num_poses + 'p.npy'
  full_y_path = dir_path + 'y_data_' + num_poses + 'p.npy'
    
  return np.load(full_X_path), np.load(full_y_path)