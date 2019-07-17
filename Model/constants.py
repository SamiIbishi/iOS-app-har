#!/usr/bin/env python

import os
import json
import numpy as np
from matplotlib import pyplot as plt

def get_one_hot(length, idx):
  vec = np.zeros(length, dtype=int)
  vec[idx] = 1
  return vec 

#########################################################
########### Constants ###################################
#########################################################

allowed_parts = [
  'nose',
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
]

# Paths 
full_basePath = '../Data/TrainingsVideo/Exercises/'
full_exercises = [
  '00_stand_by',
  '01_short_left_dives',
  '02_left_dives',
  '03_long_left_dives',
  '04_short_right_dives',
  '05_right_dives',
  '06_long_right_dives',
  #'07_low_catch',
  #'08_middle_catch',
  '09_high_catch',
]

# Label-IDs
STAND_BY_ID = '00'
SHORT_LEFT_DIVES_ID = '01'
LEFT_DIVES_ID = '02'
LONG_LEFT_DIVES_ID = '03'
SHORT_RIGHT_DIVES_ID = '04'
RIGHT_DIVES_ID = '05'
LONG_RIGHT_DIVES_ID = '06'
LOW_CATCH_ID = '07'
MIDDLE_ID = '08'
HIGH_ID = '09'

target_labels = {
  STAND_BY_ID: 0,
  SHORT_LEFT_DIVES_ID:  1,
  LEFT_DIVES_ID:  1,
  LONG_LEFT_DIVES_ID: 2,
  SHORT_RIGHT_DIVES_ID: 3,
  RIGHT_DIVES_ID: 3,
  LONG_RIGHT_DIVES_ID: 4,
  #LOW_CATCH_ID: 0,
  #MIDDLE_ID: 0,
  HIGH_ID: 5,
}

# exercise_labels = {
#   STAND_BY_ID:  label_standy_by,
#   SHORT_LEFT_DIVES_ID:  label_short_left_dive,
#   LEFT_DIVES_ID:  label_left_dive,
#   LONG_LEFT_DIVES_ID: label_long_left_dive,
#   SHORT_RIGHT_DIVES_ID: label_short_right_dive,
#   RIGHT_DIVES_ID: label_right_dive,
#   LONG_RIGHT_DIVES_ID: label_long_right_dive,
#   LOW_CATCH_ID: label_low_catch,
#   MIDDLE_ID: label_middle_catch,
#   HIGH_ID: label_high_catch,
# }

# target_labels = {
#   STAND_BY_ID: 0,
#   SHORT_LEFT_DIVES_ID:  1,
#   LEFT_DIVES_ID:  2,
#   LONG_LEFT_DIVES_ID: 3,
#   SHORT_RIGHT_DIVES_ID: 4,
#   RIGHT_DIVES_ID: 5,
#   LONG_RIGHT_DIVES_ID: 6,
#   LOW_CATCH_ID: 7,
#   MIDDLE_ID: 8,
#   HIGH_ID: 9,
# }

labels = {
  STAND_BY_ID: 'STAND BY',
  SHORT_LEFT_DIVES_ID: 'LEFT DIVE',
  LEFT_DIVES_ID:  'LEFT DIVE',
  LONG_LEFT_DIVES_ID: 'LONG LEFT DIVE',
  SHORT_RIGHT_DIVES_ID: 'RIGHT DIVE',
  RIGHT_DIVES_ID: 'RIGHT DIVE',
  LONG_RIGHT_DIVES_ID: 'LONG RIGHT DIVE',
  #LOW_CATCH_ID: 'LOW CATCH',
  #MIDDLE_ID: 'MIDDLE CATCH',
  HIGH_ID: 'HIGH CATCH'
}
