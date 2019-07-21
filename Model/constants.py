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
full_basePath = '../Exercises_Data/'
full_exercises = [
  '00_stand_by',
  '01_short_left_dives',
  '02_left_dives',
  '03_long_left_dives',
  '04_short_right_dives',
  '05_right_dives',
  '06_long_right_dives',
  '09_high_catch',
]
#'07_low_catch',
#'08_middle_catch',

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

exercise_label_map = {

}

target_labels = {
  STAND_BY_ID: 0,
  SHORT_LEFT_DIVES_ID:  1,
  LEFT_DIVES_ID:  1,
  LONG_LEFT_DIVES_ID: 1,
  SHORT_RIGHT_DIVES_ID: 2,
  RIGHT_DIVES_ID: 2,
  LONG_RIGHT_DIVES_ID: 2,
  HIGH_ID: 3,
}

labels = {
  STAND_BY_ID: 'STAND BY',
  SHORT_LEFT_DIVES_ID: 'LEFT DIVE',
  LEFT_DIVES_ID:  'LEFT DIVE',
  LONG_LEFT_DIVES_ID: 'LEFT DIVE',
  SHORT_RIGHT_DIVES_ID: 'RIGHT DIVE',
  RIGHT_DIVES_ID: 'RIGHT DIVE',
  LONG_RIGHT_DIVES_ID: 'RIGHT DIVE',
  HIGH_ID: 'HIGH CATCH'
}
