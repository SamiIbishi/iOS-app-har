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


#########################################################
########### DATA LOADER #################################
#########################################################

newPath = './Exercises'
singlefilename = 'keypoints_single.json'
filename = 'keypoints.json'
def load_all_data():

  for exercise in full_exercises:

    exPath = full_basePath + exercise
    paths = [x[0] for x in os.walk(exPath) if exPath != x[0] and not 'Videos' in x[0]]
    
    for videoPath in paths:
      path = videoPath + '/' + filename
      
      if os.path.isfile(path) is True:
        new_name = os.path.join(newPath, base, filename)

      elif os.path.isfile(videoPath + '/' filename):
        with open(videoPath + '/' + filename) as f:
          

  return trainings_data, label_data
