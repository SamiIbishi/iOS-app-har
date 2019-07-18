#!/usr/bin/env python

import os
from datetime import datetime
from constants import target_labels

#########################################################
########### Auxilary methods ############################
#########################################################

def time_stamp():
  return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def get_label_vector_by_id(exercise_id):
  return target_labels.get(exercise_id, 0)