#!/usr/bin/env python

import os
import json
import math
import pysftp
import numpy as np
from io import BytesIO
from enum import Enum
from typing import List
from matplotlib import pyplot as plt
from constants import labels, target_labels, allowed_parts
import pandas as pd

image_width = 801
image_height = 450

skelton_lines = [
  ('nose', 'leftShoulder'),
  ('nose', 'rightShoulder'),
  ('leftShoulder', 'rightShoulder'),
  ('leftHip', 'rightHip'),
  ('leftHip', 'leftShoulder'),
  ('rightHip', 'rightShoulder'),
  ('leftElbow', 'leftWrist'),
  ('leftElbow', 'leftShoulder'),
  ('rightElbow', 'rightWrist'),
  ('rightElbow', 'rightShoulder'),
  ('rightHip', 'rightKnee'),
  ('rightAnkle', 'rightKnee'),
  ('leftKnee', 'leftHip'),
  ('leftKnee', 'leftAnkle'),
]

def plot_trainings_data(trainings_data, label_data, sample=0, sequence_length=90, color = 'g', image_width = 801, image_height = 450):
  listOfKeys = [key  for (key, value) in target_labels.items() if value == label_data[sample]]

  plt.title(labels[listOfKeys[0]])
  plt.xlim((0,image_width))
  plt.ylim((image_height,0))

  for n in range(0, trainings_data.shape[1]-1):
      pose = {
        'nose': (trainings_data[sample][n][0], trainings_data[sample][n][1]),
        'leftShoulder': (trainings_data[sample][n][2], trainings_data[sample][n][3]),
        'rightShoulder': (trainings_data[sample][n][4], trainings_data[sample][n][5]),
        'leftElbow': (trainings_data[sample][n][6], trainings_data[sample][n][7]),
        'rightElbow': (trainings_data[sample][n][8], trainings_data[sample][n][9]),
        'leftWrist': (trainings_data[sample][n][10], trainings_data[sample][n][11]),
        'rightWrist': (trainings_data[sample][n][12], trainings_data[sample][n][13]),
        'leftHip': (trainings_data[sample][n][14], trainings_data[sample][n][15]),
        'rightHip': (trainings_data[sample][n][16], trainings_data[sample][n][17]),
        'leftKnee': (trainings_data[sample][n][18], trainings_data[sample][n][19]),
        'rightKnee': (trainings_data[sample][n][20], trainings_data[sample][n][21]),
        'leftAnkle': (trainings_data[sample][n][22], trainings_data[sample][n][23]),
        'rightAnkle': (trainings_data[sample][n][24], trainings_data[sample][n][25]),
      }

      for (left, right) in skelton_lines:
        plt.plot(
            [pose[left][0], pose[right][0]],
            [pose[left][1], pose[right][1]],
            color=color,
            linewidth=.5
      )

def plot_test_data(test_data, predicted_label_data, label_data, sample=0, sequence_length=90, color = 'g', image_width = 801, image_height = 450):
  
  listOfKeysLabel = [key  for (key, value) in target_labels.items() if value == label_data[sample]]
  listOfKeysPred = [key  for (key, value) in target_labels.items() if value == predicted_label_data[sample]]

  if labels[listOfKeysLabel[0]] == labels[listOfKeysPred[0]]:
    eval_color = 'g'
  else:
    eval_color = 'r'
  
  # plt.subplot(211)

  plt.title('Label: ' + labels[listOfKeysLabel[0]] + ', Model-Prediction: ' + labels[listOfKeysPred[0]], color=eval_color)
  plt.xlim((0,image_width))
  plt.ylim((image_height,0))

  for n in range(0, test_data.shape[1]-1):
    pose = {
      'nose': (test_data[sample][n][0], test_data[sample][n][1]),
      'leftShoulder': (test_data[sample][n][2], test_data[sample][n][3]),
      'rightShoulder': (test_data[sample][n][4], test_data[sample][n][5]),
      'leftElbow': (test_data[sample][n][6], test_data[sample][n][7]),
      'rightElbow': (test_data[sample][n][8], test_data[sample][n][9]),
      'leftWrist': (test_data[sample][n][10], test_data[sample][n][11]),
      'rightWrist': (test_data[sample][n][12], test_data[sample][n][13]),
      'leftHip': (test_data[sample][n][14], test_data[sample][n][15]),
      'rightHip': (test_data[sample][n][16], test_data[sample][n][17]),
      'leftKnee': (test_data[sample][n][18], test_data[sample][n][19]),
      'rightKnee': (test_data[sample][n][20], test_data[sample][n][21]),
      'leftAnkle': (test_data[sample][n][22], test_data[sample][n][23]),
      'rightAnkle': (test_data[sample][n][24], test_data[sample][n][25]),
    }

    for (left, right) in skelton_lines:
      plt.plot(
          [pose[left][0], pose[right][0]],
          [pose[left][1], pose[right][1]],
          color=color,
          linewidth=.5
    )

  # plt.subplot(212)
  # df = pd.DataFrame({'Instances':counts_test}, index=labels)
  # ax = df.plot.bar(title ="Test Data", figsize=(9,8), rot=0)
  # plt.show()

def plot_exach_class_of_training_data(trainings_data, label_data, sequence_length=90, color = 'g', image_width = 801, image_height = 450):
  
  fig = plt.figure(figsize=(18, 18), dpi= 80, facecolor='w', edgecolor='k')
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  
  for i in range(1, 6):
    ax = fig.add_subplot(5, 2, i)    

    indexes = np.where(label_data == i-1)[0]
    if indexes.shape is not 0:
      sample = indexes[0]
    else:
      continue
    
    plt.title(labels['0' + str(i-1)])
    plt.xlim((0,image_width))
    plt.ylim((image_height,0))

    for n in range(0, sequence_length-1):
      pose = {
        'nose': (trainings_data[sample][n][0], trainings_data[sample][n][1]),
        'leftShoulder': (trainings_data[sample][n][2], trainings_data[sample][n][3]),
        'rightShoulder': (trainings_data[sample][n][4], trainings_data[sample][n][5]),
        'leftElbow': (trainings_data[sample][n][6], trainings_data[sample][n][7]),
        'rightElbow': (trainings_data[sample][n][8], trainings_data[sample][n][9]),
        'leftWrist': (trainings_data[sample][n][10], trainings_data[sample][n][11]),
        'rightWrist': (trainings_data[sample][n][12], trainings_data[sample][n][13]),
        'leftHip': (trainings_data[sample][n][14], trainings_data[sample][n][15]),
        'rightHip': (trainings_data[sample][n][16], trainings_data[sample][n][17]),
        'leftKnee': (trainings_data[sample][n][18], trainings_data[sample][n][19]),
        'rightKnee': (trainings_data[sample][n][20], trainings_data[sample][n][21]),
        'leftAnkle': (trainings_data[sample][n][22], trainings_data[sample][n][23]),
        'rightAnkle': (trainings_data[sample][n][24], trainings_data[sample][n][25]),
      }

    for (left, right) in skelton_lines:
      ax.plot(
          [pose[left][0], pose[right][0]],
          [pose[left][1], pose[right][1]],
          color=color,
          linewidth=.5
      )