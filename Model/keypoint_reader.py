#!/usr/bin/env python

import os
import json
from matplotlib import pyplot as plt
import math
import pysftp
from io import BytesIO

# exercises = ['left_dives', 'right_dives']
basePath = '/home/sami/Desktop/University/GOALPLAY/Data/TrainingsVideo/Exercises/'
exercises = [
  '00_stand_by',
  '01_short_left_dives',
  '02_left_dives',
  '03_long_left_dives',
  '04_short_right_dives',
  '05_right_dives',
  '06_long_right_dives',
  '07_low_catch',
  '08_middle_catch'
  ]
# basePath = '/home/rmmm/Trainings_Data/Exercises/'
host = 'ios19goalplay.ase.in.tum.de'
username = 'rmmm'
password = 'last38*moose'

image_width = 801
image_height = 450

# with pysftp.Connection(host=host, username=username, password=password) as sftp:
#   print('connected')
#   img = BytesIO()
#   sftp.getfo('/home/rmmm/Trainings_Data/Exercises/05_right_dives/05_o_0044/05_o_0044_00078.jpg', img)
#   plt.imshow(img, 'JPG')
#   plt.show()  

colors = ['b', 'g', 'r', 'y', 'c', 'm', 'k']

def colors_gen():
  for c in colors:
    yield c
  while 1:
    yield colors[-1]

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

skelton_lines = [
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

def get_poses():
  dataset = { key: {} for key in exercises }
  for exercise in exercises:
    exPath = basePath + exercise
    paths = [x[0] for x in os.walk(exPath) if exPath != x[0] and not 'Videos' in x[0]]
    for videoPath in paths:
      path = videoPath + '/keypoints.json'
      if os.path.isfile(path) is True:
        with open(path) as f:
          data = json.load(f)
          del data['config']
          dataset[exercise][videoPath] = data

  return dataset

def get_coords_from_json_pose(json_pose):
  pose = {}
  for keypoint in json_pose['keypoints']:
    if keypoint['part'] in allowed_parts:
      pose[keypoint['part']] = {
        'x': keypoint['position']['x'],
        'y': keypoint['position']['y'],
      }
      # X.append(keypoint['position']['x'])
      # Y.append(keypoint['position']['y'])
  return pose

def distance_of_pose(pose1, pose2):
  if pose1 is None or pose2 is None:
    return None
  pose1 = get_coords_from_json_pose(pose1)
  pose2 = get_coords_from_json_pose(pose2)
  distances = []
  for key in pose1.keys():
    dx = abs(pose1[key]['x'] - pose2[key]['x'])
    dy = abs(pose1[key]['y'] - pose2[key]['y'])
    dis = math.sqrt(dx**2 + dy**2)
    distances.append(dis)
  if len(distances) == 0:
    return None
  else:
    return sum(distances) / len(distances)
  

def draw_skeleton(pose_json, color='b'):
  pose = get_coords_from_json_pose(pose_json)
  for (left, right) in skelton_lines:
    try:
      plt.plot(
        [pose[left]['x'], pose[right]['x']],
        [pose[left]['y'], pose[right]['y']],
        color=color,
        linewidth=.5
      )
    except:
      pass

def plot_poses(poses, image_path):
  plt.ylim(0, image_height)
  plt.xlim(0, image_width)
  old = None
  for pose in poses:
    px = []
    py = []
    im = plt.imread(image_path)
    implot = plt.imshow(im)
    for keypoint in pose['keypoints']:
      px.append(keypoint['position']['x'])
      py.append(keypoint['position']['y'])
    # plt.scatter(px, py)
    draw_skeleton(pose)
    # print(distance_of_pose(old, pose))
    old = pose
  plt.show()

def last_pose_of_movement(movement):
  last_img = sorted(movement.keys())[-1]
  # print(last_img)
  return movement[last_img]

def clean_keypoints(path):
  with open(path + '/keypoints.json') as f:
    data = json.load(f)
    del data['config']
    history = []
    for key in sorted(data.keys()):
      history_copy = [ elem for elem in history]
      for new_pose in data[key]:
        # print("len history", len(history))
        diff = []
        for movement in history_copy:
          # draw_skeleton(new_pose)
          # draw_skeleton(last_pose_of_movement(movement), 'g')
          dis = distance_of_pose(new_pose, last_pose_of_movement(movement))
          # print(dis)
          # plt.ylim(0, image_height)
          # plt.xlim(0, image_width)
          # plt.show()
          diff.append(dis) 
        # print(diff)
        if len(diff) > 0 and min(diff) < 40:
          idx_min = diff.index(min(diff))
          history[idx_min][key] = new_pose
        else: 
          history.append({key: new_pose})
      
    return history
        

IGNORE_ERROR = False

def clean_dataset():
  for exercise in exercises:
    exPath = basePath + exercise

    paths = [x[0] for x in os.walk(exPath) if exPath != x[0] and not 'Videos' in x[0]]
    
    i = 0

    for video_path in paths:
      
      path = video_path + '/keypoints.json'
      image_path = video_path + '/' + video_path.split('/')[-1] + '_00001.jpg'
      single_path = video_path + '/keypoints_single.json'

      if os.path.isfile(path) is True and os.path.isfile(single_path) is False and \
        (IGNORE_ERROR is False or os.path.isfile(video_path + 'error.txt') is False):

        try:
          im = plt.imread(image_path)
        except:
          continue
        
        d = clean_keypoints(video_path)
        # print(d)

        plt.show()
        print(video_path)

        for (i, movement), c in zip(enumerate(d), colors_gen()):
          for key, pose in list(movement.items())[::2]:
            draw_skeleton(pose, color=c)
        
        plt.ylim(0, image_height)
        plt.xlim(0, image_width)
        implot = plt.imshow(im)
        plt.gca().invert_yaxis()
        plt.show()
        
        # correct_color = input('What color is corect? %s? \n' % ', '.join(colors))
        # # correct_color = 'adsf'
        
        # if correct_color in colors:
        #   idx = colors.index(correct_color)
        #   with open(single_path, 'w') as f:
        #     f.write(json.dumps(d[idx]))
        #     print('%s movement saved' % correct_color)
        # else:
        #   error = input("whats the probelm?\n")
        #   res = None
        #   if error == 'c':
        #     res = 'connect'
        #   elif error == 'i':
        #     res = 'invalid data'
        #   elif error == 'h':
        #     res = 'too hard'
        #   elif error == 'n':
        #     res= 'not enough poses'
        #   else:
        #     res = error
        #   with open(video_path + 'error.txt', 'w') as f:
        #     f.write(res)
        #     print("Error was saved")

def main():
  dataset = get_poses()
  for exercise in dataset.keys():
    for video_path, video in dataset[exercise].items():
      if not '11' in video_path:
        continue
      for image_name, image in video.items():
        print(image_name)
        image_path = video_path + '/' + image_name
        plot_poses(image, image_path)

clean_dataset()
# main()
# get_poses()
