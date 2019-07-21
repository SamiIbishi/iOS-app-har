import random
import math
from load_data import *
from constants import *
from plot_pose_sequence import *
from auxilary_functions import get_label_vector_by_id

random.seed(1)

class SeqeunceProps():
  def __init__(self, poses):
    self.poses = poses
  
  @property
  def size(self):
    pose = self.poses[0]
    used_parts = ['leftHip', 'leftShoulder']
    idx = [ allowed_parts.index(part) for part in used_parts ]
    dx = abs(pose[2*idx[0]] - pose[2*idx[1]])
    dy = abs(pose[2*idx[0]+1] - pose[2*idx[1]+1])
    return math.sqrt(dx**2 + dy**2)

[140.03913566, 186.82995378, 172.5941639,  178.40273612, 150.41761651,
 190.06063223, 168.70399244, 213.54312128, 146.67211544, 218.50744483,
 147.67043626, 231.55731113, 135.99867144, 232.94808533, 195.00992889,
 224.15592397, 169.30733347, 230.63188007, 192.32495968, 263.42221674,
 165.03749132, 266.43308848, 190.85447417, 310.54015619, 167.94030407,
 307.70646323]


def scale_pose(pose):
  pose = np.array(pose)
  scale = 2 # random.uniform(0.5, 2)
  scale_matrix = np.eye(2) * scale
  coords = np.reshape(pose, (2, 13), order='F')
  scaled = np.dot(scale_matrix, coords)
  return np.reshape(scaled, pose.shape, order='F')

def scale_sequence(sequence):
  return np.array([scale_pose(pose) for pose in sequence])

def move_keypoints_of_pose(pose, delta):
  return pose + [ random.uniform(-delta, delta) for i in range(len(pose))]

def move_keypoins_of_sequence(poses):
  delta = SeqeunceProps(poses).size / 15
  return [move_keypoints_of_pose(pose, delta) for pose in poses]

def move_pose(pose, x, y):
  moved = np.reshape(pose, (13,2)) + np.array([x, y])
  return np.reshape(moved, (26))

def move_sequence(sequence):
  move_x = random.uniform(-200, 200)
  move_y = random.uniform(-200, 200)
  return [move_pose(pose, move_x, move_y) for pose in sequence]

def center_sequence():
  pass # TODO


def augment_sequence(sequence):
  s2 = move_keypoins_of_sequence(sequence)
  scaled = scale_sequence(s2)
  moved = move_sequence(scaled)
  # plot_numpy_sequence(sequence, show=False, skip=2)
  # plot_numpy_sequence(s2, color='g', show=False, skip=2)
  # plot_numpy_sequence(scaled, color='r', show=False, skip=2)
  # plot_numpy_sequence(moved, color='r', skip=2)
  return moved




class BatchGenerator():
  def __init__(self):
    self.data = json_to_raw_data(load_json_data())
  
  exercise_mapping = {
    'left': ['01_short_left_dives', '02_left_dives', '03_long_left_dives'],
    'right': ['04_short_right_dives', '05_right_dives', '06_long_right_dives'],
    'stand_by': ['00_stand_by'],
    'high_catch': ['09_high_catch']
  }
  class_mapping = {
    'stand_by': 0,
    'left': 1,
    'right': 2,
    'high_catch': 3
  }
  
  # left, right, high_catch, stand_by
  def get_batch(self, exercise, size=1000):
    sequences = []
    for ex in self.exercise_mapping[exercise]:
      sequences += (self.data[ex])
    # todo radomize
    batch = [augment_sequence(sequences[i % len(sequences)]) for i in range(size)]
    return batch

  # X, y
  def train_set(self, size_per_class=10000):
    X = []
    y = []
    for exercise in self.exercise_mapping.keys():
      exercise_batch = self.get_batch(exercise, size=size_per_class)
      y += [self.class_mapping[exercise] for _ in range(len(exercise_batch))]
      X += exercise_batch
    return X, np.array(y)
  
  def test_set(self):
    X = []
    y = []
    for exercise, sequences in self.data.items():
      X += sequences
      y += [get_label_vector_by_id(exercise.split('_')[0]) for _ in range(len(sequences))]
      return X, y

  
    


if __name__ == '__main__':
  
  bg = BatchGenerator()
  X, y = bg.test_set()
  # X, y = bg.train_set()
  print(len(X))
  print(len(y))
  print(y[0])
  print(y[500])
  # bg.get_batch(list(raw_data.keys())[0], size=1000)
  
