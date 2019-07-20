import random
import math
from load_data import *
from constants import *
from plot_pose_sequence import *

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


def augement_dataset(data):
  for sequence in data:
    sequence.shape
    s2 = move_keypoins_of_sequence(sequence)
    scaled = scale_sequence(s2)
    moved = move_sequence(scaled)
    plot_numpy_sequence(sequence, show=False, skip=2)
    plot_numpy_sequence(s2, color='g', show=False, skip=2)
    plot_numpy_sequence(scaled, color='r', show=False, skip=2)
    plot_numpy_sequence(moved, color='r', skip=2)

  


if __name__ == '__main__':
  raw_data = json_to_raw_data(load_json_data())
  for exercise, data in raw_data.items():
    augement_dataset(data)
