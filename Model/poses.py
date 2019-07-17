from enum import Enum
from typing import List


class Part(Enum):
    LEFT_ANKLE = "leftAnkle"
    LEFT_EAR = "leftEar"
    LEFT_ELBOW = "leftElbow"
    LEFT_EYE = "leftEye"
    LEFT_HIP = "leftHip"
    LEFT_KNEE = "leftKnee"
    LEFT_SHOULDER = "leftShoulder"
    LEFT_WRIST = "leftWrist"
    NOSE = "nose"
    RIGHT_ANKLE = "rightAnkle"
    RIGHT_EAR = "rightEar"
    RIGHT_ELBOW = "rightElbow"
    RIGHT_EYE = "rightEye"
    RIGHT_HIP = "rightHip"
    RIGHT_KNEE = "rightKnee"
    RIGHT_SHOULDER = "rightShoulder"
    RIGHT_WRIST = "rightWrist"


class Position:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class Keypoint:
    score: float
    part: Part
    position: Position

    def __init__(self, score: float, part: Part, position: Position) -> None:
        self.score = score
        self.part = part
        self.position = position


class PosesValue:
    score: float
    keypoints: List[Keypoint]

    def __init__(self, score: float, keypoints: List[Keypoint]) -> None:
        self.score = score
        self.keypoints = keypoints