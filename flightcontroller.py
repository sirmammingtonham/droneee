import sys
import os
import time
import cv2

import numpy as np
import tensorflow as tf

from pymultiwii import MultiWii
from picamera.array import PiRGBArray
from picamera import PiCamera

class FlightController:
    def __init__(self):
        self.im_width = 640
        self.im_height = 480
        self.board = MultiWii("COM4")

    def compute_action(self, x1, y1, x2, y2, scale_factor=0.25):
        # define the possible turning and moving action
        self.board.getData(MultiWii.RAW_IMU)
        turning, moving, vertical = self.board.rawIMU

        area = (x2-x1)*(y2-y1)
        center = [(x2+x1)/2, (y2+y1)/2]

        # obtain a x center between 0.0 and 1.0
        normalized_center = center[0] / self.im_width
        # obtain a y center between 0.0 and 1.0
        normalized_center.append(center[1] / self.im_height)

        if normalized_center[0] > 0.6:
            turning += scale_factor
        elif normalized_center[0] < 0.4:
            turning -= scale_factor

        if normalized_center[1] > 0.6:
            vertical += scale_factor
        elif normalized_center[1] < 0.4:
            vertical -= scale_factor

        # if the area is too big move backwards
        if area > 100:
            moving += scale_factor
        elif area < 80:
            moving -= scale_factor

        return [turning, moving, vertical]

    def get_observation(self, velocities): #imu [ax, ay, az]
        self.board.getData(MultiWii.RAW_IMU)
        imu = self.board.rawIMU
        return imu[:3]-velocities

    def send_action(self, motors):
        self.board.sendCMD(16, MultiWii.SET_MOTORS, motors)




