import sys
import os
import time
import cv2

import numpy as np
import tensorflow as tf

from pymultiwii import MultiWii
from flightcontroller import FlightController
from picamera.array import PiRGBArray
from picamera import PiCamera

from utils import label_map_util
from utils import visualization_utils as vis_util

'''
def sendToBoard(roll, pitch, yaw, throttle):
    motor = []
    for i in range(4):
        motor[i] = throttle

    motor[0] += (pitch + yaw - roll)
    motor[1] += (pitch - yaw + roll)
    motor[2] += (-pitch - yaw - roll)
    motor[3] += (-pitch + yaw + roll)
'''


def main():
    fc = FlightController()
    camera = PiCamera()
    pi = train(num_timesteps=1, seed=args.seed, env_id=args.env)
    U.load_state(args.model_path)

    camera.resolution = (640, 480)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = frame1.array
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        target_velocities = fc.compute_action(*boxes)

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        state = fc.get_observation(target_velocities)
        motor_pwms = pi.act(stochastic=False, ob=state)[0]
        fc.send_action(motor_pwms)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)
    camera.close()


def main():
    parser = argparse.ArgumentParser()
    logger.configure()
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'humanoid_policy'))
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--num-timesteps', type=int, default=1e7)

    args = parser.parse_args()
    print(" Making env=", args.env)
    # construct the model object, load pre-trained model and render
    pi = train(num_timesteps=1, seed=args.seed, env_id=args.env)
    U.load_state(args.model_path)

    env = gym.make(args.env)
    ob = env.reset()
    actuals = []
    desireds = []
    while True:
        desired = compute_action() #env.omega_target
        board.getData(MultiWii.RAW_IMU)
        actual = board.rawIMU #env.omega_actual
        actuals.append(actual)
        desireds.append(desired)
        print("sp=", desired, " rate=", actual)
        action = pi.act(stochastic=False, ob=ob)[0]
        ob, _, done, _ = env.step(action)
        if done:
            break
    plot_step_response(np.array(desireds), np.array(actuals))
