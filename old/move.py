import sys
import os
import time

from pymultiwii import MultiWii
import tensorflow as tf
import numpy as np

class FlightController:
    def __init__(self):

    def compute_action(x1, y1, x2, y2, velocities, scale_factor=0.25):
        # define the possible turning and moving action
        turning, moving, vertical = velocities

        area = (x2-x1)*(y2-y1)
        center = [(x2+x1)/2, (y2+y1)/2]

        # obtain a x center between 0.0 and 1.0
        normalized_center = center[0] / image.width
        # obtain a y center between 0.0 and 1.0
        normalized_center.append(center[1] / image.height)

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

    def main(self):
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


