'''
Created by Jingyun Yang on 11/30/2018.

A simple gym environment debugger that takes an environment with 
continuous control and at least 6 dimensions in action space and
provide simple keyboard control using WASD and 1-8 to play with
the environment.

For our project, this piece of code is created mainly to test the
CustomPush-v0 environment.

To run the code, type `python3 debug_env.py` in terminal under 
directory containing this file.
'''

from embed2learn.envs import MultiPointsPushEnv
from embed2learn.envs import MultiPointsReachEnv

import numpy as np
import argparse
from time import sleep
import keyboard

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default="CustomPush-v0")
    # parser.add_argument('--env', type=str, default="SingleObjectPush-v0")
    # parser.add_argument('--env', type=str, default="MultiPointsPushEnv-v0")
    parser.add_argument('--log_step', type=int, default=10)
    return parser.parse_args()

def main():
	configs = parse_args()
	# env = gym.make(configs.env)
	# env = MultiPointsPushEnv()
	env = MultiPointsReachEnv()
	env.reset()

	t = 0
	rewards = []

	while True:
		action = [0.0] * env.action_space.shape[0]

		# [Dimension 1] x
		if keyboard.is_pressed('a'):
			action[0] = -1
		elif keyboard.is_pressed('d'):
			action[0] = 1

		# [Dimension 2] y
		if keyboard.is_pressed('s'):
			action[1] = -1
		elif keyboard.is_pressed('w'):
			action[1] = 1

		# [Dimension 3] z
		if keyboard.is_pressed('1'):
			action[2] = -1
		elif keyboard.is_pressed('2'):
			action[2] = 1

		# [Dimension 4] rotate
		if keyboard.is_pressed('3'):
			action[3] = -1
		elif keyboard.is_pressed('4'):
			action[3] = 1

		# [Dimension 5] gripper 1
		if keyboard.is_pressed('5'):
			action[4] = -1
		elif keyboard.is_pressed('6'):
			action[4] = 1

		# [Dimension 6] gripper 2
		if keyboard.is_pressed('7'):
			action[5] = -1
		elif keyboard.is_pressed('8'):
			action[5] = 1

        # step environment and keep record of reward
		ob, rew, done, _ = env.step(action)

		rewards.append(rew)

        # render
		env.render()

        # log in console periodically
		if t % configs.log_step == 0:
			mean_reward = np.mean(np.array(rewards)[:-configs.log_step])
			prev_reward = rewards[-1]
			print('[Step {:4d}] Mean Reward {:.3f}; Previous Reward {:.3f}'.format(
				t, mean_reward, prev_reward))

			print('robot location: {}'.format(ob[:2]))
			print('ball location: {}'.format(ob[2:4]))
			print('ball location: {}'.format(ob[4:6]))
			print('ball location: {}'.format(ob[6:8]))
			print('ball location: {}'.format(ob[8:10]))

			print(np.linalg.norm(
				env._loc2("robot") - env._loc2("obj1")
				))



			# print('robot location: {}'.format(ob[10:13]))
			# print('ball location: {}'.format(ob[13:16]))

			# if configs.env == 'ObstaclePush-v0':
				# print('robot rotation: {}'.format(ob[-4]))

		t += 1

		sleep(0.03)

if __name__ == '__main__':
	main()
