import os
import numpy as np

from garage.core import Serializable

import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

import itertools

# objects defined in xml
N = 4
objects = [
    "red",    # rgba 1  0  0 1
    "green",  # rgba 0  .8 0 1
    "yellow", # rgba 1  1  0 1
    "blue",   # rgba .3 .3 1 1
]

# Sample Task
TASKS = [
    {'goal_description': [3, 5, 1, 2, 6, 1],
    'target_label': 'red',
    'destination_label': 'green'},
    {'goal_description': [3, 6, 1, 2, 4, 1],
    'target_label': 'green',
    'destination_label': 'yellow'},
    {'goal_description': [3, 4, 1, 2, 7, 1],
    'target_label': 'yellow',
    'destination_label': 'blue'},
    {'goal_description': [3, 7, 1, 2, 5, 1],
    'target_label': 'yellow',
    'destination_label': 'red'}
]

max_sentence_length=6
sentence_code_dim=8

def code_to_one_hot_matrix(code, sentence_code_dim):
    n_vocab = sentence_code_dim
    oh_encoding = np.zeros(
        shape=(len(code), n_vocab),
        dtype=np.float32
    )
    oh_encoding[np.arange(len(code)), code] = 1.
    return oh_encoding

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def dist_point2seg(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)


class MultiPointsReachEnv(mujoco_env.MujocoEnv, utils.EzPickle, Serializable):
    def __init__(
                self,
                max_sentence_length=max_sentence_length,
                sentence_code_dim=sentence_code_dim,
                tasks=TASKS,
                objects=objects,
                is_training=True,
                random_initialize=True, 
                init_qpos=None,
        ):
        self.initialized = False
        self.is_training = True

        # initialize ezpickle
        utils.EzPickle.__init__(self)

        # prev action
        self.prev_action = None


        self.num_objects = N
        self.tasks = tasks
        self.num_tasks = len(tasks)

        self.max_sentence_length = max_sentence_length
        self.code_sentence_dim = sentence_code_dim

        self.random_initialize=random_initialize
        self.init_qpos=init_qpos
        
        # Task Selection
        self.active_task = None
        self._task_selection = itertools.cycle(range(self.num_tasks))

        # Process Objects
        self.objects = objects
        self.object_name_id = {}
        self.id_object_name = {}
        for i, name in enumerate(objects, start=1):
            self.object_name_id[name] = i
            self.id_object_name[i] = name

        self._all_one_hots = None

        # initialize environment
        mujoco_path = os.path.join(os.path.dirname(__file__), 'assets', 'multi_points_pusher.xml')
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, mujoco_path, frame_skip)

        self.initialized = True


    @property
    def all_task_one_hots(self):
        if self._all_one_hots is None:
            self._all_one_hots = [
                code_to_one_hot_matrix(t['goal_description'], self.code_sentence_dim)
                for t in self.tasks
            ]
        return self._all_one_hots

    def reset(self):
        if self.is_training:
            self.active_task = next(self._task_selection)
        return super().reset()

    @property
    def active_task_one_hot_gt(self):
        one_hot = np.zeros(self.num_objects)
        t = 0 if self.active_task is None else self.active_task
        one_hot[t] = 1
        return one_hot

    @property
    def active_task_one_hot(self):
        t = 0 if self.active_task is None else self.active_task
        goal_description = self.tasks[t]["goal_description"]
        # goal_description = self._task_envs[t]._goal_description
        one_hot = code_to_one_hot_matrix(goal_description, self.code_sentence_dim)
        return one_hot

    @property
    def task_space(self):
        n = self.max_sentence_length
        one_hot_ub = np.ones(n) * (self.code_sentence_dim - 1)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    def _loc(self, obj_name):
        return self.get_body_com(obj_name)

    def _loc2(self, obj_name):
        return np.array(self._loc(obj_name))[:2]

    """
    reward for active task to its object
    """
    def step(self, action, debug=False):
        t = 0 if self.active_task is None else self.active_task
        task = self.tasks[t]


        # print object states before stepping
        if debug:
            print("Printing object states.")
            print("Robot Arm: {}".format(self._loc("robot")))
            print("Left Gripper: {}".format(self._loc("l_hand")))
            print("Right Gripper: {}".format(self._loc("r_hand")))
            print("Object 1: {}".format(self._loc("obj1")))
            # print("Goal 1: {}".format(self._loc("goal1")))

        # calculate reward
        reward = 0
        arm_location = self._loc2("robot")

        target_loc = self._loc2("obj{}".format(self.object_name_id[task['target_label']]))
        # destination_loc = self._loc2("obj{}".format(self.object_name_id[task['destination_label']]))

        # reward1: dist form target to dest
        # reward -= np.linalg.norm(destination_loc - target_loc)

        # reward2: arm to target
        reward -= np.linalg.norm(arm_location - target_loc)

        # part 3: arm preferred to be on the ground
        # reward -= max(0, self.get_body_com('robot')[2] - 1.12) * 0.2

        # part 4: penalize action
        # reward -= norm(action) * 0.1

        # part 5: penalize non-smooth action
        # if self.prev_action is not None:
        #     reward -= norm(action - self.prev_action) * 0.1
        # self.prev_action = action

        # step the environment
        self.do_simulation(action, self.frame_skip)

        # get observation
        obs = self._get_obs()


        # get if episode completed or not
        done = False
        if self.is_training:
            if self.initialized and np.linalg.norm(arm_location - target_loc) < 0.15:
                done = True

        # generate info
        info = 'keep going' if reward < -0.1 else 'good job'

        return obs, reward, done, dict(info=info)

    def viewer_setup(self):
        # camera distance
        self.viewer.cam.distance = 2
        
        # viewing angle
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 1.0

    def reset_model(self):
        qpos = self.init_qpos

        if self.random_initialize is True:

            self.prev_action = None

            goal_obj_pos_arr = []

            # init arm
            qpos[0] = np.random.uniform(low=-0.6, high=0.6)
            qpos[1] = np.random.uniform(low=-0.6, high=0.6)
            robot_loc = qpos[:2]

            for i in range(N):
                goal_obj_pos_arr.append(0)
                while True:
                    obj_pos = np.concatenate([
                        np.random.uniform(low=-31.4, high=31.4, size=1),
                        np.random.uniform(low=-0.6, high=0.6, size=1),
                        np.random.uniform(low=-0.6, high=0.6, size=1)
                    ])
                    _, x, y = obj_pos
                    obj_loc = np.array([x, y])
                    if not np.linalg.norm(robot_loc - obj_loc) < 0.15:
                        goal_obj_pos_arr.extend(obj_pos.tolist())
                        break

            qpos[-len(goal_obj_pos_arr):] = goal_obj_pos_arr
        else:
            qpos = self.init_qpos

        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005,
                        size=self.model.nv)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self._loc2('robot'),
            self._loc2('obj1'),
            self._loc2('obj2'),
            self._loc2('obj3'),
            self._loc2('obj4'),
        ])
